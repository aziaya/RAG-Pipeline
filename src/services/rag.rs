//! Retrieval-augmented answering: the single implementation behind POST /chat, the
//! terminal UIs and the CLI.

use crate::{
    models::{
        chat::{ChatAnswer, ChatTurn, RagStage, SourceRef},
        search::Hit,
    },
    observability::metrics::metrics,
    services::{embedder::Embedder, llm::LlmClient, vector_store::VectorStore},
};
use anyhow::Result;
use tracing::{info, info_span, Instrument};

/// Which half of a question failed, so the HTTP layer can pick the right status code.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("retrieval failed: {0:#}")]
    Retrieval(anyhow::Error),
    #[error("{0:#}")]
    Generation(anyhow::Error),
}

#[derive(Clone)]
pub struct Rag {
    embedder: Embedder,
    store: VectorStore,
    llm: LlmClient,
}

impl Rag {
    pub fn new(embedder: Embedder, store: VectorStore, llm: LlmClient) -> Self {
        Self { embedder, store, llm }
    }

    /// Hybrid (keyword + vector) search. `alpha`: 0 = keyword only, 1 = vector only.
    pub async fn search(&self, query: &str, limit: usize, alpha: f32, source: Option<&str>) -> Result<Vec<Hit>> {
        let span = info_span!("search", limit, alpha = alpha as f64, filtered = source.is_some());
        async {
            let v = self.embedder.embed_query(query).await?;
            let hits = self.store.hybrid_search(query, &v, alpha, limit, source).await?;
            metrics().inc(&metrics().searches, 1);
            info!(hits = hits.len(), "hybrid search done");
            Ok(hits)
        }
        .instrument(span)
        .await
    }

    /// Retrieve passages and generate an answer from them.
    pub async fn answer(
        &self,
        question: &str,
        history: &[ChatTurn],
        limit: usize,
        alpha: f32,
        source: Option<&str>,
        on_stage: impl Fn(RagStage),
    ) -> Result<ChatAnswer, RagError> {
        on_stage(RagStage::Searching);
        let hits = self.search(question, limit, alpha, source).await.map_err(RagError::Retrieval)?;
        if hits.is_empty() {
            info!("no passages found, skipping the model");
            return Ok(ChatAnswer {
                answer: "I couldn't find anything relevant in the indexed documents.".into(),
                sources: vec![],
            });
        }
        on_stage(RagStage::Generating { passages: hits.len() });
        let (answer, used) = self.llm.answer(question, history, &hits).await.map_err(RagError::Generation)?;
        metrics().inc(&metrics().chats, 1);
        let sources = hits
            .into_iter()
            .take(used)
            .enumerate()
            .map(|(i, h)| SourceRef {
                n: i + 1,
                source: h.source,
                headings: h.headings,
                chunk_index: h.chunk_index,
                score: h.score,
                text: h.text,
            })
            .collect();
        Ok(ChatAnswer { answer, sources })
    }
}
