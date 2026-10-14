//! Retrieval-augmented answering: the single implementation behind POST /chat, the
//! terminal UIs and the CLI.
//!
//! question -> (rewrite follow-up) -> retrieve (hybrid | semantic | keyword) -> (rerank)
//!          -> generate from the passages

use crate::{
    models::{
        chat::{ChatAnswer, ChatMemory, RagStage, SourceRef},
        search::{Hit, RetrievalOptions, SearchMode},
    },
    observability::metrics::metrics,
    services::{
        embedder::Embedder,
        llm::LlmClient,
        reranker::{self, Reranker},
        vector_store::{Scope, VectorStore},
    },
};
use anyhow::Result;
use tracing::{debug, info, info_span, warn, Instrument};

/// Which half of a question failed, so the HTTP layer can pick the right status code.
#[derive(Debug, thiserror::Error)]
pub enum RagError {
    #[error("retrieval failed: {0:#}")]
    Retrieval(anyhow::Error),
    #[error("{0:#}")]
    Generation(anyhow::Error),
}

/// Retrieved passages plus anything that degraded along the way.
pub struct Retrieval {
    pub hits: Vec<Hit>,
    pub warnings: Vec<String>,
}

/// Result of folding old turns into the summary: replace the first `dropped` turns with it.
pub struct Compaction {
    pub summary: String,
    pub dropped: usize,
}

/// Outcome of everything before generation.
enum Prep {
    /// Nothing relevant was found; this is the final reply.
    Empty(ChatAnswer),
    Ready { hits: Vec<Hit>, searched_for: Option<String>, warnings: Vec<String> },
}

fn build_answer(hits: Vec<Hit>, used: usize, answer: String, searched_for: Option<String>, warnings: Vec<String>) -> ChatAnswer {
    metrics().inc(&metrics().chats, 1);
    let sources = hits
        .into_iter()
        .take(used)
        .enumerate()
        .map(|(i, h)| SourceRef {
            n: i + 1,
            source: h.source,
            collection: h.collection,
            headings: h.headings,
            chunk_index: h.chunk_index,
            score: h.score,
            rerank_score: h.rerank_score,
            text: h.text,
        })
        .collect();
    ChatAnswer { answer, sources, searched_for, warnings }
}

#[derive(Clone)]
pub struct Rag {
    embedder: Embedder,
    store: VectorStore,
    llm: LlmClient,
    reranker: Reranker,
}

impl Rag {
    pub fn new(embedder: Embedder, store: VectorStore, llm: LlmClient, reranker: Reranker) -> Self {
        Self { embedder, store, llm, reranker }
    }

    /// Passages for `query` under the given options (no progress callbacks).
    pub async fn search(&self, query: &str, o: &RetrievalOptions) -> Result<Vec<Hit>> {
        Ok(self.retrieve(query, o, &|_| {}).await?.hits)
    }

    /// Retrieve, then optionally rerank. A reranker that is missing or failing never fails the
    /// search: the retrieval order is kept and a warning is returned.
    pub async fn retrieve(&self, query: &str, o: &RetrievalOptions, on_stage: &impl Fn(RagStage)) -> Result<Retrieval> {
        let o = o.clone().sanitized();
        let span = info_span!("search", mode = o.mode.label(), top_k = o.top_k, rerank = o.rerank, collections = o.collections.len(), filtered = o.source.is_some());
        async {
            let scope = Scope { collections: o.collections.clone(), source: o.source.clone() };
            let limit = o.fetch_limit();
            let mut hits = match o.mode {
                SearchMode::Keyword => self.store.keyword_search(query, limit, &scope).await?,
                SearchMode::Hybrid => {
                    let v = self.embedder.embed_query(query).await?;
                    self.store.hybrid_search(query, &v, o.alpha, limit, &scope).await?
                }
                SearchMode::Semantic => {
                    let v = self.embedder.embed_query(query).await?;
                    self.store.semantic_search(&v, limit, &scope).await?
                }
            };
            metrics().inc(&metrics().searches, 1);
            info!(hits = hits.len(), mode = o.mode.label(), "retrieval done");

            let mut warnings = vec![];
            if o.rerank && hits.len() > 1 {
                if !self.reranker.configured() {
                    warn!("reranking is on but RERANK_BASE_URL is not set; keeping retrieval order");
                    warnings.push("reranking is on but no reranker is configured (RERANK_BASE_URL); used retrieval order".to_string());
                    hits.truncate(o.top_k);
                } else {
                    on_stage(RagStage::Reranking { candidates: hits.len() });
                    match self.reranker.score(query, &hits).await {
                        Ok(scores) => hits = reranker::apply(hits, &scores, o.rerank_weight, o.top_k),
                        Err(e) => {
                            warn!(error = %format!("{e:#}"), "reranking failed; keeping retrieval order");
                            warnings.push(format!("reranker unavailable ({e:#}); used retrieval order"));
                            hits.truncate(o.top_k);
                        }
                    }
                }
            } else {
                hits.truncate(o.top_k);
            }
            Ok(Retrieval { hits, warnings })
        }
        .instrument(span)
        .await
    }

    /// Everything before generation: rewrite a follow-up, retrieve, rerank.
    async fn prepare(
        &self,
        question: &str,
        memory: &ChatMemory,
        o: &RetrievalOptions,
        rewrite: bool,
        on_stage: &impl Fn(RagStage),
    ) -> Result<Prep, RagError> {
        let mut warnings = vec![];
        let mut searched_for = None;
        let mut query = question.to_string();

        if rewrite && !memory.is_empty() {
            on_stage(RagStage::Rewriting);
            match self.llm.rewrite_query(question, memory).await {
                Ok(q) if q != question => {
                    info!(from = question, to = %q, "follow-up rewritten for retrieval");
                    searched_for = Some(q.clone());
                    query = q;
                }
                Ok(_) => debug!("question is already standalone"),
                Err(e) => {
                    warn!(error = %format!("{e:#}"), "follow-up rewrite failed; searching with the original wording");
                    warnings.push(format!("follow-up rewrite failed ({e:#}); searched with your wording"));
                }
            }
        }

        on_stage(RagStage::Searching);
        let r = self.retrieve(&query, o, on_stage).await.map_err(RagError::Retrieval)?;
        warnings.extend(r.warnings);
        if r.hits.is_empty() {
            info!("no passages found, skipping the model");
            return Ok(Prep::Empty(ChatAnswer {
                answer: "I couldn't find anything relevant in the indexed documents.".into(),
                sources: vec![],
                searched_for,
                warnings,
            }));
        }
        Ok(Prep::Ready { hits: r.hits, searched_for, warnings })
    }

    /// Retrieve passages and generate an answer. For follow-ups (non-empty memory) and
    /// `rewrite == true`, the question is first rewritten into a standalone search query.
    pub async fn answer(
        &self,
        question: &str,
        memory: &ChatMemory,
        o: &RetrievalOptions,
        rewrite: bool,
        on_stage: impl Fn(RagStage),
    ) -> Result<ChatAnswer, RagError> {
        match self.prepare(question, memory, o, rewrite, &on_stage).await? {
            Prep::Empty(a) => Ok(a),
            Prep::Ready { hits, searched_for, warnings } => {
                on_stage(RagStage::Generating { passages: hits.len() });
                let (answer, used) = self.llm.answer(question, memory, &hits).await.map_err(RagError::Generation)?;
                Ok(build_answer(hits, used, answer, searched_for, warnings))
            }
        }
    }

    /// Same as [`answer`](Self::answer) but streams the text: `on_token` receives each piece as
    /// the model produces it and returns `false` to stop (e.g. the client disconnected).
    pub async fn answer_stream(
        &self,
        question: &str,
        memory: &ChatMemory,
        o: &RetrievalOptions,
        rewrite: bool,
        on_stage: impl Fn(RagStage),
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<ChatAnswer, RagError> {
        match self.prepare(question, memory, o, rewrite, &on_stage).await? {
            Prep::Empty(a) => Ok(a),
            Prep::Ready { hits, searched_for, warnings } => {
                on_stage(RagStage::Generating { passages: hits.len() });
                let (answer, used) = self.llm.answer_stream(question, memory, &hits, on_token).await.map_err(RagError::Generation)?;
                Ok(build_answer(hits, used, answer, searched_for, warnings))
            }
        }
    }

    /// When the verbatim history has grown long, summarize the older turns. `None` if not needed.
    pub async fn compact_memory(&self, memory: &ChatMemory) -> Result<Option<Compaction>> {
        if !memory.needs_compaction() {
            return Ok(None);
        }
        let (old, _keep) = memory.split_for_compaction();
        let summary = self.llm.summarize(memory.summary.as_deref(), &old).await?;
        info!(folded_turns = old.len(), summary_chars = summary.len(), "conversation memory compacted");
        Ok(Some(Compaction { summary, dropped: old.len() }))
    }
}
