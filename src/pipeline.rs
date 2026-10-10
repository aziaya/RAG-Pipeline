use crate::{config::Config, embed::Embedder, extract, store::{self, NewChunk, Store}};
use anyhow::{Context, Result};
use std::{path::Path, sync::Arc};

#[derive(Clone)]
pub struct Pipeline {
    pub cfg: Arc<Config>,
    pub embedder: Embedder,
    pub store: Store,
}

/// Progress reported while a file moves through the pipeline.
#[derive(Debug, Clone)]
pub enum Stage {
    Checking,
    Extracting,
    Embedding { done: usize, total: usize },
    Indexing,
}

#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IngestOutcome {
    Skipped { source: String },
    Indexed { source: String, chunks: usize },
}

impl Pipeline {
    pub fn new(cfg: Arc<Config>, embedder: Embedder, store: Store) -> Self {
        Self { cfg, embedder, store }
    }

    /// Ingest one file. `source` is the stable identity of the document (path or filename).
    /// Idempotent: unchanged content is skipped, changed content replaces the old chunks.
    pub async fn ingest_file(&self, path: &Path, source: &str) -> Result<IngestOutcome> {
        self.ingest_file_progress(path, source, |_| {}).await
    }

    /// Same as `ingest_file`, reporting each stage to `on_stage` (used by the terminal UI).
    pub async fn ingest_file_progress(&self, path: &Path, source: &str, on_stage: impl Fn(Stage)) -> Result<IngestOutcome> {
        on_stage(Stage::Checking);
        let bytes = tokio::fs::read(path).await.with_context(|| format!("read {}", path.display()))?;
        let doc_hash = store::sha256_hex(&bytes);
        drop(bytes);

        if self.store.is_current(source, &doc_hash).await? {
            return Ok(IngestOutcome::Skipped { source: source.to_string() });
        }

        // Extraction + chunking is CPU-heavy and blocking: keep it off the async executor.
        on_stage(Stage::Extracting);
        let (p, tok, max) = (path.to_path_buf(), self.cfg.chunk_tokenizer.clone(), self.cfg.chunk_max_tokens);
        let raw = tokio::task::spawn_blocking(move || extract::extract_and_chunk(&p, &tok, max))
            .await
            .context("extraction task panicked")??;

        // Embed BEFORE deleting old chunks, so a failure never leaves the source empty.
        let texts: Vec<String> = raw.iter().map(|c| c.context_text.clone()).collect();
        on_stage(Stage::Embedding { done: 0, total: texts.len() });
        let vectors = self
            .embedder
            .embed_documents_progress(&texts, |done, total| on_stage(Stage::Embedding { done, total }))
            .await?;

        let chunks: Vec<NewChunk> = raw
            .into_iter()
            .zip(vectors)
            .enumerate()
            .map(|(i, (c, v))| NewChunk {
                text: c.text,
                context_text: c.context_text,
                headings: c.headings,
                item_refs: c.item_refs,
                chunk_index: i,
                vector: v,
            })
            .collect();

        on_stage(Stage::Indexing);
        self.store.delete_source(source).await?;
        let n = self.store.insert_chunks(source, &doc_hash, self.embedder.model(), chunks).await?;
        Ok(IngestOutcome::Indexed { source: source.to_string(), chunks: n })
    }
}
