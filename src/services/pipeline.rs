//! Ingestion: hash -> extract + chunk -> embed -> replace the document's chunks in the store.

use crate::{
    config::Config,
    models::document::{IngestOutcome, NewChunk, Stage},
    observability::metrics::metrics,
    services::{embedder::Embedder, extractor, vector_store::VectorStore},
    util::{sha256_hex, short},
};
use anyhow::{Context, Result};
use std::{path::Path, sync::Arc};
use tracing::{debug, info, info_span, warn, Instrument};

#[derive(Clone)]
pub struct Pipeline {
    pub cfg: Arc<Config>,
    pub embedder: Embedder,
    pub store: VectorStore,
}

impl Pipeline {
    pub fn new(cfg: Arc<Config>, embedder: Embedder, store: VectorStore) -> Self {
        Self { cfg, embedder, store }
    }

    /// Ingest one file into `collection`. `source` is the stable identity of the document (path or
    /// filename); a document is identified by `(collection, source)`.
    /// Idempotent: unchanged content is skipped, changed content replaces the old chunks.
    pub async fn ingest_file(&self, path: &Path, source: &str, collection: &str) -> Result<IngestOutcome> {
        self.ingest_file_progress(path, source, collection, |_| {}).await
    }

    /// Same as `ingest_file`, reporting each stage to `on_stage` (used by the terminal UIs).
    pub async fn ingest_file_progress(&self, path: &Path, source: &str, collection: &str, on_stage: impl Fn(Stage)) -> Result<IngestOutcome> {
        let span = info_span!("ingest_file", file = %short(source), collection = %collection);
        let res = self.ingest_inner(path, source, collection, on_stage).instrument(span.clone()).await;
        let m = metrics();
        match &res {
            Ok(IngestOutcome::Indexed { chunks, .. }) => {
                m.inc(&m.files_indexed, 1);
                m.inc(&m.chunks_indexed, *chunks as u64);
            }
            Ok(IngestOutcome::Skipped { .. }) => m.inc(&m.files_skipped, 1),
            Err(e) => {
                m.inc(&m.files_failed, 1);
                m.inc(&m.errors, 1);
                let _g = span.enter();
                warn!(error = %format!("{e:#}"), "ingest failed");
            }
        }
        res
    }

    async fn ingest_inner(&self, path: &Path, source: &str, collection: &str, on_stage: impl Fn(Stage)) -> Result<IngestOutcome> {
        on_stage(Stage::Checking);
        let bytes = tokio::fs::read(path).await.with_context(|| format!("read {}", path.display()))?;
        let doc_hash = sha256_hex(&bytes);
        debug!(bytes = bytes.len(), hash = %&doc_hash[..12], "file hashed");
        drop(bytes);

        if self.store.is_current(collection, source, &doc_hash).await? {
            info!("unchanged since last ingest, skipping");
            return Ok(IngestOutcome::Skipped { source: source.to_string() });
        }

        // Extraction + chunking is CPU-heavy and blocking: keep it off the async executor.
        on_stage(Stage::Extracting);
        info!("extracting and chunking");
        let (p, tok, max) = (path.to_path_buf(), self.cfg.chunk_tokenizer.clone(), self.cfg.chunk_max_tokens);
        let raw = tokio::task::spawn_blocking(move || extractor::extract_and_chunk(&p, &tok, max))
            .instrument(info_span!("extract", file = %short(source), max_tokens = max))
            .await
            .context("extraction task panicked")??;
        info!(chunks = raw.len(), "chunking finished");

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
        let n = chunks.len();
        let written = async {
            self.store.delete_source(collection, source).await?;
            self.store.insert_chunks(collection, source, &doc_hash, self.embedder.model(), chunks).await
        }
        .instrument(info_span!("weaviate_write", chunks = n))
        .await?;
        info!(chunks = written, "indexed");
        Ok(IngestOutcome::Indexed { source: source.to_string(), chunks: written })
    }
}
