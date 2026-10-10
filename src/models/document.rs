use serde::Serialize;

/// A chunk produced by the extractor, before embedding.
#[derive(Debug, Clone)]
pub struct RawChunk {
    /// Chunk text as produced by the chunker.
    pub text: String,
    /// Heading path + chunk text. This is what gets embedded and BM25-indexed.
    pub context_text: String,
    pub headings: Vec<String>,
    pub item_refs: Vec<String>,
}

/// A chunk ready to be written to the vector store.
#[derive(Debug, Clone)]
pub struct NewChunk {
    pub text: String,
    pub context_text: String,
    pub headings: Vec<String>,
    pub item_refs: Vec<String>,
    pub chunk_index: usize,
    pub vector: Vec<f32>,
}

/// Progress reported while a file moves through the ingestion pipeline.
#[derive(Debug, Clone)]
pub enum Stage {
    Checking,
    Extracting,
    Embedding { done: usize, total: usize },
    Indexing,
}

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum IngestOutcome {
    Skipped { source: String },
    Indexed { source: String, chunks: usize },
}
