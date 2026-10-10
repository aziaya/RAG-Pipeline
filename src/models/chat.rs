use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct ChatTurn {
    /// "user" or "assistant"
    pub role: String,
    pub content: String,
}

/// A retrieved passage the answer was based on.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct SourceRef {
    /// The [n] used for citations in the answer
    pub n: usize,
    pub source: String,
    pub headings: Vec<String>,
    pub chunk_index: i64,
    pub score: Option<f64>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct ChatAnswer {
    pub answer: String,
    pub sources: Vec<SourceRef>,
}

/// Where a question currently is, for progress displays.
#[derive(Debug, Clone)]
pub enum RagStage {
    Searching,
    Generating { passages: usize },
}
