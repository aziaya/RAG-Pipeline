//! Request bodies for the HTTP API (responses reuse the shared models).

use crate::models::chat::ChatTurn;
use serde::Deserialize;
use utoipa::ToSchema;

pub fn default_alpha() -> f32 {
    0.5
}

fn default_limit() -> usize {
    5
}

/// Multipart form for /ingest (documentation only).
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct UploadForm {
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
}

#[derive(Deserialize, ToSchema)]
pub struct SearchReq {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// 0 = keyword only, 1 = vector only
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    pub source: Option<String>,
}

#[derive(Deserialize, ToSchema)]
pub struct ChatReq {
    pub question: String,
    /// Earlier turns of this conversation (oldest first). Only the last 8 are used.
    #[serde(default)]
    pub history: Vec<ChatTurn>,
    /// Chunks to retrieve (defaults to CHAT_TOP_K).
    pub limit: Option<usize>,
    /// 0 = keyword only, 1 = vector only
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    /// Restrict retrieval to one ingested file
    pub source: Option<String>,
}
