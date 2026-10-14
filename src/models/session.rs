use super::chat::{ChatMemory, SourceRef};
use serde::{Deserialize, Serialize};

/// One transcript entry as saved on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    /// "user", "assistant", "system" or "error"
    pub role: String,
    pub text: String,
    #[serde(default)]
    pub sources: Vec<SourceRef>,
    pub took_secs: Option<f32>,
    pub searched_for: Option<String>,
}

/// A saved conversation: the transcript plus what the assistant remembers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub created: String,
    pub updated: String,
    pub memory: ChatMemory,
    pub messages: Vec<StoredMessage>,
}

#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub title: String,
    pub updated: String,
    pub exchanges: usize,
}
