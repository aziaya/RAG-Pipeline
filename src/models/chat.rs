use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, utoipa::ToSchema)]
pub struct ChatTurn {
    /// "user" or "assistant"
    pub role: String,
    pub content: String,
}

/// Older turns are folded into a summary once the verbatim history grows past this.
pub const COMPACT_AT: usize = 12;
/// Messages kept verbatim after compaction.
pub const KEEP_MESSAGES: usize = 8;

/// What the assistant remembers about the conversation: a rolling summary of older turns
/// plus the most recent turns verbatim.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMemory {
    pub summary: Option<String>,
    pub turns: Vec<ChatTurn>,
}

impl ChatMemory {
    pub fn is_empty(&self) -> bool {
        self.summary.is_none() && self.turns.is_empty()
    }

    pub fn push_exchange(&mut self, question: &str, answer: &str) {
        self.turns.push(ChatTurn { role: "user".into(), content: question.to_string() });
        self.turns.push(ChatTurn { role: "assistant".into(), content: answer.to_string() });
    }

    pub fn needs_compaction(&self) -> bool {
        self.turns.len() > COMPACT_AT
    }

    /// (turns to summarize, turns to keep verbatim)
    pub fn split_for_compaction(&self) -> (Vec<ChatTurn>, Vec<ChatTurn>) {
        let cut = self.turns.len().saturating_sub(KEEP_MESSAGES);
        (self.turns[..cut].to_vec(), self.turns[cut..].to_vec())
    }
}

/// A retrieved passage the answer was based on.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SourceRef {
    /// The [n] used for citations in the answer
    pub n: usize,
    pub source: String,
    /// Collection the source document belongs to
    #[serde(default)]
    pub collection: String,
    pub headings: Vec<String>,
    pub chunk_index: i64,
    pub score: Option<f64>,
    /// Raw reranker score, when reranking ran
    pub rerank_score: Option<f64>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct ChatAnswer {
    pub answer: String,
    pub sources: Vec<SourceRef>,
    /// The standalone query used for retrieval when a follow-up was rewritten
    pub searched_for: Option<String>,
    /// Things that degraded gracefully, e.g. "reranker unavailable; used retrieval order"
    pub warnings: Vec<String>,
}

/// Where a question currently is, for progress displays.
#[derive(Debug, Clone)]
pub enum RagStage {
    Rewriting,
    Searching,
    Reranking { candidates: usize },
    Generating { passages: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_splits_old_from_recent() {
        let mut m = ChatMemory::default();
        for i in 0..7 {
            m.push_exchange(&format!("q{i}"), &format!("a{i}"));
        }
        assert!(m.needs_compaction());
        let (old, keep) = m.split_for_compaction();
        assert_eq!(keep.len(), KEEP_MESSAGES);
        assert_eq!(old.len(), 14 - KEEP_MESSAGES);
        assert_eq!(keep.last().unwrap().content, "a6");
        assert_eq!(old[0].content, "q0");
    }
}
