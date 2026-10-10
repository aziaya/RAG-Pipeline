use chrono::{DateTime, Local};
use tracing::Level;

#[derive(Debug, Clone)]
pub enum EventKind {
    /// An ordinary log event.
    Log,
    /// A span finished; carries how long it took.
    SpanClosed { elapsed_ms: u64 },
}

/// One tracing event as shown in the live trace panel.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub ts: DateTime<Local>,
    pub level: Level,
    /// Innermost enclosing span, e.g. `ingest_file{file=notes.pdf}`.
    pub span: Option<String>,
    pub message: String,
    pub fields: Vec<(String, String)>,
    pub kind: EventKind,
}
