//! Logging, tracing and metrics.
//!
//! * `Plain`: human-readable logs on stdout (CLI commands and the API server).
//! * `Tui`: nothing may touch stdout, so events are forwarded to the live trace
//!   panel (and optionally to the file named by `RAG_LOG_FILE`).

pub mod events;
pub mod layer;
pub mod metrics;

use events::TraceEvent;
use std::{fs::File, sync::Mutex};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

pub enum LogMode {
    /// stdout logs at the given default level ("info", "warn", ...)
    Plain(&'static str),
    /// forward everything to the terminal UI
    Tui,
}

/// Install the global subscriber. Returns the receiver feeding the trace panel in `Tui` mode.
pub fn init(mode: LogMode) -> Option<UnboundedReceiver<TraceEvent>> {
    match mode {
        LogMode::Plain(level) => {
            let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| level.into());
            tracing_subscriber::fmt().with_env_filter(filter).init();
            None
        }
        LogMode::Tui => {
            let (tx, rx) = unbounded_channel();
            // Our own crate at full detail, dependencies only when something is wrong.
            let filter = || EnvFilter::try_from_env("RAG_TRACE").unwrap_or_else(|_| EnvFilter::new("rag_pipeline=trace,warn"));
            let ui = layer::UiLayer::new(tx).with_filter(filter());
            let file = std::env::var("RAG_LOG_FILE").ok().and_then(|p| File::create(p).ok()).map(|f| {
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(Mutex::new(f))
                    .with_filter(filter())
            });
            tracing_subscriber::registry().with(ui).with(file).init();
            Some(rx)
        }
    }
}
