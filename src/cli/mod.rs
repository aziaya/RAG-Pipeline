//! Command-line entry points. Running with no command opens the split-pane terminal UI.

pub mod classic;
pub mod commands;
pub mod plain_chat;

use crate::{
    context::AppContext,
    observability::{events::TraceEvent, LogMode},
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokio::sync::mpsc::UnboundedReceiver;

#[derive(Parser)]
#[command(
    name = "rag-pipeline",
    about = "Docling -> EmbeddingGemma -> Weaviate ingestion, hybrid retrieval and chat. Run without a command for the interactive split-pane UI."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Split-pane terminal UI with live tracing and metrics (the default)
    Tui,
    /// Prompt-and-menu terminal UI (single pane)
    Classic,
    /// Ingest a file or a folder, with progress
    Ingest { path: PathBuf },
    /// Hybrid search, printed as JSON
    Search {
        query: String,
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
        /// 0 = keyword only, 1 = vector only
        #[arg(short, long, default_value_t = 0.5)]
        alpha: f32,
    },
    /// Chat with your documents: `chat "question"` answers once, plain `chat` is interactive
    Chat {
        /// Ask one question and exit (omit for interactive mode)
        question: Option<String>,
        /// Chunks to retrieve (defaults to CHAT_TOP_K)
        #[arg(short, long)]
        limit: Option<usize>,
        /// 0 = keyword only, 1 = vector only
        #[arg(short, long, default_value_t = 0.5)]
        alpha: f32,
        /// Only search this ingested file
        #[arg(short, long)]
        source: Option<String>,
    },
    /// Run the HTTP API (POST /ingest, /search, /chat; Swagger UI at /docs)
    Serve,
}

impl Cli {
    pub fn wants_tui(&self) -> bool {
        matches!(self.command, None | Some(Command::Tui))
    }

    /// How logging should behave for this command.
    pub fn log_mode(&self) -> LogMode {
        match self.command {
            None | Some(Command::Tui) => LogMode::Tui,
            Some(Command::Serve) => LogMode::Plain("info"),
            _ => LogMode::Plain("warn"),
        }
    }

    /// Commands that should start even when Weaviate is down (they can show why).
    pub fn lenient_startup(&self) -> bool {
        matches!(self.command, None | Some(Command::Tui) | Some(Command::Classic) | Some(Command::Serve))
    }
}

pub async fn run(cli: Cli, ctx: AppContext, trace_rx: Option<UnboundedReceiver<TraceEvent>>) -> Result<()> {
    use std::io::IsTerminal;
    let is_tty = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
    match cli.command {
        None | Some(Command::Tui) => {
            let rx = trace_rx.ok_or_else(|| anyhow::anyhow!("tracing was not initialised for the UI"))?;
            crate::tui::run(ctx, rx).await
        }
        Some(Command::Classic) => classic::run_menu(&ctx).await,
        Some(Command::Ingest { path }) => commands::ingest(&ctx, &path).await,
        Some(Command::Search { query, limit, alpha }) => commands::search(&ctx, &query, limit, alpha).await,
        Some(Command::Chat { question, limit, alpha, source }) => {
            let limit = limit.unwrap_or(ctx.cfg.chat_top_k).clamp(1, 20);
            let alpha = alpha.clamp(0.0, 1.0);
            if question.is_none() && is_tty {
                classic::chat_session(&ctx, classic::ChatOpts { limit, alpha, source }).await
            } else {
                // One-shot questions and piped input use the plain, script-friendly mode.
                plain_chat::run(&ctx, question, plain_chat::Opts { limit, alpha, source }).await
            }
        }
        Some(Command::Serve) => commands::serve(ctx).await,
    }
}
