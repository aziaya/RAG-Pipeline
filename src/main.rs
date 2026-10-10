mod api;
mod chat;
mod cli_chat;
mod config;
mod embed;
mod extract;
mod pipeline;
mod store;
mod tui;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::Semaphore;

#[derive(Parser)]
#[command(
    name = "rag-pipeline",
    about = "Docling -> EmbeddingGemma -> Weaviate ingestion and hybrid retrieval. Run without a command for the interactive menu."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Ingest a file or a whole directory (recursive)
    Ingest { path: PathBuf },
    /// Hybrid search
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
    /// Run the HTTP API (POST /ingest, POST /search, POST /chat, GET /health)
    Serve,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();
    // Quiet logs for the interactive UI; the API server keeps info-level logs.
    let default_level = if matches!(cli.cmd, Some(Cmd::Serve)) { "info" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| default_level.into()))
        .init();

    use std::io::IsTerminal;
    let is_tty = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
    if cli.cmd.is_none() && !is_tty {
        Cli::command().print_help()?;
        eprintln!("\n\nThe interactive menu needs a terminal. Use a subcommand when piping.");
        std::process::exit(2);
    }
    let cfg = Arc::new(config::Config::from_env()?);
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let chat_http = http.clone();
    let embedder = embed::Embedder::new(&cfg, http.clone());
    let store = store::Store::new(&cfg, http);
    // Hard-fail for CLI commands; for `serve`, start anyway and retry per request.
    // The server and the interactive menu start anyway (the menu has a status screen).
    let lenient = matches!(cli.cmd, Some(Cmd::Serve) | None);
    if let Err(e) = store.ensure_schema().await {
        if lenient {
            tracing::warn!("Weaviate not ready at startup ({e:#}); will retry when needed");
        } else {
            return Err(e);
        }
    }
    let pipeline = pipeline::Pipeline::new(cfg.clone(), embedder, store);
    let chat_client = chat::ChatClient::new(&cfg, chat_http);

    match cli.cmd {
        None => tui::run_menu(&pipeline, &chat_client).await?,
        Some(Cmd::Ingest { path }) => {
            let files = tui::collect_files(&path);
            if files.is_empty() {
                anyhow::bail!("no files found at {}", path.display());
            }
            tui::ingest_paths(&pipeline, files).await?;
        }
        Some(Cmd::Search { query, limit, alpha }) => {
            let hits = api::run_search(&pipeline, &query, limit, alpha, None).await?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Some(Cmd::Chat { question, limit, alpha, source }) => {
            let limit = limit.unwrap_or(cfg.chat_top_k).clamp(1, 20);
            let alpha = alpha.clamp(0.0, 1.0);
            if question.is_none() && is_tty {
                tui::chat_session(&pipeline, &chat_client, tui::ChatOpts { limit, alpha, source }).await?;
            } else {
                // One-shot questions and piped input use the plain, script-friendly mode.
                cli_chat::run(&pipeline, &chat_client, question, cli_chat::Opts { limit, alpha, source }).await?;
            }
        }
        Some(Cmd::Serve) => {
            let state = api::AppState {
                permits: Arc::new(Semaphore::new(cfg.max_concurrent_ingests.max(1))),
                chat: chat_client,
                chat_permits: Arc::new(Semaphore::new(cfg.chat_max_concurrent.max(1))),
                pipeline,
            };
            let listener = tokio::net::TcpListener::bind(&cfg.bind_addr).await?;
            tracing::info!("listening on http://{}  (Swagger UI: /docs)", cfg.bind_addr);
            axum::serve(listener, api::router(state)).await?;
        }
    }
    Ok(())
}
