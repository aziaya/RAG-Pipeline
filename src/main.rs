mod api;
mod cli;
mod config;
mod context;
mod models;
mod observability;
mod services;
mod tui;
mod util;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use cli::Cli;
use std::io::IsTerminal;

#[tokio::main]
async fn main() -> Result<()> {
    // A missing .env is fine; a malformed one must not fail silently (it would drop every
    // variable below the bad line).
    if let Err(e) = dotenvy::dotenv() {
        if !e.not_found() {
            eprintln!("warning: .env was not fully loaded: {e}");
        }
    }
    let cli = Cli::parse();

    let is_tty = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
    if cli.wants_tui() && !is_tty {
        Cli::command().print_help()?;
        eprintln!("\n\nThe interactive UI needs a terminal. Use a subcommand when piping.");
        std::process::exit(2);
    }

    let trace_rx = observability::init(cli.log_mode());
    let ctx = context::AppContext::build(config::Config::from_env()?)?;

    if let Err(e) = ctx.store.ensure_schema().await {
        if cli.lenient_startup() {
            tracing::warn!("Weaviate not ready at startup ({e:#}); will retry when needed");
        } else {
            return Err(e);
        }
    }
    cli::run(cli, ctx, trace_rx).await
}
