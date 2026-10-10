//! Non-interactive commands.

use crate::{api, cli::classic, context::AppContext, util};
use anyhow::Result;
use std::path::Path;
use tracing::info;

pub async fn ingest(ctx: &AppContext, path: &Path) -> Result<()> {
    let files = util::collect_files(path);
    if files.is_empty() {
        anyhow::bail!("no files found at {}", path.display());
    }
    classic::ingest_paths(&ctx.pipeline, files).await
}

pub async fn search(ctx: &AppContext, query: &str, limit: usize, alpha: f32) -> Result<()> {
    let hits = ctx.rag.search(query, limit.clamp(1, 50), alpha.clamp(0.0, 1.0), None).await?;
    println!("{}", serde_json::to_string_pretty(&hits)?);
    Ok(())
}

pub async fn serve(ctx: AppContext) -> Result<()> {
    let addr = ctx.cfg.bind_addr.clone();
    let state = api::state::AppState::new(ctx);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("listening on http://{addr}  (Swagger UI: /docs)");
    axum::serve(listener, api::router(state)).await?;
    Ok(())
}
