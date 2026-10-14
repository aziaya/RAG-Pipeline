//! Non-interactive commands.

use crate::{api, cli::classic, context::AppContext, models::collection::sanitize_collection, util};
use anyhow::Result;
use std::path::Path;
use tracing::info;

pub async fn ingest(ctx: &AppContext, path: &Path, collection: Option<String>) -> Result<()> {
    let default = &ctx.cfg.default_collection;
    let collection = collection.map(|c| sanitize_collection(&c, default)).unwrap_or_else(|| default.clone());
    let files = util::collect_files(path);
    if files.is_empty() {
        anyhow::bail!("no files found at {}", path.display());
    }
    classic::ingest_paths(&ctx.pipeline, files, &collection).await
}

pub async fn search(ctx: &AppContext, query: &str, limit: usize, alpha: f32, collections: Vec<String>) -> Result<()> {
    let mut opts = ctx.cfg.retrieval_defaults();
    opts.top_k = limit;
    opts.alpha = alpha;
    opts.collections = collections.iter().map(|n| sanitize_collection(n, "")).filter(|n| !n.is_empty()).collect();
    let hits = ctx.rag.search(query, &opts.sanitized()).await?;
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
