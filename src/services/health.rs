//! Readiness checks for every dependency, shared by the terminal UIs.

use crate::{
    context::AppContext,
    models::health::{Check, Level},
};
use anyhow::Result;
use std::{future::Future, path::Path, time::Duration};

async fn limited<T>(f: impl Future<Output = Result<T>>) -> Result<T> {
    match tokio::time::timeout(Duration::from_secs(6), f).await {
        Ok(r) => r,
        Err(_) => Err(anyhow::anyhow!("timed out after 6s")),
    }
}

pub async fn run_checks(ctx: &AppContext) -> Vec<Check> {
    let cfg = &ctx.cfg;
    let rerank_probe = async {
        if ctx.reranker.configured() {
            Some(limited(ctx.reranker.probe()).await)
        } else {
            None
        }
    };
    let (weaviate, docs, embed, llm, rerank) = tokio::join!(
        limited(ctx.store.is_ready()),
        limited(ctx.store.list_documents(None)),
        limited(ctx.embedder.probe()),
        limited(ctx.llm.probe()),
        rerank_probe,
    );

    let mut out = Vec::new();
    match &weaviate {
        Ok(()) => out.push(Check { name: "Weaviate", level: Level::Ok, detail: cfg.weaviate_url.clone() }),
        Err(e) => out.push(Check { name: "Weaviate", level: Level::Fail, detail: format!("{} — {e:#}", cfg.weaviate_url) }),
    }
    if weaviate.is_ok() {
        if let Ok(d) = docs {
            let chunks: usize = d.iter().map(|x| x.chunks).sum();
            let mut cols: Vec<&str> = d.iter().map(|x| x.collection.as_str()).collect();
            cols.dedup();
            out.push(Check { name: "Index", level: Level::Ok, detail: format!("{} documents in {} collections, {chunks} chunks", d.len(), cols.len()) });
        }
    }
    match embed {
        Ok(dim) => out.push(Check {
            name: "Embeddings",
            level: Level::Ok,
            detail: format!("{} · {dim} dimensions · {}", cfg.embed_model, cfg.embed_base_url),
        }),
        Err(e) => out.push(Check { name: "Embeddings", level: Level::Fail, detail: format!("{} — {e:#}", cfg.embed_base_url) }),
    }
    match llm {
        Ok(m) => out.push(Check { name: "Chat model", level: Level::Ok, detail: format!("{m} · {}", cfg.chat_base_url) }),
        Err(e) => out.push(Check { name: "Chat model", level: Level::Fail, detail: format!("{} — {e:#}", cfg.chat_base_url) }),
    }
    match rerank {
        Some(Ok(m)) => out.push(Check {
            name: "Reranker",
            level: Level::Ok,
            detail: format!("{m} · {} · {}", ctx.reranker.endpoint().unwrap_or(""), if cfg.rerank_enabled { "on by default" } else { "off by default" }),
        }),
        Some(Err(e)) => out.push(Check { name: "Reranker", level: Level::Fail, detail: format!("{} — {e:#}", ctx.reranker.endpoint().unwrap_or("")) }),
        None if cfg.rerank_enabled => out.push(Check {
            name: "Reranker",
            level: Level::Warn,
            detail: "RERANK_ENABLED=true but RERANK_BASE_URL is not set".into(),
        }),
        None => out.push(Check { name: "Reranker", level: Level::Ok, detail: "off (set RERANK_BASE_URL to enable reranking)".into() }),
    }
    match std::fs::metadata(&cfg.chunk_tokenizer) {
        Ok(m) if m.len() > 1_000_000 => out.push(Check {
            name: "Tokenizer",
            level: Level::Ok,
            detail: format!("{} ({:.1} MB)", cfg.chunk_tokenizer.display(), m.len() as f64 / 1e6),
        }),
        Ok(m) => out.push(Check {
            name: "Tokenizer",
            level: Level::Warn,
            detail: format!("{} is only {} bytes: wrong file?", cfg.chunk_tokenizer.display(), m.len()),
        }),
        Err(_) => out.push(Check { name: "Tokenizer", level: Level::Fail, detail: format!("not found: {}", cfg.chunk_tokenizer.display()) }),
    }
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) if Path::new(&p).exists() => out.push(Check { name: "ONNX Runtime", level: Level::Ok, detail: p }),
        Ok(p) => out.push(Check { name: "ONNX Runtime", level: Level::Fail, detail: format!("ORT_DYLIB_PATH points to a missing file: {p}") }),
        Err(_) => out.push(Check { name: "ONNX Runtime", level: Level::Warn, detail: "ORT_DYLIB_PATH not set (needed for PDFs and images)".into() }),
    }
    out
}
