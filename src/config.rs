use anyhow::{Context, Result};
use std::{env, path::PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub weaviate_url: String,
    pub weaviate_api_key: Option<String>,
    pub embed_base_url: String,
    pub embed_model: String,
    pub embed_api_key: Option<String>,
    pub embed_doc_prefix: String,
    pub embed_query_prefix: String,
    pub chunk_tokenizer: PathBuf,
    pub chunk_max_tokens: usize,
    pub bind_addr: String,
    pub max_upload_mb: usize,
    pub max_concurrent_ingests: usize,
    pub ingest_timeout_secs: u64,
    pub chat_base_url: String,
    pub chat_model: String,
    pub chat_api_key: Option<String>,
    pub chat_max_tokens: usize,
    pub chat_temperature: f32,
    pub chat_timeout_secs: u64,
    pub chat_top_k: usize,
    pub chat_context_chars: usize,
    pub chat_disable_thinking: bool,
    pub chat_max_concurrent: usize,
}

fn var(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn opt(name: &str) -> Option<String> {
    env::var(name).ok().filter(|v| !v.trim().is_empty())
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            weaviate_url: var("WEAVIATE_URL", "http://localhost:8080").trim_end_matches('/').to_string(),
            weaviate_api_key: opt("WEAVIATE_API_KEY"),
            embed_base_url: var("EMBED_BASE_URL", "http://localhost:11434/v1").trim_end_matches('/').to_string(),
            embed_model: var("EMBED_MODEL", "embeddinggemma-2"),
            embed_api_key: opt("EMBED_API_KEY"),
            embed_doc_prefix: env::var("EMBED_DOC_PREFIX").unwrap_or_default(),
            embed_query_prefix: env::var("EMBED_QUERY_PREFIX").unwrap_or_default(),
            chunk_tokenizer: PathBuf::from(var("CHUNK_TOKENIZER", "models/embeddinggemma-2/tokenizer.json")),
            chunk_max_tokens: var("CHUNK_MAX_TOKENS", "512").parse().context("CHUNK_MAX_TOKENS")?,
            bind_addr: var("BIND_ADDR", "0.0.0.0:3000"),
            max_upload_mb: var("MAX_UPLOAD_MB", "50").parse().context("MAX_UPLOAD_MB")?,
            max_concurrent_ingests: var("MAX_CONCURRENT_INGESTS", "2").parse().context("MAX_CONCURRENT_INGESTS")?,
            ingest_timeout_secs: var("INGEST_TIMEOUT_SECS", "600").parse().context("INGEST_TIMEOUT_SECS")?,
            chat_base_url: var("CHAT_BASE_URL", "http://127.0.0.1:8082/v1").trim_end_matches('/').to_string(),
            chat_model: var("CHAT_MODEL", "gemma-4-e2b-it"),
            chat_api_key: opt("CHAT_API_KEY"),
            chat_max_tokens: var("CHAT_MAX_TOKENS", "512").parse().context("CHAT_MAX_TOKENS")?,
            chat_temperature: var("CHAT_TEMPERATURE", "0.2").parse().context("CHAT_TEMPERATURE")?,
            chat_timeout_secs: var("CHAT_TIMEOUT_SECS", "300").parse().context("CHAT_TIMEOUT_SECS")?,
            chat_top_k: var("CHAT_TOP_K", "4").parse().context("CHAT_TOP_K")?,
            chat_context_chars: var("CHAT_CONTEXT_CHARS", "9000").parse().context("CHAT_CONTEXT_CHARS")?,
            chat_disable_thinking: var("CHAT_DISABLE_THINKING", "true").parse().context("CHAT_DISABLE_THINKING")?,
            chat_max_concurrent: var("CHAT_MAX_CONCURRENT", "2").parse().context("CHAT_MAX_CONCURRENT")?,
        })
    }
}
