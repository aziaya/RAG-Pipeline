//! Embeddings via any OpenAI-compatible `/embeddings` endpoint.

use crate::config::Config;
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::json;

#[derive(Clone)]
pub struct Embedder {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
    doc_prefix: String,
    query_prefix: String,
}

#[derive(Deserialize)]
struct EmbResponse {
    data: Vec<EmbItem>,
}

#[derive(Deserialize)]
struct EmbItem {
    index: usize,
    embedding: Vec<f32>,
}

impl Embedder {
    pub fn new(cfg: &Config, http: reqwest::Client) -> Self {
        Self {
            http,
            base_url: cfg.embed_base_url.clone(),
            model: cfg.embed_model.clone(),
            api_key: cfg.embed_api_key.clone(),
            doc_prefix: cfg.embed_doc_prefix.clone(),
            query_prefix: cfg.embed_query_prefix.clone(),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    async fn embed_raw(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let n = inputs.len();
        let mut req = self
            .http
            .post(format!("{}/embeddings", self.base_url))
            .json(&json!({ "model": self.model, "input": inputs }));
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.context("embedding request failed")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("embedding endpoint returned {status}: {body}"));
        }
        let mut parsed: EmbResponse = resp.json().await.context("bad embedding response")?;
        if parsed.data.len() != n {
            return Err(anyhow!("expected {n} embeddings, got {}", parsed.data.len()));
        }
        parsed.data.sort_by_key(|d| d.index);
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Embed document chunks (applies the document task prefix), reporting (done, total)
    /// after every batch.
    pub async fn embed_documents_progress(
        &self,
        texts: &[String],
        mut on_progress: impl FnMut(usize, usize),
    ) -> Result<Vec<Vec<f32>>> {
        let total = texts.len();
        let mut all = Vec::with_capacity(total);
        for batch in texts.chunks(32) {
            let inputs = batch.iter().map(|t| format!("{}{}", self.doc_prefix, t)).collect();
            all.extend(self.embed_raw(inputs).await?);
            on_progress(all.len(), total);
        }
        Ok(all)
    }

    /// Health check: embed one short string and return the vector length.
    pub async fn probe(&self) -> Result<usize> {
        Ok(self.embed_raw(vec!["ping".to_string()]).await?.first().map(|v| v.len()).unwrap_or(0))
    }

    /// Embed a search query (applies the query task prefix).
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let mut v = self.embed_raw(vec![format!("{}{}", self.query_prefix, query)]).await?;
        v.pop().ok_or_else(|| anyhow!("empty embedding response"))
    }
}
