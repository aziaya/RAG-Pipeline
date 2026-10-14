//! Cross-encoder reranking through a `/rerank` endpoint (llama-server `--reranking`,
//! Jina, Cohere-compatible servers): `POST {base}/rerank` with
//! `{model, query, documents, top_n}` -> `{results: [{index, relevance_score}]}`.

use crate::{config::Config, models::search::Hit};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;
use tracing::{debug, info_span, Instrument};

/// Reranker context is short; long passages are cut to this many characters.
const MAX_DOC_CHARS: usize = 1800;

#[derive(Clone)]
pub struct Reranker {
    http: reqwest::Client,
    base_url: Option<String>,
    model: String,
    api_key: Option<String>,
    timeout: Duration,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f64,
}

impl Reranker {
    pub fn new(cfg: &Config, http: reqwest::Client) -> Self {
        Self {
            http,
            base_url: cfg.rerank_base_url.clone(),
            model: cfg.rerank_model.clone(),
            api_key: cfg.rerank_api_key.clone(),
            timeout: Duration::from_secs(cfg.rerank_timeout_secs),
        }
    }

    pub fn configured(&self) -> bool {
        self.base_url.is_some()
    }

    pub fn endpoint(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    /// Raw relevance scores aligned with `hits` (higher = more relevant).
    pub async fn score(&self, query: &str, hits: &[Hit]) -> Result<Vec<f64>> {
        let base = self.base_url.as_ref().ok_or_else(|| anyhow!("RERANK_BASE_URL is not set"))?;
        let docs: Vec<String> = hits.iter().map(doc_text).collect();
        let span = info_span!("rerank", candidates = docs.len(), model = %self.model);
        async {
            let mut req = self
                .http
                .post(format!("{base}/rerank"))
                .timeout(self.timeout)
                .json(&json!({ "model": self.model, "query": query, "documents": docs, "top_n": docs.len() }));
            if let Some(k) = &self.api_key {
                req = req.bearer_auth(k);
            }
            let resp = req.send().await.context("rerank request failed (is the reranker running?)")?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(anyhow!("rerank endpoint returned {status}: {body}"));
            }
            let parsed: RerankResponse = resp.json().await.context("bad rerank response")?;
            let mut scores = vec![f64::NEG_INFINITY; docs.len()];
            for r in parsed.results {
                if let Some(s) = scores.get_mut(r.index) {
                    *s = r.relevance_score;
                }
            }
            if scores.iter().any(|s| s.is_infinite()) {
                return Err(anyhow!("rerank response did not score every candidate"));
            }
            debug!(candidates = scores.len(), "rerank scores received");
            Ok(scores)
        }
        .instrument(span)
        .await
    }

    /// Health check.
    pub async fn probe(&self) -> Result<String> {
        let base = self.base_url.as_ref().ok_or_else(|| anyhow!("RERANK_BASE_URL is not set"))?;
        let mut req = self.http.get(format!("{base}/models")).timeout(Duration::from_secs(5));
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.context("reranker unreachable")?;
        if !resp.status().is_success() {
            return Err(anyhow!("reranker returned HTTP {}", resp.status()));
        }
        Ok(self.model.clone())
    }
}

fn doc_text(h: &Hit) -> String {
    let head = if h.headings.is_empty() { String::new() } else { format!("{}\n", h.headings.join(" > ")) };
    let full = format!("{head}{}", h.text);
    full.chars().take(MAX_DOC_CHARS).collect()
}

/// Min-max normalise a score list into 0..=1 (a constant list maps to 0.5).
fn normalize(xs: &[f64]) -> Vec<f64> {
    let (min, max) = xs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
    if !(max - min).is_finite() || (max - min).abs() < 1e-12 {
        return vec![0.5; xs.len()];
    }
    xs.iter().map(|x| (x - min) / (max - min)).collect()
}

/// Blend retrieval and reranker scores: `weight` 0 keeps the retrieval order, 1 uses the
/// reranker alone. Both are normalised first because rerankers often emit raw logits.
pub fn blend(retrieval: &[f64], rerank: &[f64], weight: f32) -> Vec<f64> {
    let w = weight.clamp(0.0, 1.0) as f64;
    let (r, s) = (normalize(retrieval), normalize(rerank));
    r.iter().zip(&s).map(|(a, b)| (1.0 - w) * a + w * b).collect()
}

/// Reorder `hits` by blended score and keep the best `top_k`. Each kept hit records the raw
/// reranker score and its final blended score.
pub fn apply(mut hits: Vec<Hit>, rerank_scores: &[f64], weight: f32, top_k: usize) -> Vec<Hit> {
    let retrieval: Vec<f64> = hits.iter().map(|h| h.retrieval_score.unwrap_or(0.0)).collect();
    let blended = blend(&retrieval, rerank_scores, weight);
    for ((h, b), raw) in hits.iter_mut().zip(&blended).zip(rerank_scores) {
        h.rerank_score = Some(*raw);
        h.score = Some(*b);
    }
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(top_k);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(name: &str, retrieval: f64) -> Hit {
        Hit { source: name.into(), retrieval_score: Some(retrieval), score: Some(retrieval), ..Default::default() }
    }

    #[test]
    fn weight_zero_keeps_retrieval_order_and_one_follows_reranker() {
        let hits = vec![hit("a", 0.9), hit("b", 0.6), hit("c", 0.3)];
        let rerank = [-5.0, 8.0, 2.0]; // raw logits favour b, then c, then a
        let kept: Vec<String> = apply(hits.clone(), &rerank, 0.0, 3).iter().map(|h| h.source.clone()).collect();
        assert_eq!(kept, ["a", "b", "c"]);
        let kept: Vec<String> = apply(hits, &rerank, 1.0, 3).iter().map(|h| h.source.clone()).collect();
        assert_eq!(kept, ["b", "c", "a"]);
    }

    #[test]
    fn blend_mixes_and_top_k_truncates() {
        let hits = vec![hit("a", 1.0), hit("b", 0.0)];
        let out = apply(hits, &[0.0, 1.0], 0.5, 1);
        assert_eq!(out.len(), 1);
        let s = out[0].score.unwrap();
        assert!((s - 0.5).abs() < 1e-9, "{s}");
        assert!(out[0].rerank_score.is_some());
    }

    #[test]
    fn constant_scores_do_not_divide_by_zero() {
        assert_eq!(normalize(&[3.0, 3.0]), vec![0.5, 0.5]);
        assert_eq!(blend(&[1.0, 1.0], &[2.0, 2.0], 0.7), vec![0.5, 0.5]);
    }

    #[test]
    fn document_text_includes_headings_and_is_capped() {
        let h = Hit { headings: vec!["A".into(), "B".into()], text: "x".repeat(5000), ..Default::default() };
        let d = doc_text(&h);
        assert!(d.starts_with("A > B\n") && d.chars().count() == MAX_DOC_CHARS);
    }
}
