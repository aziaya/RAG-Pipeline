//! Request bodies for the HTTP API (responses reuse the shared models).

use crate::models::{
    chat::ChatTurn,
    collection::sanitize_collection,
    search::{RetrievalOptions, SearchMode},
};
use serde::Deserialize;
use utoipa::ToSchema;

/// Multipart form for /ingest (documentation only).
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct UploadForm {
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
    /// Collection to ingest into (default: the server's DEFAULT_COLLECTION)
    pub collection: Option<String>,
}

/// Retrieval overrides shared by /search and /chat; anything omitted uses the server defaults.
#[derive(Deserialize, ToSchema, Default)]
pub struct RetrievalParams {
    /// hybrid (default), semantic or keyword
    pub mode: Option<SearchMode>,
    /// Hybrid only: 0 = keyword, 1 = vector
    pub alpha: Option<f32>,
    /// Passages to return / use
    pub top_k: Option<usize>,
    /// Deprecated alias for `top_k`
    pub limit: Option<usize>,
    pub rerank: Option<bool>,
    /// 0 = keep retrieval order, 1 = reranker order only
    pub rerank_weight: Option<f32>,
    /// Passages fetched for the reranker to choose from
    pub candidates: Option<usize>,
    /// Restrict to one ingested document
    pub source: Option<String>,
    /// Collections to search (default: all of them)
    pub collections: Option<Vec<String>>,
    /// Shorthand for a single collection
    pub collection: Option<String>,
}

impl RetrievalParams {
    /// Apply the overrides on top of `base`, then clamp.
    pub fn apply(&self, mut base: RetrievalOptions) -> RetrievalOptions {
        if let Some(m) = self.mode {
            base.mode = m;
        }
        if let Some(a) = self.alpha {
            base.alpha = a;
        }
        if let Some(k) = self.top_k.or(self.limit) {
            base.top_k = k;
        }
        if let Some(r) = self.rerank {
            base.rerank = r;
        }
        if let Some(w) = self.rerank_weight {
            base.rerank_weight = w;
        }
        if let Some(c) = self.candidates {
            base.candidates = c;
        }
        base.source = self.source.clone().or(base.source);
        let names: Option<Vec<String>> = match (&self.collections, &self.collection) {
            (Some(list), _) => Some(list.clone()),
            (None, Some(one)) => Some(vec![one.clone()]),
            _ => None,
        };
        if let Some(names) = names {
            base.collections = names.iter().map(|n| sanitize_collection(n, "")).filter(|n| !n.is_empty()).collect();
        }
        base.sanitized()
    }
}

#[derive(Deserialize, ToSchema)]
pub struct SearchReq {
    pub query: String,
    #[serde(flatten)]
    pub params: RetrievalParams,
}

#[derive(Deserialize, ToSchema)]
pub struct ChatReq {
    pub question: String,
    /// Earlier turns of this conversation (oldest first). Only the last 8 are used.
    #[serde(default)]
    pub history: Vec<ChatTurn>,
    /// Optional running summary of even older turns
    pub summary: Option<String>,
    /// Rewrite follow-up questions into standalone search queries (default: server setting)
    pub rewrite: Option<bool>,
    #[serde(flatten)]
    pub params: RetrievalParams,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> RetrievalOptions {
        RetrievalOptions { alpha: 0.5, top_k: 4, rerank_weight: 0.7, candidates: 20, ..Default::default() }
    }

    #[test]
    fn overrides_apply_and_clamp() {
        let p: RetrievalParams = serde_json::from_str(r#"{"mode":"semantic","limit":6,"rerank":true,"rerank_weight":3,"candidates":2}"#).unwrap();
        let o = p.apply(base());
        assert_eq!((o.mode, o.top_k, o.rerank), (SearchMode::Semantic, 6, true));
        assert_eq!(o.rerank_weight, 1.0);
        assert_eq!(o.candidates, 6, "candidates never drop below top_k");
        let untouched = RetrievalParams::default().apply(base());
        assert_eq!((untouched.top_k, untouched.alpha), (4, 0.5));
    }

    #[test]
    fn collections_default_to_all_and_names_are_normalised() {
        assert!(RetrievalParams::default().apply(base()).collections.is_empty(), "no collection given means all collections");
        let one: RetrievalParams = serde_json::from_str(r#"{"collection":"Legal Docs"}"#).unwrap();
        assert_eq!(one.apply(base()).collections, vec!["legal-docs"]);
        let many: RetrievalParams = serde_json::from_str(r#"{"collections":["hr","legal","hr","  "]}"#).unwrap();
        assert_eq!(many.apply(base()).collections, vec!["hr", "legal"]);
    }

    #[test]
    fn flattened_requests_deserialize() {
        let r: SearchReq = serde_json::from_str(r#"{"query":"q","top_k":3,"mode":"keyword"}"#).unwrap();
        assert_eq!((r.query.as_str(), r.params.top_k, r.params.mode), ("q", Some(3), Some(SearchMode::Keyword)));
        let c: ChatReq = serde_json::from_str(r#"{"question":"q","rewrite":false,"history":[{"role":"user","content":"a"}]}"#).unwrap();
        assert_eq!((c.rewrite, c.history.len()), (Some(false), 1));
    }
}
