//! Weaviate access over REST + GraphQL (there is no official Rust client; Weaviate's
//! official clients are Python, JS/TS, Java, C# and Go). Vectors are supplied by us
//! (`vectorizer: none`), so Weaviate only stores and searches them.

use crate::{
    config::Config,
    models::{document::NewChunk, search::Hit},
};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tracing::{debug, info_span, Instrument};
use uuid::Uuid;

pub const CLASS: &str = "DocChunk";

#[derive(Clone)]
pub struct VectorStore {
    http: reqwest::Client,
    base: String,
    api_key: Option<String>,
}

/// Deterministic object id: re-ingesting identical content overwrites instead of duplicating.
fn object_id(source: &str, doc_hash: &str, chunk_index: usize) -> Uuid {
    let name = format!("{source}\u{1f}{doc_hash}\u{1f}{chunk_index}");
    Uuid::new_v5(&Uuid::NAMESPACE_URL, name.as_bytes())
}

impl VectorStore {
    pub fn new(cfg: &Config, http: reqwest::Client) -> Self {
        Self { http, base: cfg.weaviate_url.clone(), api_key: cfg.weaviate_api_key.clone() }
    }

    fn req(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut r = self.http.request(method, format!("{}{}", self.base, path));
        if let Some(k) = &self.api_key {
            r = r.bearer_auth(k);
        }
        r
    }

    async fn check(resp: reqwest::Response, what: &str) -> Result<Value> {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("weaviate {what} failed ({status}): {body}"));
        }
        Ok(serde_json::from_str(&body).unwrap_or(Value::Null))
    }

    /// Create the collection if it does not exist yet.
    pub async fn ensure_schema(&self) -> Result<()> {
        let existing = self.req(reqwest::Method::GET, &format!("/v1/schema/{CLASS}")).send().await?;
        if existing.status().is_success() {
            return Ok(());
        }
        // "field" tokenization = exact match, which is what we want for source / docHash filters.
        let class = json!({
            "class": CLASS,
            "vectorizer": "none",
            "vectorIndexConfig": { "distance": "cosine" },
            "properties": [
                { "name": "text",        "dataType": ["text"] },
                { "name": "contextText", "dataType": ["text"] },
                { "name": "headings",    "dataType": ["text[]"] },
                { "name": "source",      "dataType": ["text"], "tokenization": "field" },
                { "name": "docHash",     "dataType": ["text"], "tokenization": "field" },
                { "name": "chunkIndex",  "dataType": ["int"] },
                { "name": "itemRefs",    "dataType": ["text[]"] },
                { "name": "embedModel",  "dataType": ["text"], "tokenization": "field" }
            ]
        });
        let resp = self.req(reqwest::Method::POST, "/v1/schema").json(&class).send().await?;
        Self::check(resp, "create schema").await?;
        Ok(())
    }

    async fn graphql(&self, query: &str) -> Result<Value> {
        async {
            let resp = self
                .req(reqwest::Method::POST, "/v1/graphql")
                .json(&json!({ "query": query }))
                .send()
                .await
                .context("graphql request")?;
            let v = Self::check(resp, "graphql").await?;
            if let Some(errs) = v.get("errors") {
                if !errs.is_null() {
                    return Err(anyhow!("weaviate graphql errors: {errs}"));
                }
            }
            debug!("weaviate graphql ok");
            Ok(v)
        }
        .instrument(info_span!("weaviate_query"))
        .await
    }

    /// Weaviate readiness probe.
    pub async fn is_ready(&self) -> Result<()> {
        let resp = self.req(reqwest::Method::GET, "/v1/.well-known/ready").send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(anyhow!("not ready (HTTP {})", resp.status()))
        }
    }

    /// Every ingested source with its chunk count, sorted by name.
    pub async fn list_sources(&self) -> Result<Vec<(String, usize)>> {
        let q = format!("{{ Aggregate {{ {CLASS}(groupBy: [\"source\"]) {{ groupedBy {{ value }} meta {{ count }} }} }} }}");
        let v = self.graphql(&q).await?;
        let mut out: Vec<(String, usize)> = v["data"]["Aggregate"][CLASS]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| {
                        let name = r["groupedBy"]["value"].as_str()?.to_string();
                        let n = r["meta"]["count"].as_u64()? as usize;
                        Some((name, n))
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
        Ok(out)
    }

    /// True if this exact content for this source is already indexed.
    pub async fn is_current(&self, source: &str, doc_hash: &str) -> Result<bool> {
        let where_clause = json!({
            "operator": "And",
            "operands": [
                { "path": ["source"],  "operator": "Equal", "valueText": source },
                { "path": ["docHash"], "operator": "Equal", "valueText": doc_hash }
            ]
        });
        let q = format!(
            "{{ Get {{ {CLASS}(where: {}, limit: 1) {{ docHash }} }} }}",
            to_graphql_input(&where_clause)
        );
        let v = self.graphql(&q).await?;
        let n = v["data"]["Get"][CLASS].as_array().map(|a| a.len()).unwrap_or(0);
        Ok(n > 0)
    }

    /// Remove every chunk of a source (used before re-ingesting a changed file).
    pub async fn delete_source(&self, source: &str) -> Result<()> {
        let body = json!({
            "match": {
                "class": CLASS,
                "where": { "path": ["source"], "operator": "Equal", "valueText": source }
            }
        });
        let resp = self.req(reqwest::Method::DELETE, "/v1/batch/objects").json(&body).send().await?;
        Self::check(resp, "delete by source").await?;
        Ok(())
    }

    pub async fn insert_chunks(&self, source: &str, doc_hash: &str, embed_model: &str, chunks: Vec<NewChunk>) -> Result<usize> {
        let total = chunks.len();
        for batch in chunks.chunks(100) {
            let objects: Vec<Value> = batch
                .iter()
                .map(|c| {
                    json!({
                        "class": CLASS,
                        "id": object_id(source, doc_hash, c.chunk_index).to_string(),
                        "properties": {
                            "text": c.text,
                            "contextText": c.context_text,
                            "headings": c.headings,
                            "source": source,
                            "docHash": doc_hash,
                            "chunkIndex": c.chunk_index,
                            "itemRefs": c.item_refs,
                            "embedModel": embed_model
                        },
                        "vector": c.vector
                    })
                })
                .collect();
            let resp = self
                .req(reqwest::Method::POST, "/v1/batch/objects")
                .json(&json!({ "objects": objects }))
                .send()
                .await?;
            let v = Self::check(resp, "batch insert").await?;
            // The batch endpoint returns 200 even when individual objects fail.
            if let Some(items) = v.as_array() {
                for item in items {
                    if let Some(errs) = item["result"]["errors"].as_object() {
                        if !errs.is_empty() {
                            return Err(anyhow!("object insert error: {}", item["result"]["errors"]));
                        }
                    }
                }
            }
        }
        Ok(total)
    }

    /// Hybrid (BM25 + vector) search. `alpha`: 0 = keyword only, 1 = vector only.
    pub async fn hybrid_search(&self, query: &str, vector: &[f32], alpha: f32, limit: usize, source: Option<&str>) -> Result<Vec<Hit>> {
        let filter = match source {
            Some(s) => format!(
                ", where: {}",
                to_graphql_input(&json!({ "path": ["source"], "operator": "Equal", "valueText": s }))
            ),
            None => String::new(),
        };
        let q = format!(
            "{{ Get {{ {CLASS}(hybrid: {{ query: {query_lit}, vector: {vec}, alpha: {alpha}, properties: [\"contextText\"] }}, limit: {limit}{filter}) \
             {{ text headings source chunkIndex _additional {{ id score }} }} }} }}",
            query_lit = serde_json::to_string(query)?,
            vec = serde_json::to_string(vector)?,
        );
        let v = self.graphql(&q).await?;
        let rows = v["data"]["Get"][CLASS].as_array().cloned().unwrap_or_default();
        Ok(rows.iter().map(row_to_hit).collect())
    }
}

fn row_to_hit(r: &Value) -> Hit {
    Hit {
        id: r["_additional"]["id"].as_str().map(String::from),
        // Weaviate returns the hybrid score as a string.
        score: r["_additional"]["score"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| r["_additional"]["score"].as_f64()),
        text: r["text"].as_str().unwrap_or_default().to_string(),
        headings: r["headings"]
            .as_array()
            .map(|a| a.iter().filter_map(|h| h.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        source: r["source"].as_str().unwrap_or_default().to_string(),
        chunk_index: r["chunkIndex"].as_i64().unwrap_or(0),
    }
}

/// Render a JSON value as a GraphQL *input object* literal: keys unquoted, strings quoted
/// and escaped (JSON string escaping is valid GraphQL string escaping), enum values bare.
fn to_graphql_input(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let inner: Vec<String> = m
                .iter()
                .map(|(k, val)| {
                    // `operator` takes a bare enum value, everything else is rendered normally.
                    if k == "operator" {
                        format!("{k}: {}", val.as_str().unwrap_or_default())
                    } else {
                        format!("{k}: {}", to_graphql_input(val))
                    }
                })
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(to_graphql_input).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphql_literal_shape() {
        let w = json!({"operator":"And","operands":[{"path":["source"],"operator":"Equal","valueText":"a \"b\".pdf"}]});
        let s = to_graphql_input(&w);
        assert!(s.contains("operator: And"));
        assert!(s.contains("operator: Equal"));
        assert!(s.contains(r#"valueText: "a \"b\".pdf""#));
        assert!(s.contains(r#"path: ["source"]"#));
    }

    #[test]
    fn ids_are_deterministic() {
        assert_eq!(object_id("a", "h", 1), object_id("a", "h", 1));
        assert_ne!(object_id("a", "h", 1), object_id("a", "h", 2));
    }
}
