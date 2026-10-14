//! Weaviate access over REST + GraphQL (there is no official Rust client; Weaviate's
//! official clients are Python, JS/TS, Java, C# and Go). Vectors are supplied by us
//! (`vectorizer: none`), so Weaviate only stores and searches them.
//!
//! Everything lives in one Weaviate class. A *collection* is a text property on every chunk, so
//! queries can span all collections (no filter) or any subset (a filter), and a document is
//! identified by `(collection, source)`.

use crate::{
    config::Config,
    models::{
        collection::{ChunkInfo, DocumentInfo},
        document::NewChunk,
        search::Hit,
    },
};
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tracing::{debug, info, info_span, warn, Instrument};
use uuid::Uuid;

pub const CLASS: &str = "DocChunk";

#[derive(Clone)]
pub struct VectorStore {
    http: reqwest::Client,
    base: String,
    api_key: Option<String>,
    default_collection: String,
}

/// Which chunks a query may see: any of `collections` (empty = all), optionally one document.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    pub collections: Vec<String>,
    pub source: Option<String>,
}

impl Scope {
    /// GraphQL `where` filter for this scope, if it narrows anything.
    fn filter(&self) -> Option<Value> {
        let mut operands: Vec<Value> = vec![];
        match self.collections.len() {
            0 => {}
            1 => operands.push(eq("collection", &self.collections[0])),
            _ => operands.push(json!({
                "operator": "Or",
                "operands": self.collections.iter().map(|c| eq("collection", c)).collect::<Vec<_>>()
            })),
        }
        if let Some(s) = &self.source {
            operands.push(eq("source", s));
        }
        match operands.len() {
            0 => None,
            1 => operands.pop(),
            _ => Some(json!({ "operator": "And", "operands": operands })),
        }
    }
}

fn eq(path: &str, value: &str) -> Value {
    json!({ "path": [path], "operator": "Equal", "valueText": value })
}

fn and(operands: Vec<Value>) -> Value {
    json!({ "operator": "And", "operands": operands })
}

fn where_arg(filter: Option<Value>) -> String {
    filter.map(|f| format!(", where: {}", to_graphql_input(&f))).unwrap_or_default()
}

/// Deterministic object id: re-ingesting identical content overwrites instead of duplicating.
fn object_id(collection: &str, source: &str, doc_hash: &str, chunk_index: usize) -> Uuid {
    let name = format!("{collection}\u{1f}{source}\u{1f}{doc_hash}\u{1f}{chunk_index}");
    Uuid::new_v5(&Uuid::NAMESPACE_URL, name.as_bytes())
}

impl VectorStore {
    pub fn new(cfg: &Config, http: reqwest::Client) -> Self {
        Self {
            http,
            base: cfg.weaviate_url.clone(),
            api_key: cfg.weaviate_api_key.clone(),
            default_collection: cfg.default_collection.clone(),
        }
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

    /// Create the class if it does not exist. If it exists from before collections were
    /// introduced, add the `collection` property and put the existing chunks in the default
    /// collection (Weaviate does not backfill new properties).
    pub async fn ensure_schema(&self) -> Result<()> {
        let existing = self.req(reqwest::Method::GET, &format!("/v1/schema/{CLASS}")).send().await?;
        if existing.status().is_success() {
            let schema: Value = existing.json().await.unwrap_or(Value::Null);
            let has_collection = schema["properties"].as_array().map(|p| p.iter().any(|x| x["name"] == "collection")).unwrap_or(false);
            if !has_collection {
                info!("adding the `collection` property to the existing schema");
                let prop = json!({ "name": "collection", "dataType": ["text"], "tokenization": "field" });
                let resp = self.req(reqwest::Method::POST, &format!("/v1/schema/{CLASS}/properties")).json(&prop).send().await?;
                Self::check(resp, "add collection property").await?;
                match self.migrate_legacy().await {
                    Ok(n) => info!(chunks = n, collection = %self.default_collection, "existing chunks moved into the default collection"),
                    Err(e) => warn!(error = %format!("{e:#}"), "could not move existing chunks into a collection; they stay searchable but belong to none"),
                }
            }
            return Ok(());
        }
        // "field" tokenization = exact match, which is what we want for the filter properties.
        let class = json!({
            "class": CLASS,
            "vectorizer": "none",
            "vectorIndexConfig": { "distance": "cosine" },
            "properties": [
                { "name": "text",        "dataType": ["text"] },
                { "name": "contextText", "dataType": ["text"] },
                { "name": "headings",    "dataType": ["text[]"] },
                { "name": "collection",  "dataType": ["text"], "tokenization": "field" },
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

    /// Give every chunk that has no collection the default one (merge-patch, vectors untouched).
    async fn migrate_legacy(&self) -> Result<usize> {
        let (mut after, mut patched) = (None::<String>, 0usize);
        loop {
            let mut path = format!("/v1/objects?class={CLASS}&limit=100");
            if let Some(a) = &after {
                path.push_str(&format!("&after={a}"));
            }
            let v = Self::check(self.req(reqwest::Method::GET, &path).send().await?, "list objects").await?;
            let objects = v["objects"].as_array().cloned().unwrap_or_default();
            let Some(last) = objects.last().and_then(|o| o["id"].as_str()).map(String::from) else { break };
            for o in &objects {
                if !o["properties"]["collection"].is_null() {
                    continue;
                }
                let Some(id) = o["id"].as_str() else { continue };
                let body = json!({ "class": CLASS, "properties": { "collection": self.default_collection } });
                let resp = self.req(reqwest::Method::PATCH, &format!("/v1/objects/{CLASS}/{id}")).json(&body).send().await?;
                Self::check(resp, "patch object").await?;
                patched += 1;
            }
            after = Some(last);
        }
        Ok(patched)
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

    /// Group chunk counts by a property, optionally filtered. Returns `(value, count)`.
    async fn group_counts(&self, property: &str, filter: Option<Value>) -> Result<Vec<(String, usize)>> {
        let q = format!(
            "{{ Aggregate {{ {CLASS}(groupBy: [\"{property}\"]{}) {{ groupedBy {{ value }} meta {{ count }} }} }} }}",
            where_arg(filter)
        );
        let v = self.graphql(&q).await?;
        let mut out: Vec<(String, usize)> = v["data"]["Aggregate"][CLASS]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| Some((r["groupedBy"]["value"].as_str()?.to_string(), r["meta"]["count"].as_u64()? as usize)))
                    .collect()
            })
            .unwrap_or_default();
        out.sort_by_key(|a| a.0.to_lowercase());
        Ok(out)
    }

    /// Every collection with its chunk count.
    pub async fn list_collections(&self) -> Result<Vec<(String, usize)>> {
        self.group_counts("collection", None).await
    }

    /// Ingested documents, in one collection or in all of them.
    pub async fn list_documents(&self, collection: Option<&str>) -> Result<Vec<DocumentInfo>> {
        let collections: Vec<String> = match collection {
            Some(c) => vec![c.to_string()],
            None => self.list_collections().await?.into_iter().map(|(c, _)| c).collect(),
        };
        let mut out = vec![];
        for c in collections {
            for (source, chunks) in self.group_counts("source", Some(eq("collection", &c))).await? {
                out.push(DocumentInfo { collection: c.clone(), source, chunks });
            }
        }
        out.sort_by_key(|d| (d.collection.to_lowercase(), d.source.to_lowercase()));
        Ok(out)
    }

    /// Number of chunks stored for one document.
    pub async fn count_chunks(&self, collection: &str, source: &str) -> Result<usize> {
        let filter = and(vec![eq("collection", collection), eq("source", source)]);
        let q = format!("{{ Aggregate {{ {CLASS}({}) {{ meta {{ count }} }} }} }}", where_arg(Some(filter)).trim_start_matches(", "));
        let v = self.graphql(&q).await?;
        Ok(v["data"]["Aggregate"][CLASS][0]["meta"]["count"].as_u64().unwrap_or(0) as usize)
    }

    /// One page of a document's chunks in document order.
    pub async fn list_chunks(&self, collection: &str, source: &str, limit: usize, offset: usize) -> Result<Vec<ChunkInfo>> {
        let filter = and(vec![eq("collection", collection), eq("source", source)]);
        let fields = "text headings chunkIndex _additional { id }";
        let sorted = format!(
            "{{ Get {{ {CLASS}(where: {}, sort: [{{path: [\"chunkIndex\"], order: asc}}], limit: {limit}, offset: {offset}) {{ {fields} }} }} }}",
            to_graphql_input(&filter)
        );
        let rows = match self.graphql(&sorted).await {
            Ok(v) => v["data"]["Get"][CLASS].as_array().cloned().unwrap_or_default(),
            Err(e) => {
                // Fall back to sorting here if this Weaviate rejects `sort`.
                warn!(error = %format!("{e:#}"), "sorted chunk query failed; sorting client-side");
                let plain = format!("{{ Get {{ {CLASS}(where: {}, limit: {}) {{ {fields} }} }} }}", to_graphql_input(&filter), limit + offset);
                let mut rows = self.graphql(&plain).await?["data"]["Get"][CLASS].as_array().cloned().unwrap_or_default();
                rows.sort_by_key(|r| r["chunkIndex"].as_i64().unwrap_or(0));
                rows.into_iter().skip(offset).take(limit).collect()
            }
        };
        Ok(rows
            .iter()
            .map(|r| ChunkInfo {
                id: r["_additional"]["id"].as_str().map(String::from),
                chunk_index: r["chunkIndex"].as_i64().unwrap_or(0),
                headings: strings(&r["headings"]),
                text: r["text"].as_str().unwrap_or_default().to_string(),
            })
            .collect())
    }

    /// True if this exact content for this document is already indexed in this collection.
    pub async fn is_current(&self, collection: &str, source: &str, doc_hash: &str) -> Result<bool> {
        let filter = and(vec![eq("collection", collection), eq("source", source), eq("docHash", doc_hash)]);
        let q = format!("{{ Get {{ {CLASS}({}, limit: 1) {{ docHash }} }} }}", where_arg(Some(filter)).trim_start_matches(", "));
        let v = self.graphql(&q).await?;
        Ok(v["data"]["Get"][CLASS].as_array().map(|a| !a.is_empty()).unwrap_or(false))
    }

    async fn delete_where(&self, filter: Value, what: &str) -> Result<()> {
        let body = json!({ "match": { "class": CLASS, "where": filter } });
        let resp = self.req(reqwest::Method::DELETE, "/v1/batch/objects").json(&body).send().await?;
        Self::check(resp, what).await?;
        Ok(())
    }

    /// Remove every chunk of one document (also used before re-ingesting a changed file).
    pub async fn delete_source(&self, collection: &str, source: &str) -> Result<()> {
        self.delete_where(and(vec![eq("collection", collection), eq("source", source)]), "delete document").await
    }

    /// Remove a whole collection and all its documents.
    pub async fn delete_collection(&self, collection: &str) -> Result<()> {
        self.delete_where(eq("collection", collection), "delete collection").await
    }

    pub async fn insert_chunks(&self, collection: &str, source: &str, doc_hash: &str, embed_model: &str, chunks: Vec<NewChunk>) -> Result<usize> {
        let total = chunks.len();
        for batch in chunks.chunks(100) {
            let objects: Vec<Value> = batch
                .iter()
                .map(|c| {
                    json!({
                        "class": CLASS,
                        "id": object_id(collection, source, doc_hash, c.chunk_index).to_string(),
                        "properties": {
                            "text": c.text,
                            "contextText": c.context_text,
                            "headings": c.headings,
                            "collection": collection,
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
            let resp = self.req(reqwest::Method::POST, "/v1/batch/objects").json(&json!({ "objects": objects })).send().await?;
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

    /// Run a `Get` with the given search operator (`hybrid`, `nearVector` or `bm25`).
    async fn get(&self, search_arg: &str, limit: usize, scope: &Scope, extra: &str) -> Result<Vec<Hit>> {
        let q = format!(
            "{{ Get {{ {CLASS}({search_arg}, limit: {limit}{}) \
             {{ text headings collection source chunkIndex _additional {{ id {extra} }} }} }} }}",
            where_arg(scope.filter())
        );
        let v = self.graphql(&q).await?;
        let rows = v["data"]["Get"][CLASS].as_array().cloned().unwrap_or_default();
        debug!(rows = rows.len(), "weaviate search returned");
        Ok(rows.iter().map(row_to_hit).collect())
    }

    /// Hybrid (BM25 + vector) search. `alpha`: 0 = keyword only, 1 = vector only.
    pub async fn hybrid_search(&self, query: &str, vector: &[f32], alpha: f32, limit: usize, scope: &Scope) -> Result<Vec<Hit>> {
        let arg = format!(
            "hybrid: {{ query: {}, vector: {}, alpha: {alpha}, properties: [\"contextText\"] }}",
            serde_json::to_string(query)?,
            serde_json::to_string(vector)?,
        );
        self.get(&arg, limit, scope, "score").await
    }

    /// Pure vector search; the score is cosine similarity (1 - distance).
    pub async fn semantic_search(&self, vector: &[f32], limit: usize, scope: &Scope) -> Result<Vec<Hit>> {
        let arg = format!("nearVector: {{ vector: {} }}", serde_json::to_string(vector)?);
        self.get(&arg, limit, scope, "distance").await
    }

    /// Pure BM25 keyword search.
    pub async fn keyword_search(&self, query: &str, limit: usize, scope: &Scope) -> Result<Vec<Hit>> {
        let arg = format!("bm25: {{ query: {}, properties: [\"contextText\"] }}", serde_json::to_string(query)?);
        self.get(&arg, limit, scope, "score").await
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|h| h.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn row_to_hit(r: &Value) -> Hit {
    let add = &r["_additional"];
    // Hybrid and BM25 return `score` as a string; vector search returns `distance`.
    let score = add["score"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| add["score"].as_f64())
        .or_else(|| add["distance"].as_f64().map(|d| 1.0 - d));
    Hit {
        id: add["id"].as_str().map(String::from),
        collection: r["collection"].as_str().unwrap_or_default().to_string(),
        score,
        retrieval_score: score,
        rerank_score: None,
        text: r["text"].as_str().unwrap_or_default().to_string(),
        headings: strings(&r["headings"]),
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
    fn scope_filters_cover_all_combinations() {
        // no restriction: no filter at all, so every collection is searched
        assert!(Scope::default().filter().is_none());
        // one collection
        let one = Scope { collections: vec!["legal".into()], source: None }.filter().unwrap();
        // (key order is irrelevant to GraphQL; serde_json happens to emit keys alphabetically)
        let one = to_graphql_input(&one);
        assert!(one.contains("operator: Equal") && one.contains(r#"path: ["collection"]"#) && one.contains(r#"valueText: "legal""#), "{one}");
        // several collections are OR-ed
        let many = to_graphql_input(&Scope { collections: vec!["a".into(), "b".into()], source: None }.filter().unwrap());
        assert!(many.contains("operator: Or") && many.contains(r#"valueText: "a""#) && many.contains(r#"valueText: "b""#));
        // collections AND a document
        let both = to_graphql_input(&Scope { collections: vec!["a".into(), "b".into()], source: Some("f.pdf".into()) }.filter().unwrap());
        assert!(both.contains("operator: And") && both.contains("operator: Or") && both.contains(r#"path: ["source"]"#) && both.contains(r#"valueText: "f.pdf""#), "{both}");
    }

    #[test]
    fn rows_parse_hybrid_and_vector_scores_and_collection() {
        let hybrid = json!({"text":"t","headings":["A"],"collection":"legal","source":"s","chunkIndex":2,"_additional":{"id":"u","score":"0.75"}});
        let h = row_to_hit(&hybrid);
        assert_eq!((h.score, h.retrieval_score), (Some(0.75), Some(0.75)));
        assert_eq!(h.collection, "legal");
        let vector = json!({"text":"t","source":"s","chunkIndex":0,"_additional":{"id":"u","distance":0.2}});
        let h = row_to_hit(&vector);
        assert!((h.score.unwrap() - 0.8).abs() < 1e-9);
        assert!(h.rerank_score.is_none());
        assert_eq!(h.collection, "", "legacy rows without a collection still parse");
    }

    #[test]
    fn ids_are_deterministic_and_collection_aware() {
        assert_eq!(object_id("c", "a", "h", 1), object_id("c", "a", "h", 1));
        assert_ne!(object_id("c", "a", "h", 1), object_id("c", "a", "h", 2));
        assert_ne!(object_id("c1", "a", "h", 1), object_id("c2", "a", "h", 1), "the same file in two collections must not collide");
    }
}
