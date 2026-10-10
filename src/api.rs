use crate::{chat::{ChatClient, ChatTurn}, pipeline::{IngestOutcome, Pipeline}, store::Hit};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Redirect, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;
use std::{path::Path, sync::Arc, time::Duration};
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct AppState {
    pub pipeline: Pipeline,
    pub permits: Arc<Semaphore>,
    pub chat: ChatClient,
    pub chat_permits: Arc<Semaphore>,
}

#[derive(OpenApi)]
#[openapi(
    info(title = "rag-pipeline API", description = "Docling -> embeddings -> Weaviate ingestion and hybrid search"),
    paths(health, ingest, search, chat),
    components(schemas(SearchReq, UploadForm, Hit, IngestOutcome, ChatReq, ChatResp, ChatSource, ChatTurn))
)]
struct ApiDoc;

pub fn router(state: AppState) -> Router {
    let limit = state.pipeline.cfg.max_upload_mb * 1024 * 1024;
    Router::new()
        .route("/", get(|| async { Redirect::to("/docs") }))
        .route("/health", get(health))
        .route("/ingest", post(ingest))
        .route("/search", post(search))
        .route("/chat", post(chat))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
        // Swagger UI at /docs, spec at /openapi.json
        .merge(SwaggerUi::new("/docs").url("/openapi.json", ApiDoc::openapi()))
}

/// Liveness check.
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Server is up", body = String)))]
async fn health() -> &'static str {
    "ok"
}

/// Multipart form for /ingest (documentation only).
#[derive(ToSchema)]
#[allow(dead_code)]
struct UploadForm {
    #[schema(value_type = String, format = Binary)]
    file: Vec<u8>,
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

fn internal(e: impl std::fmt::Display) -> ApiError {
    tracing::error!("{e}");
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

/// Upload and ingest one file (multipart field name `file`).
#[utoipa::path(
    post, path = "/ingest",
    request_body(content = UploadForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Indexed or skipped (unchanged)", body = IngestOutcome),
        (status = 400, description = "Missing or invalid file"),
        (status = 504, description = "Ingest timed out"),
        (status = 500, description = "Extraction, embedding or Weaviate error")
    )
)]
async fn ingest(State(st): State<AppState>, mut mp: Multipart) -> Result<Json<IngestOutcome>, ApiError> {
    let field = mp
        .next_field()
        .await
        .map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?
        .ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, "missing file field".into()))?;

    // Never trust the client's path: keep only the final component.
    let name = field
        .file_name()
        .and_then(|n| Path::new(n).file_name())
        .and_then(|n| n.to_str())
        .map(String::from)
        .ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, "file needs a filename".into()))?;
    let data = field.bytes().await.map_err(|e| ApiError(StatusCode::BAD_REQUEST, e.to_string()))?;

    st.pipeline.store.ensure_schema().await.map_err(internal)?;
    let _permit = st.permits.acquire().await.map_err(internal)?;
    let dir = tempfile::tempdir().map_err(internal)?;
    let path = dir.path().join(&name); // keep the extension: format detection may rely on it
    tokio::fs::write(&path, &data).await.map_err(internal)?;

    let secs = st.pipeline.cfg.ingest_timeout_secs;
    let out = tokio::time::timeout(Duration::from_secs(secs), st.pipeline.ingest_file(&path, &name))
        .await
        .map_err(|_| ApiError(StatusCode::GATEWAY_TIMEOUT, format!("ingest exceeded {secs}s")))?
        .map_err(internal)?;
    Ok(Json(out))
}

#[derive(Deserialize, ToSchema)]
struct SearchReq {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    /// 0 = keyword only, 1 = vector only
    #[serde(default = "default_alpha")]
    alpha: f32,
    source: Option<String>,
}
fn default_limit() -> usize { 5 }
fn default_alpha() -> f32 { 0.5 }

/// Hybrid (keyword + vector) search. `alpha`: 0 = keyword only, 1 = vector only.
#[utoipa::path(
    post, path = "/search",
    request_body = SearchReq,
    responses(
        (status = 200, description = "Ranked chunks", body = Vec<Hit>),
        (status = 400, description = "Empty query"),
        (status = 500, description = "Embedding or Weaviate error")
    )
)]
async fn search(State(st): State<AppState>, Json(req): Json<SearchReq>) -> Result<Json<Vec<Hit>>, ApiError> {
    if req.query.trim().is_empty() {
        return Err(ApiError(StatusCode::BAD_REQUEST, "empty query".into()));
    }
    let hits = run_search(&st.pipeline, &req.query, req.limit.clamp(1, 50), req.alpha.clamp(0.0, 1.0), req.source.as_deref())
        .await
        .map_err(internal)?;
    Ok(Json(hits))
}

pub async fn run_search(p: &Pipeline, query: &str, limit: usize, alpha: f32, source: Option<&str>) -> anyhow::Result<Vec<Hit>> {
    let v = p.embedder.embed_query(query).await?;
    p.store.hybrid_search(query, &v, alpha, limit, source).await
}

#[derive(Deserialize, ToSchema)]
struct ChatReq {
    question: String,
    /// Earlier turns of this conversation (oldest first). Only the last 8 are used.
    #[serde(default)]
    history: Vec<ChatTurn>,
    /// Chunks to retrieve (defaults to CHAT_TOP_K).
    limit: Option<usize>,
    /// 0 = keyword only, 1 = vector only
    #[serde(default = "default_alpha")]
    alpha: f32,
    /// Restrict retrieval to one ingested file
    source: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct ChatSource {
    /// The [n] used for citations in the answer
    n: usize,
    source: String,
    headings: Vec<String>,
    chunk_index: i64,
    score: Option<f64>,
    text: String,
}

#[derive(Serialize, ToSchema)]
struct ChatResp {
    answer: String,
    sources: Vec<ChatSource>,
}

/// Answer a question from the indexed documents (retrieve, then generate).
#[utoipa::path(
    post, path = "/chat",
    request_body = ChatReq,
    responses(
        (status = 200, description = "Answer with the chunks it was based on", body = ChatResp),
        (status = 400, description = "Empty or too long question"),
        (status = 500, description = "Embedding or Weaviate error"),
        (status = 502, description = "Chat model unreachable or returned nothing")
    )
)]
async fn chat(State(st): State<AppState>, Json(req): Json<ChatReq>) -> Result<Json<ChatResp>, ApiError> {
    let question = req.question.trim();
    if question.is_empty() || question.chars().count() > 4000 {
        return Err(ApiError(StatusCode::BAD_REQUEST, "question must be 1-4000 characters".into()));
    }
    let limit = req.limit.unwrap_or(st.pipeline.cfg.chat_top_k).clamp(1, 20);
    let hits = run_search(&st.pipeline, question, limit, req.alpha.clamp(0.0, 1.0), req.source.as_deref())
        .await
        .map_err(internal)?;
    if hits.is_empty() {
        return Ok(Json(ChatResp {
            answer: "I couldn't find anything relevant in the indexed documents.".into(),
            sources: vec![],
        }));
    }

    let _permit = st.chat_permits.acquire().await.map_err(internal)?;
    let (answer, used) = st.chat.answer(question, &req.history, &hits).await.map_err(|e| {
        tracing::error!("chat failed: {e:#}");
        ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}"))
    })?;

    let sources = hits
        .into_iter()
        .take(used)
        .enumerate()
        .map(|(i, h)| ChatSource { n: i + 1, source: h.source, headings: h.headings, chunk_index: h.chunk_index, score: h.score, text: h.text })
        .collect();
    Ok(Json(ChatResp { answer, sources }))
}
