//! HTTP API: router, OpenAPI document and Swagger UI.

pub mod dto;
pub mod error;
pub mod handlers;
pub mod state;

use axum::{
    extract::DefaultBodyLimit,
    response::Redirect,
    routing::{get, post},
    Router,
};
use state::AppState;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

#[derive(OpenApi)]
#[openapi(
    info(title = "rag-pipeline API", description = "Docling -> embeddings -> Weaviate ingestion, hybrid search and chat"),
    paths(handlers::health::health, handlers::ingest::ingest, handlers::search::search, handlers::chat::chat, handlers::chat_stream::chat_stream,
        handlers::collections::list_collections,
        handlers::collections::list_documents,
        handlers::collections::document_chunks
    ),
    components(schemas(
        dto::SearchReq,
        dto::ChatReq,
        dto::RetrievalParams,
        crate::models::search::SearchMode,
        crate::models::collection::CollectionInfo,
        crate::models::collection::DocumentInfo,
        crate::models::collection::ChunkInfo,
        crate::models::collection::ChunkPage,
        dto::UploadForm,
        crate::models::search::Hit,
        crate::models::document::IngestOutcome,
        crate::models::chat::ChatAnswer,
        crate::models::chat::SourceRef,
        crate::models::chat::ChatTurn
    ))
)]
struct ApiDoc;

pub fn router(state: AppState) -> Router {
    let limit = state.ctx.cfg.max_upload_mb * 1024 * 1024;
    Router::new()
        .route("/", get(|| async { Redirect::to("/docs") }))
        .route("/health", get(handlers::health::health))
        .route("/ingest", post(handlers::ingest::ingest))
        .route("/search", post(handlers::search::search))
        .route("/chat", post(handlers::chat::chat))
        .route("/chat/stream", post(handlers::chat_stream::chat_stream))
        .route("/collections", get(handlers::collections::list_collections))
        .route("/documents", get(handlers::collections::list_documents))
        .route("/documents/chunks", get(handlers::collections::document_chunks))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
        // Swagger UI at /docs, spec at /openapi.json
        .merge(SwaggerUi::new("/docs").url("/openapi.json", ApiDoc::openapi()))
}
