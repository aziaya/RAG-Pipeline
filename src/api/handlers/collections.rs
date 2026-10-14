//! Read-only views of what is stored: collections, documents and a document's chunks.

use crate::{
    api::{
        error::{internal, ApiError},
        state::AppState,
    },
    models::collection::{sanitize_collection, ChunkPage, CollectionInfo, DocumentInfo},
};
use axum::{
    extract::{Query, State},
    Json,
};
use serde::Deserialize;

#[derive(Deserialize, utoipa::IntoParams)]
pub struct DocumentsQuery {
    /// Only this collection (default: all)
    pub collection: Option<String>,
}

#[derive(Deserialize, utoipa::IntoParams)]
pub struct ChunksQuery {
    /// Collection of the document (default: the server's DEFAULT_COLLECTION)
    pub collection: Option<String>,
    /// The document's source id, as listed by `GET /documents`
    pub source: String,
    /// Page size, 1-200 (default 20)
    pub limit: Option<usize>,
    /// Chunks to skip (default 0)
    pub offset: Option<usize>,
}

/// Every collection with its document and chunk counts.
#[utoipa::path(get, path = "/collections", responses((status = 200, body = Vec<CollectionInfo>), (status = 500, description = "Weaviate error")))]
pub async fn list_collections(State(st): State<AppState>) -> Result<Json<Vec<CollectionInfo>>, ApiError> {
    let chunks = st.ctx.store.list_collections().await.map_err(internal)?;
    let docs = st.ctx.store.list_documents(None).await.map_err(internal)?;
    Ok(Json(
        chunks
            .into_iter()
            .map(|(name, chunks)| CollectionInfo { documents: docs.iter().filter(|d| d.collection == name).count(), name, chunks })
            .collect(),
    ))
}

/// Ingested documents, optionally limited to one collection.
#[utoipa::path(get, path = "/documents", params(DocumentsQuery), responses((status = 200, body = Vec<DocumentInfo>), (status = 500, description = "Weaviate error")))]
pub async fn list_documents(State(st): State<AppState>, Query(q): Query<DocumentsQuery>) -> Result<Json<Vec<DocumentInfo>>, ApiError> {
    let collection = q.collection.map(|c| sanitize_collection(&c, &st.ctx.cfg.default_collection));
    Ok(Json(st.ctx.store.list_documents(collection.as_deref()).await.map_err(internal)?))
}

/// A page of one document's stored chunks, in document order.
#[utoipa::path(
    get, path = "/documents/chunks", params(ChunksQuery),
    responses((status = 200, body = ChunkPage), (status = 400, description = "Missing source"), (status = 500, description = "Weaviate error"))
)]
pub async fn document_chunks(State(st): State<AppState>, Query(q): Query<ChunksQuery>) -> Result<Json<ChunkPage>, ApiError> {
    if q.source.trim().is_empty() {
        return Err(ApiError::bad_request("source is required"));
    }
    let default = &st.ctx.cfg.default_collection;
    let collection = q.collection.map(|c| sanitize_collection(&c, default)).unwrap_or_else(|| default.clone());
    let (limit, offset) = (q.limit.unwrap_or(20).clamp(1, 200), q.offset.unwrap_or(0));
    let total = st.ctx.store.count_chunks(&collection, &q.source).await.map_err(internal)?;
    let chunks = st.ctx.store.list_chunks(&collection, &q.source, limit, offset).await.map_err(internal)?;
    Ok(Json(ChunkPage { collection, source: q.source, total, offset, chunks }))
}
