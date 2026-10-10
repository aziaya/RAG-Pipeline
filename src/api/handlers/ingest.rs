use crate::{
    api::{
        dto::UploadForm,
        error::{internal, ApiError},
        state::AppState,
    },
    models::document::IngestOutcome,
};
use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    Json,
};
use std::{path::Path, time::Duration};

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
pub async fn ingest(State(st): State<AppState>, mut mp: Multipart) -> Result<Json<IngestOutcome>, ApiError> {
    let field = mp
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(e.to_string()))?
        .ok_or_else(|| ApiError::bad_request("missing file field"))?;

    // Never trust the client's path: keep only the final component.
    let name = field
        .file_name()
        .and_then(|n| Path::new(n).file_name())
        .and_then(|n| n.to_str())
        .map(String::from)
        .ok_or_else(|| ApiError::bad_request("file needs a filename"))?;
    let data = field.bytes().await.map_err(|e| ApiError::bad_request(e.to_string()))?;

    st.ctx.store.ensure_schema().await.map_err(internal)?;
    let _permit = st.ingest_permits.acquire().await.map_err(internal)?;
    let dir = tempfile::tempdir().map_err(internal)?;
    let path = dir.path().join(&name); // keep the extension: format detection may rely on it
    tokio::fs::write(&path, &data).await.map_err(internal)?;

    let secs = st.ctx.cfg.ingest_timeout_secs;
    let out = tokio::time::timeout(Duration::from_secs(secs), st.ctx.pipeline.ingest_file(&path, &name))
        .await
        .map_err(|_| ApiError(StatusCode::GATEWAY_TIMEOUT, format!("ingest exceeded {secs}s")))?
        .map_err(internal)?;
    Ok(Json(out))
}
