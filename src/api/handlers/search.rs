use crate::{
    api::{
        dto::SearchReq,
        error::{internal, ApiError},
        state::AppState,
    },
    models::search::Hit,
};
use axum::{extract::State, Json};

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
pub async fn search(State(st): State<AppState>, Json(req): Json<SearchReq>) -> Result<Json<Vec<Hit>>, ApiError> {
    if req.query.trim().is_empty() {
        return Err(ApiError::bad_request("empty query"));
    }
    let mut base = st.ctx.cfg.retrieval_defaults();
    base.top_k = 5; // /search has always defaulted to 5
    let opts = req.params.apply(base);
    let hits = st.ctx.rag.search(&req.query, &opts).await.map_err(internal)?;
    Ok(Json(hits))
}
