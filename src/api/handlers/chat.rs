use crate::{
    api::{
        dto::ChatReq,
        error::{internal, ApiError},
        state::AppState,
    },
    models::chat::ChatAnswer,
    services::rag::RagError,
};
use axum::{extract::State, http::StatusCode, Json};

/// Answer a question from the indexed documents (retrieve, then generate).
#[utoipa::path(
    post, path = "/chat",
    request_body = ChatReq,
    responses(
        (status = 200, description = "Answer with the chunks it was based on", body = ChatAnswer),
        (status = 400, description = "Empty or too long question"),
        (status = 500, description = "Embedding or Weaviate error"),
        (status = 502, description = "Chat model unreachable or returned nothing")
    )
)]
pub async fn chat(State(st): State<AppState>, Json(req): Json<ChatReq>) -> Result<Json<ChatAnswer>, ApiError> {
    let question = req.question.trim();
    if question.is_empty() || question.chars().count() > 4000 {
        return Err(ApiError::bad_request("question must be 1-4000 characters"));
    }
    let limit = req.limit.unwrap_or(st.ctx.cfg.chat_top_k).clamp(1, 20);
    let _permit = st.chat_permits.acquire().await.map_err(internal)?;
    let answer = st
        .ctx
        .rag
        .answer(question, &req.history, limit, req.alpha.clamp(0.0, 1.0), req.source.as_deref(), |_| {})
        .await
        .map_err(|e| match e {
            RagError::Retrieval(_) => internal(e),
            RagError::Generation(_) => {
                tracing::error!("chat failed: {e}");
                ApiError(StatusCode::BAD_GATEWAY, e.to_string())
            }
        })?;
    Ok(Json(answer))
}
