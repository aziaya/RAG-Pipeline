use crate::{
    api::{
        dto::ChatReq,
        error::{internal, ApiError},
        state::AppState,
    },
    models::chat::{ChatAnswer, ChatMemory},
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
    let opts = req.params.apply(st.ctx.cfg.retrieval_defaults());
    let memory = ChatMemory { summary: req.summary.clone().filter(|s| !s.trim().is_empty()), turns: req.history.clone() };
    let rewrite = req.rewrite.unwrap_or(st.ctx.cfg.chat_rewrite);
    let _permit = st.chat_permits.acquire().await.map_err(internal)?;
    let answer = st
        .ctx
        .rag
        .answer(question, &memory, &opts, rewrite, |_| {})
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
