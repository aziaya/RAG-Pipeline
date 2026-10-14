use crate::{
    api::{dto::ChatReq, error::ApiError, state::AppState},
    models::chat::{ChatMemory, RagStage},
};
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use futures_util::Stream;
use serde_json::{json, Value};
use std::convert::Infallible;
use tokio::sync::mpsc::unbounded_channel;

fn stage_json(s: &RagStage) -> Value {
    match s {
        RagStage::Rewriting => json!({ "stage": "rewriting" }),
        RagStage::Searching => json!({ "stage": "searching" }),
        RagStage::Reranking { candidates } => json!({ "stage": "reranking", "candidates": candidates }),
        RagStage::Generating { passages } => json!({ "stage": "generating", "passages": passages }),
    }
}

/// Answer a question and stream it as Server-Sent Events.
///
/// Events: `stage` (`{"stage": "rewriting|searching|reranking|generating", ...}`), `token`
/// (`{"text": "..."}`, one per piece of generated text), then exactly one of `done` (the full
/// answer with its sources, same shape as `POST /chat`) or `error` (`{"error": "..."}`).
/// Closing the connection stops generation.
#[utoipa::path(
    post, path = "/chat/stream",
    request_body = ChatReq,
    responses(
        (status = 200, description = "text/event-stream of stage, token, done and error events", content_type = "text/event-stream", body = String),
        (status = 400, description = "Empty or too long question")
    )
)]
pub async fn chat_stream(State(st): State<AppState>, Json(req): Json<ChatReq>) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let question = req.question.trim().to_string();
    if question.is_empty() || question.chars().count() > 4000 {
        return Err(ApiError::bad_request("question must be 1-4000 characters"));
    }
    let opts = req.params.apply(st.ctx.cfg.retrieval_defaults());
    let memory = ChatMemory { summary: req.summary.clone().filter(|s| !s.trim().is_empty()), turns: req.history.clone() };
    let rewrite = req.rewrite.unwrap_or(st.ctx.cfg.chat_rewrite);

    let (tx, rx) = unbounded_channel::<Event>();
    let (ctx, permits) = (st.ctx.clone(), st.chat_permits.clone());
    tokio::spawn(async move {
        let Ok(_permit) = permits.acquire_owned().await else { return };
        let (tx_stage, tx_token) = (tx.clone(), tx.clone());
        let res = ctx
            .rag
            .answer_stream(
                &question,
                &memory,
                &opts,
                rewrite,
                move |s| {
                    let _ = tx_stage.send(Event::default().event("stage").json_data(stage_json(&s)).unwrap_or_default());
                },
                // A failed send means the client went away: returning false stops generation.
                move |t| tx_token.send(Event::default().event("token").json_data(json!({ "text": t })).unwrap_or_default()).is_ok(),
            )
            .await;
        let last = match res {
            Ok(answer) => Event::default().event("done").json_data(&answer).unwrap_or_default(),
            Err(e) => {
                tracing::error!("streamed chat failed: {e}");
                Event::default().event("error").json_data(json!({ "error": e.to_string() })).unwrap_or_default()
            }
        };
        let _ = tx.send(last);
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|e| (Ok::<Event, Infallible>(e), rx)) });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
