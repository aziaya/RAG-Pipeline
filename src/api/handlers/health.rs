/// Liveness check.
#[utoipa::path(get, path = "/health", responses((status = 200, description = "Server is up", body = String)))]
pub async fn health() -> &'static str {
    "ok"
}
