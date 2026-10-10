use serde::Serialize;

/// One retrieved chunk.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct Hit {
    pub id: Option<String>,
    pub score: Option<f64>,
    pub text: String,
    pub headings: Vec<String>,
    pub source: String,
    pub chunk_index: i64,
}
