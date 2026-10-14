use serde::Serialize;

/// A named group of documents (e.g. "legal", "hr"). Documents live in exactly one collection.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct CollectionInfo {
    pub name: String,
    pub documents: usize,
    pub chunks: usize,
}

/// One ingested document and where it lives.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct DocumentInfo {
    pub collection: String,
    pub source: String,
    pub chunks: usize,
}

/// One stored chunk of a document, as shown by the chunk viewer.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct ChunkInfo {
    pub id: Option<String>,
    pub chunk_index: i64,
    pub headings: Vec<String>,
    pub text: String,
}

/// A page of a document's chunks, in document order.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct ChunkPage {
    pub collection: String,
    pub source: String,
    pub total: usize,
    pub offset: usize,
    pub chunks: Vec<ChunkInfo>,
}

/// Normalise a user-typed collection name: lowercase, spaces become `-`, only `a-z 0-9 - _`,
/// at most 40 characters. An empty result falls back to `fallback`.
pub fn sanitize_collection(raw: &str, fallback: &str) -> String {
    let cleaned: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_whitespace() { '-' } else { c })
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(40)
        .collect();
    let cleaned = cleaned.trim_matches(|c| c == '-' || c == '_').to_string();
    if cleaned.is_empty() {
        fallback.to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_names_are_normalised() {
        assert_eq!(sanitize_collection("  Legal Docs ", "default"), "legal-docs");
        assert_eq!(sanitize_collection("HR_2026!", "default"), "hr_2026");
        assert_eq!(sanitize_collection("  ", "default"), "default");
        assert_eq!(sanitize_collection("../../etc", "default"), "etc");
        assert_eq!(sanitize_collection(&"a".repeat(80), "default").len(), 40);
    }
}
