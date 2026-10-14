use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// How passages are retrieved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// BM25 keywords + vector similarity, blended by `alpha`
    #[default]
    Hybrid,
    /// Vector similarity only (cosine)
    Semantic,
    /// BM25 keywords only
    Keyword,
}

impl SearchMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Semantic => "semantic",
            Self::Keyword => "keyword",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Hybrid => Self::Semantic,
            Self::Semantic => Self::Keyword,
            Self::Keyword => Self::Hybrid,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Hybrid => Self::Keyword,
            Self::Semantic => Self::Hybrid,
            Self::Keyword => Self::Semantic,
        }
    }
}

impl FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "hybrid" => Ok(Self::Hybrid),
            "semantic" | "vector" => Ok(Self::Semantic),
            "keyword" | "bm25" => Ok(Self::Keyword),
            other => Err(format!("unknown search mode `{other}` (use hybrid, semantic or keyword)")),
        }
    }
}

/// One retrieved chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Hit {
    pub id: Option<String>,
    /// Collection the passage's document was ingested into
    pub collection: String,
    /// Final relevance: the reranked blend when reranking ran, otherwise the retrieval score.
    pub score: Option<f64>,
    pub text: String,
    pub headings: Vec<String>,
    pub source: String,
    pub chunk_index: i64,
    /// Score from the search itself (hybrid/BM25 score, or 1 - cosine distance for semantic)
    pub retrieval_score: Option<f64>,
    /// Raw reranker score, when reranking ran
    pub rerank_score: Option<f64>,
}

/// Everything that shapes one retrieval.
#[derive(Debug, Clone, Default, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RetrievalOptions {
    pub mode: SearchMode,
    /// Hybrid only: 0 = keyword, 1 = vector
    pub alpha: f32,
    /// Passages handed on (to the model, or back to the caller)
    pub top_k: usize,
    pub rerank: bool,
    /// 0 = keep retrieval order, 1 = reranker order only
    pub rerank_weight: f32,
    /// How many passages to fetch for the reranker to choose from
    pub candidates: usize,
    /// Restrict to one ingested document
    pub source: Option<String>,
    /// Collections to search; empty means all of them
    #[serde(default)]
    pub collections: Vec<String>,
}

impl RetrievalOptions {
    /// Clamp every field into its valid range.
    pub fn sanitized(mut self) -> Self {
        self.alpha = self.alpha.clamp(0.0, 1.0);
        self.rerank_weight = self.rerank_weight.clamp(0.0, 1.0);
        self.top_k = self.top_k.clamp(1, 50);
        self.candidates = self.candidates.clamp(self.top_k, 100);
        self.collections.retain(|c| !c.trim().is_empty());
        self.collections.sort();
        self.collections.dedup();
        self
    }

    /// How many passages to ask the store for.
    pub fn fetch_limit(&self) -> usize {
        if self.rerank {
            self.candidates.max(self.top_k)
        } else {
            self.top_k
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parsing_and_cycling() {
        assert_eq!("Semantic".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
        assert_eq!("bm25".parse::<SearchMode>().unwrap(), SearchMode::Keyword);
        assert!("fuzzy".parse::<SearchMode>().is_err());
        assert_eq!(SearchMode::Hybrid.next().next().next(), SearchMode::Hybrid);
        assert_eq!(SearchMode::Hybrid.prev(), SearchMode::Keyword);
    }

    #[test]
    fn options_are_clamped_and_fetch_limit_follows_rerank() {
        let o = RetrievalOptions {
            mode: SearchMode::Hybrid,
            alpha: 3.0,
            top_k: 0,
            rerank: true,
            rerank_weight: -1.0,
            candidates: 2,
            ..Default::default()
        }
        .sanitized();
        assert_eq!((o.alpha, o.rerank_weight, o.top_k, o.candidates), (1.0, 0.0, 1, 2));
        let mut o = RetrievalOptions { top_k: 5, candidates: 30, rerank: true, ..o };
        assert_eq!(o.fetch_limit(), 30);
        o.rerank = false;
        assert_eq!(o.fetch_limit(), 5);
    }
}
