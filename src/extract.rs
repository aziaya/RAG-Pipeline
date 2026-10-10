//! Extraction + token-aware chunking via docling.rs (HybridChunker).
//! This is the only module that touches docling, so it can be swapped for the
//! docling-serve (Python) sidecar later without changing the rest of the pipeline.

use anyhow::Result;
use std::path::Path;

/// Our own chunk type: the rest of the pipeline never sees docling types.
#[derive(Debug, Clone)]
pub struct RawChunk {
    /// Chunk text as produced by the chunker.
    pub text: String,
    /// Heading path + chunk text. This is what gets embedded and BM25-indexed.
    pub context_text: String,
    pub headings: Vec<String>,
    pub item_refs: Vec<String>,
}

#[cfg(feature = "docling")]
pub fn extract_and_chunk(path: &Path, tokenizer: &Path, max_tokens: usize) -> Result<Vec<RawChunk>> {
    use anyhow::anyhow;
    use docling::chunker::{contextualize, HuggingFaceTokenizer, HybridChunker};
    use docling::{DocumentConverter, SourceDocument};

    // Built per call so nothing non-Send crosses threads. If profiling shows model
    // load time matters, keep a converter per worker thread instead.
    let converter = DocumentConverter::new();
    let source = SourceDocument::from_file(path).map_err(|e| anyhow!("open {}: {e:?}", path.display()))?;
    let result = converter
        .convert(source)
        .map_err(|e| anyhow!("convert {}: {e:?}", path.display()))?;

    let tok = HuggingFaceTokenizer::from_file(tokenizer, max_tokens)
        .map_err(|e| anyhow!("load tokenizer {}: {e:?}", tokenizer.display()))?;

    let mut out = Vec::new();
    for chunk in HybridChunker::new(tok).chunk(&result.document) {
        if chunk.text.trim().is_empty() {
            continue;
        }
        // Go through serde so we don't depend on the exact field types.
        let headings: Vec<String> = serde_json::to_value(&chunk.headings)
            .ok()
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        let item_refs: Vec<String> = chunk.doc_items.iter().map(|i| i.self_ref.to_string()).collect();
        out.push(RawChunk {
            context_text: contextualize(&chunk),
            text: chunk.text.clone(),
            headings,
            item_refs,
        });
    }
    Ok(out)
}

/// Fallback when built without the `docling` feature: plain text only (txt, md, csv),
/// split into paragraph-based chunks. Handy for trying the pipeline without ONNX/PDF support.
#[cfg(not(feature = "docling"))]
pub fn extract_and_chunk(path: &Path, _tokenizer: &Path, _max_tokens: usize) -> Result<Vec<RawChunk>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if !matches!(ext.as_str(), "txt" | "md" | "markdown" | "csv") {
        anyhow::bail!("built without the `docling` feature; only txt/md/csv can be read ({})", path.display());
    }
    let text = std::fs::read_to_string(path)?;
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for para in text.split("\n\n") {
        if !cur.is_empty() && cur.len() + para.len() > 1500 {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(para.trim());
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    Ok(chunks
        .into_iter()
        .map(|t| RawChunk { context_text: t.clone(), text: t, headings: vec![], item_refs: vec![] })
        .collect())
}
