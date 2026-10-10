# rag-pipeline

Rust ingestion + hybrid retrieval: **docling.rs** (extract + token-aware HybridChunker)
-> **EmbeddingGemma 2** (via an OpenAI-compatible endpoint) -> **Weaviate** (bring-your-own vectors).

```
file -> sha256 -> unchanged? skip
     -> docling.rs convert -> HybridChunker (tokenizer.json of the embedding model)
     -> embed (context text = heading path + chunk) -> delete old chunks of that source -> batch insert
search: embed query -> Weaviate hybrid (BM25 + vector, alpha) -> hits
```

## Setup

1. Rust **1.92+** (docling 1.103 requires it).
2. `docker compose up -d` (Weaviate on :8080). Bump `WEAVIATE_VERSION` to a current release.
3. Serve EmbeddingGemma 2 behind an OpenAI-compatible `/v1/embeddings` endpoint (Ollama, vLLM, llama.cpp server...).
   Set `EMBED_BASE_URL` / `EMBED_MODEL`.
4. Download `tokenizer.json` from the **google/embeddinggemma-2** repo (not the 300m one) to
   `models/embeddinggemma-2/tokenizer.json`. If the repo does not ship one, see "Open items".
5. `cp .env.example .env` and fill in `EMBED_DOC_PREFIX` / `EMBED_QUERY_PREFIX` from the model card.
6. Use bfloat16/float32 when serving the model (the model card says activations exceed float16 range).

## Interactive terminal UI

Run with no command in a terminal:

```
cargo run --release
```

A menu (arrow keys, Enter, Esc) offers: chat, search, ingest, browse/manage documents, and a system status screen
that checks Weaviate, the embedding server, the chat server, the tokenizer file and ONNX Runtime.
Ingesting shows an overall progress bar, a live stage line (extracting, embedding n/m chunks, writing) and a
summary table. Inside chat: `/source` (pick a document), `/show N`, `/alpha`, `/limit`, `/clear`, `/exit`; Up/Down
recall earlier questions. Documents ingested from the terminal are identified by their full path; uploads through
`/ingest` are identified by filename.

## Run

Swagger UI is at **http://localhost:3000/docs** (OpenAPI JSON at `/openapi.json`); `/` redirects to it.
`serve` starts even if Weaviate is unreachable (it logs a warning and retries on `/ingest`).
PDFs need ONNX Runtime at runtime when docling's `ort-load-dynamic` feature is on:
`export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib` (or `.so`).

```
cargo run --release -- ingest ./docs          # file or directory, recursive
cargo run --release -- search "what is the notice period?" --limit 5 --alpha 0.5
cargo run --release -- serve                   # HTTP API
cargo run --release -- chat                    # interactive chat in the terminal
cargo run --release -- chat "what is kokoro"   # one question, then exit
```

`POST /chat` retrieves with hybrid search, then asks the chat model (llama-server on CHAT_BASE_URL) to answer from the chunks, returning the answer plus numbered sources. For Gemma 4 start the chat server with `--jinja --chat-template-kwargs '{"enable_thinking":false}'`.

```
curl -F file=@report.pdf http://localhost:3000/ingest
curl -X POST http://localhost:3000/search -H 'content-type: application/json' \
     -d '{"query":"termination clause","limit":5,"alpha":0.5}'
```

## Open items (verify on your side)

- **Tokenizer file**: I could not confirm `tokenizer.json` ships in the embeddinggemma-2 repo. Check the Files tab.
- **Prefixes**: exact task-prefix strings come from the model card; they are left empty in `.env.example`.
- **Max input length** of the model: check the card; keep `CHUNK_MAX_TOKENS` well below it.
- **Sandbox verification**: the project was compile-checked and unit-tested here with Rust 1.91
  (`--ignore-rust-version`), with docling built **without** its ONNX/PDF features because the sandbox could not
  download ONNX Runtime. So the docling API calls (convert, HybridChunker, contextualize) are verified
  to compile against 1.103.0, but PDF/OCR/audio runtime behavior and the Weaviate/embedding calls
  against live services are **not** tested. Run the 10-20 mixed-file inspection before trusting it.
- Weaviate is queried via REST + GraphQL (no official Rust client).
- Ingestion deletes a source's old chunks only after the new ones are embedded, but delete + insert is not atomic.
