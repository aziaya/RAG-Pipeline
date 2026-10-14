# RAG Spark

Blazing-fast High-performance RAG pipeline in Rust — **CLI + HTTP API**.  
Document ingestion, hybrid search, and local LLM chat with a live TUI.
## Interfaces
RAG Spark ships with two first-class interfaces that share the same core pipeline:

| Interface | How to use | Best for |
|-----------|------------|----------|
| **CLI** | `cargo run --release -- <command>` | Scripting, one-off ingest/search/chat, automation |
| **HTTP API** | `cargo run --release -- serve` then hit `/ingest`, `/search`, `/chat` | Integration with other services, frontends, agents |

- **CLI modes**
  - Split-pane TUI (default) — live metrics, trace, and multi-screen workflow
  - Classic single-pane UI
  - Direct commands: `ingest`, `search`, `chat`
- **API**
  - OpenAPI / Swagger UI at `/docs`
  - Endpoints: `POST /ingest`, `POST /search`, `POST /chat`, `GET /health`



Both interfaces talk to the same Weaviate index and the same embedding + chat models.

Ingestion, hybrid retrieval and chat over your documents:  
**docling.rs** (extract + token-aware chunking) → **EmbeddingGemma 2** (OpenAI-compatible endpoint) → **Weaviate** (bring-your-own vectors) → answers from a local chat model (llama-server).

```
file -> sha256 -> unchanged? skip
     -> docling.rs convert -> HybridChunker (tokenizer.json of the embedding model)
     -> embed (heading path + chunk) -> replace the document's chunks -> Weaviate
question -> (rewrite follow-up using the conversation) -> hybrid | semantic | keyword search
         -> (rerank candidates, blend with retrieval score) -> top-k passages -> chat model -> answer + sources
```

## Project layout

```
src/
  main.rs               bootstrap: env, logging mode, services, dispatch
  config.rs             environment configuration
  context.rs            AppContext: wires the services once, cheap to clone
  util.rs               path / text helpers
  models/               plain data types (chat, document, health, search)
  services/             business logic and clients
    extractor.rs          docling.rs extraction + chunking (the only docling touchpoint)
    embedder.rs           embeddings client
    vector_store.rs       Weaviate over REST + GraphQL
    llm.rs                chat-model client and prompt building
    pipeline.rs           ingestion orchestration
    rag.rs                rewrite follow-up -> retrieve -> rerank -> answer (shared by API, TUI, CLI)
    reranker.rs           cross-encoder /rerank client + score blending
    sessions.rs           saved conversations (JSON files)
    health.rs             readiness checks
  api/                  HTTP layer
    mod.rs                router + OpenAPI / Swagger UI
    state.rs  error.rs  dto.rs
    handlers/             one file per endpoint: health, ingest, search, chat
  cli/                  command-line entry points
    mod.rs                clap definitions + dispatch
    commands.rs           ingest, search, serve
    plain_chat.rs         script-friendly chat
    classic.rs            single-pane prompt-and-menu UI
  observability/        tracing layer, event types, metrics registry
  tui/                  split-pane terminal UI
    app.rs  ui.rs  input.rs  markdown.rs  theme.rs  widgets.rs
    screens/              chat, ingest, search, docs, status
    obs/                  in-flight operations, metrics, live trace log
```

## Setup

1. Rust **1.92+** (docling 1.103 requires it).
2. `docker compose up -d` (Weaviate on :8080). Bump `WEAVIATE_VERSION` to a current release.
3. Serve the embedding model and a chat model with llama-server (or any OpenAI-compatible server) and set
   `EMBED_*` / `CHAT_*` in `.env`.
4. Download `tokenizer.json` from the **google/embeddinggemma-2** repo and set `CHUNK_TOKENIZER` to its absolute path.
5. `cp .env.example .env`. Values containing spaces or `|` (the task prefixes) **must be quoted**.
6. For PDFs and images export `ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib` (docling uses `ort-load-dynamic`).

## Run

```
cargo run --release                  # split-pane UI with live tracing (the default)
cargo run --release -- classic       # single-pane prompt-and-menu UI
cargo run --release -- serve         # HTTP API, Swagger UI at /docs
cargo run --release -- ingest ./docs
cargo run --release -- search "question"
cargo run --release -- chat "question"
```

## The split-pane UI

The active screen (Chat, Ingest, Search, Docs, Status) is on the left; observability is on the right
(or underneath on narrow terminals) and updates live while a question or an ingestion runs:

- **In flight**: every running operation (`ingest_file`, `extract`, `embed_batch`, `weaviate_write`, `search`,
  `llm_generate`) with its elapsed time.
- **Metrics**: files indexed / skipped / failed, chunks, embeddings, searches, chats, errors, LLM tokens and tok/s, and
  last / average / p95 latency per operation with a small trend chart.
- **Trace**: a timestamped, color-coded log of tracing events and span timings.

Keys: `Esc` leaves the text field (NAV mode), `i`/`Enter` types again, `1`-`5` / `h` `l` switch screens, `Tab` focuses
the trace pane, `v` verbosity (INFO/DEBUG/TRACE), `p` pause, `X` clear, `L` layout (right / bottom / hidden),
`?` help, `q` or `Ctrl-C` quit. Chat has slash-commands (`/help`). Set `RAG_LOG_FILE` to also write the trace to a file.

## Collections

A collection is a named group of documents (e.g. `legal`, `hr`). Every chunk stores its collection, and a document is
identified by *(collection, source)*, so the same file can live in two collections without clashing.

- **Ingest into a collection**: TUI Ingest screen (`Ctrl-K` picks an existing collection or creates one), CLI
  `ingest ./docs --collection legal`, API `POST /ingest` with a `collection` form field. No name = `DEFAULT_COLLECTION`.
- **Search and chat cover all collections by default.** Narrow them with `/collection` (picker, or `/collection a,b`) in
  chat, `Ctrl-K` on the Search screen, `--collection a,b` on the CLI, or `collection` / `collections` in API requests.
  Names are normalised (`Legal Docs` becomes `legal-docs`).
- **Upgrading an existing index**: on first start the `collection` property is added to the existing Weaviate class and your
  earlier chunks are moved into the default collection.
- **Browse**: the Docs screen lists documents with their collection (`f` filters, `d` deletes a document, `D` a whole
  collection). `Enter` opens the **chunk viewer** (paged, in document order; `Enter` reads a chunk, `n`/`p` page).
  API: `GET /collections`, `GET /documents`, `GET /documents/chunks?collection=&source=&limit=&offset=`.

## Chat: follow-ups, memory, retrieval controls

- **Follow-ups**: with history, the question is first rewritten into a standalone search query ("does it need a GPU?"
  becomes "kokoro GPU requirements"); the answer still sees your original wording. The chat shows what was searched for.
  This costs one short extra model call per follow-up (`/rewrite off` or `CHAT_REWRITE=false` disables it).
- **Memory**: the last turns are kept verbatim; past 12 messages the older ones are folded into a rolling summary that is
  fed back into the prompt. Conversations are saved automatically in `DATA_DIR/sessions`. `/new`, `/sessions` (resume),
  `/forget` (delete), `/memory` (inspect).
- **Retrieval modes**: `hybrid` (BM25 + vector, blended by alpha), `semantic` (vector only, cosine), `keyword` (BM25 only).
- **Reranking**: fetch `candidates` passages, have a cross-encoder score them, blend with the retrieval score
  (`rerank weight` 0 = retrieval order, 1 = reranker only; both scores are normalised first), keep the best `top-k`.
  If the reranker is down the answer still works and a warning is shown.
- **Streaming**: answers appear token by token with a live cursor and tokens/s readout; `Ctrl-X` (or `x` in navigate mode,
  `/cancel`) stops a running answer, which closes the connection so the model stops generating, and keeps the partial
  text out of memory. `/stream on|off`, a row in `/settings`, or `CHAT_STREAM=false` turns it off. The plain CLI streams
  to stdout too, and `POST /chat/stream` streams Server-Sent Events: `stage`, then one `token` event per piece of text,
  then `done` (same body as `POST /chat`) or `error`. Closing the connection stops generation. The trace pane shows
  time to first token (`first_token`) alongside the other timings.
- **Controls**: `/settings` opens an interactive panel (↑↓ row, ←→ change); or `/mode`, `/alpha`, `/topk`, `/candidates`,
  `/rerank on|off`, `/weight`, `/rewrite on|off`. The same options are accepted by `POST /chat` and `POST /search`.

Running a reranker locally: `llama-server -m <reranker>.gguf --reranking --embedding --pooling rank --port 8083`, then set
`RERANK_BASE_URL=http://127.0.0.1:8083/v1` and `RERANK_ENABLED=true`. Reranker quality through llama-server varies by model,
so compare results on your own documents.

## API

`POST /ingest` (multipart `file`), `POST /search`, `POST /chat`, `GET /health`; Swagger UI at `/docs`.
`/search` and `/chat` accept `mode`, `alpha`, `top_k`, `rerank`, `rerank_weight`, `candidates`, `source`, `collection`/`collections`; `/chat` also takes
`history`, `summary` and `rewrite`, and returns `searched_for` and `warnings`.

## Notes

- Weaviate is queried over REST + GraphQL (there is no official Rust client).
- Documents ingested from the terminal are identified by their full path; uploads through `/ingest` by filename (always together with their collection).
- Ingestion embeds before it deletes a document's old chunks, but delete + insert is not atomic.
- Without the `docling` feature only `.txt`, `.md` and `.csv` can be read.
