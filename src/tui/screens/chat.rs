use super::{RenderCtx, Screen};
use crate::{
    context::AppContext,
    models::{
        chat::{ChatAnswer, ChatMemory, RagStage, SourceRef},
        collection::{sanitize_collection, DocumentInfo},
        search::{RetrievalOptions, SearchMode},
        session::{Session, StoredMessage},
    },
    services::rag::Compaction,
    tui::{
        app::{AppMsg, Mode},
        input::{InputEvent, InputLine},
        markdown, theme,
        widgets::{centered, Picker, PickerEvent},
    },
    util::{score_bar, short, trunc},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, Paragraph},
    Frame,
};
use std::time::{Duration, Instant};
use tokio::{sync::mpsc::UnboundedSender, task::AbortHandle};

pub enum ChatMsg {
    Stage(RagStage),
    /// One piece of the answer as the model writes it.
    Token(String),
    Done(Result<ChatAnswer, String>),
    Docs(Result<Vec<DocumentInfo>, String>),
    Collections(Result<Vec<(String, usize)>, String>),
    Compacted(Compaction),
    CompactFailed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Assistant,
    System,
    Error,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Error => "error",
        }
    }

    fn from_str(s: &str) -> Role {
        match s {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "error" => Role::Error,
            _ => Role::System,
        }
    }
}

struct Msg {
    role: Role,
    text: String,
    sources: Vec<SourceRef>,
    took: Option<Duration>,
    searched_for: Option<String>,
    warnings: Vec<String>,
}

impl Msg {
    fn plain(role: Role, text: impl Into<String>) -> Self {
        Self { role, text: text.into(), sources: vec![], took: None, searched_for: None, warnings: vec![] }
    }
}

/// The answer as it is being written.
struct Streaming {
    text: String,
    tokens: usize,
    first: Option<Instant>,
}

struct Busy {
    started: Instant,
    stage: String,
}

enum PickerKind {
    /// (collections, document) per entry; `None` = all documents
    Source(Vec<Option<(String, String)>>),
    /// collection per entry; `None` = all collections
    Collection(Vec<Option<String>>),
    Session(Vec<String>),
}

const ROWS: [&str; 8] = ["Retrieval mode", "Hybrid alpha", "Top-K passages", "Rerank candidates", "Reranking", "Rerank weight", "Rewrite follow-ups", "Stream answers"];

const HELP: &str = "**Retrieval**\n- `/settings` interactive panel (↑↓ row, ←→ change)\n- `/mode hybrid|semantic|keyword`\n- `/alpha 0-1` hybrid blend: 0 = keywords, 1 = meaning\n- `/topk N` passages used (alias `/limit`)\n- `/rerank on|off` · `/weight 0-1` reranker influence · `/candidates N` pool size\n- `/rewrite on|off` rewrite follow-ups into standalone searches\n- `/stream on|off` show answers as they are written · **Ctrl-X** cancels a running answer\n- `/collection` pick collections to search (default: all) · `/collection a,b` · `/collection all`\n- `/source` search within one document\n\n**Conversation**\n- `/new` start fresh (the old one stays saved) · `/sessions` resume · `/forget` delete this one\n- `/memory` show what is remembered · `/show N` read source N in full\n\nPress **Esc** to leave the input, then **q** to quit.";

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_lowercase().as_str() {
        "on" | "true" | "yes" | "1" => Some(true),
        "off" | "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

fn round2(x: f32) -> f32 {
    (x * 100.0).round() / 100.0
}

pub struct ChatScreen {
    ctx: AppContext,
    tx: UnboundedSender<AppMsg>,
    input: InputLine,
    msgs: Vec<Msg>,
    memory: ChatMemory,
    session: Session,
    pending_question: Option<String>,
    busy: Option<Busy>,
    compacting: bool,
    scroll: usize,
    opts: RetrievalOptions,
    rewrite: bool,
    stream: bool,
    streaming: Option<Streaming>,
    task: Option<AbortHandle>,
    picker: Option<(Picker, PickerKind)>,
    settings: Option<usize>,
    last_sources: Vec<SourceRef>,
}

impl ChatScreen {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        let opts = ctx.cfg.retrieval_defaults();
        let rewrite = ctx.cfg.chat_rewrite;
        let stream = ctx.cfg.chat_stream;
        let session = ctx.sessions.new_session();
        let mut s = Self {
            ctx,
            tx,
            input: InputLine::default(),
            msgs: vec![],
            memory: ChatMemory::default(),
            session,
            pending_question: None,
            busy: None,
            compacting: false,
            scroll: 0,
            opts,
            rewrite,
            stream,
            streaming: None,
            task: None,
            picker: None,
            settings: None,
            last_sources: vec![],
        };
        s.reset_transcript();
        s
    }

    fn reset_transcript(&mut self) {
        self.msgs = vec![Msg::plain(Role::System, "Ask a question about your documents. Type /help for commands, /settings to tune retrieval.")];
    }

    fn new_conversation(&mut self) {
        self.cancel();
        self.memory = ChatMemory::default();
        self.last_sources.clear();
        self.session = self.ctx.sessions.new_session();
        self.reset_transcript();
    }

    /// Narrow retrieval to these collections and optionally one document.
    pub fn set_scope(&mut self, collections: Vec<String>, source: Option<String>) {
        self.opts.collections = collections;
        self.opts.source = source;
        let scope = self.scope_label();
        self.system(format!("Searching within **{scope}**."));
    }

    fn scope_label(&self) -> String {
        match (&self.opts.source, self.opts.collections.is_empty()) {
            (Some(s), _) => format!("{} ({})", short(s), self.opts.collections.join(", ")),
            (None, false) => format!("collections: {}", self.opts.collections.join(", ")),
            (None, true) => "all collections".to_string(),
        }
    }

    fn system(&mut self, text: String) {
        self.msgs.push(Msg::plain(Role::System, text));
        self.scroll = 0;
    }

    fn settings_line(&self) -> String {
        let o = &self.opts;
        let mode = match o.mode {
            SearchMode::Hybrid => format!("hybrid α{:.2}", o.alpha),
            m => m.label().to_string(),
        };
        let rerank = if o.rerank { format!("rerank ×{:.2} of {}", o.rerank_weight, o.candidates) } else { "rerank off".into() };
        format!("{mode} · top {} · {rerank} · rewrite {}", o.top_k, if self.rewrite { "on" } else { "off" })
    }

    /// Save the conversation (only once there is something to save).
    fn persist(&mut self) {
        if !self.msgs.iter().any(|m| m.role == Role::User) {
            return;
        }
        self.session.memory = self.memory.clone();
        self.session.messages = self
            .msgs
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| StoredMessage {
                role: m.role.as_str().into(),
                text: m.text.clone(),
                sources: m.sources.clone(),
                took_secs: m.took.map(|d| d.as_secs_f32()),
                searched_for: m.searched_for.clone(),
            })
            .collect();
        if let Err(e) = self.ctx.sessions.save(&mut self.session) {
            self.system(format!("Could not save the conversation: {e:#}"));
        }
    }

    fn resume(&mut self, id: &str) {
        match self.ctx.sessions.load(id) {
            Ok(s) => {
                self.msgs = s
                    .messages
                    .iter()
                    .map(|m| Msg {
                        role: Role::from_str(&m.role),
                        text: m.text.clone(),
                        sources: m.sources.clone(),
                        took: m.took_secs.map(Duration::from_secs_f32),
                        searched_for: m.searched_for.clone(),
                        warnings: vec![],
                    })
                    .collect();
                self.last_sources = self.msgs.iter().rev().find(|m| !m.sources.is_empty()).map(|m| m.sources.clone()).unwrap_or_default();
                self.memory = s.memory.clone();
                let title = s.title.clone();
                self.session = s;
                self.system(format!("Resumed **{title}**. I remember the earlier turns."));
            }
            Err(e) => self.system(format!("Could not load that conversation: {e:#}")),
        }
    }

    fn submit(&mut self, text: String) {
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        if let Some(cmd) = text.strip_prefix('/') {
            self.command(cmd);
            return;
        }
        if self.busy.is_some() {
            self.system("Still working on the previous question…".into());
            return;
        }
        self.msgs.push(Msg::plain(Role::User, text.clone()));
        self.scroll = 0;
        self.pending_question = Some(text.clone());
        self.busy = Some(Busy { started: Instant::now(), stage: "Searching your documents…".into() });

        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        let (memory, opts, rewrite) = (self.memory.clone(), self.opts.clone(), self.rewrite);
        self.streaming = None;
        let handle = if self.stream {
            tokio::spawn(async move {
                let (tx_stage, tx_token) = (tx.clone(), tx.clone());
                let res = ctx
                    .rag
                    .answer_stream(
                        &text,
                        &memory,
                        &opts,
                        rewrite,
                        move |s| {
                            let _ = tx_stage.send(AppMsg::Chat(ChatMsg::Stage(s)));
                        },
                        move |t| tx_token.send(AppMsg::Chat(ChatMsg::Token(t.to_string()))).is_ok(),
                    )
                    .await;
                let _ = tx.send(AppMsg::Chat(ChatMsg::Done(res.map_err(|e| e.to_string()))));
            })
        } else {
            tokio::spawn(async move {
                let tx2 = tx.clone();
                let res = ctx
                    .rag
                    .answer(&text, &memory, &opts, rewrite, move |s| {
                        let _ = tx2.send(AppMsg::Chat(ChatMsg::Stage(s)));
                    })
                    .await;
                let _ = tx.send(AppMsg::Chat(ChatMsg::Done(res.map_err(|e| e.to_string()))));
            })
        };
        self.task = Some(handle.abort_handle());
    }

    /// Stop the running answer. Dropping the request closes the connection to the model server,
    /// which stops generation. The partial text is kept on screen but not added to memory.
    fn cancel(&mut self) {
        if self.busy.is_none() {
            return;
        }
        if let Some(h) = self.task.take() {
            h.abort();
        }
        self.busy = None;
        self.pending_question = None;
        let partial = self.streaming.take().map(|s| s.text).unwrap_or_default();
        if !partial.trim().is_empty() {
            self.msgs.push(Msg::plain(Role::Assistant, partial.trim().to_string()));
        }
        self.system("Cancelled. The partial answer was not added to the conversation memory.".into());
    }

    fn command(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, ' ');
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match name {
            "help" => self.system(HELP.into()),
            "settings" | "set" => self.settings = Some(0),
            "new" | "clear" => {
                self.new_conversation();
            }
            "forget" => {
                let id = self.session.id.clone();
                let _ = self.ctx.sessions.delete(&id);
                self.new_conversation();
                self.system("Deleted the previous conversation from disk and started a new one.".into());
            }
            "memory" => {
                let summary = self.memory.summary.clone().unwrap_or_else(|| "(none yet)".into());
                self.system(format!(
                    "**Memory**\n- summary of older turns: {summary}\n- {} recent messages kept verbatim\n- saved as `{}` in {}",
                    self.memory.turns.len(),
                    self.session.id,
                    self.ctx.cfg.data_dir.join("sessions").display()
                ));
            }
            "sessions" => {
                let list = self.ctx.sessions.list();
                if list.is_empty() {
                    self.system("No saved conversations yet. They are saved automatically after each answer.".into());
                } else {
                    let items: Vec<String> = list
                        .iter()
                        .map(|m| format!("{}  ({} q · {})", trunc(&m.title, 48), m.exchanges, m.updated.get(..16).unwrap_or(&m.updated).replace('T', " ")))
                        .collect();
                    let ids = list.into_iter().map(|m| m.id).collect();
                    self.picker = Some((Picker::new("Resume a conversation", items), PickerKind::Session(ids)));
                }
            }
            "mode" => match arg.parse::<SearchMode>() {
                Ok(m) => {
                    self.opts.mode = m;
                    self.system(format!("Retrieval: **{}**", self.settings_line()));
                }
                Err(e) => self.system(e),
            },
            "alpha" => match arg.parse::<f32>() {
                Ok(a) => {
                    self.opts.alpha = round2(a.clamp(0.0, 1.0));
                    self.system(format!("Retrieval: **{}**", self.settings_line()));
                }
                Err(_) => self.system("Usage: `/alpha 0.5` (0 = keywords, 1 = meaning)".into()),
            },
            "topk" | "limit" => match arg.parse::<usize>() {
                Ok(n) => {
                    self.opts.top_k = n.clamp(1, 20);
                    self.opts.candidates = self.opts.candidates.max(self.opts.top_k);
                    self.system(format!("Retrieval: **{}**", self.settings_line()));
                }
                Err(_) => self.system("Usage: `/topk 4`".into()),
            },
            "candidates" => match arg.parse::<usize>() {
                Ok(n) => {
                    self.opts.candidates = n.clamp(self.opts.top_k, 100);
                    self.system(format!("Retrieval: **{}**", self.settings_line()));
                }
                Err(_) => self.system("Usage: `/candidates 20`".into()),
            },
            "rerank" => match parse_bool(arg) {
                Some(b) => {
                    self.opts.rerank = b;
                    let warn = if b && !self.ctx.reranker.configured() { "\n⚠ no reranker configured yet (set RERANK_BASE_URL); retrieval order will be used" } else { "" };
                    self.system(format!("Retrieval: **{}**{warn}", self.settings_line()));
                }
                None => self.system("Usage: `/rerank on` or `/rerank off`".into()),
            },
            "weight" => match arg.parse::<f32>() {
                Ok(w) => {
                    self.opts.rerank_weight = round2(w.clamp(0.0, 1.0));
                    self.system(format!("Retrieval: **{}**", self.settings_line()));
                }
                Err(_) => self.system("Usage: `/weight 0.7` (0 = retrieval order, 1 = reranker only)".into()),
            },
            "stream" => match parse_bool(arg) {
                Some(b) => {
                    self.stream = b;
                    self.system(format!("Streaming answers: **{}**", if b { "on" } else { "off" }));
                }
                None => self.system("Usage: `/stream on` or `/stream off`".into()),
            },
            "cancel" => self.cancel(),
            "rewrite" => match parse_bool(arg) {
                Some(b) => {
                    self.rewrite = b;
                    self.system(format!("Retrieval: **{}**", self.settings_line()));
                }
                None => self.system("Usage: `/rewrite on` or `/rewrite off`".into()),
            },
            "show" => match arg.parse::<usize>().ok().and_then(|n| n.checked_sub(1)).and_then(|i| self.last_sources.get(i)).cloned() {
                Some(s) => {
                    let h = if s.headings.is_empty() { String::new() } else { format!("{}\n\n", s.headings.join(" > ")) };
                    self.system(format!("**[{}] {}** (chunk {})\n\n{h}{}", s.n, s.source, s.chunk_index, s.text));
                }
                None => self.system("Usage: `/show N` (a source number from the last answer)".into()),
            },
            "collection" | "collections" => {
                if arg.is_empty() {
                    let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
                    tokio::spawn(async move {
                        let r = ctx.store.list_collections().await.map_err(|e| format!("{e:#}"));
                        let _ = tx.send(AppMsg::Chat(ChatMsg::Collections(r)));
                    });
                } else if arg.eq_ignore_ascii_case("all") {
                    self.set_scope(vec![], None);
                } else {
                    let names: Vec<String> = arg.split(',').map(|n| sanitize_collection(n, "")).filter(|n| !n.is_empty()).collect();
                    self.set_scope(names, None);
                }
            }
            "source" => {
                let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
                tokio::spawn(async move {
                    let r = ctx.store.list_documents(None).await.map_err(|e| format!("{e:#}"));
                    let _ = tx.send(AppMsg::Chat(ChatMsg::Docs(r)));
                });
            }
            _ => self.system(format!("Unknown command `/{name}`. Try `/help`.")),
        }
    }

    pub fn on_msg(&mut self, msg: ChatMsg) {
        match msg {
            ChatMsg::Stage(s) => {
                if let Some(b) = self.busy.as_mut() {
                    b.stage = match s {
                        RagStage::Rewriting => "Understanding your follow-up…".into(),
                        RagStage::Searching => "Searching your documents…".into(),
                        RagStage::Reranking { candidates } => format!("Reranking {candidates} candidates…"),
                        RagStage::Generating { passages } => format!("Reading {passages} passages and writing an answer…"),
                    };
                }
            }
            ChatMsg::Token(t) => {
                if self.busy.is_some() {
                    let s = self.streaming.get_or_insert_with(|| Streaming { text: String::new(), tokens: 0, first: None });
                    if s.first.is_none() {
                        s.first = Some(Instant::now());
                    }
                    s.text.push_str(&t);
                    s.tokens += 1;
                }
            }
            // A result that arrives after a cancel is stale.
            ChatMsg::Done(_) if self.busy.is_none() => {}
            ChatMsg::Done(Ok(ans)) => {
                self.streaming = None;
                self.task = None;
                let took = self.busy.take().map(|b| b.started.elapsed());
                if let Some(q) = self.pending_question.take() {
                    self.memory.push_exchange(&q, &ans.answer);
                }
                self.last_sources = ans.sources.clone();
                self.msgs.push(Msg {
                    role: Role::Assistant,
                    text: ans.answer,
                    sources: ans.sources,
                    took,
                    searched_for: ans.searched_for,
                    warnings: ans.warnings,
                });
                self.scroll = 0;
                self.persist();
                if self.memory.needs_compaction() && !self.compacting {
                    self.compacting = true;
                    let (ctx, tx, mem) = (self.ctx.clone(), self.tx.clone(), self.memory.clone());
                    tokio::spawn(async move {
                        let msg = match ctx.rag.compact_memory(&mem).await {
                            Ok(Some(c)) => ChatMsg::Compacted(c),
                            _ => ChatMsg::CompactFailed,
                        };
                        let _ = tx.send(AppMsg::Chat(msg));
                    });
                }
            }
            ChatMsg::Done(Err(e)) => {
                self.busy = None;
                self.task = None;
                if let Some(s) = self.streaming.take() {
                    if !s.text.trim().is_empty() {
                        self.msgs.push(Msg::plain(Role::Assistant, s.text.trim().to_string()));
                    }
                }
                self.pending_question = None;
                self.msgs.push(Msg::plain(Role::Error, e));
                self.scroll = 0;
            }
            ChatMsg::Compacted(c) => {
                self.compacting = false;
                if self.memory.turns.len() >= c.dropped {
                    self.memory.turns.drain(..c.dropped);
                    self.memory.summary = Some(c.summary);
                    self.system("Older turns were folded into a summary to keep the conversation memory compact.".into());
                    self.persist();
                }
            }
            ChatMsg::CompactFailed => self.compacting = false,
            ChatMsg::Docs(Ok(docs)) => {
                if docs.is_empty() {
                    self.system("No documents ingested yet.".into());
                } else {
                    let mut items = vec!["All documents".to_string()];
                    let mut values = vec![None];
                    for d in docs {
                        items.push(format!("{} / {}  ({} chunks)", d.collection, short(&d.source), d.chunks));
                        values.push(Some((d.collection, d.source)));
                    }
                    self.picker = Some((Picker::new("Search within a document", items), PickerKind::Source(values)));
                }
            }
            ChatMsg::Collections(Ok(cols)) => {
                if cols.is_empty() {
                    self.system("No collections yet. Ingest a document first.".into());
                } else {
                    let mut items = vec!["All collections (default)".to_string()];
                    let mut values = vec![None];
                    for (name, chunks) in cols {
                        items.push(format!("{name}  ({chunks} chunks)"));
                        values.push(Some(name));
                    }
                    self.picker = Some((Picker::new("Search within a collection", items), PickerKind::Collection(values)));
                }
            }
            ChatMsg::Collections(Err(e)) => self.msgs.push(Msg::plain(Role::Error, e)),
            ChatMsg::Docs(Err(e)) => self.msgs.push(Msg::plain(Role::Error, e)),
        }
    }

    fn adjust(&mut self, row: usize, dir: i32) {
        let o = &mut self.opts;
        let d = dir as f32;
        match row {
            0 => o.mode = if dir > 0 { o.mode.next() } else { o.mode.prev() },
            1 => o.alpha = round2((o.alpha + 0.05 * d).clamp(0.0, 1.0)),
            2 => {
                o.top_k = (o.top_k as i32 + dir).clamp(1, 20) as usize;
                o.candidates = o.candidates.max(o.top_k);
            }
            3 => o.candidates = (o.candidates as i32 + 5 * dir).clamp(o.top_k as i32, 100) as usize,
            4 => o.rerank = !o.rerank,
            5 => o.rerank_weight = round2((o.rerank_weight + 0.05 * d).clamp(0.0, 1.0)),
            6 => self.rewrite = !self.rewrite,
            7 => self.stream = !self.stream,
            _ => {}
        }
    }

    fn build_lines(&self, width: usize, tick: u64) -> Vec<Line<'static>> {
        let mut lines: Vec<Line<'static>> = vec![];
        for m in &self.msgs {
            let (label, color) = match m.role {
                Role::User => ("you", Color::Blue),
                Role::Assistant => ("assistant", Color::Cyan),
                Role::System => ("·", Color::DarkGray),
                Role::Error => ("error", Color::Red),
            };
            lines.push(Line::from(Span::styled(label, theme::bold(color))));
            let base = match m.role {
                Role::Error => Style::new().fg(Color::Red),
                Role::System => theme::dim(),
                _ => Style::new(),
            };
            lines.extend(markdown::render(&m.text, width, base));
            if !m.sources.is_empty() {
                lines.push(Line::from(Span::styled("Sources", theme::bold(Color::White))));
                for s in &m.sources {
                    let section = if s.headings.is_empty() { String::new() } else { format!(" › {}", trunc(&s.headings.join(" › "), 40)) };
                    let mut spans = vec![
                        Span::styled(format!("  [{}] ", s.n), theme::bold(Color::Yellow)),
                        Span::styled(short(&s.source), Style::new().fg(Color::Green)),
                        Span::styled(if s.collection.is_empty() { String::new() } else { format!(" · {}", s.collection) }, Style::new().fg(Color::Magenta)),
                        Span::styled(section, theme::dim()),
                        Span::styled(format!("  {}", score_bar(s.score)), theme::dim()),
                    ];
                    if let Some(r) = s.rerank_score {
                        spans.push(Span::styled(format!("  rerank {r:+.2}"), Style::new().fg(Color::Magenta)));
                    }
                    lines.push(Line::from(spans));
                }
            }
            for w in &m.warnings {
                lines.extend(markdown::render(&format!("⚠ {w}"), width, Style::new().fg(Color::Yellow)));
            }
            if m.took.is_some() || m.searched_for.is_some() {
                let mut foot = String::new();
                if let Some(t) = m.took {
                    foot.push_str(&format!("answered in {:.1}s", t.as_secs_f32()));
                }
                if let Some(q) = &m.searched_for {
                    if !foot.is_empty() {
                        foot.push_str(" · ");
                    }
                    foot.push_str(&format!("searched for “{q}”"));
                }
                lines.extend(markdown::render(&foot, width, theme::dim()));
            }
            lines.push(Line::default());
        }
        if let Some(s) = &self.streaming {
            lines.push(Line::from(Span::styled("assistant", theme::bold(Color::Cyan))));
            lines.extend(markdown::render(&format!("{}▌", s.text), width, Style::new()));
            let rate = s
                .first
                .map(|f| f.elapsed().as_secs_f64())
                .filter(|secs| *secs > 0.5)
                .map(|secs| format!(" · {:.1} tok/s", s.tokens as f64 / secs))
                .unwrap_or_default();
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", theme::spinner(tick)), Style::new().fg(Color::Cyan)),
                Span::styled(format!("writing · {} tokens{rate} · Ctrl-X to cancel", s.tokens), theme::dim()),
            ]));
        } else if let Some(b) = &self.busy {
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", theme::spinner(tick)), Style::new().fg(Color::Cyan)),
                Span::raw(b.stage.clone()),
                Span::styled(format!("  {}s · Ctrl-X to cancel", b.started.elapsed().as_secs()), theme::dim()),
            ]));
        }
        lines
    }

    fn render_settings(&self, f: &mut Frame, row: usize) {
        let area = f.area();
        let rect = centered(area, 78.min(area.width.saturating_sub(4)), (ROWS.len() as u16 * 2 + 6).min(area.height.saturating_sub(2)));
        f.render_widget(Clear, rect);
        let o = &self.opts;
        let values = [
            o.mode.label().to_string(),
            format!("{:.2}", o.alpha),
            o.top_k.to_string(),
            o.candidates.to_string(),
            if o.rerank { "on".into() } else { "off".into() },
            format!("{:.2}", o.rerank_weight),
            if self.rewrite { "on".into() } else { "off".into() },
            if self.stream { "on".into() } else { "off".into() },
        ];
        let notes = [
            "hybrid = keywords + meaning · semantic = meaning only · keyword = exact terms",
            "hybrid only: 0 = keywords, 1 = meaning",
            "passages handed to the model",
            "fetched for the reranker to choose from",
            "a cross-encoder reorders the candidates",
            "0 = keep retrieval order, 1 = reranker order only",
            "use the conversation to rewrite follow-ups before searching",
            "show the answer as it is written instead of all at once",
        ];
        let disabled = [false, o.mode != SearchMode::Hybrid, false, !o.rerank, false, !o.rerank, false, false];
        let mut lines: Vec<Line> = vec![Line::default()];
        for i in 0..ROWS.len() {
            let sel = i == row;
            let dim = disabled[i];
            let label_style = if sel {
                Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else if dim {
                theme::dim()
            } else {
                Style::new().fg(Color::White)
            };
            let val_style = if dim { theme::dim() } else { theme::bold(Color::Yellow) };
            lines.push(Line::from(vec![
                Span::styled(format!(" {:<20}", ROWS[i]), label_style),
                Span::styled(format!("  ◀ {:^10} ▶", values[i]), val_style),
            ]));
            lines.push(Line::from(Span::styled(format!("   {}", notes[i]), theme::dim())));
        }
        let reranker = if self.ctx.reranker.configured() {
            format!("reranker: {}", self.ctx.reranker.endpoint().unwrap_or(""))
        } else {
            "reranker: not configured (set RERANK_BASE_URL to use it)".to_string()
        };
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(format!(" {reranker}"), if o.rerank && !self.ctx.reranker.configured() { Style::new().fg(Color::Yellow) } else { theme::dim() })));
        let block = theme::pane(Line::from(Span::styled(" Retrieval settings ", theme::bold(Color::White))), true)
            .title_bottom(Line::from(Span::styled(" ↑↓ row · ←→ change · Esc close ", theme::dim())));
        f.render_widget(Paragraph::new(lines).block(block), rect);
    }
}

impl Screen for ChatScreen {
    fn has_input(&self) -> bool {
        true
    }

    fn popup_open(&self) -> bool {
        self.picker.is_some() || self.settings.is_some()
    }

    fn on_key(&mut self, key: KeyEvent, mode: Mode) -> bool {
        if self.picker.is_some() {
            let ev = self.picker.as_mut().map(|(p, _)| p.on_key(key)).unwrap_or(PickerEvent::None);
            match ev {
                PickerEvent::Chosen(i) => {
                    if let Some((_, kind)) = self.picker.take() {
                        match kind {
                            PickerKind::Source(values) => match values.get(i).cloned().flatten() {
                                Some((collection, source)) => self.set_scope(vec![collection], Some(source)),
                                None => self.set_scope(self.opts.collections.clone(), None),
                            },
                            PickerKind::Collection(values) => match values.get(i).cloned().flatten() {
                                Some(name) => self.set_scope(vec![name], None),
                                None => self.set_scope(vec![], None),
                            },
                            PickerKind::Session(ids) => {
                                if let Some(id) = ids.get(i) {
                                    let id = id.clone();
                                    self.resume(&id);
                                }
                            }
                        }
                    }
                }
                PickerEvent::Cancel => self.picker = None,
                PickerEvent::None => {}
            }
            return true;
        }
        if let Some(row) = self.settings {
            match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => {
                    self.settings = None;
                    let line = self.settings_line();
                    self.system(format!("Retrieval: **{line}**"));
                }
                KeyCode::Up | KeyCode::Char('k') => self.settings = Some(if row == 0 { ROWS.len() - 1 } else { row - 1 }),
                KeyCode::Down | KeyCode::Char('j') => self.settings = Some((row + 1) % ROWS.len()),
                KeyCode::Left | KeyCode::Char('h') => self.adjust(row, -1),
                KeyCode::Right | KeyCode::Char('l') => self.adjust(row, 1),
                _ => {}
            }
            return true;
        }
        match key.code {
            KeyCode::PageUp => {
                self.scroll += 8;
                return true;
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_sub(8);
                return true;
            }
            _ => {}
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('x') {
            self.cancel();
            return true;
        }
        if mode == Mode::Input {
            if let InputEvent::Submit(t) = self.input.on_key(key) {
                self.submit(t);
            }
            return true;
        }
        match key.code {
            KeyCode::Char('s') => self.settings = Some(0),
            KeyCode::Char('x') => self.cancel(),
            KeyCode::Up | KeyCode::Char('k') => self.scroll += 1,
            KeyCode::Down | KeyCode::Char('j') => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::End | KeyCode::Char('G') => self.scroll = 0,
            _ => return false,
        }
        true
    }

    fn render(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx) {
        let [log_area, input_area] = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).areas(area);

        let block = theme::pane(Line::from(Span::styled(" Chat ", theme::bold(Color::White))), rc.focused && rc.mode == Mode::Nav);
        let inner = block.inner(log_area);
        let lines = self.build_lines(inner.width as usize, rc.tick);
        let h = inner.height as usize;
        let max_scroll = lines.len().saturating_sub(h);
        self.scroll = self.scroll.min(max_scroll);
        let offset = max_scroll - self.scroll;
        f.render_widget(block, log_area);
        f.render_widget(Paragraph::new(lines).scroll((offset.min(u16::MAX as usize) as u16, 0)), inner);

        let scope = self.scope_label();
        let title = Line::from(vec![
            Span::styled(" Ask ", theme::bold(Color::White)),
            Span::styled(format!("· {scope} · {} ", self.settings_line()), theme::dim()),
        ]);
        let typing = rc.focused && rc.mode == Mode::Input;
        let ib = theme::pane(title, typing);
        let iin = ib.inner(input_area);
        f.render_widget(ib, input_area);
        self.input.render(f, iin, "Ask about your documents… (/help · /settings)", typing && !self.popup_open());
    }

    fn render_popup(&mut self, f: &mut Frame) {
        if let Some((p, _)) = self.picker.as_mut() {
            p.render(f);
        }
        if let Some(row) = self.settings {
            self.render_settings(f, row);
        }
    }

    fn hints(&self, mode: Mode) -> &'static str {
        match mode {
            Mode::Input => "Enter send · Ctrl-X cancel answer · /settings · /help · PgUp/PgDn scroll · Esc → navigate",
            Mode::Nav => "i or Enter type · s settings · j/k PgUp/PgDn scroll · 1-5 screens · Tab pane · q quit",
        }
    }
}
