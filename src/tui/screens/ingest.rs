use super::{RenderCtx, Screen};
use crate::{
    context::AppContext,
    models::{
        collection::sanitize_collection,
        document::{IngestOutcome, Stage},
    },
    tui::{
        app::{AppMsg, Mode},
        input::{InputEvent, InputLine},
        theme,
        widgets::{Picker, PickerEvent},
    },
    util::{clean_path, collect_files, complete_path, trunc},
};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Cell, Gauge, Paragraph, Row, Table, TableState},
    Frame,
};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc::UnboundedSender;

pub enum IngestMsg {
    /// Existing collections, for the collection picker.
    Collections(Result<Vec<(String, usize)>, String>),
    Started { idx: usize },
    Stage { idx: usize, stage: Stage },
    Done { idx: usize, result: Result<IngestOutcome, String> },
    Aborted(String),
    Finished { cancelled: bool },
}

enum JobState {
    Queued,
    Running(Stage),
    Indexed(usize),
    Skipped,
    Failed(String),
}

struct Job {
    name: String,
    state: JobState,
    started: Option<Instant>,
    took: Option<Duration>,
}

pub struct IngestScreen {
    ctx: AppContext,
    tx: UnboundedSender<AppMsg>,
    input: InputLine,
    jobs: Vec<Job>,
    running: bool,
    cancel: Arc<AtomicBool>,
    run_started: Option<Instant>,
    notice: Option<String>,
    table: TableState,
    /// Collection the next run saves into.
    collection: String,
    /// The input box is asking for a new collection name instead of a path.
    naming: bool,
    /// Collection picker; `None` entry = "new collection…".
    picker: Option<(Picker, Vec<Option<String>>)>,
}

impl IngestScreen {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        let collection = ctx.cfg.default_collection.clone();
        Self {
            ctx,
            tx,
            input: InputLine::default(),
            jobs: vec![],
            running: false,
            cancel: Arc::new(AtomicBool::new(false)),
            run_started: None,
            notice: Some("Type a file or folder path and press Enter. Tab completes paths. Ctrl-K chooses the collection.".into()),
            table: TableState::default(),
            collection,
            naming: false,
            picker: None,
        }
    }

    fn submit(&mut self, raw: String) {
        if raw.trim().is_empty() {
            return;
        }
        if self.naming {
            self.naming = false;
            self.collection = sanitize_collection(&raw, &self.ctx.cfg.default_collection);
            self.notice = Some(format!("Saving into collection “{}”. Now type a path.", self.collection));
            return;
        }
        if self.running {
            self.notice = Some("An ingestion is already running (x in navigate mode cancels it).".into());
            return;
        }
        let path = clean_path(&raw);
        if !path.exists() {
            self.notice = Some(format!("Path not found: {}", path.display()));
            return;
        }
        let files = collect_files(&path);
        if files.is_empty() {
            self.notice = Some("No files found there.".into());
            return;
        }
        self.jobs = files
            .iter()
            .map(|f| Job {
                name: f.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
                state: JobState::Queued,
                started: None,
                took: None,
            })
            .collect();
        self.running = true;
        self.run_started = Some(Instant::now());
        self.cancel.store(false, Ordering::Relaxed);
        self.notice = Some(format!("Ingesting {} file(s) into “{}”…", files.len(), self.collection));
        self.table.select(Some(0));
        self.spawn(files);
    }

    fn spawn(&self, files: Vec<PathBuf>) {
        let (ctx, tx, cancel) = (self.ctx.clone(), self.tx.clone(), self.cancel.clone());
        let collection = self.collection.clone();
        tokio::spawn(async move {
            if let Err(e) = ctx.store.ensure_schema().await {
                let _ = tx.send(AppMsg::Ingest(IngestMsg::Aborted(format!("Weaviate is not reachable: {e:#}"))));
                return;
            }
            let mut cancelled = false;
            for (idx, path) in files.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    cancelled = true;
                    break;
                }
                let source = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()).to_string_lossy().to_string();
                let _ = tx.send(AppMsg::Ingest(IngestMsg::Started { idx }));
                let txs = tx.clone();
                let result = ctx
                    .pipeline
                    .ingest_file_progress(path, &source, &collection, move |stage| {
                        let _ = txs.send(AppMsg::Ingest(IngestMsg::Stage { idx, stage }));
                    })
                    .await
                    .map_err(|e| format!("{e:#}"));
                let _ = tx.send(AppMsg::Ingest(IngestMsg::Done { idx, result }));
            }
            let _ = tx.send(AppMsg::Ingest(IngestMsg::Finished { cancelled }));
        });
    }

    pub fn on_msg(&mut self, msg: IngestMsg) {
        match msg {
            IngestMsg::Collections(Ok(cols)) => {
                let mut items = vec![];
                let mut values = vec![];
                for (name, chunks) in cols {
                    let mark = if name == self.collection { "  ← current" } else { "" };
                    items.push(format!("{name}  ({chunks} chunks){mark}"));
                    values.push(Some(name));
                }
                items.push("＋ New collection…".to_string());
                values.push(None);
                self.picker = Some((Picker::new("Save into collection", items), values));
            }
            IngestMsg::Collections(Err(e)) => self.notice = Some(format!("Could not list collections: {e}")),
            IngestMsg::Started { idx } => {
                if let Some(j) = self.jobs.get_mut(idx) {
                    j.started = Some(Instant::now());
                    j.state = JobState::Running(Stage::Checking);
                }
                self.table.select(Some(idx));
            }
            IngestMsg::Stage { idx, stage } => {
                if let Some(j) = self.jobs.get_mut(idx) {
                    j.state = JobState::Running(stage);
                }
            }
            IngestMsg::Done { idx, result } => {
                if let Some(j) = self.jobs.get_mut(idx) {
                    j.took = j.started.map(|s| s.elapsed());
                    j.state = match result {
                        Ok(IngestOutcome::Indexed { chunks, .. }) => JobState::Indexed(chunks),
                        Ok(IngestOutcome::Skipped { .. }) => JobState::Skipped,
                        Err(e) => JobState::Failed(e),
                    };
                }
            }
            IngestMsg::Aborted(e) => {
                self.running = false;
                self.notice = Some(e);
            }
            IngestMsg::Finished { cancelled } => {
                self.running = false;
                let (mut ok, mut skip, mut fail) = (0, 0, 0);
                for j in &self.jobs {
                    match j.state {
                        JobState::Indexed(_) => ok += 1,
                        JobState::Skipped => skip += 1,
                        JobState::Failed(_) => fail += 1,
                        _ => {}
                    }
                }
                let took = self.run_started.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);
                self.notice = Some(format!(
                    "{} in {took:.1}s: {ok} indexed, {skip} unchanged, {fail} failed",
                    if cancelled { "Cancelled" } else { "Finished" }
                ));
            }
        }
    }

    fn done_count(&self) -> usize {
        self.jobs.iter().filter(|j| matches!(j.state, JobState::Indexed(_) | JobState::Skipped | JobState::Failed(_))).count()
    }
}

fn stage_cells(state: &JobState, tick: u64) -> (String, Style, String) {
    let sp = theme::spinner(tick);
    match state {
        JobState::Queued => ("queued".into(), theme::dim(), String::new()),
        JobState::Running(Stage::Checking) => (format!("{sp} checking for changes"), Style::new().fg(Color::Cyan), String::new()),
        JobState::Running(Stage::Extracting) => (format!("{sp} extracting and chunking"), Style::new().fg(Color::Cyan), String::new()),
        JobState::Running(Stage::Embedding { done, total }) => {
            let w = 12;
            let filled = if *total == 0 { 0 } else { done * w / total };
            (
                format!("{sp} embedding {done}/{total}"),
                Style::new().fg(Color::Magenta),
                format!("{}{}", "█".repeat(filled), "░".repeat(w - filled)),
            )
        }
        JobState::Running(Stage::Indexing) => (format!("{sp} writing to Weaviate"), Style::new().fg(Color::Blue), String::new()),
        JobState::Indexed(n) => (format!("✔ indexed {n} chunks"), Style::new().fg(Color::Green), String::new()),
        JobState::Skipped => ("↷ unchanged, skipped".into(), Style::new().fg(Color::Yellow), String::new()),
        JobState::Failed(e) => (format!("✖ {}", trunc(e, 60)), Style::new().fg(Color::Red), String::new()),
    }
}

impl Screen for IngestScreen {
    fn has_input(&self) -> bool {
        true
    }

    fn popup_open(&self) -> bool {
        self.picker.is_some()
    }

    fn on_key(&mut self, key: KeyEvent, mode: Mode) -> bool {
        if self.picker.is_some() {
            let ev = self.picker.as_mut().map(|(p, _)| p.on_key(key)).unwrap_or(PickerEvent::None);
            match ev {
                PickerEvent::Chosen(idx) => {
                    if let Some((_, values)) = self.picker.take() {
                        match values.get(idx).cloned().flatten() {
                            Some(name) => {
                                self.notice = Some(format!("Saving into collection “{name}”."));
                                self.collection = name;
                            }
                            None => {
                                self.naming = true;
                                self.notice = Some("Type the new collection name and press Enter.".into());
                            }
                        }
                    }
                }
                PickerEvent::Cancel => self.picker = None,
                PickerEvent::None => {}
            }
            return true;
        }
        let wants_picker = (mode == Mode::Input && key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('k'))
            || (mode == Mode::Nav && key.code == KeyCode::Char('k'));
        if wants_picker {
            let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
            tokio::spawn(async move {
                let r = ctx.store.list_collections().await.map_err(|e| format!("{e:#}"));
                let _ = tx.send(AppMsg::Ingest(IngestMsg::Collections(r)));
            });
            return true;
        }
        if mode == Mode::Input {
            match self.input.on_key(key) {
                InputEvent::Submit(t) => self.submit(t),
                InputEvent::Tab => {
                    if let Some(c) = complete_path(&self.input.text()) {
                        self.input.set(&c);
                    }
                }
                InputEvent::None => {}
            }
            return true;
        }
        match key.code {
            KeyCode::Char('x') if self.running => {
                self.cancel.store(true, Ordering::Relaxed);
                self.notice = Some("Cancelling after the current file…".into());
            }
            KeyCode::Char('c') if !self.running => {
                self.jobs.clear();
                self.notice = None;
            }
            _ => return false,
        }
        true
    }

    fn render(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx) {
        let [input_area, gauge_area, table_area, notice_area] =
            Layout::vertical([Constraint::Length(3), Constraint::Length(3), Constraint::Min(3), Constraint::Length(1)]).areas(area);

        let typing = rc.focused && rc.mode == Mode::Input;
        let title = if self.naming {
            Line::from(Span::styled(" New collection name (Enter to confirm) ", theme::bold(Color::Magenta)))
        } else {
            Line::from(vec![
                Span::styled(" Ingest ", theme::bold(Color::White)),
                Span::styled("→ collection ", theme::dim()),
                Span::styled(self.collection.clone(), theme::bold(Color::Magenta)),
                Span::styled(" · Ctrl-K change · path below ", theme::dim()),
            ])
        };
        let ib = theme::pane(title, typing);
        let iin = ib.inner(input_area);
        f.render_widget(ib, input_area);
        self.input.render(f, iin, if self.naming { "e.g. legal, hr, contracts-2026" } else { "/path/to/documents" }, typing && self.picker.is_none());

        let (done, total) = (self.done_count(), self.jobs.len());
        let ratio = if total == 0 { 0.0 } else { done as f64 / total as f64 };
        let elapsed = self.run_started.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        let label = if total == 0 { "idle".to_string() } else { format!("{done}/{total} files · {:02}:{:02}", elapsed / 60, elapsed % 60) };
        f.render_widget(
            Gauge::default()
                .block(theme::pane(" Overall ", false))
                .gauge_style(Style::new().fg(Color::Cyan).bg(Color::DarkGray))
                .ratio(ratio)
                .label(label),
            gauge_area,
        );

        let rows: Vec<Row> = self
            .jobs
            .iter()
            .map(|j| {
                let (text, style, bar) = stage_cells(&j.state, rc.tick);
                let time = match (j.took, j.started) {
                    (Some(t), _) => format!("{:.1}s", t.as_secs_f32()),
                    (None, Some(s)) => format!("{:.0}s", s.elapsed().as_secs_f32()),
                    _ => String::new(),
                };
                Row::new(vec![Cell::from(trunc(&j.name, 36)), Cell::from(text).style(style), Cell::from(bar).style(Style::new().fg(Color::Magenta)), Cell::from(time)])
            })
            .collect();
        let table = Table::new(rows, [Constraint::Percentage(34), Constraint::Percentage(40), Constraint::Length(13), Constraint::Length(7)])
            .header(Row::new(vec!["File", "Stage", "Progress", "Time"]).style(Style::new().fg(Color::Cyan)))
            .block(theme::pane(" Files ", rc.focused && rc.mode == Mode::Nav))
            .row_highlight_style(Style::new().add_modifier(ratatui::style::Modifier::BOLD));
        f.render_stateful_widget(table, table_area, &mut self.table);

        if let Some(n) = &self.notice {
            f.render_widget(Paragraph::new(Span::styled(trunc(n, notice_area.width as usize), Style::new().fg(Color::Yellow))), notice_area);
        }
    }

    fn render_popup(&mut self, f: &mut Frame) {
        if let Some((p, _)) = self.picker.as_mut() {
            p.render(f);
        }
    }

    fn hints(&self, mode: Mode) -> &'static str {
        match mode {
            Mode::Input => "Enter start · Ctrl-K collection · Tab complete path · ↑↓ history · Esc → navigate",
            Mode::Nav => "i or Enter type path · k choose collection · x cancel run · c clear list · 1-5 screens · Tab pane · q quit",
        }
    }
}
