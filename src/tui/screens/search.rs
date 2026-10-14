use super::{RenderCtx, Screen};
use crate::{
    context::AppContext,
    models::search::{Hit, SearchMode},
    tui::{
        app::{AppMsg, Mode},
        input::{InputEvent, InputLine},
        markdown, theme,
        widgets::{centered, Picker, PickerEvent},
    },
    util::{score_bar, short, trunc},
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Cell, Clear, Paragraph, Row, Table, TableState},
    Frame,
};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

pub struct SearchScreen {
    ctx: AppContext,
    tx: UnboundedSender<AppMsg>,
    input: InputLine,
    mode: SearchMode,
    hits: Vec<Hit>,
    table: TableState,
    last_query: String,
    busy: Option<Instant>,
    status: String,
    detail: Option<usize>,
    detail_scroll: usize,
    /// Collections to search; empty = all.
    collections: Vec<String>,
    /// Collection picker; `None` entry = all collections.
    picker: Option<(Picker, Vec<Option<String>>)>,
}

impl SearchScreen {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        Self {
            ctx,
            tx,
            input: InputLine::default(),
            mode: SearchMode::Hybrid,
            hits: vec![],
            table: TableState::default(),
            last_query: String::new(),
            busy: None,
            status: "Type a query and press Enter. Tab changes the search mode.".into(),
            detail: None,
            detail_scroll: 0,
            collections: vec![],
            picker: None,
        }
    }

    fn run(&mut self, query: String) {
        let q = query.trim().to_string();
        if q.is_empty() || self.busy.is_some() {
            return;
        }
        self.last_query = q.clone();
        self.busy = Some(Instant::now());
        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        let mut opts = ctx.cfg.retrieval_defaults();
        opts.mode = self.mode;
        opts.top_k = 10;
        opts.collections = self.collections.clone();
        tokio::spawn(async move {
            let r = ctx.rag.search(&q, &opts.sanitized()).await.map_err(|e| format!("{e:#}"));
            let _ = tx.send(AppMsg::Search(r));
        });
    }

    pub fn on_msg(&mut self, r: Result<Vec<Hit>, String>) {
        let took = self.busy.take().map(|s| s.elapsed().as_millis()).unwrap_or(0);
        match r {
            Ok(hits) => {
                self.status = format!("{} result(s) in {took} ms · mode: {}", hits.len(), self.mode.label());
                self.table.select(if hits.is_empty() { None } else { Some(0) });
                self.hits = hits;
            }
            Err(e) => {
                self.hits.clear();
                self.status = format!("Search failed: {e}");
            }
        }
    }

    fn request_collections(&self) {
        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        tokio::spawn(async move {
            let r = ctx.store.list_collections().await.map_err(|e| format!("{e:#}"));
            let _ = tx.send(AppMsg::SearchCollections(r));
        });
    }

    pub fn on_collections(&mut self, r: Result<Vec<(String, usize)>, String>) {
        match r {
            Ok(cols) if !cols.is_empty() => {
                let mut items = vec!["All collections (default)".to_string()];
                let mut values = vec![None];
                for (name, chunks) in cols {
                    items.push(format!("{name}  ({chunks} chunks)"));
                    values.push(Some(name));
                }
                self.picker = Some((Picker::new("Search within a collection", items), values));
            }
            Ok(_) => self.status = "No collections yet. Ingest a document first.".into(),
            Err(e) => self.status = format!("Could not list collections: {e}"),
        }
    }

    fn select(&mut self, delta: isize) {
        if self.hits.is_empty() {
            return;
        }
        let cur = self.table.selected().unwrap_or(0) as isize;
        self.table.select(Some((cur + delta).clamp(0, self.hits.len() as isize - 1) as usize));
    }

    fn open_detail(&mut self) {
        if let Some(i) = self.table.selected().filter(|i| *i < self.hits.len()) {
            self.detail = Some(i);
            self.detail_scroll = 0;
        }
    }
}

impl Screen for SearchScreen {
    fn has_input(&self) -> bool {
        true
    }

    fn popup_open(&self) -> bool {
        self.detail.is_some() || self.picker.is_some()
    }

    fn on_key(&mut self, key: KeyEvent, mode: Mode) -> bool {
        if self.picker.is_some() {
            let ev = self.picker.as_mut().map(|(p, _)| p.on_key(key)).unwrap_or(PickerEvent::None);
            match ev {
                PickerEvent::Chosen(i) => {
                    if let Some((_, values)) = self.picker.take() {
                        self.collections = values.get(i).cloned().flatten().into_iter().collect();
                        self.status = if self.collections.is_empty() { "Searching all collections.".into() } else { format!("Searching collection “{}”.", self.collections[0]) };
                        self.last_query.clear(); // the next Enter searches again with the new scope
                    }
                }
                PickerEvent::Cancel => self.picker = None,
                PickerEvent::None => {}
            }
            return true;
        }
        if (mode == Mode::Input && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) && key.code == KeyCode::Char('k')) || (mode == Mode::Nav && key.code == KeyCode::Char('f')) {
            self.request_collections();
            return true;
        }
        if self.detail.is_some() {
            match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => self.detail = None,
                KeyCode::Up | KeyCode::Char('k') => self.detail_scroll = self.detail_scroll.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => self.detail_scroll += 1,
                KeyCode::PageUp => self.detail_scroll = self.detail_scroll.saturating_sub(10),
                KeyCode::PageDown => self.detail_scroll += 10,
                _ => {}
            }
            return true;
        }
        if mode == Mode::Input {
            match key.code {
                KeyCode::Up if !self.hits.is_empty() => self.select(-1),
                KeyCode::Down if !self.hits.is_empty() => self.select(1),
                KeyCode::Enter => {
                    let text = self.input.text();
                    if text.trim() != self.last_query || self.hits.is_empty() {
                        self.input.on_key(key); // records history and clears the field
                        self.input.set(text.trim());
                        self.run(text);
                    } else {
                        self.open_detail();
                    }
                }
                _ => {
                    if let InputEvent::Tab = self.input.on_key(key) {
                        self.mode = self.mode.next();
                        self.status = format!("Search mode: {}", self.mode.label());
                    }
                }
            }
            return true;
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.select(-1),
            KeyCode::Down | KeyCode::Char('j') => self.select(1),
            KeyCode::Enter => self.open_detail(),
            _ => return false,
        }
        true
    }

    fn render(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx) {
        let [input_area, status_area, table_area] = Layout::vertical([Constraint::Length(3), Constraint::Length(1), Constraint::Min(3)]).areas(area);

        let typing = rc.focused && rc.mode == Mode::Input;
        let title = Line::from(vec![
            Span::styled(" Search ", theme::bold(Color::White)),
            Span::styled(format!("· {} · {} ", self.mode.label(), if self.collections.is_empty() { "all collections".to_string() } else { self.collections.join(", ") }), theme::dim()),
        ]);
        let ib = theme::pane(title, typing);
        let iin = ib.inner(input_area);
        f.render_widget(ib, input_area);
        self.input.render(f, iin, "what are you looking for?", typing && self.detail.is_none());

        let status = match self.busy {
            Some(s) => Line::from(vec![
                Span::styled(format!("{} ", theme::spinner(rc.tick)), Style::new().fg(Color::Cyan)),
                Span::raw(format!("searching… {} ms", s.elapsed().as_millis())),
            ]),
            None => Line::from(Span::styled(trunc(&self.status, status_area.width as usize), Style::new().fg(Color::Yellow))),
        };
        f.render_widget(Paragraph::new(status), status_area);

        let rows: Vec<Row> = self
            .hits
            .iter()
            .enumerate()
            .map(|(i, h)| {
                Row::new(vec![
                    Cell::from(format!("{}", i + 1)).style(Style::new().fg(Color::Yellow)),
                    Cell::from(score_bar(h.score)).style(Style::new().fg(Color::Green)),
                    Cell::from(trunc(&short(&h.source), 24)),
                    Cell::from(trunc(&h.headings.join(" > "), 24)),
                    Cell::from(trunc(&h.text, 200)),
                ])
            })
            .collect();
        let table = Table::new(
            rows,
            [Constraint::Length(3), Constraint::Length(13), Constraint::Percentage(24), Constraint::Percentage(22), Constraint::Min(10)],
        )
        .header(Row::new(vec!["#", "Score", "Document", "Section", "Preview"]).style(Style::new().fg(Color::Cyan)))
        .block(theme::pane(" Results ", rc.focused && rc.mode == Mode::Nav))
        .row_highlight_style(Style::new().add_modifier(Modifier::REVERSED))
        .highlight_symbol("▶ ");
        f.render_stateful_widget(table, table_area, &mut self.table);
    }

    fn render_popup(&mut self, f: &mut Frame) {
        if let Some((p, _)) = self.picker.as_mut() {
            p.render(f);
        }
        if let Some(i) = self.detail {
            if let Some(h) = self.hits.get(i) {
                let full = f.area();
                let rect = centered(full, full.width.saturating_sub(6).min(100), full.height.saturating_sub(4));
                f.render_widget(Clear, rect);
                let title = Line::from(Span::styled(format!(" {} · chunk {} ", short(&h.source), h.chunk_index), theme::bold(Color::Green)));
                let block = theme::pane(title, true).title_bottom(Line::from(Span::styled(" ↑↓ scroll · Esc close ", theme::dim())));
                let inner = block.inner(rect);
                let mut lines: Vec<Line> = vec![];
                if !h.headings.is_empty() {
                    lines.push(Line::from(Span::styled(h.headings.join(" › "), theme::dim())));
                    lines.push(Line::default());
                }
                lines.extend(markdown::render(&h.text, inner.width as usize, Style::new()));
                let max = lines.len().saturating_sub(inner.height as usize);
                self.detail_scroll = self.detail_scroll.min(max);
                f.render_widget(block, rect);
                f.render_widget(Paragraph::new(lines).scroll((self.detail_scroll as u16, 0)), inner);
            }
        }
    }

    fn hints(&self, mode: Mode) -> &'static str {
        match mode {
            Mode::Input => "Enter search / open result · ↑↓ select · Tab mode · Ctrl-K collection · Esc → navigate",
            Mode::Nav => "i type · j/k select · Enter open · f collection · 1-5 screens · Tab pane · q quit",
        }
    }
}
