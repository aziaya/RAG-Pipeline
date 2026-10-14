use super::{RenderCtx, Screen};
use crate::{
    context::AppContext,
    models::collection::{ChunkInfo, ChunkPage, DocumentInfo},
    tui::{
        app::{AppMsg, Mode},
        markdown, theme,
        widgets::{centered, confirm_key, render_confirm},
    },
    util::{short, trunc},
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Cell, Clear, Paragraph, Row, Table, TableState},
    Frame,
};
use tokio::sync::mpsc::UnboundedSender;
use tracing::Instrument;

/// Chunks fetched per page in the chunk viewer.
const PAGE: usize = 25;

enum Confirm {
    Document { collection: String, source: String },
    Collection(String),
}

/// The chunk viewer for one document.
struct ChunkView {
    collection: String,
    source: String,
    total: usize,
    offset: usize,
    rows: Vec<ChunkInfo>,
    table: TableState,
    loading: bool,
    error: Option<String>,
    detail: Option<usize>,
    detail_scroll: usize,
}

pub struct DocsScreen {
    ctx: AppContext,
    tx: UnboundedSender<AppMsg>,
    docs: Vec<DocumentInfo>,
    table: TableState,
    loading: bool,
    error: Option<String>,
    /// Show only this collection (cycled with `f`).
    filter: Option<String>,
    confirm: Option<Confirm>,
    view: Option<ChunkView>,
}

impl DocsScreen {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        Self { ctx, tx, docs: vec![], table: TableState::default(), loading: false, error: None, filter: None, confirm: None, view: None }
    }

    pub fn refresh(&mut self) {
        if self.loading {
            return;
        }
        self.loading = true;
        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        tokio::spawn(async move {
            let r = ctx.store.list_documents(None).await.map_err(|e| format!("{e:#}"));
            let _ = tx.send(AppMsg::Docs(r));
        });
    }

    pub fn on_docs(&mut self, r: Result<Vec<DocumentInfo>, String>) {
        self.loading = false;
        match r {
            Ok(docs) => {
                self.error = None;
                self.docs = docs;
                // A filter that no longer matches anything is dropped.
                if let Some(f) = &self.filter {
                    if !self.docs.iter().any(|d| &d.collection == f) {
                        self.filter = None;
                    }
                }
                let n = self.visible().len();
                let keep = self.table.selected().unwrap_or(0);
                self.table.select(if n == 0 { None } else { Some(keep.min(n - 1)) });
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn visible(&self) -> Vec<&DocumentInfo> {
        self.docs.iter().filter(|d| self.filter.as_ref().map(|f| &d.collection == f).unwrap_or(true)).collect()
    }

    fn selected(&self) -> Option<DocumentInfo> {
        let v = self.visible();
        self.table.selected().and_then(|i| v.get(i).map(|d| (*d).clone()))
    }

    fn mv(&mut self, delta: isize) {
        let n = self.visible().len();
        if n == 0 {
            return;
        }
        let cur = self.table.selected().unwrap_or(0) as isize;
        self.table.select(Some((cur + delta).clamp(0, n as isize - 1) as usize));
    }

    fn cycle_filter(&mut self) {
        let mut names: Vec<String> = self.docs.iter().map(|d| d.collection.clone()).collect();
        names.dedup();
        self.filter = match &self.filter {
            None => names.first().cloned(),
            Some(cur) => names.iter().position(|n| n == cur).and_then(|i| names.get(i + 1)).cloned(),
        };
        self.table.select(if self.visible().is_empty() { None } else { Some(0) });
    }

    fn delete(&mut self, what: Confirm) {
        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        tokio::spawn(async move {
            let (label, span, res) = match what {
                Confirm::Document { collection, source } => {
                    let span = tracing::info_span!("weaviate_delete", file = %short(&source), collection = %collection);
                    (short(&source), span.clone(), ctx.store.delete_source(&collection, &source).instrument(span).await)
                }
                Confirm::Collection(name) => {
                    let span = tracing::info_span!("weaviate_delete", collection = %name);
                    (format!("collection {name}"), span.clone(), ctx.store.delete_collection(&name).instrument(span).await)
                }
            };
            drop(span);
            let _ = tx.send(AppMsg::DocDeleted(res.map(|_| label).map_err(|e| format!("{e:#}"))));
        });
    }

    // ── chunk viewer ──

    fn open_view(&mut self, d: &DocumentInfo) {
        self.view = Some(ChunkView {
            collection: d.collection.clone(),
            source: d.source.clone(),
            total: d.chunks,
            offset: 0,
            rows: vec![],
            table: TableState::default(),
            loading: false,
            error: None,
            detail: None,
            detail_scroll: 0,
        });
        self.load_page(0);
    }

    fn load_page(&mut self, offset: usize) {
        let Some(v) = self.view.as_mut() else { return };
        if v.loading {
            return;
        }
        v.loading = true;
        let (ctx, tx, collection, source) = (self.ctx.clone(), self.tx.clone(), v.collection.clone(), v.source.clone());
        tokio::spawn(async move {
            let res = async {
                let total = ctx.store.count_chunks(&collection, &source).await?;
                let chunks = ctx.store.list_chunks(&collection, &source, PAGE, offset).await?;
                Ok::<_, anyhow::Error>(ChunkPage { collection, source, total, offset, chunks })
            }
            .await
            .map_err(|e| format!("{e:#}"));
            let _ = tx.send(AppMsg::Chunks(res));
        });
    }

    pub fn on_chunks(&mut self, r: Result<ChunkPage, String>) {
        let Some(v) = self.view.as_mut() else { return };
        v.loading = false;
        match r {
            Ok(page) => {
                v.error = None;
                v.total = page.total;
                v.offset = page.offset;
                v.table.select(if page.chunks.is_empty() { None } else { Some(0) });
                v.rows = page.chunks;
            }
            Err(e) => v.error = Some(e),
        }
    }

    fn on_view_key(&mut self, key: KeyEvent) -> bool {
        // Detail popup first.
        if let Some(v) = self.view.as_mut() {
            if let Some(i) = v.detail {
                match key.code {
                    KeyCode::Esc | KeyCode::Enter => v.detail = None,
                    KeyCode::Up | KeyCode::Char('k') => v.detail_scroll = v.detail_scroll.saturating_sub(1),
                    KeyCode::Down | KeyCode::Char('j') => v.detail_scroll += 1,
                    KeyCode::PageUp => v.detail_scroll = v.detail_scroll.saturating_sub(10),
                    KeyCode::PageDown => v.detail_scroll += 10,
                    KeyCode::Char('n') if i + 1 < v.rows.len() => {
                        v.detail = Some(i + 1);
                        v.detail_scroll = 0;
                        v.table.select(Some(i + 1));
                    }
                    KeyCode::Char('p') if i > 0 => {
                        v.detail = Some(i - 1);
                        v.detail_scroll = 0;
                        v.table.select(Some(i - 1));
                    }
                    _ => {}
                }
                return true;
            }
        }
        let (mut goto, mut chat) = (None::<usize>, false);
        {
            let Some(v) = self.view.as_mut() else { return false };
            let cur = v.table.selected().unwrap_or(0);
            match key.code {
                KeyCode::Esc | KeyCode::Backspace | KeyCode::Char('b') => {
                    self.view = None;
                    return true;
                }
                KeyCode::Up | KeyCode::Char('k') => v.table.select(Some(cur.saturating_sub(1))),
                KeyCode::Down | KeyCode::Char('j') => v.table.select(Some((cur + 1).min(v.rows.len().saturating_sub(1)))),
                KeyCode::Enter | KeyCode::Char('v') => {
                    if cur < v.rows.len() {
                        v.detail = Some(cur);
                        v.detail_scroll = 0;
                    }
                }
                KeyCode::Char('n') | KeyCode::PageDown if v.offset + PAGE < v.total => goto = Some(v.offset + PAGE),
                KeyCode::Char('p') | KeyCode::PageUp if v.offset > 0 => goto = Some(v.offset.saturating_sub(PAGE)),
                KeyCode::Char('c') => chat = true,
                _ => return false,
            }
        }
        if let Some(o) = goto {
            self.load_page(o);
        }
        if chat {
            if let Some(v) = &self.view {
                let _ = self.tx.send(AppMsg::ChatAbout { collection: v.collection.clone(), source: v.source.clone() });
            }
        }
        true
    }

    fn render_view(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx) {
        let Some(v) = self.view.as_mut() else { return };
        let end = v.offset + v.rows.len();
        let title = Line::from(vec![
            Span::styled(format!(" {} ", short(&v.source)), theme::bold(Color::White)),
            Span::styled(format!("· {} · chunks {}–{} of {} ", v.collection, if v.rows.is_empty() { 0 } else { v.offset + 1 }, end, v.total), theme::dim()),
            if v.loading { Span::styled(format!("{} loading", theme::spinner(rc.tick)), Style::new().fg(Color::Cyan)) } else { Span::raw("") },
        ]);
        let mut block = theme::pane(title, rc.focused)
            .title_bottom(Line::from(Span::styled(" Enter read · j/k move · n/p page · c chat · Esc back ", theme::dim())));
        if let Some(e) = &v.error {
            block = block.title_bottom(Line::from(Span::styled(format!(" {e} "), Style::new().fg(Color::Red))).right_aligned());
        }
        let rows: Vec<Row> = v
            .rows
            .iter()
            .map(|c| {
                Row::new(vec![
                    Cell::from(c.chunk_index.to_string()).style(Style::new().fg(Color::Yellow)),
                    Cell::from(trunc(&c.headings.join(" › "), 26)).style(Style::new().fg(Color::Cyan)),
                    Cell::from(c.text.chars().count().to_string()).style(theme::dim()),
                    Cell::from(trunc(&c.text, 240)),
                ])
            })
            .collect();
        let table = Table::new(rows, [Constraint::Length(4), Constraint::Percentage(28), Constraint::Length(6), Constraint::Min(10)])
            .header(Row::new(vec!["#", "Section", "Chars", "Preview"]).style(Style::new().fg(Color::Cyan)))
            .block(block)
            .row_highlight_style(Style::new().add_modifier(Modifier::REVERSED))
            .highlight_symbol("▶ ");
        f.render_stateful_widget(table, area, &mut v.table);
        if v.rows.is_empty() && !v.loading {
            let msg = Paragraph::new(Span::styled("No chunks found for this document.", theme::dim()));
            f.render_widget(msg, Rect { x: area.x + 2, y: area.y + 3, width: area.width.saturating_sub(4), height: 1 });
        }
    }

    fn render_detail(&mut self, f: &mut Frame) {
        let Some(v) = self.view.as_mut() else { return };
        let Some(i) = v.detail else { return };
        let Some(c) = v.rows.get(i) else { return };
        let full = f.area();
        let rect = centered(full, full.width.saturating_sub(6).min(100), full.height.saturating_sub(4));
        f.render_widget(Clear, rect);
        let title = Line::from(Span::styled(
            format!(" {} · chunk {} of {} ", short(&v.source), c.chunk_index, v.total),
            theme::bold(Color::Green),
        ));
        let block = theme::pane(title, true).title_bottom(Line::from(Span::styled(" ↑↓ scroll · n/p next/previous chunk · Esc close ", theme::dim())));
        let inner = block.inner(rect);
        let mut lines: Vec<Line> = vec![Line::from(Span::styled(
            format!("{} characters{}", c.text.chars().count(), c.id.as_ref().map(|id| format!(" · id {}", &id[..id.len().min(8)])).unwrap_or_default()),
            theme::dim(),
        ))];
        if !c.headings.is_empty() {
            lines.push(Line::from(Span::styled(c.headings.join(" › "), Style::new().fg(Color::Cyan))));
        }
        lines.push(Line::default());
        lines.extend(markdown::render(&c.text, inner.width as usize, Style::new()));
        let max = lines.len().saturating_sub(inner.height as usize);
        v.detail_scroll = v.detail_scroll.min(max);
        f.render_widget(block, rect);
        f.render_widget(Paragraph::new(lines).scroll((v.detail_scroll as u16, 0)), inner);
    }
}

impl Screen for DocsScreen {
    fn has_input(&self) -> bool {
        false
    }

    fn popup_open(&self) -> bool {
        self.confirm.is_some() || self.view.as_ref().map(|v| v.detail.is_some()).unwrap_or(false)
    }

    fn on_enter(&mut self) {
        self.view = None;
        self.refresh();
    }

    fn on_key(&mut self, key: KeyEvent, _mode: Mode) -> bool {
        if self.confirm.is_some() {
            if let Some(yes) = confirm_key(key) {
                if let Some(c) = self.confirm.take() {
                    if yes {
                        self.delete(c);
                    }
                }
            }
            return true;
        }
        if self.view.is_some() {
            return self.on_view_key(key);
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.mv(-1),
            KeyCode::Down | KeyCode::Char('j') => self.mv(1),
            KeyCode::Char('r') => self.refresh(),
            KeyCode::Char('f') => self.cycle_filter(),
            KeyCode::Char('d') => self.confirm = self.selected().map(|d| Confirm::Document { collection: d.collection, source: d.source }),
            KeyCode::Char('D') => self.confirm = self.selected().map(|d| Confirm::Collection(d.collection)),
            KeyCode::Enter | KeyCode::Char('v') => {
                if let Some(d) = self.selected() {
                    self.open_view(&d);
                }
            }
            KeyCode::Char('c') => {
                if let Some(d) = self.selected() {
                    let _ = self.tx.send(AppMsg::ChatAbout { collection: d.collection, source: d.source });
                }
            }
            _ => return false,
        }
        true
    }

    fn render(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx) {
        if self.view.is_some() {
            self.render_view(f, area, rc);
            return;
        }
        let visible: Vec<DocumentInfo> = self.visible().into_iter().cloned().collect();
        let total: usize = visible.iter().map(|d| d.chunks).sum();
        let mut collections: Vec<&str> = visible.iter().map(|d| d.collection.as_str()).collect();
        collections.dedup();
        let title = Line::from(vec![
            Span::styled(" Documents ", theme::bold(Color::White)),
            Span::styled(
                format!("· {} docs · {} collections · {total} chunks ", visible.len(), collections.len()),
                theme::dim(),
            ),
            match &self.filter {
                Some(c) => Span::styled(format!("· only “{c}” "), Style::new().fg(Color::Magenta)),
                None => Span::raw(""),
            },
            if self.loading { Span::styled(format!("{} loading", theme::spinner(rc.tick)), Style::new().fg(Color::Cyan)) } else { Span::raw("") },
        ]);
        let mut block = theme::pane(title, rc.focused);
        if let Some(e) = &self.error {
            block = block.title_bottom(Line::from(Span::styled(format!(" {e} "), Style::new().fg(Color::Red))));
        }
        let rows: Vec<Row> = visible
            .iter()
            .map(|d| {
                Row::new(vec![
                    Cell::from(d.collection.clone()).style(Style::new().fg(Color::Magenta)),
                    Cell::from(short(&d.source)),
                    Cell::from(d.chunks.to_string()),
                    Cell::from(d.source.clone()).style(theme::dim()),
                ])
            })
            .collect();
        let table = Table::new(rows, [Constraint::Percentage(18), Constraint::Percentage(30), Constraint::Length(7), Constraint::Min(10)])
            .header(Row::new(vec!["Collection", "Document", "Chunks", "Source id"]).style(Style::new().fg(Color::Cyan)))
            .block(block)
            .row_highlight_style(Style::new().add_modifier(Modifier::REVERSED))
            .highlight_symbol("▶ ");
        f.render_stateful_widget(table, area, &mut self.table);
        if visible.is_empty() && !self.loading {
            let msg = Paragraph::new(Span::styled("Nothing ingested yet. Open the Ingest screen (2).", theme::dim()));
            f.render_widget(msg, Rect { x: area.x + 2, y: area.y + 3, width: area.width.saturating_sub(4), height: 1 });
        }
    }

    fn render_popup(&mut self, f: &mut Frame) {
        self.render_detail(f);
        match &self.confirm {
            Some(Confirm::Document { source, collection }) => render_confirm(
                f,
                "Delete document",
                &format!("Delete every chunk of “{}” from collection “{collection}”? The original file is not touched.", short(source)),
            ),
            Some(Confirm::Collection(name)) => render_confirm(
                f,
                "Delete collection",
                &format!("Delete the whole collection “{name}” and ALL its documents from Weaviate? The original files are not touched."),
            ),
            None => {}
        }
    }

    fn hints(&self, _mode: Mode) -> &'static str {
        if self.view.is_some() {
            "j/k select · Enter read chunk · n/p next/previous page · c chat about it · Esc back · 1-5 screens"
        } else {
            "j/k select · Enter view chunks · c chat · f filter collection · d delete doc · D delete collection · r refresh"
        }
    }
}
