use super::{RenderCtx, Screen};
use crate::{
    context::AppContext,
    tui::{
        app::{AppMsg, Mode},
        theme,
        widgets::{confirm_key, render_confirm},
    },
    util::short,
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Cell, Row, Table, TableState},
    Frame,
};
use tokio::sync::mpsc::UnboundedSender;
use tracing::Instrument;

pub struct DocsScreen {
    ctx: AppContext,
    tx: UnboundedSender<AppMsg>,
    docs: Vec<(String, usize)>,
    table: TableState,
    loading: bool,
    error: Option<String>,
    confirm: Option<String>,
}

impl DocsScreen {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        Self { ctx, tx, docs: vec![], table: TableState::default(), loading: false, error: None, confirm: None }
    }

    pub fn refresh(&mut self) {
        if self.loading {
            return;
        }
        self.loading = true;
        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        tokio::spawn(async move {
            let r = ctx.store.list_sources().await.map_err(|e| format!("{e:#}"));
            let _ = tx.send(AppMsg::Docs(r));
        });
    }

    pub fn on_docs(&mut self, r: Result<Vec<(String, usize)>, String>) {
        self.loading = false;
        match r {
            Ok(docs) => {
                self.error = None;
                let keep = self.table.selected().unwrap_or(0);
                self.table.select(if docs.is_empty() { None } else { Some(keep.min(docs.len() - 1)) });
                self.docs = docs;
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn selected(&self) -> Option<&(String, usize)> {
        self.table.selected().and_then(|i| self.docs.get(i))
    }

    fn mv(&mut self, delta: isize) {
        if self.docs.is_empty() {
            return;
        }
        let cur = self.table.selected().unwrap_or(0) as isize;
        self.table.select(Some((cur + delta).clamp(0, self.docs.len() as isize - 1) as usize));
    }

    fn delete(&mut self, source: String) {
        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        tokio::spawn(async move {
            let span = tracing::info_span!("weaviate_delete", file = %short(&source));
            let r = ctx.store.delete_source(&source).instrument(span).await.map(|_| source.clone()).map_err(|e| format!("{e:#}"));
            let _ = tx.send(AppMsg::DocDeleted(r));
        });
    }
}

impl Screen for DocsScreen {
    fn has_input(&self) -> bool {
        false
    }

    fn popup_open(&self) -> bool {
        self.confirm.is_some()
    }

    fn on_enter(&mut self) {
        self.refresh();
    }

    fn on_key(&mut self, key: KeyEvent, _mode: Mode) -> bool {
        if let Some(src) = self.confirm.clone() {
            if let Some(yes) = confirm_key(key) {
                self.confirm = None;
                if yes {
                    self.delete(src);
                }
            }
            return true;
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.mv(-1),
            KeyCode::Down | KeyCode::Char('j') => self.mv(1),
            KeyCode::Char('r') => self.refresh(),
            KeyCode::Char('d') => self.confirm = self.selected().map(|d| d.0.clone()),
            KeyCode::Char('c') | KeyCode::Enter => {
                if let Some((src, _)) = self.selected().cloned() {
                    let _ = self.tx.send(AppMsg::ChatAbout(src));
                }
            }
            _ => return false,
        }
        true
    }

    fn render(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx) {
        let total: usize = self.docs.iter().map(|d| d.1).sum();
        let title = Line::from(vec![
            Span::styled(" Documents ", theme::bold(Color::White)),
            Span::styled(format!("· {} docs · {total} chunks ", self.docs.len()), theme::dim()),
            if self.loading { Span::styled(format!("{} loading", theme::spinner(rc.tick)), Style::new().fg(Color::Cyan)) } else { Span::raw("") },
        ]);
        let mut block = theme::pane(title, rc.focused);
        if let Some(e) = &self.error {
            block = block.title_bottom(Line::from(Span::styled(format!(" {e} "), Style::new().fg(Color::Red))));
        }
        let rows: Vec<Row> = self
            .docs
            .iter()
            .map(|(name, n)| Row::new(vec![Cell::from(short(name)), Cell::from(n.to_string()), Cell::from(name.clone()).style(theme::dim())]))
            .collect();
        let table = Table::new(rows, [Constraint::Percentage(35), Constraint::Length(8), Constraint::Min(10)])
            .header(Row::new(vec!["Document", "Chunks", "Source id"]).style(Style::new().fg(Color::Cyan)))
            .block(block)
            .row_highlight_style(Style::new().add_modifier(Modifier::REVERSED))
            .highlight_symbol("▶ ");
        f.render_stateful_widget(table, area, &mut self.table);
        if self.docs.is_empty() && !self.loading {
            let msg = ratatui::widgets::Paragraph::new(Span::styled("Nothing ingested yet. Open the Ingest screen (2).", theme::dim()));
            f.render_widget(msg, Rect { x: area.x + 2, y: area.y + 3, width: area.width.saturating_sub(4), height: 1 });
        }
    }

    fn render_popup(&mut self, f: &mut Frame) {
        if let Some(src) = &self.confirm {
            render_confirm(f, "Delete document", &format!("Delete every chunk of “{}” from Weaviate? The original file is not touched.", short(src)));
        }
    }

    fn hints(&self, _mode: Mode) -> &'static str {
        "j/k select · Enter or c chat about it · d delete · r refresh · 1-5 screens · Tab pane · q quit"
    }
}

