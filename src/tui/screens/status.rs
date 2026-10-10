use super::{RenderCtx, Screen};
use crate::{
    context::AppContext,
    models::health::{Check, Level},
    services::health,
    tui::{
        app::{AppMsg, Mode},
        theme,
    },
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use tokio::sync::mpsc::UnboundedSender;

pub struct StatusScreen {
    ctx: AppContext,
    tx: UnboundedSender<AppMsg>,
    checks: Vec<Check>,
    loading: bool,
}

impl StatusScreen {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        Self { ctx, tx, checks: vec![], loading: false }
    }

    pub fn refresh(&mut self) {
        if self.loading {
            return;
        }
        self.loading = true;
        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        tokio::spawn(async move {
            let checks = health::run_checks(&ctx).await;
            let _ = tx.send(AppMsg::Status(checks));
        });
    }

    pub fn on_checks(&mut self, checks: Vec<Check>) {
        self.loading = false;
        self.checks = checks;
    }
}

impl Screen for StatusScreen {
    fn has_input(&self) -> bool {
        false
    }

    fn on_enter(&mut self) {
        self.refresh();
    }

    fn on_key(&mut self, key: KeyEvent, _mode: Mode) -> bool {
        match key.code {
            KeyCode::Char('r') => {
                self.refresh();
                true
            }
            _ => false,
        }
    }

    fn render(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx) {
        let title = Line::from(vec![
            Span::styled(" Status ", theme::bold(Color::White)),
            if self.loading { Span::styled(format!("{} checking", theme::spinner(rc.tick)), Style::new().fg(Color::Cyan)) } else { Span::raw("") },
        ]);
        let mut lines: Vec<Line> = vec![Line::default()];
        for c in &self.checks {
            let (mark, color) = match c.level {
                Level::Ok => ("✔", Color::Green),
                Level::Warn => ("!", Color::Yellow),
                Level::Fail => ("✖", Color::Red),
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {mark} "), theme::bold(color)),
                Span::styled(format!("{:<14}", c.name), theme::bold(Color::White)),
                Span::styled(c.detail.clone(), theme::dim()),
            ]));
        }
        if self.checks.is_empty() && !self.loading {
            lines.push(Line::from(Span::styled("  press r to run the checks", theme::dim())));
        }
        f.render_widget(Paragraph::new(lines).block(theme::pane(title, rc.focused)), area);
    }

    fn hints(&self, _mode: Mode) -> &'static str {
        "r re-run checks · 1-5 screens · Tab pane · q quit"
    }
}
