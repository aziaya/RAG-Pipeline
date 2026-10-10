//! Popups shared by the screens.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Flex, Layout, Rect},
    style::{Modifier, Style},
    text::Line,
    widgets::{Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};

use super::theme;

/// A centered rectangle `w` x `h` cells (clamped to `area`).
pub fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    let [v] = Layout::vertical([Constraint::Length(h)]).flex(Flex::Center).areas(area);
    let [r] = Layout::horizontal([Constraint::Length(w)]).flex(Flex::Center).areas(v);
    r
}

pub enum PickerEvent {
    None,
    Chosen(usize),
    Cancel,
}

pub struct Picker {
    pub title: String,
    pub items: Vec<String>,
    state: ListState,
}

impl Picker {
    pub fn new(title: impl Into<String>, items: Vec<String>) -> Self {
        let mut state = ListState::default();
        state.select(Some(0));
        Self { title: title.into(), items, state }
    }

    pub fn on_key(&mut self, key: KeyEvent) -> PickerEvent {
        let n = self.items.len();
        let cur = self.state.selected().unwrap_or(0);
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.state.select(Some(if cur == 0 { n.saturating_sub(1) } else { cur - 1 })),
            KeyCode::Down | KeyCode::Char('j') => self.state.select(Some(if cur + 1 >= n { 0 } else { cur + 1 })),
            KeyCode::Enter => return PickerEvent::Chosen(cur),
            KeyCode::Esc | KeyCode::Char('q') => return PickerEvent::Cancel,
            _ => {}
        }
        PickerEvent::None
    }

    pub fn render(&mut self, f: &mut Frame) {
        let area = f.area();
        let h = (self.items.len() as u16 + 4).min(area.height.saturating_sub(2)).max(5);
        let w = (self.items.iter().map(|i| i.chars().count()).max().unwrap_or(20) as u16 + 8).clamp(30, area.width.saturating_sub(4));
        let rect = centered(area, w, h);
        f.render_widget(Clear, rect);
        let items: Vec<ListItem> = self.items.iter().map(|i| ListItem::new(i.clone())).collect();
        let list = List::new(items)
            .block(theme::pane(Line::from(format!(" {} ", self.title)), true).title_bottom(Line::from(" ↑↓ move · Enter choose · Esc cancel ").style(theme::dim())))
            .highlight_style(Style::new().fg(theme::ACCENT).add_modifier(Modifier::BOLD | Modifier::REVERSED))
            .highlight_symbol("▶ ");
        f.render_stateful_widget(list, rect, &mut self.state);
    }
}

/// A yes/no question; returns Some(true/false) once answered.
pub fn confirm_key(key: KeyEvent) -> Option<bool> {
    match key.code {
        KeyCode::Char('y') | KeyCode::Char('Y') => Some(true),
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc | KeyCode::Enter => Some(false),
        _ => None,
    }
}

pub fn render_confirm(f: &mut Frame, title: &str, message: &str) {
    let area = f.area();
    let rect = centered(area, 64.min(area.width.saturating_sub(4)), 7);
    f.render_widget(Clear, rect);
    let body = Paragraph::new(vec![
        Line::from(message.to_string()),
        Line::default(),
        Line::from("y = yes, delete    n / Esc = cancel").style(theme::dim()),
    ])
    .wrap(Wrap { trim: true })
    .block(theme::pane(Line::from(format!(" {title} ")), true));
    f.render_widget(body, rect);
}
