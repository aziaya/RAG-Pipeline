//! A single-line text input with cursor movement and history.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::{
    layout::{Position, Rect},
    style::Style,
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use unicode_width::UnicodeWidthChar;

use super::theme;

pub enum InputEvent {
    None,
    Submit(String),
    Tab,
}

#[derive(Default)]
pub struct InputLine {
    buf: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    hist_pos: Option<usize>,
    draft: String,
}

fn width_of(chars: &[char]) -> usize {
    chars.iter().map(|c| c.width().unwrap_or(0)).sum()
}

impl InputLine {
    pub fn text(&self) -> String {
        self.buf.iter().collect()
    }

    pub fn set(&mut self, s: &str) {
        self.buf = s.chars().collect();
        self.cursor = self.buf.len();
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;
        self.hist_pos = None;
    }

    pub fn on_key(&mut self, key: KeyEvent) -> InputEvent {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Enter => {
                let text = self.text();
                if !text.trim().is_empty() && self.history.last() != Some(&text) {
                    self.history.push(text.clone());
                }
                self.clear();
                return InputEvent::Submit(text);
            }
            KeyCode::Tab => return InputEvent::Tab,
            KeyCode::Char('a') if ctrl => self.cursor = 0,
            KeyCode::Char('e') if ctrl => self.cursor = self.buf.len(),
            KeyCode::Char('u') if ctrl => {
                self.buf.drain(..self.cursor);
                self.cursor = 0;
            }
            KeyCode::Char('k') if ctrl => self.buf.truncate(self.cursor),
            KeyCode::Char('w') if ctrl => {
                let mut start = self.cursor;
                while start > 0 && self.buf[start - 1] == ' ' {
                    start -= 1;
                }
                while start > 0 && self.buf[start - 1] != ' ' {
                    start -= 1;
                }
                self.buf.drain(start..self.cursor);
                self.cursor = start;
            }
            KeyCode::Char(c) if !ctrl => {
                self.buf.insert(self.cursor, c);
                self.cursor += 1;
            }
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                self.buf.remove(self.cursor);
            }
            KeyCode::Delete if self.cursor < self.buf.len() => {
                self.buf.remove(self.cursor);
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.buf.len()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.buf.len(),
            KeyCode::Up => self.history_prev(),
            KeyCode::Down => self.history_next(),
            _ => {}
        }
        InputEvent::None
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let pos = match self.hist_pos {
            None => {
                self.draft = self.text();
                self.history.len() - 1
            }
            Some(p) => p.saturating_sub(1),
        };
        self.hist_pos = Some(pos);
        let h = self.history[pos].clone();
        self.set(&h);
    }

    fn history_next(&mut self) {
        let Some(p) = self.hist_pos else { return };
        if p + 1 < self.history.len() {
            self.hist_pos = Some(p + 1);
            let h = self.history[p + 1].clone();
            self.set(&h);
        } else {
            self.hist_pos = None;
            let d = std::mem::take(&mut self.draft);
            self.set(&d);
        }
    }

    /// Draw inside `area` (no border; the caller provides the block). Places the cursor when `focused`.
    pub fn render(&self, f: &mut Frame, area: Rect, placeholder: &str, focused: bool) {
        let avail = area.width.saturating_sub(1) as usize;
        if avail == 0 {
            return;
        }
        if self.buf.is_empty() {
            f.render_widget(Paragraph::new(Span::styled(placeholder.to_string(), theme::dim())), area);
        } else {
            // Scroll horizontally so the cursor stays visible.
            let mut start = 0;
            while width_of(&self.buf[start..self.cursor]) >= avail {
                start += 1;
            }
            let mut visible = String::new();
            let mut w = 0;
            for c in &self.buf[start..] {
                let cw = c.width().unwrap_or(0);
                if w + cw > avail {
                    break;
                }
                visible.push(*c);
                w += cw;
            }
            f.render_widget(Paragraph::new(Line::from(Span::styled(visible, Style::new()))), area);
        }
        if focused {
            let mut start = 0;
            while width_of(&self.buf[start..self.cursor]) >= avail {
                start += 1;
            }
            let x = area.x + width_of(&self.buf[start..self.cursor]) as u16;
            f.set_cursor_position(Position::new(x.min(area.x + area.width.saturating_sub(1)), area.y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEvent;

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    #[test]
    fn typing_editing_and_history() {
        let mut i = InputLine::default();
        for c in "helo".chars() {
            i.on_key(key(KeyCode::Char(c)));
        }
        i.on_key(key(KeyCode::Left));
        i.on_key(key(KeyCode::Char('l')));
        assert_eq!(i.text(), "hello");
        match i.on_key(key(KeyCode::Enter)) {
            InputEvent::Submit(s) => assert_eq!(s, "hello"),
            _ => panic!("expected submit"),
        }
        assert_eq!(i.text(), "");
        i.on_key(key(KeyCode::Up));
        assert_eq!(i.text(), "hello");
        i.on_key(key(KeyCode::Down));
        assert_eq!(i.text(), "");
    }

    #[test]
    fn ctrl_w_deletes_a_word() {
        let mut i = InputLine::default();
        i.set("one two");
        i.on_key(KeyEvent::new(KeyCode::Char('w'), KeyModifiers::CONTROL));
        assert_eq!(i.text(), "one ");
    }
}
