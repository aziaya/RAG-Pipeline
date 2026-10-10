//! The screens shown in the left pane.

pub mod chat;
pub mod docs;
pub mod ingest;
pub mod search;
pub mod status;

use crate::tui::app::Mode;
use crossterm::event::KeyEvent;
use ratatui::{layout::Rect, Frame};

pub struct RenderCtx {
    pub focused: bool,
    pub mode: Mode,
    pub tick: u64,
}

pub trait Screen {
    /// Does this screen have a text field (so Input mode makes sense)?
    fn has_input(&self) -> bool;
    /// A popup is open and should receive every key.
    fn popup_open(&self) -> bool {
        false
    }
    /// Called when the screen becomes active.
    fn on_enter(&mut self) {}
    /// Returns true when the key was consumed.
    fn on_key(&mut self, key: KeyEvent, mode: Mode) -> bool;
    fn render(&mut self, f: &mut Frame, area: Rect, rc: &RenderCtx);
    /// Popups are drawn last, over both panes.
    fn render_popup(&mut self, _f: &mut Frame) {}
    fn hints(&self, mode: Mode) -> &'static str;
}
