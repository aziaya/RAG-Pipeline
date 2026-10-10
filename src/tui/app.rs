//! Application state, key routing and background-task messages.

use super::{
    obs::ObsState,
    screens::{
        chat::{ChatMsg, ChatScreen},
        docs::DocsScreen,
        ingest::{IngestMsg, IngestScreen},
        search::SearchScreen,
        status::StatusScreen,
        Screen,
    },
};
use crate::{
    context::AppContext,
    models::{health::Check, search::Hit},
    util::short,
};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

/// Typing into a text field, or navigating.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Input,
    Nav,
}

/// Which pane has keyboard focus.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pane {
    Main,
    Obs,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScreenId {
    Chat,
    Ingest,
    Search,
    Docs,
    Status,
}

impl ScreenId {
    pub const ALL: [ScreenId; 5] = [Self::Chat, Self::Ingest, Self::Search, Self::Docs, Self::Status];

    pub fn title(self) -> &'static str {
        match self {
            Self::Chat => "Chat",
            Self::Ingest => "Ingest",
            Self::Search => "Search",
            Self::Docs => "Docs",
            Self::Status => "Status",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ObsLayout {
    Right,
    Bottom,
    Hidden,
}

/// Results of background work, delivered to the UI loop.
pub enum AppMsg {
    Chat(ChatMsg),
    Ingest(IngestMsg),
    Search(Result<Vec<Hit>, String>),
    Docs(Result<Vec<(String, usize)>, String>),
    DocDeleted(Result<String, String>),
    Status(Vec<Check>),
    ChatAbout(String),
}

pub struct App {
    pub screen: ScreenId,
    pub chat: ChatScreen,
    pub ingest: IngestScreen,
    pub search: SearchScreen,
    pub docs: DocsScreen,
    pub status: StatusScreen,
    pub obs: ObsState,
    pub mode: Mode,
    pub focus: Pane,
    pub layout: ObsLayout,
    pub help: bool,
    pub notice: Option<(String, Instant)>,
    pub tick: u64,
    pub should_quit: bool,
}

impl App {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        let mut app = Self {
            screen: ScreenId::Chat,
            chat: ChatScreen::new(ctx.clone(), tx.clone()),
            ingest: IngestScreen::new(ctx.clone(), tx.clone()),
            search: SearchScreen::new(ctx.clone(), tx.clone()),
            docs: DocsScreen::new(ctx.clone(), tx.clone()),
            status: StatusScreen::new(ctx, tx),
            obs: ObsState::new(),
            mode: Mode::Input,
            focus: Pane::Main,
            layout: ObsLayout::Right,
            help: false,
            notice: None,
            tick: 0,
            should_quit: false,
        };
        tracing::info!("terminal UI started");
        // Warm the status screen so the trace pane shows activity straight away.
        app.status.refresh();
        app
    }

    pub fn screen_ref(&self) -> &dyn Screen {
        match self.screen {
            ScreenId::Chat => &self.chat,
            ScreenId::Ingest => &self.ingest,
            ScreenId::Search => &self.search,
            ScreenId::Docs => &self.docs,
            ScreenId::Status => &self.status,
        }
    }

    pub fn screen_mut(&mut self) -> &mut dyn Screen {
        match self.screen {
            ScreenId::Chat => &mut self.chat,
            ScreenId::Ingest => &mut self.ingest,
            ScreenId::Search => &mut self.search,
            ScreenId::Docs => &mut self.docs,
            ScreenId::Status => &mut self.status,
        }
    }

    pub fn notify(&mut self, text: impl Into<String>) {
        self.notice = Some((text.into(), Instant::now()));
    }

    pub fn goto(&mut self, id: ScreenId) {
        self.screen = id;
        self.focus = Pane::Main;
        self.mode = if self.screen_ref().has_input() { Mode::Input } else { Mode::Nav };
        self.screen_mut().on_enter();
    }

    fn step_screen(&mut self, delta: isize) {
        let n = ScreenId::ALL.len() as isize;
        let i = (self.screen.index() as isize + delta).rem_euclid(n) as usize;
        self.goto(ScreenId::ALL[i]);
    }

    pub fn on_tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        if let Some((_, at)) = &self.notice {
            if at.elapsed() > Duration::from_secs(6) {
                self.notice = None;
            }
        }
    }

    pub fn on_event(&mut self, ev: Event) {
        if let Event::Key(key) = ev {
            self.on_key(key);
        }
    }

    pub fn on_msg(&mut self, msg: AppMsg) {
        match msg {
            AppMsg::Chat(m) => self.chat.on_msg(m),
            AppMsg::Ingest(m) => self.ingest.on_msg(m),
            AppMsg::Search(r) => self.search.on_msg(r),
            AppMsg::Docs(r) => self.docs.on_docs(r),
            AppMsg::DocDeleted(Ok(name)) => {
                self.notify(format!("Deleted {}", short(&name)));
                self.docs.refresh();
            }
            AppMsg::DocDeleted(Err(e)) => self.notify(format!("Delete failed: {e}")),
            AppMsg::Status(c) => self.status.on_checks(c),
            AppMsg::ChatAbout(src) => {
                self.chat.set_source(Some(src));
                self.goto(ScreenId::Chat);
            }
        }
    }

    fn cycle_layout(&mut self) {
        self.layout = match self.layout {
            ObsLayout::Right => ObsLayout::Bottom,
            ObsLayout::Bottom => ObsLayout::Hidden,
            ObsLayout::Hidden => ObsLayout::Right,
        };
        if self.layout == ObsLayout::Hidden {
            self.focus = Pane::Main;
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return;
        }
        if self.help {
            self.help = false;
            return;
        }
        // A popup (picker, confirmation, result detail) owns the keyboard.
        if self.screen_ref().popup_open() {
            let mode = self.mode;
            self.screen_mut().on_key(key, mode);
            return;
        }

        if self.focus == Pane::Obs {
            match key.code {
                KeyCode::Esc | KeyCode::Tab => self.focus = Pane::Main,
                KeyCode::Char('q') => self.should_quit = true,
                KeyCode::Char('v') => self.obs.cycle_verbosity(),
                KeyCode::Char('p') => self.obs.toggle_pause(),
                KeyCode::Char('X') | KeyCode::Char('c') => self.obs.clear(),
                KeyCode::Char('L') => self.cycle_layout(),
                KeyCode::Char('?') => self.help = true,
                KeyCode::Char(c @ '1'..='5') => self.goto(ScreenId::ALL[(c as usize) - ('1' as usize)]),
                _ => self.obs.on_key(key),
            }
            return;
        }

        if self.mode == Mode::Input && self.screen_ref().has_input() {
            if key.code == KeyCode::Esc {
                self.mode = Mode::Nav;
            } else {
                self.screen_mut().on_key(key, Mode::Input);
            }
            return;
        }

        // Navigate mode.
        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char(c @ '1'..='5') => self.goto(ScreenId::ALL[(c as usize) - ('1' as usize)]),
            KeyCode::Char('h') | KeyCode::Left => self.step_screen(-1),
            KeyCode::Char('l') | KeyCode::Right => self.step_screen(1),
            KeyCode::Tab => {
                if self.layout == ObsLayout::Hidden {
                    self.notify("The trace pane is hidden: press L to show it");
                } else {
                    self.focus = Pane::Obs;
                }
            }
            KeyCode::Char('L') => self.cycle_layout(),
            KeyCode::Char('v') => self.obs.cycle_verbosity(),
            KeyCode::Char('p') => self.obs.toggle_pause(),
            KeyCode::Char('X') => self.obs.clear(),
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('i') if self.screen_ref().has_input() => self.mode = Mode::Input,
            _ => {
                let consumed = self.screen_mut().on_key(key, Mode::Nav);
                if !consumed && key.code == KeyCode::Enter && self.screen_ref().has_input() {
                    self.mode = Mode::Input;
                }
            }
        }
    }
}
