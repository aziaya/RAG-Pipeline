use super::{RenderCtx, Screen};
use crate::{
    context::AppContext,
    models::chat::{ChatAnswer, ChatTurn, RagStage, SourceRef},
    tui::{
        app::{AppMsg, Mode},
        input::{InputEvent, InputLine},
        markdown, theme,
        widgets::{Picker, PickerEvent},
    },
    util::{score_bar, short, trunc},
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::UnboundedSender;

pub enum ChatMsg {
    Stage(RagStage),
    Done(Result<ChatAnswer, String>),
    Docs(Result<Vec<(String, usize)>, String>),
}

enum Role {
    User,
    Assistant,
    System,
    Error,
}

struct Msg {
    role: Role,
    text: String,
    sources: Vec<SourceRef>,
    took: Option<Duration>,
}

struct Busy {
    started: Instant,
    stage: String,
}

const HELP: &str = "- `/source` pick a document to search within (or all)\n- `/show N` read source N of the last answer in full\n- `/alpha 0-1` 0 = keywords only, 1 = meaning only\n- `/limit N` passages retrieved per question\n- `/clear` forget this conversation\n\nPress **Esc** to leave the input, then **q** to quit.";

pub struct ChatScreen {
    ctx: AppContext,
    tx: UnboundedSender<AppMsg>,
    input: InputLine,
    msgs: Vec<Msg>,
    history: Vec<ChatTurn>,
    pending_question: Option<String>,
    busy: Option<Busy>,
    scroll: usize,
    source: Option<String>,
    limit: usize,
    alpha: f32,
    picker: Option<(Picker, Vec<Option<String>>)>,
    last_sources: Vec<SourceRef>,
}

impl ChatScreen {
    pub fn new(ctx: AppContext, tx: UnboundedSender<AppMsg>) -> Self {
        let limit = ctx.cfg.chat_top_k;
        let mut s = Self {
            ctx,
            tx,
            input: InputLine::default(),
            msgs: vec![],
            history: vec![],
            pending_question: None,
            busy: None,
            scroll: 0,
            source: None,
            limit,
            alpha: 0.5,
            picker: None,
            last_sources: vec![],
        };
        s.reset_transcript();
        s
    }

    fn reset_transcript(&mut self) {
        self.msgs = vec![Msg {
            role: Role::System,
            text: "Ask a question about your documents. Type /help for commands.".into(),
            sources: vec![],
            took: None,
        }];
    }

    pub fn set_source(&mut self, source: Option<String>) {
        self.source = source;
        let scope = self.source.as_deref().map(short).unwrap_or_else(|| "all documents".into());
        self.system(format!("Searching within **{scope}**."));
    }

    fn system(&mut self, text: String) {
        self.msgs.push(Msg { role: Role::System, text, sources: vec![], took: None });
        self.scroll = 0;
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
        self.msgs.push(Msg { role: Role::User, text: text.clone(), sources: vec![], took: None });
        self.scroll = 0;
        self.pending_question = Some(text.clone());
        self.busy = Some(Busy { started: Instant::now(), stage: "Searching your documents…".into() });

        let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
        let (history, limit, alpha, source) = (self.history.clone(), self.limit, self.alpha, self.source.clone());
        tokio::spawn(async move {
            let tx2 = tx.clone();
            let res = ctx
                .rag
                .answer(&text, &history, limit, alpha, source.as_deref(), move |s| {
                    let _ = tx2.send(AppMsg::Chat(ChatMsg::Stage(s)));
                })
                .await;
            let _ = tx.send(AppMsg::Chat(ChatMsg::Done(res.map_err(|e| e.to_string()))));
        });
    }

    fn command(&mut self, cmd: &str) {
        let mut parts = cmd.splitn(2, ' ');
        let name = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        match name {
            "help" => self.system(HELP.into()),
            "clear" => {
                self.history.clear();
                self.last_sources.clear();
                self.reset_transcript();
            }
            "alpha" => match arg.parse::<f32>() {
                Ok(a) => {
                    self.alpha = a.clamp(0.0, 1.0);
                    self.system(format!("alpha is now **{:.2}**", self.alpha));
                }
                Err(_) => self.system("Usage: `/alpha 0.5`".into()),
            },
            "limit" => match arg.parse::<usize>() {
                Ok(n) => {
                    self.limit = n.clamp(1, 20);
                    self.system(format!("retrieving **{}** passages per question", self.limit));
                }
                Err(_) => self.system("Usage: `/limit 4`".into()),
            },
            "show" => match arg.parse::<usize>().ok().and_then(|n| n.checked_sub(1)).and_then(|i| self.last_sources.get(i)).cloned() {
                Some(s) => {
                    let h = if s.headings.is_empty() { String::new() } else { format!("{}\n\n", s.headings.join(" > ")) };
                    self.system(format!("**[{}] {}** (chunk {})\n\n{h}{}", s.n, s.source, s.chunk_index, s.text));
                }
                None => self.system("Usage: `/show N` (a source number from the last answer)".into()),
            },
            "source" => {
                let (ctx, tx) = (self.ctx.clone(), self.tx.clone());
                tokio::spawn(async move {
                    let r = ctx.store.list_sources().await.map_err(|e| format!("{e:#}"));
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
                        RagStage::Searching => "Searching your documents…".into(),
                        RagStage::Generating { passages } => format!("Reading {passages} passages and writing an answer…"),
                    };
                }
            }
            ChatMsg::Done(Ok(ans)) => {
                let took = self.busy.take().map(|b| b.started.elapsed());
                if let Some(q) = self.pending_question.take() {
                    self.history.push(ChatTurn { role: "user".into(), content: q });
                    self.history.push(ChatTurn { role: "assistant".into(), content: ans.answer.clone() });
                }
                self.last_sources = ans.sources.clone();
                self.msgs.push(Msg { role: Role::Assistant, text: ans.answer, sources: ans.sources, took });
                self.scroll = 0;
            }
            ChatMsg::Done(Err(e)) => {
                self.busy = None;
                self.pending_question = None;
                self.msgs.push(Msg { role: Role::Error, text: e, sources: vec![], took: None });
                self.scroll = 0;
            }
            ChatMsg::Docs(Ok(docs)) => {
                if docs.is_empty() {
                    self.system("No documents ingested yet.".into());
                } else {
                    let mut items = vec!["All documents".to_string()];
                    let mut values = vec![None];
                    for (name, n) in docs {
                        items.push(format!("{}  ({n} chunks)", short(&name)));
                        values.push(Some(name));
                    }
                    self.picker = Some((Picker::new("Search within", items), values));
                }
            }
            ChatMsg::Docs(Err(e)) => self.msgs.push(Msg { role: Role::Error, text: e, sources: vec![], took: None }),
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
                    lines.push(Line::from(vec![
                        Span::styled(format!("  [{}] ", s.n), theme::bold(Color::Yellow)),
                        Span::styled(short(&s.source), Style::new().fg(Color::Green)),
                        Span::styled(section, theme::dim()),
                        Span::styled(format!("  {}", score_bar(s.score)), theme::dim()),
                    ]));
                }
            }
            if let Some(t) = m.took {
                lines.push(Line::from(Span::styled(format!("answered in {:.1}s · /show N reads a source", t.as_secs_f32()), theme::dim())));
            }
            lines.push(Line::default());
        }
        if let Some(b) = &self.busy {
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", theme::spinner(tick)), Style::new().fg(Color::Cyan)),
                Span::raw(b.stage.clone()),
                Span::styled(format!("  {}s", b.started.elapsed().as_secs()), theme::dim()),
            ]));
        }
        lines
    }
}

impl Screen for ChatScreen {
    fn has_input(&self) -> bool {
        true
    }

    fn popup_open(&self) -> bool {
        self.picker.is_some()
    }

    fn on_key(&mut self, key: KeyEvent, mode: Mode) -> bool {
        if let Some((picker, values)) = self.picker.as_mut() {
            match picker.on_key(key) {
                PickerEvent::Chosen(i) => {
                    let v = values.get(i).cloned().flatten();
                    self.picker = None;
                    self.set_source(v);
                }
                PickerEvent::Cancel => self.picker = None,
                PickerEvent::None => {}
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
        if mode == Mode::Input {
            if let InputEvent::Submit(t) = self.input.on_key(key) {
                self.submit(t);
            }
            return true;
        }
        match key.code {
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

        let scope = self.source.as_deref().map(short).unwrap_or_else(|| "all documents".into());
        let title = Line::from(vec![
            Span::styled(" Ask ", theme::bold(Color::White)),
            Span::styled(format!("· {scope} · top {} · α {:.2} ", self.limit, self.alpha), theme::dim()),
        ]);
        let typing = rc.focused && rc.mode == Mode::Input;
        let ib = theme::pane(title, typing);
        let iin = ib.inner(input_area);
        f.render_widget(ib, input_area);
        self.input.render(f, iin, "Ask about your documents… (/help)", typing && self.picker.is_none());
    }

    fn render_popup(&mut self, f: &mut Frame) {
        if let Some((p, _)) = self.picker.as_mut() {
            p.render(f);
        }
    }

    fn hints(&self, mode: Mode) -> &'static str {
        match mode {
            Mode::Input => "Enter send · ↑↓ history · PgUp/PgDn scroll · /help · Esc → navigate",
            Mode::Nav => "i or Enter type · j/k PgUp/PgDn scroll · 1-5 screens · Tab pane · q quit",
        }
    }
}
