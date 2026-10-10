//! The observability pane: in-flight operations, metrics and a live, filterable trace log.

use crate::observability::{
    events::{EventKind, TraceEvent},
    metrics::metrics,
};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use std::collections::VecDeque;
use tracing::Level;
use unicode_width::UnicodeWidthStr;

use super::theme::{self, human_ms};

const CAP: usize = 3000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verbosity {
    Info,
    Debug,
    Trace,
}

impl Verbosity {
    pub fn next(self) -> Self {
        match self {
            Self::Info => Self::Debug,
            Self::Debug => Self::Trace,
            Self::Trace => Self::Info,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }

    fn allows(self, level: Level) -> bool {
        let rank = |l: Level| match l {
            Level::ERROR => 0,
            Level::WARN => 1,
            Level::INFO => 2,
            Level::DEBUG => 3,
            Level::TRACE => 4,
        };
        let max = match self {
            Self::Info => 2,
            Self::Debug => 3,
            Self::Trace => 4,
        };
        rank(level) <= max
    }
}

pub struct ObsState {
    events: VecDeque<TraceEvent>,
    /// Lines scrolled up from the newest; 0 follows the live tail.
    pub scroll: usize,
    /// While paused, the view is frozen at this many events.
    paused_len: Option<usize>,
    pub verbosity: Verbosity,
}

impl ObsState {
    pub fn new() -> Self {
        Self { events: VecDeque::new(), scroll: 0, paused_len: None, verbosity: Verbosity::Info }
    }

    pub fn push(&mut self, ev: TraceEvent) {
        if self.events.len() == CAP {
            self.events.pop_front();
            if let Some(n) = self.paused_len.as_mut() {
                *n = n.saturating_sub(1);
            }
        }
        self.events.push_back(ev);
    }

    pub fn cycle_verbosity(&mut self) {
        self.verbosity = self.verbosity.next();
        self.scroll = 0;
    }

    pub fn toggle_pause(&mut self) {
        self.paused_len = match self.paused_len {
            Some(_) => None,
            None => Some(self.events.len()),
        };
    }

    pub fn clear(&mut self) {
        self.events.clear();
        self.paused_len = self.paused_len.map(|_| 0);
        self.scroll = 0;
    }

    /// Keys while the pane has focus.
    pub fn on_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll += 1,
            KeyCode::Down | KeyCode::Char('j') => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::PageUp => self.scroll += 10,
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::End | KeyCode::Char('G') => self.scroll = 0,
            _ => {}
        }
    }

    fn visible(&self) -> Vec<&TraceEvent> {
        let n = self.paused_len.unwrap_or(self.events.len()).min(self.events.len());
        self.events
            .iter()
            .take(n)
            .filter(|e| match e.kind {
                EventKind::SpanClosed { .. } => true,
                EventKind::Log => self.verbosity.allows(e.level),
            })
            .collect()
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect, focused: bool, tick: u64) {
        let active = metrics().active_snapshot();
        let snap = metrics().snapshot();

        let active_h = active.len().clamp(1, 4) as u16 + 2;
        let metrics_h = (snap.latency.len().max(1) as u16 + 6).min(area.height.saturating_sub(active_h + 5));
        let [a, m, t] = Layout::vertical([Constraint::Length(active_h), Constraint::Length(metrics_h), Constraint::Min(4)]).areas(area);

        // ── in flight ──
        let mut lines: Vec<Line> = vec![];
        if active.is_empty() {
            lines.push(Line::from(Span::styled("idle", theme::dim())));
        }
        for op in active.iter().take(4) {
            let mut spans = vec![
                Span::styled(format!("{} ", theme::spinner(tick)), Style::new().fg(Color::Green)),
                Span::styled(op.name.to_string(), theme::bold(Color::Magenta)),
                Span::raw("  "),
                Span::styled(human_ms(op.started.elapsed().as_millis() as u64), Style::new().fg(Color::Yellow)),
            ];
            if !op.detail.is_empty() {
                spans.push(Span::styled(format!("  {}", op.detail), theme::dim()));
            }
            lines.push(fit(spans, a.width.saturating_sub(2) as usize));
        }
        f.render_widget(Paragraph::new(lines).block(theme::pane(" In flight ", false)), a);

        // ── metrics ──
        let mut ml: Vec<Line> = vec![
            Line::from(vec![
                Span::styled("files  ", theme::dim()),
                Span::styled(format!("{} ✔", snap.files_indexed), Style::new().fg(Color::Green)),
                Span::raw("  "),
                Span::styled(format!("{} ↷", snap.files_skipped), Style::new().fg(Color::Yellow)),
                Span::raw("  "),
                Span::styled(format!("{} ✖", snap.files_failed), Style::new().fg(if snap.files_failed > 0 { Color::Red } else { Color::DarkGray })),
            ]),
            Line::from(vec![
                Span::styled("chunks ", theme::dim()),
                Span::raw(format!("{}  ", snap.chunks_indexed)),
                Span::styled("embeds ", theme::dim()),
                Span::raw(format!("{}  ", snap.embeddings)),
                Span::styled("search ", theme::dim()),
                Span::raw(format!("{}  ", snap.searches)),
                Span::styled("chats ", theme::dim()),
                Span::raw(format!("{}  ", snap.chats)),
                Span::styled("errors ", theme::dim()),
                Span::styled(snap.errors.to_string(), Style::new().fg(if snap.errors > 0 { Color::Red } else { Color::White })),
            ]),
            Line::from(vec![
                Span::styled("llm    ", theme::dim()),
                if snap.llm_tokens > 0 {
                    Span::raw(format!("{} tokens · {:.1} tok/s", snap.llm_tokens, snap.llm_rate))
                } else {
                    Span::styled("no generation yet", theme::dim())
                },
            ]),
            Line::from(Span::styled(format!("{:<15}{:>7}{:>7}{:>7}  trend", "operation", "last", "avg", "p95"), theme::dim())),
        ];
        if snap.latency.is_empty() {
            ml.push(Line::from(Span::styled("no timings yet", theme::dim())));
        }
        for r in &snap.latency {
            let color = if r.last < 500 { Color::Green } else if r.last < 5000 { Color::Yellow } else { Color::Red };
            ml.push(Line::from(vec![
                Span::styled(format!("{:<15}", r.name), Style::new().fg(Color::Magenta)),
                Span::styled(format!("{:>7}", human_ms(r.last)), Style::new().fg(color)),
                Span::raw(format!("{:>7}{:>7}  ", human_ms(r.avg), human_ms(r.p95))),
                Span::styled(theme::trend(&r.history, 12), Style::new().fg(Color::Cyan)),
            ]));
        }
        f.render_widget(Paragraph::new(ml).block(theme::pane(" Metrics ", false)), m);

        // ── trace log ──
        let (lines, scroll) = {
            let vis = self.visible();
            let inner_h = t.height.saturating_sub(2) as usize;
            let inner_w = t.width.saturating_sub(2) as usize;
            let total = vis.len();
            let scroll = self.scroll.min(total.saturating_sub(inner_h));
            let end = total - scroll.min(total);
            let start = end.saturating_sub(inner_h);
            let lines: Vec<Line> = vis[start..end].iter().map(|e| event_line(e, inner_w)).collect();
            (lines, scroll)
        };
        self.scroll = scroll;

        let state = if self.paused_len.is_some() {
            Span::styled(" ⏸ paused ", Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        } else if self.scroll > 0 {
            Span::styled(format!(" ↑ {} lines back ", self.scroll), Style::new().fg(Color::Yellow))
        } else {
            Span::styled(" ● live ", Style::new().fg(Color::Green))
        };
        let title = Line::from(vec![
            Span::styled(" Trace ", theme::bold(Color::White)),
            Span::styled(format!("· {} ", self.verbosity.label()), Style::new().fg(Color::Cyan)),
            state,
        ]);
        let hint = if focused { " ↑↓ PgUp/PgDn scroll · End follow " } else { " Tab to focus " };
        let block = theme::pane(title, focused).title_bottom(Line::from(Span::styled(hint, theme::dim())).right_aligned());
        f.render_widget(Paragraph::new(lines).block(block), t);
    }
}

fn level_style(level: Level) -> (&'static str, Color) {
    match level {
        Level::ERROR => ("ERROR", Color::Red),
        Level::WARN => ("WARN ", Color::Yellow),
        Level::INFO => ("INFO ", Color::Green),
        Level::DEBUG => ("DEBUG", Color::Blue),
        Level::TRACE => ("TRACE", Color::DarkGray),
    }
}

fn event_line(ev: &TraceEvent, width: usize) -> Line<'static> {
    let mut spans = vec![Span::styled(ev.ts.format("%H:%M:%S%.3f").to_string(), theme::dim()), Span::raw(" ")];
    match ev.kind {
        EventKind::SpanClosed { elapsed_ms } => {
            let color = if elapsed_ms < 500 { Color::Green } else if elapsed_ms < 5000 { Color::Yellow } else { Color::Red };
            spans.push(Span::styled("✓ ", Style::new().fg(color)));
            spans.push(Span::styled(ev.message.clone(), Style::new().fg(Color::Magenta)));
            spans.push(Span::styled(format!("  {}", human_ms(elapsed_ms)), Style::new().fg(color).add_modifier(Modifier::BOLD)));
            if let Some((_, detail)) = ev.fields.first() {
                spans.push(Span::styled(format!("  {detail}"), theme::dim()));
            }
        }
        EventKind::Log => {
            let (label, color) = level_style(ev.level);
            spans.push(Span::styled(label.to_string(), Style::new().fg(color)));
            spans.push(Span::raw(" "));
            if let Some(span) = &ev.span {
                let name = span.split('{').next().unwrap_or(span);
                spans.push(Span::styled(format!("{name} "), Style::new().fg(Color::Magenta)));
            }
            spans.push(Span::raw(ev.message.clone()));
            for (k, v) in &ev.fields {
                spans.push(Span::styled(format!(" {k}={v}"), Style::new().fg(Color::Cyan).add_modifier(Modifier::DIM)));
            }
        }
    }
    fit(spans, width)
}

/// Truncate a line of spans to `width` columns with an ellipsis.
fn fit(spans: Vec<Span<'static>>, width: usize) -> Line<'static> {
    let total: usize = spans.iter().map(|s| s.content.width()).sum();
    if total <= width {
        return Line::from(spans);
    }
    let mut out = vec![];
    let mut used = 0;
    for s in spans {
        let w = s.content.width();
        if used + w < width {
            used += w;
            out.push(s);
        } else {
            let room = width.saturating_sub(used + 1);
            let cut: String = s.content.chars().scan(0usize, |acc, c| {
                *acc += unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
                (*acc <= room).then_some(c)
            }).collect();
            out.push(Span::styled(format!("{cut}…"), s.style));
            break;
        }
    }
    Line::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verbosity_filters_levels() {
        assert!(Verbosity::Info.allows(Level::WARN));
        assert!(Verbosity::Info.allows(Level::INFO));
        assert!(!Verbosity::Info.allows(Level::DEBUG));
        assert!(Verbosity::Debug.allows(Level::DEBUG));
        assert!(!Verbosity::Debug.allows(Level::TRACE));
        assert!(Verbosity::Trace.allows(Level::TRACE));
    }

    #[test]
    fn fit_truncates_with_ellipsis() {
        let line = fit(vec![Span::raw("hello world, this is long")], 10);
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(text.ends_with('…') && text.width() <= 10, "{text}");
    }
}
