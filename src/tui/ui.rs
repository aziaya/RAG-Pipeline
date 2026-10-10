//! Layout and drawing.

use super::{
    app::{App, ObsLayout, Pane, ScreenId},
    screens::RenderCtx,
    theme, widgets,
};
use crate::tui::app::Mode;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, Paragraph, Tabs, Wrap},
    Frame,
};

const MIN_W: u16 = 80;
const MIN_H: u16 = 20;

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    if area.width < MIN_W || area.height < MIN_H {
        let msg = format!("Terminal too small ({}x{}). Need at least {MIN_W}x{MIN_H}.", area.width, area.height);
        f.render_widget(Paragraph::new(msg).wrap(Wrap { trim: true }), area);
        return;
    }
    let [header, body, footer] = Layout::vertical([Constraint::Length(3), Constraint::Min(0), Constraint::Length(1)]).areas(area);
    draw_header(f, header, app);

    // On narrow terminals a right-hand pane would squeeze the screens: stack instead.
    let layout = if app.layout == ObsLayout::Right && area.width < 120 { ObsLayout::Bottom } else { app.layout };
    let (main, obs): (Rect, Option<Rect>) = match layout {
        ObsLayout::Right => {
            let [l, r] = Layout::horizontal([Constraint::Percentage(57), Constraint::Percentage(43)]).areas(body);
            (l, Some(r))
        }
        ObsLayout::Bottom => {
            let [t, b] = Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(body);
            (t, Some(b))
        }
        ObsLayout::Hidden => (body, None),
    };

    let rc = RenderCtx { focused: app.focus == Pane::Main, mode: app.mode, tick: app.tick };
    app.screen_mut().render(f, main, &rc);
    if let Some(o) = obs {
        let focused = app.focus == Pane::Obs;
        let tick = app.tick;
        app.obs.render(f, o, focused, tick);
    }
    app.screen_mut().render_popup(f);
    draw_footer(f, footer, app);
    if app.help {
        draw_help(f);
    }
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let (mode_text, mode_color) = match (app.focus, app.mode) {
        (Pane::Obs, _) => ("TRACE", Color::Magenta),
        (_, Mode::Input) if app.screen_ref().has_input() => ("INPUT", Color::Green),
        _ => ("NAV", Color::Yellow),
    };
    let badge = Line::from(vec![
        Span::styled(format!(" {mode_text} "), Style::new().fg(Color::Black).bg(mode_color).add_modifier(Modifier::BOLD)),
        Span::styled(" ? help ", theme::dim()),
    ])
    .right_aligned();
    let block = theme::pane(Line::from(Span::styled(" RAG Pipeline ", theme::bold(Color::White))), false).title_top(badge);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let titles: Vec<Line> = ScreenId::ALL
        .iter()
        .enumerate()
        .map(|(i, s)| Line::from(format!(" {} {} ", i + 1, s.title())))
        .collect();
    let selected = ScreenId::ALL.iter().position(|s| *s == app.screen).unwrap_or(0);
    f.render_widget(
        Tabs::new(titles)
            .select(selected)
            .style(theme::dim())
            .highlight_style(Style::new().fg(Color::Black).bg(theme::ACCENT).add_modifier(Modifier::BOLD))
            .divider("│"),
        inner,
    );
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let line = if let Some((text, _)) = &app.notice {
        Line::from(Span::styled(format!(" {text}"), Style::new().fg(Color::Yellow)))
    } else if app.focus == Pane::Obs {
        Line::from(Span::styled(
            " ↑↓ PgUp/PgDn scroll · End follow · v verbosity · p pause · X clear · L layout · Esc/Tab back",
            theme::dim(),
        ))
    } else {
        Line::from(Span::styled(format!(" {}", app.screen_ref().hints(app.mode)), theme::dim()))
    };
    f.render_widget(Paragraph::new(line), area);
}

fn draw_help(f: &mut Frame) {
    let area = f.area();
    let rect = widgets::centered(area, 94.min(area.width.saturating_sub(4)), 22.min(area.height.saturating_sub(2)));
    f.render_widget(Clear, rect);
    let h = |t: &str| Line::from(Span::styled(t.to_string(), theme::bold(Color::Cyan)));
    let lines = vec![
        h("Moving around"),
        Line::from("  Esc            leave the text field (NAV mode); i or Enter to type again"),
        Line::from("  1-5, h/l, ←/→  switch screen          q  quit          Ctrl-C  quit anywhere"),
        Line::from("  Tab            move focus to the trace pane and back"),
        Line::default(),
        h("Trace pane (right/bottom)"),
        Line::from("  v  verbosity INFO → DEBUG → TRACE      p  pause / resume     X  clear"),
        Line::from("  L  layout: right → bottom → hidden     ↑↓ PgUp PgDn scroll   End follow live"),
        Line::default(),
        h("Screens"),
        Line::from("  Chat    type a question · /help lists slash-commands · PgUp/PgDn scroll"),
        Line::from("  Ingest  type a path · Tab completes · x cancels the run (NAV) · c clears"),
        Line::from("  Search  Enter searches, then opens the selected result · Tab changes mode"),
        Line::from("  Docs    j/k select · Enter or c chat about it · d delete · r refresh"),
        Line::from("  Status  r re-runs the health checks"),
        Line::default(),
        Line::from(Span::styled("press any key to close", theme::dim())),
    ];
    f.render_widget(Paragraph::new(lines).block(theme::pane(" Help ", true)), rect);
}
