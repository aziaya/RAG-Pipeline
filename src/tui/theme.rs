use ratatui::{
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, BorderType},
};

pub const ACCENT: Color = Color::Cyan;

pub fn dim() -> Style {
    Style::new().fg(Color::DarkGray)
}

pub fn bold(color: Color) -> Style {
    Style::new().fg(color).add_modifier(Modifier::BOLD)
}

/// A rounded pane; the border lights up when the pane has focus.
pub fn pane<'a>(title: impl Into<Line<'a>>, focused: bool) -> Block<'a> {
    let border = if focused { Style::new().fg(ACCENT) } else { dim() };
    Block::bordered().border_type(BorderType::Rounded).border_style(border).title(title.into())
}

pub fn spinner(tick: u64) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[(tick as usize) % FRAMES.len()]
}

/// `1.2s` / `340ms`
pub fn human_ms(ms: u64) -> String {
    if ms >= 60_000 {
        format!("{}m{:02}s", ms / 60_000, (ms % 60_000) / 1000)
    } else if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    }
}

/// Tiny inline chart: ▁▂▃▄▅▆▇█
pub fn trend(values: &[u64], width: usize) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let tail: Vec<u64> = values.iter().rev().take(width).rev().copied().collect();
    let max = tail.iter().copied().max().unwrap_or(0).max(1);
    tail.iter().map(|v| BARS[((*v as f64 / max as f64) * 7.0).round() as usize]).collect()
}
