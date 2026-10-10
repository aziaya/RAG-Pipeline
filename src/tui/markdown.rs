//! A small markdown subset (headings, bullets, **bold**, `code`, fenced code, [n] citations)
//! rendered into pre-wrapped ratatui lines, so scrolling is exact.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use unicode_width::UnicodeWidthStr;

type Seg = (String, Style);

fn parse_inline(text: &str, base: Style) -> Vec<Seg> {
    let chars: Vec<char> = text.chars().collect();
    let (mut out, mut cur): (Vec<Seg>, String) = (vec![], String::new());
    let (mut bold, mut code) = (false, false);
    let style = |bold: bool, code: bool| {
        let mut s = base;
        if bold {
            s = s.add_modifier(Modifier::BOLD);
        }
        if code {
            s = s.fg(Color::Yellow);
        }
        s
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '*' && chars.get(i + 1) == Some(&'*') {
            if !cur.is_empty() {
                out.push((std::mem::take(&mut cur), style(bold, code)));
            }
            bold = !bold;
            i += 2;
            continue;
        }
        if c == '`' {
            if !cur.is_empty() {
                out.push((std::mem::take(&mut cur), style(bold, code)));
            }
            code = !code;
            i += 1;
            continue;
        }
        if c == '[' && !code {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && chars.get(j) == Some(&']') {
                if !cur.is_empty() {
                    out.push((std::mem::take(&mut cur), style(bold, code)));
                }
                out.push((chars[i..=j].iter().collect(), Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD)));
                i = j + 1;
                continue;
            }
        }
        cur.push(c);
        i += 1;
    }
    if !cur.is_empty() {
        out.push((cur, style(bold, code)));
    }
    out
}

fn flush(lines: &mut Vec<Line<'static>>, cur: &mut Vec<Span<'static>>, prefix: &str, prefix_style: Style) {
    let mut spans = vec![Span::styled(prefix.to_string(), prefix_style)];
    spans.append(cur);
    lines.push(Line::from(spans));
}

/// Greedy word wrap of styled segments. A space at the edge of one styled run still
/// separates it from the next run ("It has **82M** parameters").
fn wrap(segs: Vec<Seg>, width: usize, first: &str, rest: &str, prefix_style: Style) -> Vec<Line<'static>> {
    let avail = width.saturating_sub(first.width().max(rest.width())).max(8);
    let mut lines: Vec<Line<'static>> = vec![];
    let mut cur: Vec<Span<'static>> = vec![];
    let mut cur_w = 0usize;
    let mut prefix = first;
    let mut pending_space = false;

    for (text, style) in segs {
        let mut first_piece = true;
        for piece in text.split(' ') {
            if !first_piece {
                pending_space = true;
            }
            first_piece = false;
            if piece.is_empty() {
                continue;
            }
            let mut word = piece.to_string();
            let sep = if cur_w > 0 && pending_space { 1 } else { 0 };
            while word.width() > avail {
                // Hard-split a word longer than a line.
                let mut head = String::new();
                for ch in word.chars() {
                    if head.width() + ch.to_string().width() > avail.saturating_sub(cur_w) && !head.is_empty() {
                        break;
                    }
                    head.push(ch);
                    if cur_w + head.width() >= avail {
                        break;
                    }
                }
                let rest_word: String = word.chars().skip(head.chars().count()).collect();
                cur.push(Span::styled(head, style));
                flush(&mut lines, &mut cur, prefix, prefix_style);
                prefix = rest;
                cur_w = 0;
                word = rest_word;
            }
            if word.is_empty() {
                pending_space = false;
                continue;
            }
            if cur_w > 0 && cur_w + sep + word.width() > avail {
                flush(&mut lines, &mut cur, prefix, prefix_style);
                prefix = rest;
                cur_w = 0;
            } else if sep == 1 {
                cur.push(Span::raw(" "));
                cur_w += 1;
            }
            cur_w += word.width();
            cur.push(Span::styled(word, style));
            pending_space = false;
        }
    }
    if !cur.is_empty() || lines.is_empty() {
        flush(&mut lines, &mut cur, prefix, prefix_style);
    }
    lines
}

pub fn render(text: &str, width: usize, base: Style) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = vec![];
    let mut in_code = false;
    for raw in text.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            lines.extend(wrap(vec![(raw.to_string(), Style::new().fg(Color::Yellow))], width, "  ", "  ", Style::new()));
            continue;
        }
        if trimmed.is_empty() {
            lines.push(Line::default());
            continue;
        }
        if trimmed.starts_with('#') {
            let title = trimmed.trim_start_matches('#').trim();
            lines.extend(wrap(
                vec![(title.to_string(), Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD))],
                width,
                "",
                "",
                Style::new(),
            ));
            continue;
        }
        let indent = raw.len() - trimmed.len();
        let pad = " ".repeat(indent.min(8));
        if let Some(item) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
            let first = format!("{pad}• ");
            let rest = format!("{pad}  ");
            lines.extend(wrap(parse_inline(item, base), width, &first, &rest, Style::new().fg(Color::Cyan)));
            continue;
        }
        let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 && trimmed[digits..].starts_with(". ") {
            let first = format!("{pad}{} ", &trimmed[..digits + 1]);
            let rest = format!("{pad}{}", " ".repeat(digits + 2));
            lines.extend(wrap(parse_inline(&trimmed[digits + 2..], base), width, &first, &rest, Style::new().fg(Color::Cyan)));
            continue;
        }
        lines.extend(wrap(parse_inline(trimmed, base), width, &pad, &pad, Style::new()));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line]) -> Vec<String> {
        lines.iter().map(|l| l.spans.iter().map(|s| s.content.to_string()).collect::<String>()).collect()
    }

    #[test]
    fn wraps_at_word_boundaries_within_width() {
        let out = plain(&render("the quick brown fox jumps over the lazy dog", 20, Style::new()));
        assert!(out.iter().all(|l| l.chars().count() <= 20), "{out:?}");
        assert_eq!(out.join(" ").split_whitespace().count(), 9);
    }

    #[test]
    fn bullets_bold_and_citations() {
        let lines = render("- It has **82M** parameters [1]", 60, Style::new());
        let p = plain(&lines);
        assert_eq!(p[0], "• It has 82M parameters [1]");
        let cite = lines[0].spans.iter().find(|s| s.content == "[1]").unwrap();
        assert_eq!(cite.style.fg, Some(Color::Yellow));
        let bold = lines[0].spans.iter().find(|s| s.content == "82M").unwrap();
        assert!(bold.style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn very_long_word_is_split() {
        let out = plain(&render(&"x".repeat(50), 20, Style::new()));
        assert!(out.len() >= 3 && out.iter().all(|l| l.chars().count() <= 20), "{out:?}");
    }

    #[test]
    fn code_fences_are_hidden() {
        let out = plain(&render("a\n```\nlet x = 1;\n```\nb", 40, Style::new()));
        assert_eq!(out, vec!["a", "  let x = 1;", "b"]);
    }
}
