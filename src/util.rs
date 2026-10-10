//! Small helpers shared by the CLI and the terminal UIs.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// File name of a path-like source id.
pub fn short(source: &str) -> String {
    Path::new(source)
        .file_name()
        .and_then(|n| n.to_str())
        .map(String::from)
        .unwrap_or_else(|| source.to_string())
}

/// Collapse whitespace and cut to `n` characters.
pub fn trunc(s: &str, n: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= n {
        flat
    } else {
        format!("{}…", flat.chars().take(n.saturating_sub(1)).collect::<String>())
    }
}

/// `████░░░░ 0.52`
pub fn score_bar(score: Option<f64>) -> String {
    match score {
        Some(sc) => {
            let n = (sc.clamp(0.0, 1.0) * 8.0).round() as usize;
            format!("{}{} {:.2}", "█".repeat(n), "░".repeat(8 - n), sc)
        }
        None => String::new(),
    }
}

/// Turn what a user typed or dragged into a terminal into a real path.
pub fn clean_path(input: &str) -> PathBuf {
    let mut s = input.trim().to_string();
    if (s.starts_with('\'') && s.ends_with('\'') || s.starts_with('"') && s.ends_with('"')) && s.len() >= 2 {
        s = s[1..s.len() - 1].to_string();
    }
    s = s.replace("\\ ", " ");
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(s)
}

/// Files under `path` (or `path` itself), skipping hidden files and folders.
pub fn collect_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut files: Vec<PathBuf> = WalkDir::new(path)
        .into_iter()
        .filter_entry(|e| e.depth() == 0 || !e.file_name().to_string_lossy().starts_with('.'))
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    files.sort();
    files
}

/// Tab-completion for filesystem paths: the longest common prefix of the matches.
pub fn complete_path(input: &str) -> Option<String> {
    let typed = clean_path(input);
    let (dir, prefix) = if input.ends_with('/') {
        (typed.clone(), String::new())
    } else {
        (
            typed.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(".")),
            typed.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
        )
    };
    let dir = if dir.as_os_str().is_empty() { PathBuf::from(".") } else { dir };
    let mut names: Vec<(String, bool)> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| Some((e.file_name().to_str()?.to_string(), e.path().is_dir())))
        .filter(|(n, _)| n.starts_with(&prefix) && (prefix.starts_with('.') || !n.starts_with('.')))
        .collect();
    names.sort();
    let first = names.first()?.clone();
    let mut common = first.0.clone();
    for (n, _) in &names {
        while !n.starts_with(&common) {
            common.pop();
        }
    }
    let completed = if names.len() == 1 && first.1 { format!("{common}/") } else { common };
    let base = input.rsplit_once('/').map(|(b, _)| format!("{b}/")).unwrap_or_default();
    Some(format!("{base}{completed}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trunc_and_short() {
        assert_eq!(trunc("a   b\nc", 10), "a b c");
        assert_eq!(trunc("abcdefghij", 5), "abcd…");
        assert_eq!(short("/a/b/c.pdf"), "c.pdf");
    }

    #[test]
    fn clean_path_handles_drag_and_drop() {
        assert_eq!(clean_path("'/tmp/my file.txt'"), PathBuf::from("/tmp/my file.txt"));
        assert_eq!(clean_path("/tmp/my\\ file.txt"), PathBuf::from("/tmp/my file.txt"));
    }
}
