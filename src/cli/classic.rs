//! Prompt-and-menu terminal interface (single pane): menus, prompts, spinners, progress
//! bars, tables and a colored chat view. Presentation only; the work is done by the services.

use crate::{
    context::AppContext,
    models::{
        collection::sanitize_collection,
        chat::{ChatMemory, ChatTurn, RagStage, SourceRef},
        document::{IngestOutcome, Stage},
        health::Level,
        search::Hit,
    },
    services::{health, pipeline::Pipeline},
    util::{clean_path, collect_files, score_bar, short, trunc},
};
use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};
use console::{style, Term};
use dialoguer::{theme::ColorfulTheme, BasicHistory, Completion, Confirm, FuzzySelect, Input, MultiSelect, Select};
use indicatif::{HumanDuration, MultiProgress, ProgressBar, ProgressStyle};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use termimad::MadSkin;

pub struct ChatOpts {
    pub limit: usize,
    pub alpha: f32,
    pub source: Option<String>,
    /// Collections to search; empty means all
    pub collections: Vec<String>,
}

// ───────────────────────────── small helpers ─────────────────────────────

fn theme() -> ColorfulTheme {
    ColorfulTheme::default()
}

fn spinner(msg: &str) -> ProgressBar {
    let sp = ProgressBar::new_spinner();
    sp.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg} {elapsed:.dim}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏", "✔"]),
    );
    sp.set_message(msg.to_string());
    sp.enable_steady_tick(Duration::from_millis(90));
    sp
}

/// Wrap citation markers like [1] in bold so the markdown renderer highlights them.
fn highlight_citations(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == '[' {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && j < b.len() && b[j] == ']' {
                out.push_str("**");
                out.extend(&b[i..=j]);
                out.push_str("**");
                i = j + 1;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

fn skin() -> MadSkin {
    let mut skin = MadSkin::default();
    skin.bold.set_fg(termimad::crossterm::style::Color::Yellow);
    skin
}

/// Tab-completion for filesystem paths in the ingest prompt.
struct PathCompletion;

impl Completion for PathCompletion {
    fn get(&self, input: &str) -> Option<String> {
        crate::util::complete_path(input)
    }
}

fn banner(p: &Pipeline) {
    let _ = Term::stdout().clear_screen();
    println!("{}", style("  RAG PIPELINE  ").bold().white().on_blue());
    println!(
        "{}\n",
        style(format!("docling → {} → Weaviate @ {}", p.cfg.embed_model, p.cfg.weaviate_url)).dim()
    );
}

fn pause() {
    println!("\n{}", style("Press any key to return to the menu…").dim());
    let _ = Term::stdout().read_key();
}

fn new_table(header: &[&str]) -> Table {
    let mut t = Table::new();
    t.load_style(UTF8_FULL.with_rounded_corners())
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(header.iter().map(|h| Cell::new(h).fg(Color::Cyan)).collect::<Vec<_>>());
    t
}

fn hits_table(hits: &[Hit]) -> Table {
    let mut t = new_table(&["#", "Score", "Document", "Section", "Preview"]);
    for (i, h) in hits.iter().enumerate() {
        t.add_row(vec![
            Cell::new(i + 1).fg(Color::Yellow),
            Cell::new(score_bar(h.score)).fg(Color::Green),
            Cell::new(short(&h.source)),
            Cell::new(trunc(&h.headings.join(" > "), 40)),
            Cell::new(trunc(&h.text, 110)),
        ]);
    }
    t
}

// ───────────────────────────── main menu ─────────────────────────────

pub async fn run_menu(ctx: &AppContext) -> Result<()> {
    let p = &ctx.pipeline;
    let items = [
        "Chat with your documents",
        "Search your documents",
        "Ingest files or a folder",
        "Browse and manage documents",
        "System status",
        "Quit",
    ];
    loop {
        banner(p);
        let choice = Select::with_theme(&theme())
            .with_prompt("What would you like to do?  (↑/↓ move, Enter select, Esc quit)")
            .items(&items)
            .default(0)
            .interact_opt()?;
        let result = match choice {
            Some(0) => {
                chat_session(ctx, ChatOpts { limit: p.cfg.chat_top_k, alpha: 0.5, source: None, collections: vec![] }).await
            }
            Some(1) => search_flow(ctx).await,
            Some(2) => ingest_flow(p).await,
            Some(3) => browse_flow(ctx).await,
            Some(4) => status(ctx).await,
            _ => break,
        };
        match result {
            Err(e) => {
                println!("\n{} {e:#}", style("Error:").red().bold());
                pause();
            }
            Ok(()) if choice != Some(0) => pause(),
            Ok(()) => {}
        }
    }
    let _ = Term::stdout().show_cursor();
    println!("Goodbye.");
    Ok(())
}

// ───────────────────────────── ingest ─────────────────────────────

async fn ingest_flow(p: &Pipeline) -> Result<()> {
    let th = theme();
    let default = p.cfg.default_collection.clone();
    let typed: String = Input::with_theme(&th)
        .with_prompt("Collection to save into (Enter for the default)")
        .default(default.clone())
        .interact_text()?;
    let collection = sanitize_collection(&typed, &default);
    println!("{}", style(format!("Saving into collection “{collection}”")).dim());
    let raw: String = Input::with_theme(&th)
        .with_prompt("Path to a file or folder (Tab completes, drag & drop works)")
        .completion_with(&PathCompletion)
        .validate_with(|s: &String| -> std::result::Result<(), String> {
            if clean_path(s).exists() {
                Ok(())
            } else {
                Err("that path does not exist".into())
            }
        })
        .interact_text()?;
    let path = clean_path(&raw);
    let mut files = collect_files(&path);
    if files.is_empty() {
        println!("{}", style("No files found there.").yellow());
        return Ok(());
    }

    if files.len() > 1 {
        let mut exts: std::collections::BTreeMap<String, usize> = Default::default();
        for f in &files {
            let e = f.extension().and_then(|e| e.to_str()).unwrap_or("(none)").to_lowercase();
            *exts.entry(e).or_default() += 1;
        }
        let summary = exts.iter().map(|(e, n)| format!("{n} .{e}")).collect::<Vec<_>>().join(", ");
        println!("Found {} files: {}", style(files.len()).bold(), style(summary).dim());
        let pick = Select::with_theme(&th)
            .with_prompt("How do you want to proceed?")
            .items(&[format!("Ingest all {} files", files.len()), "Choose which files…".to_string(), "Cancel".to_string()])
            .default(0)
            .interact_opt()?;
        match pick {
            Some(0) => {}
            Some(1) => {
                let names: Vec<String> = files
                    .iter()
                    .map(|f| f.strip_prefix(&path).unwrap_or(f).display().to_string())
                    .collect();
                let chosen = MultiSelect::with_theme(&th)
                    .with_prompt("Space toggles, a toggles all, Enter confirms")
                    .items(&names)
                    .defaults(&vec![true; names.len()])
                    .max_length(15)
                    .interact_opt()?
                    .unwrap_or_default();
                files = chosen.into_iter().map(|i| files[i].clone()).collect();
                if files.is_empty() {
                    println!("Nothing selected.");
                    return Ok(());
                }
            }
            _ => return Ok(()),
        }
    }
    ingest_paths(p, files, &collection).await
}

fn stage_text(name: &str, s: &Stage) -> String {
    match s {
        Stage::Checking => format!("{name}  checking for changes"),
        Stage::Extracting => format!("{name}  extracting and chunking (docling)"),
        Stage::Embedding { done, total } => {
            let width = 20;
            let filled = if *total == 0 { 0 } else { done * width / total };
            format!(
                "{name}  embedding [{}{}] {done}/{total} chunks",
                "█".repeat(filled),
                "░".repeat(width - filled)
            )
        }
        Stage::Indexing => format!("{name}  writing to Weaviate"),
    }
}

/// Ingest files one after another with an overall progress bar and a live stage line,
/// then print a summary table. Used by the menu and by `rag-pipeline ingest`.
pub async fn ingest_paths(p: &Pipeline, files: Vec<PathBuf>, collection: &str) -> Result<()> {
    p.store.ensure_schema().await?;
    let mp = MultiProgress::new();
    let overall = mp.add(ProgressBar::new(files.len() as u64));
    overall.set_style(
        ProgressStyle::with_template("{prefix:.bold} [{bar:30.cyan/blue}] {pos}/{len} files  {elapsed_precise}  ETA {eta}")
            .unwrap()
            .progress_chars("█▓░"),
    );
    overall.set_prefix(format!("Ingesting → {collection}"));
    let stage = mp.add(ProgressBar::new_spinner());
    stage.set_style(ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    stage.enable_steady_tick(Duration::from_millis(90));

    let mut rows: Vec<(String, std::result::Result<IngestOutcome, String>, Duration)> = Vec::new();
    for f in &files {
        let name = f.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let source = std::fs::canonicalize(f).unwrap_or_else(|_| f.clone()).to_string_lossy().to_string();
        stage.set_message(stage_text(&name, &Stage::Checking));
        let started = Instant::now();
        let res = p
            .ingest_file_progress(f, &source, collection, |s| stage.set_message(stage_text(&name, &s)))
            .await
            .map_err(|e| format!("{e:#}"));
        rows.push((name, res, started.elapsed()));
        overall.inc(1);
    }
    stage.finish_and_clear();
    overall.finish_and_clear();

    let mut t = new_table(&["File", "Status", "Chunks", "Time"]);
    let (mut ok, mut skipped, mut failed) = (0, 0, 0);
    for (name, res, took) in &rows {
        let (status, color, chunks) = match res {
            Ok(IngestOutcome::Indexed { chunks, .. }) => {
                ok += 1;
                ("indexed", Color::Green, chunks.to_string())
            }
            Ok(IngestOutcome::Skipped { .. }) => {
                skipped += 1;
                ("unchanged, skipped", Color::Yellow, "-".into())
            }
            Err(_) => {
                failed += 1;
                ("failed", Color::Red, "-".into())
            }
        };
        t.add_row(vec![
            Cell::new(name),
            Cell::new(status).fg(color),
            Cell::new(chunks),
            Cell::new(format!("{:.1}s", took.as_secs_f32())),
        ]);
    }
    println!("{t}");
    println!(
        "{} indexed, {} skipped, {} failed",
        style(ok).green().bold(),
        style(skipped).yellow().bold(),
        style(failed).red().bold()
    );
    for (name, res, _) in &rows {
        if let Err(e) = res {
            println!("  {} {}: {}", style("✖").red(), style(name).bold(), trunc(e, 300));
        }
    }
    Ok(())
}

// ───────────────────────────── search ─────────────────────────────

async fn search_flow(ctx: &AppContext) -> Result<()> {
    let th = theme();
    let mode = Select::with_theme(&th)
        .with_prompt("Search mode")
        .items(&[
            "Hybrid: keywords + meaning (recommended)",
            "Keyword-leaning (names, codes, exact terms)",
            "Meaning-leaning (paraphrased questions)",
        ])
        .default(0)
        .interact_opt()?;
    let alpha = match mode {
        Some(0) => 0.5,
        Some(1) => 0.15,
        Some(2) => 0.85,
        _ => return Ok(()),
    };
    let mut hist = BasicHistory::new().max_entries(50).no_duplicates(true);
    loop {
        let q: String = Input::with_theme(&th)
            .with_prompt("Search query (empty to go back)")
            .allow_empty(true)
            .history_with(&mut hist)
            .interact_text()?;
        if q.trim().is_empty() {
            return Ok(());
        }
        let sp = spinner("Embedding the query and searching Weaviate…");
        let res = ctx.rag.search(q.trim(), &{ let mut o = ctx.cfg.retrieval_defaults(); o.top_k = 8; o.alpha = alpha; o.sanitized() }).await;
        sp.finish_and_clear();
        let hits = res?;
        if hits.is_empty() {
            println!("{}", style("No results.").yellow());
            continue;
        }
        println!("{}", hits_table(&hits));
        let sk = skin();
        loop {
            let mut items: Vec<String> = hits
                .iter()
                .enumerate()
                .map(|(i, h)| format!("Read result {} — {}", i + 1, trunc(&format!("{} {}", short(&h.source), h.headings.join(" > ")), 60)))
                .collect();
            items.push("New search".into());
            let pick = Select::with_theme(&th).with_prompt("Open a result").items(&items).default(0).interact_opt()?;
            match pick {
                Some(i) if i < hits.len() => {
                    let h = &hits[i];
                    println!("\n{}", style(format!("{} — chunk {}", h.source, h.chunk_index)).green().bold());
                    if !h.headings.is_empty() {
                        println!("{}", style(h.headings.join(" > ")).dim());
                    }
                    println!();
                    sk.print_text(&h.text);
                }
                _ => break,
            }
        }
    }
}

// ───────────────────────────── chat ─────────────────────────────

struct Session {
    history: Vec<ChatTurn>,
    limit: usize,
    alpha: f32,
    source: Option<String>,
    collections: Vec<String>,
    sources: Vec<SourceRef>,
}

const CHAT_HELP: &str = "  /source        pick a document to search within (or all)\n  /collection a,b  search only these collections (no argument = all)\n  /show N        read source N in full\n  /alpha 0-1     0 = keywords only, 1 = meaning only\n  /limit N       passages retrieved per question\n  /clear         forget this conversation\n  /exit          back to the menu";

fn chat_header(s: &Session) {
    let scope = match (&s.source, s.collections.is_empty()) {
        (Some(f), _) => format!("document: {}", short(f)),
        (None, false) => format!("collections: {}", s.collections.join(", ")),
        (None, true) => "all collections".to_string(),
    };
    println!(
        "{}  {}",
        style("Chat").bold().cyan(),
        style(format!("scope: {scope} · top {} · alpha {:.2} · /help for commands", s.limit, s.alpha)).dim()
    );
}

fn print_source(s: &SourceRef, sk: &MadSkin) {
    println!("\n{}", style(format!("{} — chunk {}", s.source, s.chunk_index)).green().bold());
    if !s.headings.is_empty() {
        println!("{}", style(s.headings.join(" > ")).dim());
    }
    println!();
    sk.print_text(&s.text);
}

pub async fn chat_session(ctx: &AppContext, opts: ChatOpts) -> Result<()> {
    let th = theme();
    let sk = skin();
    let mut s = Session { history: vec![], limit: opts.limit, alpha: opts.alpha, source: opts.source, collections: opts.collections, sources: vec![] };
    let mut hist = BasicHistory::new().max_entries(100).no_duplicates(true);
    println!();
    chat_header(&s);
    println!();
    loop {
        let line: String = match Input::<String>::with_theme(&th).with_prompt("you").allow_empty(true).history_with(&mut hist).interact_text() {
            Ok(l) => l.trim().to_string(),
            Err(_) => break, // Ctrl-C
        };
        if line.is_empty() {
            continue;
        }
        if let Some(cmd) = line.strip_prefix('/') {
            let mut parts = cmd.splitn(2, ' ');
            let name = parts.next().unwrap_or("");
            let arg = parts.next().unwrap_or("").trim();
            match name {
                "exit" | "quit" | "menu" => break,
                "help" => println!("{CHAT_HELP}"),
                "clear" => {
                    s.history.clear();
                    s.sources.clear();
                    println!("{}", style("Conversation cleared.").dim());
                }
                "alpha" => match arg.parse::<f32>() {
                    Ok(a) => {
                        s.alpha = a.clamp(0.0, 1.0);
                        chat_header(&s);
                    }
                    Err(_) => println!("Usage: /alpha 0.5"),
                },
                "limit" => match arg.parse::<usize>() {
                    Ok(n) => {
                        s.limit = n.clamp(1, 20);
                        chat_header(&s);
                    }
                    Err(_) => println!("Usage: /limit 4"),
                },
                "show" => match arg.parse::<usize>().ok().and_then(|n| n.checked_sub(1)).and_then(|i| s.sources.get(i)) {
                    Some(src) => print_source(src, &sk),
                    None => println!("Usage: /show N (a source number from the last answer)"),
                },
                "collection" => {
                    s.collections = arg.split(',').map(|n| sanitize_collection(n, "")).filter(|n| !n.is_empty()).collect();
                    s.source = None;
                    chat_header(&s);
                }
                "source" => {
                    let docs = ctx.store.list_documents(None).await?;
                    if docs.is_empty() {
                        println!("{}", style("No documents ingested yet.").yellow());
                    } else {
                        let mut items = vec!["All documents".to_string()];
                        items.extend(docs.iter().map(|d| format!("{} / {}  ({} chunks)", d.collection, short(&d.source), d.chunks)));
                        let sel = FuzzySelect::with_theme(&th)
                            .with_prompt("Search within (type to filter, Esc to cancel)")
                            .items(&items)
                            .default(0)
                            .interact_opt()?;
                        match sel {
                            Some(0) => s.source = None,
                            Some(i) => {
                                s.source = Some(docs[i - 1].source.clone());
                                s.collections = vec![docs[i - 1].collection.clone()];
                            }
                            None => {}
                        }
                        chat_header(&s);
                    }
                }
                _ => println!("Unknown command.\n{CHAT_HELP}"),
            }
            continue;
        }
        if let Err(e) = ask(ctx, &mut s, &line, &sk).await {
            println!("\n{} {e:#}\n", style("Error:").red().bold());
        }
    }
    Ok(())
}

async fn ask(ctx: &AppContext, s: &mut Session, q: &str, sk: &MadSkin) -> Result<()> {
    let started = Instant::now();
    let sp = spinner("Searching your documents…");
    let sp2 = sp.clone();
    let memory = ChatMemory { summary: None, turns: s.history.clone() };
    let mut opts = ctx.cfg.retrieval_defaults();
    opts.top_k = s.limit;
    opts.alpha = s.alpha;
    opts.source = s.source.clone();
    opts.collections = s.collections.clone();
    let res = ctx
        .rag
        .answer(q, &memory, &opts.sanitized(), ctx.cfg.chat_rewrite, move |stage| match stage {
            RagStage::Rewriting => sp2.set_message("Understanding your follow-up…"),
            RagStage::Searching => sp2.set_message("Searching your documents…"),
            RagStage::Reranking { candidates } => sp2.set_message(format!("Reranking {candidates} candidates…")),
            RagStage::Generating { passages } => sp2.set_message(format!("Reading {passages} passages and writing an answer…")),
        })
        .await;
    sp.finish_and_clear();
    let ans = res?;
    if ans.sources.is_empty() {
        println!("\n{}\n", style(&ans.answer).yellow());
        return Ok(());
    }

    println!("\n{}", style("assistant").cyan().bold());
    sk.print_text(&highlight_citations(&ans.answer));
    println!("{}", style("Sources").bold());
    for src in &ans.sources {
        let section = if src.headings.is_empty() { String::new() } else { format!(" › {}", trunc(&src.headings.join(" › "), 50)) };
        println!(
            "  {} {}{}  {}",
            style(format!("[{}]", src.n)).yellow().bold(),
            style(short(&src.source)).green(),
            style(section).dim(),
            style(score_bar(src.score)).dim()
        );
    }
    println!(
        "{}\n",
        style(format!("answered in {} · /show N reads a source · /help", HumanDuration(started.elapsed()))).dim()
    );

    s.history.push(ChatTurn { role: "user".into(), content: q.to_string() });
    s.history.push(ChatTurn { role: "assistant".into(), content: ans.answer });
    s.sources = ans.sources;
    Ok(())
}

// ───────────────────────────── browse / manage ─────────────────────────────

async fn browse_flow(ctx: &AppContext) -> Result<()> {
    let p = &ctx.pipeline;
    let th = theme();
    loop {
        let sp = spinner("Loading documents from Weaviate…");
        let docs = p.store.list_documents(None).await;
        sp.finish_and_clear();
        let docs = docs?;
        if docs.is_empty() {
            println!("{}", style("Nothing ingested yet. Use “Ingest files or a folder”.").yellow());
            return Ok(());
        }
        let total: usize = docs.iter().map(|d| d.chunks).sum();
        let mut t = new_table(&["#", "Collection", "Document", "Chunks"]);
        for (i, d) in docs.iter().enumerate() {
            t.add_row(vec![Cell::new(i + 1).fg(Color::Yellow), Cell::new(&d.collection).fg(Color::Magenta), Cell::new(short(&d.source)), Cell::new(d.chunks)]);
        }
        println!("{t}");
        println!("{} documents, {} chunks\n", style(docs.len()).bold(), style(total).bold());

        let mut items: Vec<String> = docs.iter().map(|d| format!("{} / {}  ({} chunks)", d.collection, d.source, d.chunks)).collect();
        items.push("Back to menu".into());
        let pick = FuzzySelect::with_theme(&th)
            .with_prompt("Select a document (type to filter, Esc to go back)")
            .items(&items)
            .default(0)
            .interact_opt()?;
        let Some(i) = pick.filter(|i| *i < docs.len()) else { return Ok(()) };
        let (collection, source) = (docs[i].collection.clone(), docs[i].source.clone());

        let action = Select::with_theme(&th)
            .with_prompt(format!("{}", short(&source)))
            .items(&["Chat about this document", "Delete this document from the index", "Back"])
            .default(0)
            .interact_opt()?;
        match action {
            Some(0) => {
                return chat_session(ctx, ChatOpts { limit: p.cfg.chat_top_k, alpha: 0.5, source: Some(source), collections: vec![collection] }).await;
            }
            Some(1) => {
                let sure = Confirm::with_theme(&th)
                    .with_prompt(format!("Delete all chunks of “{}” from Weaviate? (the original file is not touched)", short(&source)))
                    .default(false)
                    .interact()?;
                if sure {
                    let sp = spinner("Deleting…");
                    let r = p.store.delete_source(&collection, &source).await;
                    sp.finish_and_clear();
                    r?;
                    println!("{} deleted {}", style("✔").green(), short(&source));
                }
            }
            _ => {}
        }
    }
}

// ───────────────────────────── status ─────────────────────────────

pub async fn status(ctx: &AppContext) -> Result<()> {
    println!("{}\n", style("System status").bold().cyan());
    let sp = spinner("Checking services…");
    let checks = health::run_checks(ctx).await;
    sp.finish_and_clear();
    for c in &checks {
        let mark = match c.level {
            Level::Ok => style("✔").green().bold(),
            Level::Warn => style("!").yellow().bold(),
            Level::Fail => style("✖").red().bold(),
        };
        println!("  {mark} {:<18} {}", style(c.name).bold(), style(&c.detail).dim());
    }
    Ok(())
}
