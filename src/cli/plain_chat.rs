//! Plain-text chat: one-shot (`chat "question"`) or a simple line-by-line loop.
//! Script- and pipe-friendly; no colors, no screen control.

use crate::{context::AppContext, models::chat::ChatTurn};
use anyhow::Result;
use std::io::Write;
use tokio::io::{AsyncBufReadExt, BufReader};

pub struct Opts {
    pub limit: usize,
    pub alpha: f32,
    pub source: Option<String>,
}

const HELP: &str = "Commands:\n  /source <file>  only search this ingested file\n  /source         clear the file filter\n  /clear          forget the conversation so far\n  /help           show this help\n  /exit           quit (Ctrl-D also works)";

async fn ask(ctx: &AppContext, question: &str, history: &[ChatTurn], o: &Opts) -> Result<String> {
    eprintln!("(searching and thinking, this can take a while on CPU...)");
    let ans = ctx.rag.answer(question, history, o.limit, o.alpha, o.source.as_deref(), |_| {}).await?;
    println!("\n{}\n", ans.answer);
    if !ans.sources.is_empty() {
        println!("Sources:");
        for s in &ans.sources {
            let section = if s.headings.is_empty() { String::new() } else { format!(" > {}", s.headings.join(" > ")) };
            let score = s.score.map(|v| format!("  (score {v:.2})")).unwrap_or_default();
            println!("  [{}] {}{}{}", s.n, s.source, section, score);
        }
        println!();
    }
    Ok(ans.answer)
}

pub async fn run(ctx: &AppContext, question: Option<String>, mut o: Opts) -> Result<()> {
    if let Some(q) = question {
        ask(ctx, &q, &[], &o).await?;
        return Ok(());
    }

    println!("Chat with your documents. Type /help for commands, /exit to quit.\n");
    let mut history: Vec<ChatTurn> = Vec::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        print!("you> ");
        std::io::stdout().flush()?;
        let Some(line) = lines.next_line().await? else { break };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/exit" | "/quit" => break,
            "/help" => {
                println!("{HELP}");
                continue;
            }
            "/clear" => {
                history.clear();
                println!("(conversation cleared)");
                continue;
            }
            _ => {}
        }
        if let Some(rest) = line.strip_prefix("/source") {
            let s = rest.trim();
            o.source = if s.is_empty() { None } else { Some(s.to_string()) };
            match &o.source {
                Some(f) => println!("(searching only: {f})"),
                None => println!("(searching all documents)"),
            }
            continue;
        }
        if line.starts_with('/') {
            println!("Unknown command. {HELP}");
            continue;
        }
        match ask(ctx, line, &history, &o).await {
            Ok(answer) => {
                history.push(ChatTurn { role: "user".into(), content: line.to_string() });
                history.push(ChatTurn { role: "assistant".into(), content: answer });
            }
            Err(e) => eprintln!("error: {e:#}\n"),
        }
    }
    Ok(())
}
