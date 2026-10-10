//! Terminal chat: one-shot (`chat "question"`) or an interactive loop (`chat`).
//! Uses the same retrieval and chat code as POST /chat, without the HTTP server.

use crate::{api, chat::{ChatClient, ChatTurn}, pipeline::Pipeline};
use anyhow::Result;
use std::io::Write;
use tokio::io::{AsyncBufReadExt, BufReader};

pub struct Opts {
    pub limit: usize,
    pub alpha: f32,
    pub source: Option<String>,
}

const HELP: &str = "Commands:\n  /source <file>  only search this ingested file\n  /source         clear the file filter\n  /clear          forget the conversation so far\n  /help           show this help\n  /exit           quit (Ctrl-D also works)";

/// Retrieve, generate, print the answer and its sources. Returns the answer text.
async fn ask(p: &Pipeline, chat: &ChatClient, question: &str, history: &[ChatTurn], o: &Opts) -> Result<String> {
    eprintln!("(searching and thinking, this can take a while on CPU...)");
    let hits = api::run_search(p, question, o.limit, o.alpha, o.source.as_deref()).await?;
    if hits.is_empty() {
        let msg = "I couldn't find anything relevant in the indexed documents.";
        println!("\n{msg}\n");
        return Ok(msg.to_string());
    }
    let (answer, used) = chat.answer(question, history, &hits).await?;
    println!("\n{answer}\n");
    println!("Sources:");
    for (i, h) in hits.iter().take(used).enumerate() {
        let section = if h.headings.is_empty() { String::new() } else { format!(" > {}", h.headings.join(" > ")) };
        let score = h.score.map(|s| format!("  (score {s:.2})")).unwrap_or_default();
        println!("  [{}] {}{}{}", i + 1, h.source, section, score);
    }
    println!();
    Ok(answer)
}

pub async fn run(p: &Pipeline, chat: &ChatClient, question: Option<String>, mut o: Opts) -> Result<()> {
    if let Some(q) = question {
        ask(p, chat, &q, &[], &o).await?;
        return Ok(());
    }

    println!("Chat with your documents. Type /help for commands, /exit to quit.\n");
    let mut history: Vec<ChatTurn> = Vec::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        print!("you> ");
        std::io::stdout().flush()?;
        let Some(line) = lines.next_line().await? else { break }; // Ctrl-D
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/quit" {
            break;
        }
        if line == "/help" {
            println!("{HELP}");
            continue;
        }
        if line == "/clear" {
            history.clear();
            println!("(conversation cleared)");
            continue;
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

        match ask(p, chat, line, &history, &o).await {
            Ok(answer) => {
                history.push(ChatTurn { role: "user".into(), content: line.to_string() });
                history.push(ChatTurn { role: "assistant".into(), content: answer });
            }
            Err(e) => eprintln!("error: {e:#}\n"),
        }
    }
    Ok(())
}
