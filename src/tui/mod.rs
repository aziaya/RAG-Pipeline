//! Split-pane terminal UI: the active screen on one side, live tracing and metrics on the other.

mod app;
mod input;
mod markdown;
mod obs;
mod screens;
mod theme;
mod ui;
mod widgets;

use crate::{context::AppContext, observability::events::TraceEvent};
use anyhow::Result;
use app::{App, AppMsg};
use crossterm::event::EventStream;
use futures_util::StreamExt;
use ratatui::DefaultTerminal;
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

pub async fn run(ctx: AppContext, mut trace_rx: UnboundedReceiver<TraceEvent>) -> Result<()> {
    let (tx, mut rx) = unbounded_channel::<AppMsg>();
    let mut app = App::new(ctx, tx);
    // `init` enters the alternate screen + raw mode and installs a panic hook that restores the terminal.
    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut app, &mut rx, &mut trace_rx).await;
    ratatui::restore();
    result
}

async fn event_loop(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    rx: &mut UnboundedReceiver<AppMsg>,
    trace_rx: &mut UnboundedReceiver<TraceEvent>,
) -> Result<()> {
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        terminal.draw(|f| ui::draw(f, app))?;
        tokio::select! {
            ev = events.next() => match ev {
                Some(Ok(ev)) => app.on_event(ev),
                Some(Err(e)) => return Err(e.into()),
                None => break,
            },
            Some(msg) = rx.recv() => {
                app.on_msg(msg);
                while let Ok(m) = rx.try_recv() {
                    app.on_msg(m);
                }
            }
            Some(ev) = trace_rx.recv() => {
                app.obs.push(ev);
                while let Ok(e) = trace_rx.try_recv() {
                    app.obs.push(e);
                }
            }
            _ = tick.tick() => app.on_tick(),
        }
        if app.should_quit {
            break;
        }
    }
    Ok(())
}
