//! A `tracing` layer that forwards events and span lifecycles to the terminal UI.

use super::{
    events::{EventKind, TraceEvent},
    metrics::metrics,
};
use chrono::Local;
use std::{fmt::Debug, time::Instant};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{
    field::{Field, Visit},
    span::{Attributes, Id},
    Event, Level, Subscriber,
};
use tracing_subscriber::{layer::Context, registry::LookupSpan, Layer};

pub struct UiLayer {
    tx: UnboundedSender<TraceEvent>,
}

impl UiLayer {
    pub fn new(tx: UnboundedSender<TraceEvent>) -> Self {
        Self { tx }
    }
}

#[derive(Default)]
struct Fields {
    message: String,
    fields: Vec<(String, String)>,
}

impl Fields {
    fn detail(&self) -> String {
        self.fields.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
    }
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields.push((field.name().to_string(), value.to_string()));
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            self.fields.push((field.name().to_string(), format!("{value:?}")));
        }
    }
}

/// Stored in span extensions.
struct SpanInfo {
    started: Instant,
    detail: String,
}

impl<S> Layer<S> for UiLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut f = Fields::default();
        attrs.record(&mut f);
        let detail = f.detail();
        if let Some(span) = ctx.span(id) {
            metrics().active_start(id.into_u64(), span.name(), detail.clone());
            span.extensions_mut().insert(SpanInfo { started: Instant::now(), detail });
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut f = Fields::default();
        event.record(&mut f);
        let span = ctx.event_scope(event).and_then(|mut scope| {
            scope.next().map(|s| {
                let detail = s.extensions().get::<SpanInfo>().map(|i| i.detail.clone()).unwrap_or_default();
                if detail.is_empty() {
                    s.name().to_string()
                } else {
                    format!("{}{{{}}}", s.name(), detail)
                }
            })
        });
        let _ = self.tx.send(TraceEvent {
            ts: Local::now(),
            level: *event.metadata().level(),
            span,
            message: f.message,
            fields: f.fields,
            kind: EventKind::Log,
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        metrics().active_end(id.into_u64());
        let Some(span) = ctx.span(&id) else { return };
        let Some((elapsed_ms, detail)) = span.extensions().get::<SpanInfo>().map(|i| (i.started.elapsed().as_millis() as u64, i.detail.clone())) else {
            return;
        };
        metrics().record_latency(span.name(), elapsed_ms);
        let _ = self.tx.send(TraceEvent {
            ts: Local::now(),
            level: Level::INFO,
            span: Some(span.name().to_string()),
            message: span.name().to_string(),
            fields: if detail.is_empty() { vec![] } else { vec![("".to_string(), detail)] },
            kind: EventKind::SpanClosed { elapsed_ms },
        });
    }
}
