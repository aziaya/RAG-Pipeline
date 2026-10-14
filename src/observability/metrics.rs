//! Process-wide counters, latency history and the registry of in-flight operations.
//! Services bump counters directly; the tracing layer records span latencies and
//! maintains the in-flight list.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Mutex, OnceLock,
    },
    time::Instant,
};

const HISTORY: usize = 40;

#[derive(Default)]
pub struct Metrics {
    pub files_indexed: AtomicU64,
    pub files_skipped: AtomicU64,
    pub files_failed: AtomicU64,
    pub chunks_indexed: AtomicU64,
    pub embeddings: AtomicU64,
    pub searches: AtomicU64,
    pub chats: AtomicU64,
    pub errors: AtomicU64,
    pub llm_tokens: AtomicU64,
    llm_rate_milli: AtomicU64,
    latency: Mutex<HashMap<&'static str, VecDeque<u64>>>,
    active: Mutex<HashMap<u64, ActiveOp>>,
}

#[derive(Clone)]
pub struct ActiveOp {
    pub name: &'static str,
    pub detail: String,
    pub started: Instant,
}

/// Point-in-time copy for rendering.
pub struct Snapshot {
    pub files_indexed: u64,
    pub files_skipped: u64,
    pub files_failed: u64,
    pub chunks_indexed: u64,
    pub embeddings: u64,
    pub searches: u64,
    pub chats: u64,
    pub errors: u64,
    pub llm_tokens: u64,
    pub llm_rate: f64,
    pub latency: Vec<LatencyRow>,
}

pub struct LatencyRow {
    pub name: &'static str,
    pub last: u64,
    pub avg: u64,
    pub p95: u64,
    pub history: Vec<u64>,
}

/// Order in which operations are shown.
const OPS: [&str; 10] = [
    "extract",
    "embed_batch",
    "weaviate_query",
    "weaviate_write",
    "search",
    "rerank",
    "rewrite_query",
    "first_token",
    "llm_generate",
    "summarize_memory",
];

impl Metrics {
    pub fn inc(&self, counter: &AtomicU64, by: u64) {
        counter.fetch_add(by, Relaxed);
    }

    pub fn set_llm_rate(&self, tokens_per_sec: f64) {
        self.llm_rate_milli.store((tokens_per_sec * 1000.0) as u64, Relaxed);
    }

    pub fn record_latency(&self, name: &'static str, ms: u64) {
        let mut map = self.latency.lock().unwrap();
        let q = map.entry(name).or_default();
        if q.len() == HISTORY {
            q.pop_front();
        }
        q.push_back(ms);
    }

    pub fn active_start(&self, id: u64, name: &'static str, detail: String) {
        self.active.lock().unwrap().insert(id, ActiveOp { name, detail, started: Instant::now() });
    }

    pub fn active_end(&self, id: u64) {
        self.active.lock().unwrap().remove(&id);
    }

    pub fn active_snapshot(&self) -> Vec<ActiveOp> {
        let mut v: Vec<ActiveOp> = self.active.lock().unwrap().values().cloned().collect();
        v.sort_by_key(|a| a.started);
        v
    }

    pub fn snapshot(&self) -> Snapshot {
        let map = self.latency.lock().unwrap();
        let mut latency = Vec::new();
        for name in OPS {
            if let Some(q) = map.get(name).filter(|q| !q.is_empty()) {
                let mut sorted: Vec<u64> = q.iter().copied().collect();
                sorted.sort_unstable();
                let p95 = sorted[((sorted.len() as f64 * 0.95).ceil() as usize).saturating_sub(1).min(sorted.len() - 1)];
                latency.push(LatencyRow {
                    name,
                    last: *q.back().unwrap(),
                    avg: q.iter().sum::<u64>() / q.len() as u64,
                    p95,
                    history: q.iter().copied().collect(),
                });
            }
        }
        Snapshot {
            files_indexed: self.files_indexed.load(Relaxed),
            files_skipped: self.files_skipped.load(Relaxed),
            files_failed: self.files_failed.load(Relaxed),
            chunks_indexed: self.chunks_indexed.load(Relaxed),
            embeddings: self.embeddings.load(Relaxed),
            searches: self.searches.load(Relaxed),
            chats: self.chats.load(Relaxed),
            errors: self.errors.load(Relaxed),
            llm_tokens: self.llm_tokens.load(Relaxed),
            llm_rate: self.llm_rate_milli.load(Relaxed) as f64 / 1000.0,
            latency,
        }
    }
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

/// The process-wide metrics registry.
pub fn metrics() -> &'static Metrics {
    METRICS.get_or_init(Metrics::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_summary() {
        let m = Metrics::default();
        for ms in [10, 20, 30, 40, 100] {
            m.record_latency("search", ms);
        }
        let snap = m.snapshot();
        let row = snap.latency.iter().find(|r| r.name == "search").unwrap();
        assert_eq!(row.last, 100);
        assert_eq!(row.avg, 40);
        assert_eq!(row.p95, 100);
    }

    #[test]
    fn active_ops_track_start_and_end() {
        let m = Metrics::default();
        m.active_start(1, "embed_batch", "inputs=8".into());
        assert_eq!(m.active_snapshot().len(), 1);
        m.active_end(1);
        assert!(m.active_snapshot().is_empty());
    }
}
