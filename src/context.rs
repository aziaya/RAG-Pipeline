//! Wires the services together once; cheap to clone into tasks and handlers.

use crate::{
    config::Config,
    services::{embedder::Embedder, llm::LlmClient, pipeline::Pipeline, rag::Rag, vector_store::VectorStore},
};
use anyhow::Result;
use std::{sync::Arc, time::Duration};

#[derive(Clone)]
pub struct AppContext {
    pub cfg: Arc<Config>,
    pub store: VectorStore,
    pub embedder: Embedder,
    pub llm: LlmClient,
    pub pipeline: Pipeline,
    pub rag: Rag,
}

impl AppContext {
    pub fn build(cfg: Config) -> Result<Self> {
        let cfg = Arc::new(cfg);
        let http = reqwest::Client::builder().timeout(Duration::from_secs(120)).build()?;
        let embedder = Embedder::new(&cfg, http.clone());
        let store = VectorStore::new(&cfg, http.clone());
        let llm = LlmClient::new(&cfg, http);
        let pipeline = Pipeline::new(cfg.clone(), embedder.clone(), store.clone());
        let rag = Rag::new(embedder.clone(), store.clone(), llm.clone());
        Ok(Self { cfg, store, embedder, llm, pipeline, rag })
    }
}
