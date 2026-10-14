//! Conversations saved as JSON files so they survive restarts.

use crate::models::session::{Session, SessionMeta};
use anyhow::{Context, Result};
use chrono::Local;
use std::path::PathBuf;

#[derive(Clone)]
pub struct SessionStore {
    dir: PathBuf,
}

impl SessionStore {
    pub fn new(data_dir: &std::path::Path) -> Self {
        Self { dir: data_dir.join("sessions") }
    }

    fn path(&self, id: &str) -> PathBuf {
        // ids are generated here; refuse anything that could escape the directory.
        let safe: String = id.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_').collect();
        self.dir.join(format!("{safe}.json"))
    }

    pub fn new_session(&self) -> Session {
        let now = Local::now();
        Session {
            id: now.format("%Y%m%d-%H%M%S").to_string(),
            title: "New conversation".into(),
            created: now.to_rfc3339(),
            updated: now.to_rfc3339(),
            memory: Default::default(),
            messages: vec![],
        }
    }

    pub fn save(&self, s: &mut Session) -> Result<()> {
        std::fs::create_dir_all(&self.dir).with_context(|| format!("create {}", self.dir.display()))?;
        s.updated = Local::now().to_rfc3339();
        if let Some(first) = s.messages.iter().find(|m| m.role == "user") {
            s.title = crate::util::trunc(&first.text, 60);
        }
        let tmp = self.path(&format!("{}.tmp", s.id));
        std::fs::write(&tmp, serde_json::to_vec_pretty(s)?)?;
        std::fs::rename(&tmp, self.path(&s.id))?; // atomic replace: a crash never leaves a half-written file
        Ok(())
    }

    pub fn load(&self, id: &str) -> Result<Session> {
        let p = self.path(id);
        serde_json::from_slice(&std::fs::read(&p).with_context(|| format!("read {}", p.display()))?)
            .with_context(|| format!("parse {}", p.display()))
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let p = self.path(id);
        if p.exists() {
            std::fs::remove_file(&p)?;
        }
        Ok(())
    }

    /// Saved conversations, newest first. Unreadable files are skipped.
    pub fn list(&self) -> Vec<SessionMeta> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return vec![] };
        let mut out: Vec<SessionMeta> = rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
            .filter_map(|e| {
                let s: Session = serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok()?;
                let exchanges = s.messages.iter().filter(|m| m.role == "user").count();
                Some(SessionMeta { id: s.id, title: s.title, updated: s.updated, exchanges })
            })
            .collect();
        out.sort_by(|a, b| b.updated.cmp(&a.updated));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::session::StoredMessage;

    fn message(role: &str, text: &str) -> StoredMessage {
        StoredMessage { role: role.into(), text: text.into(), sources: vec![], took_secs: None, searched_for: None }
    }

    #[test]
    fn save_load_list_delete_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());
        assert!(store.list().is_empty());

        let mut s = store.new_session();
        s.messages.push(message("user", "what is kokoro"));
        s.messages.push(message("assistant", "a TTS model"));
        s.memory.push_exchange("what is kokoro", "a TTS model");
        store.save(&mut s).unwrap();

        let loaded = store.load(&s.id).unwrap();
        assert_eq!(loaded.title, "what is kokoro");
        assert_eq!(loaded.memory.turns.len(), 2);
        assert_eq!(loaded.messages.len(), 2);

        let metas = store.list();
        assert_eq!((metas.len(), metas[0].exchanges), (1, 1));

        store.delete(&s.id).unwrap();
        assert!(store.list().is_empty());
    }

    #[test]
    fn ids_cannot_escape_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());
        let p = store.path("../../etc/passwd");
        assert!(p.starts_with(dir.path().join("sessions")));
        assert!(!p.to_string_lossy().contains(".."));
    }
}
