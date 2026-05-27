// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Session controller — Rust mirror of
//! `python/sglang/srt/session/session_controller.py`.
//!
//! A "session" lets the API caller continue from a previous request's
//! KV state.  The Python controller maintains a parent/child tree of
//! `SessionReqNode`s so that branching from a session offset works
//! cleanly.
//!
//! ## Scope of this cut
//!
//! Lightweight bookkeeping only — enough to:
//!   * Open a session id (uuid).
//!   * Look up the most recent rid for that session so a follow-up
//!     `TokenizedGenerateReqInput.session_params.id` can resume.
//!   * Close a session and abort any outstanding children.
//!
//! The full Python `SessionReqNode` tree (with `clear_children`,
//! `_str_helper`, etc.) is not ported.  Branching from a non-leaf
//! offset is rejected for now — once the scheduler grows full
//! prefix-cache integration the tree will follow.

use std::collections::HashMap;

/// One open session.  Holds the rid of the last admitted request in
/// this session so a follow-up can resume from its KV.
#[derive(Debug, Clone, Default)]
pub struct Session {
    pub session_id: String,
    pub last_rid: Option<String>,
    /// Server-side creation time as a Unix epoch millis (purely for
    /// diagnostics today).
    pub created_at_ms: i64,
}

#[derive(Debug, Default)]
pub struct SessionController {
    sessions: HashMap<String, Session>,
}

impl SessionController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new session.  Returns its id (a synthetic 16-byte
    /// uuid-like string — matches Python's `uuid.uuid4().hex`).
    pub fn open(&mut self) -> String {
        let id = pseudo_uuid_hex();
        let session = Session {
            session_id: id.clone(),
            last_rid: None,
            created_at_ms: now_ms(),
        };
        self.sessions.insert(id.clone(), session);
        id
    }

    /// Look up an open session.
    pub fn get(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Record the rid of the most recent request belonging to a
    /// session.  Idempotent: setting the same rid twice is a no-op.
    pub fn note_request(&mut self, session_id: &str, rid: &str) -> bool {
        match self.sessions.get_mut(session_id) {
            Some(s) => {
                s.last_rid = Some(rid.into());
                true
            }
            None => false,
        }
    }

    /// Close a session.  Returns the closed session if it existed —
    /// callers can walk `last_rid` to issue an abort for the in-flight
    /// continuation, if any.
    pub fn close(&mut self, session_id: &str) -> Option<Session> {
        self.sessions.remove(session_id)
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

fn pseudo_uuid_hex() -> String {
    // A 16-byte random string in hex.  We avoid an extra crate
    // dependency on `uuid` for this stub by hashing the current time +
    // a process-local counter.
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let now = now_ns();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let lo = now.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(seq);
    let hi = (now >> 32).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    format!("{:016x}{:016x}", hi, lo)
}

fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn now_ms() -> i64 {
    (now_ns() / 1_000_000) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_then_close_returns_session() {
        let mut sc = SessionController::new();
        let id = sc.open();
        assert!(sc.get(&id).is_some());
        assert_eq!(sc.len(), 1);

        sc.note_request(&id, "rid-1");
        assert_eq!(sc.get(&id).unwrap().last_rid.as_deref(), Some("rid-1"));

        let closed = sc.close(&id).expect("close returns the session");
        assert_eq!(closed.session_id, id);
        assert!(sc.is_empty());
    }

    #[test]
    fn unknown_session_lookups_are_safe() {
        let mut sc = SessionController::new();
        assert!(sc.get("does-not-exist").is_none());
        assert!(!sc.note_request("does-not-exist", "rid"));
        assert!(sc.close("does-not-exist").is_none());
    }

    #[test]
    fn pseudo_uuids_are_unique_per_call() {
        let mut sc = SessionController::new();
        let a = sc.open();
        let b = sc.open();
        assert_ne!(a, b);
        assert_eq!(sc.len(), 2);
    }
}
