// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Sanity tests for the queue / policy / req layer.

use std::sync::{Arc, RwLock};

use sglang_scheduler::queue::{SchedulePolicy, SchedulePolicyKind, WaitingQueue};
use sglang_scheduler::types::{FinishReason, Req, SamplingParams};

fn make_req(rid: &str, max_new_tokens: u32) -> Arc<RwLock<Req>> {
    let mut sp = SamplingParams::default();
    sp.max_new_tokens = max_new_tokens;
    Arc::new(RwLock::new(Req::new(rid.into(), vec![1, 2, 3], sp)))
}

#[test]
fn req_check_finished_length() {
    let mut r = Req::new("r1".into(), vec![10, 11], {
        let mut sp = SamplingParams::default();
        sp.max_new_tokens = 2;
        sp
    });
    assert!(!r.finished());
    r.output_ids.push(7);
    assert!(r.check_finished().is_none());
    r.output_ids.push(8);
    let reason = r.check_finished().cloned();
    assert!(matches!(reason, Some(FinishReason::Length { length: 2 })));
    assert!(r.finished());
}

#[test]
fn req_check_finished_stop_token() {
    let mut sp = SamplingParams::default();
    sp.max_new_tokens = 100;
    sp.stop_token_ids = Some(vec![42]);
    let mut r = Req::new("r1".into(), vec![1], sp);
    r.output_ids.push(7);
    assert!(r.check_finished().is_none());
    r.output_ids.push(42);
    let reason = r.check_finished().cloned();
    assert!(matches!(reason, Some(FinishReason::MatchedToken { token_id: 42 })));
}

#[test]
fn req_to_finish_path() {
    let mut r = Req::new("r1".into(), vec![1], SamplingParams::default());
    r.to_finish = Some(FinishReason::Aborted {
        message: "client closed".into(),
    });
    let reason = r.check_finished().cloned();
    assert!(matches!(reason, Some(FinishReason::Aborted { .. })));
    assert!(r.finished());
}

#[test]
fn schedule_policy_fcfs_preserves_order() {
    let p = SchedulePolicy::new(SchedulePolicyKind::Fcfs);
    let reqs = vec![make_req("a", 10), make_req("b", 5), make_req("c", 20)];
    assert_eq!(p.order(&reqs), vec![0, 1, 2]);
}

#[test]
fn schedule_policy_lof_picks_shortest_first() {
    let p = SchedulePolicy::new(SchedulePolicyKind::Lof);
    let reqs = vec![make_req("a", 10), make_req("b", 5), make_req("c", 20)];
    let order = p.order(&reqs);
    // Shortest max_new_tokens first.
    let first = reqs[order[0]].read().unwrap().rid.clone();
    assert_eq!(first, "b");
}

#[test]
fn waiting_queue_admits_in_policy_order() {
    let mut q = WaitingQueue::new();
    q.push(make_req("a", 10));
    q.push(make_req("b", 5));
    q.push(make_req("c", 20));
    assert_eq!(q.len(), 3);

    let policy = SchedulePolicy::new(SchedulePolicyKind::Lof);
    let admitted = q.drain(2, &policy);
    assert_eq!(admitted.len(), 2);
    // LoF picks the two shortest: rid "b" (5) then rid "a" (10).
    let rids: Vec<String> = admitted
        .iter()
        .map(|r| r.read().unwrap().rid.clone())
        .collect();
    assert_eq!(rids, vec!["b".to_string(), "a".to_string()]);

    // Only "c" remains.
    assert_eq!(q.len(), 1);
    let remaining = q.drain(10, &policy);
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].read().unwrap().rid, "c");
}

#[test]
fn waiting_queue_drop_by_rid() {
    let mut q = WaitingQueue::new();
    q.push(make_req("a", 10));
    q.push(make_req("b", 5));
    q.push(make_req("c", 20));
    let dropped = q.drop_by_rid("b");
    assert_eq!(dropped, 1);
    assert_eq!(q.len(), 2);
}

#[test]
fn unknown_policy_falls_back_to_fcfs() {
    assert_eq!(
        SchedulePolicyKind::parse("not-a-real-policy"),
        SchedulePolicyKind::Fcfs
    );
}
