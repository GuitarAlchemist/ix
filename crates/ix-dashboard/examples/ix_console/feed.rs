use crate::state::{ConsoleState, EvidenceItem, FeedEvent, NodeHealth, NodeSpec, UnknownReason};

/// Build the default deterministic synthetic hierarchy for `ix_console`.
/// All entries are explicitly tagged `[SYNTHETIC]` so an operator never confuses
/// this demo feed with a live `ixctl` or agent session.
pub fn load_synthetic_demo_state() -> ConsoleState {
    let mut state = ConsoleState::new(64);
    for event in initial_synthetic_feed_events() {
        state.apply_feed(event);
    }
    state
}

pub fn initial_synthetic_feed_events() -> Vec<FeedEvent> {
    let nodes = vec![
        NodeSpec {
            id: "root.factory".to_string(),
            parent_id: None,
            label: "factory-root".to_string(),
            task_summary: "[SYNTHETIC] IX Agentic Factory Topology (Demo Root)".to_string(),
            incarnation: "inc-root-20261031-01".to_string(),
            reported_health: NodeHealth::Working,
            last_heartbeat_ms: Some(10_000),
            stale_after_ms: 15_000,
            progress_bp: Some(6200),
            quality_bp: Some(9400),
            evidence: vec![EvidenceItem {
                kind: "contract".to_string(),
                label: "SYNTHETIC-MANDATE-IX-CONSOLE-ASCII-V1".to_string(),
                uri: "synthetic://mandates/ix-console-ascii/v1".to_string(),
                synthetic: true,
            }],
        },
        NodeSpec {
            id: "plane.ixctl".to_string(),
            parent_id: Some("root.factory".to_string()),
            label: "ixctl-plane".to_string(),
            task_summary: "[SYNTHETIC] Local Control Plane & Probe Multiplexer".to_string(),
            incarnation: "inc-ixctl-epoch-07".to_string(),
            reported_health: NodeHealth::Working,
            last_heartbeat_ms: Some(10_000),
            stale_after_ms: 10_000,
            progress_bp: Some(8000),
            quality_bp: Some(9800),
            evidence: vec![
                EvidenceItem {
                    kind: "ws-envelope".to_string(),
                    label: "SYNTHETIC-ixctl.v1-cursor-seq-104".to_string(),
                    uri: "synthetic://ixctl/v1/ws#seq=104".to_string(),
                    synthetic: true,
                },
                EvidenceItem {
                    kind: "service".to_string(),
                    label: "SYNTHETIC-session0-peer-desk".to_string(),
                    uri: "synthetic://ixctl/peers/desk".to_string(),
                    synthetic: true,
                },
            ],
        },
        NodeSpec {
            id: "svc.desk.loop".to_string(),
            parent_id: Some("plane.ixctl".to_string()),
            label: "desk/loop-worker".to_string(),
            task_summary: "[SYNTHETIC] Bounded ix-loop worker run on user peer desk".to_string(),
            incarnation: "inc-20261031-01".to_string(),
            reported_health: NodeHealth::Working,
            last_heartbeat_ms: Some(9_950),
            stale_after_ms: 5_000,
            progress_bp: Some(4500),
            quality_bp: Some(9100),
            evidence: vec![EvidenceItem {
                kind: "run-identity".to_string(),
                label: "SYNTHETIC-run(origin=desk,epoch=7,seq=19)".to_string(),
                uri: "synthetic://ixctl/svc/desk.loop/runs/19".to_string(),
                synthetic: true,
            }],
        },
        NodeSpec {
            id: "run.desk.loop.step1".to_string(),
            parent_id: Some("svc.desk.loop".to_string()),
            label: "step-1-verify".to_string(),
            task_summary: "[SYNTHETIC] Leaf verification step inside desk/loop-worker".to_string(),
            incarnation: "inc-step1-01".to_string(),
            reported_health: NodeHealth::Done,
            last_heartbeat_ms: Some(9_900),
            stale_after_ms: 10_000,
            progress_bp: Some(10_000),
            quality_bp: Some(9950),
            evidence: vec![EvidenceItem {
                kind: "test-oracle".to_string(),
                label: "SYNTHETIC-oracle-green".to_string(),
                uri: "synthetic://ix-loop/runs/19/step1".to_string(),
                synthetic: true,
            }],
        },
        NodeSpec {
            id: "probe.mux.root".to_string(),
            parent_id: Some("plane.ixctl".to_string()),
            label: "probe/mux@1".to_string(),
            task_summary: "[SYNTHETIC] Probe tree multiplexer (heartbeat@1 + cpu@1 + delta@1)"
                .to_string(),
            incarnation: "inc-mux-01".to_string(),
            reported_health: NodeHealth::Idle,
            last_heartbeat_ms: Some(10_000),
            stale_after_ms: 6_000,
            progress_bp: Some(10_000),
            quality_bp: Some(9600),
            evidence: vec![EvidenceItem {
                kind: "cursor".to_string(),
                label: "SYNTHETIC-probe.list?with_cursor=1".to_string(),
                uri: "synthetic://ixctl/probes/mux-root".to_string(),
                synthetic: true,
            }],
        },
        NodeSpec {
            id: "plane.wmux".to_string(),
            parent_id: Some("root.factory".to_string()),
            label: "wmux-surfaces".to_string(),
            task_summary: "[SYNTHETIC] Read-only WMUX surface snapshot (Tier 1)".to_string(),
            incarnation: "inc-wmux-snap-01".to_string(),
            reported_health: NodeHealth::Blocked,
            last_heartbeat_ms: Some(4_000),
            stale_after_ms: 4_000,
            progress_bp: Some(3000),
            quality_bp: Some(8500),
            evidence: vec![EvidenceItem {
                kind: "refusal-guard".to_string(),
                label: "SYNTHETIC-DRAFT_SAFE_TRANSPORT_UNAVAILABLE".to_string(),
                uri: "synthetic://wmux/guard/draft-safe".to_string(),
                synthetic: true,
            }],
        },
        NodeSpec {
            id: "surface.claude.indigo".to_string(),
            parent_id: Some("plane.wmux".to_string()),
            label: "surface/indigo".to_string(),
            task_summary: "[SYNTHETIC] Unverified interactive surface (Silence != Idle demo)"
                .to_string(),
            incarnation: "unverified-null".to_string(),
            reported_health: NodeHealth::Unknown(UnknownReason::Unobserved),
            last_heartbeat_ms: None,
            stale_after_ms: 5_000,
            progress_bp: None,
            quality_bp: None,
            evidence: vec![EvidenceItem {
                kind: "identity-state".to_string(),
                label: "SYNTHETIC-TARGET_IDENTITY_UNVERIFIED".to_string(),
                uri: "synthetic://wmux/surfaces/indigo".to_string(),
                synthetic: true,
            }],
        },
        NodeSpec {
            id: "plane.governance".to_string(),
            parent_id: Some("root.factory".to_string()),
            label: "demerzel-gov".to_string(),
            task_summary: "[SYNTHETIC] Governance & Epistemic Audit Checks".to_string(),
            incarnation: "inc-gov-20261031".to_string(),
            reported_health: NodeHealth::Stale { silent_ms: 7_500 },
            last_heartbeat_ms: Some(2_500),
            stale_after_ms: 5_000,
            progress_bp: Some(7500),
            quality_bp: Some(8800),
            evidence: vec![EvidenceItem {
                kind: "audit".to_string(),
                label: "SYNTHETIC-blackbox-policy-check".to_string(),
                uri: "synthetic://governance/blackbox".to_string(),
                synthetic: true,
            }],
        },
    ];

    let mut events: Vec<FeedEvent> = nodes.into_iter().map(FeedEvent::UpsertNode).collect();

    events.push(FeedEvent::AppendLog {
        node_id: "root.factory".to_string(),
        seq: 1,
        timestamp_ms: 9_800,
        raw_text: "[SYNTHETIC] Bootstrapped IX ASCII console topology (8 demo nodes)".to_string(),
    });
    events.push(FeedEvent::AppendLog {
        node_id: "root.factory".to_string(),
        seq: 2,
        timestamp_ms: 9_900,
        raw_text: "[SYNTHETIC] Escaped ANSI sample: \x1b[31mRED_ALERT\x1b[0m and CR\\r neutralized"
            .to_string(),
    });
    events.push(FeedEvent::AppendLog {
        node_id: "plane.ixctl".to_string(),
        seq: 10,
        timestamp_ms: 9_920,
        raw_text: "[SYNTHETIC] ixctl.run.begin run={origin:desk,epoch:7,seq:19}".to_string(),
    });
    events.push(FeedEvent::AppendLog {
        node_id: "svc.desk.loop".to_string(),
        seq: 101,
        timestamp_ms: 9_940,
        raw_text: "[SYNTHETIC] PreparedAsk format=ix.semantic.prepared.v0 req=req-demo-01"
            .to_string(),
    });
    events.push(FeedEvent::AppendLog {
        node_id: "svc.desk.loop".to_string(),
        seq: 102,
        timestamp_ms: 9_950,
        raw_text: "[SYNTHETIC] Sanitized terminal escape test: \x1b]0;spoof-title\x07 [ok]"
            .to_string(),
    });
    events.extend(synthetic_log_burst("svc.desk.loop", 2, 103));
    events.push(FeedEvent::RecordGap {
        node_id: "surface.claude.indigo".to_string(),
        missed_seq: 7,
        timestamp_ms: 9_960,
    });
    events.push(FeedEvent::UpsertNode(NodeSpec {
        id: "run.ephemeral.canary".to_string(),
        parent_id: Some("plane.ixctl".to_string()),
        label: "canary/ephemeral".to_string(),
        task_summary: "[SYNTHETIC] Completed ephemeral canary (tombstoned demo)".to_string(),
        incarnation: "inc-canary-01".to_string(),
        reported_health: NodeHealth::Done,
        last_heartbeat_ms: Some(9_500),
        stale_after_ms: 10_000,
        progress_bp: Some(10_000),
        quality_bp: Some(10_000),
        evidence: vec![],
    }));
    events.push(FeedEvent::RemoveNode {
        id: "run.ephemeral.canary".to_string(),
        timestamp_ms: 9_980,
    });
    events.push(FeedEvent::AdvanceClock { now_ms: 10_000 });

    events
}

pub fn synthetic_log_burst(node_id: &str, count: usize, start_seq: u64) -> Vec<FeedEvent> {
    (0..count)
        .map(|i| FeedEvent::AppendLog {
            node_id: node_id.to_string(),
            seq: start_seq + i as u64,
            timestamp_ms: 10_000 + i as u64,
            raw_text: format!(
                "[SYNTHETIC-BURST] event={} node={} payload=deterministic",
                start_seq + i as u64,
                node_id
            ),
        })
        .collect()
}

/// Build a deterministic synthetic hierarchy of `target_nodes` (e.g. 10,000 nodes)
/// across 4 levels under `root.factory` to benchmark `reduce + layout` and `input -> render`.
pub fn build_synthetic_10k_state(target_nodes: usize) -> ConsoleState {
    let total = target_nodes.max(4);
    let mut state = ConsoleState::new(16);

    state.apply_feed(FeedEvent::UpsertNode(NodeSpec {
        id: "root.factory".to_string(),
        parent_id: None,
        label: "factory-root-10k".to_string(),
        task_summary: format!("[SYNTHETIC-10K] Scale benchmark root ({total} nodes)"),
        incarnation: "inc-10k-root".to_string(),
        reported_health: NodeHealth::Working,
        last_heartbeat_ms: Some(100_000),
        stale_after_ms: 30_000,
        progress_bp: Some(5000),
        quality_bp: Some(9500),
        evidence: vec![],
    }));

    let mut created = 1usize;
    let l1_count = 10usize.min(total - created);
    for i in 0..l1_count {
        let id = format!("dom.{i:02}");
        state.apply_feed(FeedEvent::UpsertNode(NodeSpec {
            id,
            parent_id: Some("root.factory".to_string()),
            label: format!("domain-{i:02}"),
            task_summary: format!("[SYNTHETIC-10K] Domain partition {i}"),
            incarnation: format!("inc-dom-{i:02}"),
            reported_health: if i % 5 == 0 {
                NodeHealth::Idle
            } else {
                NodeHealth::Working
            },
            last_heartbeat_ms: Some(100_000),
            stale_after_ms: 30_000,
            progress_bp: Some(((i * 900) % 10_000) as u16),
            quality_bp: Some(9000),
            evidence: vec![],
        }));
        created += 1;
    }

    let l2_count = 100usize.min(total.saturating_sub(created));
    for j in 0..l2_count {
        let parent_id = format!("dom.{:02}", j % l1_count.max(1));
        let id = format!("svc.{j:03}");
        state.apply_feed(FeedEvent::UpsertNode(NodeSpec {
            id,
            parent_id: Some(parent_id),
            label: format!("service-{j:03}"),
            task_summary: format!("[SYNTHETIC-10K] Service group {j}"),
            incarnation: format!("inc-svc-{j:03}"),
            reported_health: NodeHealth::Working,
            last_heartbeat_ms: Some(99_500),
            stale_after_ms: 20_000,
            progress_bp: Some(((j * 130) % 10_000) as u16),
            quality_bp: Some(9200),
            evidence: vec![],
        }));
        created += 1;
    }

    let mut leaf_idx = 0usize;
    while created < total {
        let parent_id = if l2_count > 0 {
            format!("svc.{:03}", leaf_idx % l2_count)
        } else {
            "root.factory".to_string()
        };
        let id = format!("node.{leaf_idx:05}");
        let reported_health = match leaf_idx % 16 {
            0 => NodeHealth::Unknown(UnknownReason::Unobserved),
            1 => NodeHealth::Blocked,
            2 => NodeHealth::Done,
            _ => NodeHealth::Working,
        };
        let last_hb = if leaf_idx % 16 == 0 {
            None
        } else if leaf_idx % 8 == 0 {
            Some(50_000) // will be Stale at now_ms = 100_000
        } else {
            Some(99_900)
        };
        state.apply_feed(FeedEvent::UpsertNode(NodeSpec {
            id,
            parent_id: Some(parent_id),
            label: format!("unit-{leaf_idx:05}"),
            task_summary: format!("[SYNTHETIC-10K] Work unit #{leaf_idx}"),
            incarnation: format!("inc-u-{leaf_idx:05}"),
            reported_health,
            last_heartbeat_ms: last_hb,
            stale_after_ms: 15_000,
            progress_bp: Some(((leaf_idx * 37) % 10_000) as u16),
            quality_bp: Some((8000 + (leaf_idx % 2000)) as u16),
            evidence: vec![],
        }));
        created += 1;
        leaf_idx += 1;
    }

    state
}
