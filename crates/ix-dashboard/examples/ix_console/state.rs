#![allow(dead_code)]
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub type NodeId = String;
pub type IncarnationId = String;

pub const DEFAULT_MAX_TREE_DEPTH: usize = 1024;
pub const DEFAULT_MAX_LOG_LINE_BYTES: usize = 512;
pub const DEFAULT_MAX_NODE_LOG_BYTES: usize = 16 * 1024;
pub const DEFAULT_MAX_TOTAL_LOG_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownReason {
    Unobserved,
    GapDetected { missed_seq: u64 },
    EpochMismatch { expected: String, observed: String },
}

impl UnknownReason {
    pub fn short_code(&self) -> String {
        match self {
            Self::Unobserved => "UNOBSERVED".to_string(),
            Self::GapDetected { missed_seq } => format!("GAP(seq={missed_seq})"),
            Self::EpochMismatch { expected, observed } => {
                format!("EPOCH({expected}!={observed})")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeHealth {
    Working,
    Idle,
    Blocked,
    Done,
    Failed,
    Stale { silent_ms: u64 },
    Unknown(UnknownReason),
}

impl NodeHealth {
    pub fn badge(&self) -> &'static str {
        match self {
            Self::Working => "WORK",
            Self::Idle => "IDLE",
            Self::Blocked => "BLKD",
            Self::Done => "DONE",
            Self::Failed => "FAIL",
            Self::Stale { .. } => "STLE",
            Self::Unknown(_) => "UNKN",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub kind: String,
    pub label: String,
    pub uri: String,
    pub synthetic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SanitizedLogLine {
    pub seq: u64,
    pub timestamp_ms: u64,
    pub incarnation: IncarnationId,
    pub text: String,
    pub byte_len: usize,
    pub was_escaped: bool,
    pub was_truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeSpec {
    pub id: NodeId,
    pub parent_id: Option<NodeId>,
    pub label: String,
    pub task_summary: String,
    pub incarnation: IncarnationId,
    pub reported_health: NodeHealth,
    pub last_heartbeat_ms: Option<u64>,
    pub stale_after_ms: u64,
    /// Task progress in basis points (0..=10_000). Orthogonal to depth, quality, and visual size.
    pub progress_bp: Option<u16>,
    /// Verification quality score in basis points (0..=10_000). Orthogonal to depth, progress, and visual size.
    pub quality_bp: Option<u16>,
    pub evidence: Vec<EvidenceItem>,
}

#[derive(Debug, Clone)]
pub struct NodeEntry {
    pub spec: NodeSpec,
    pub children: Vec<NodeId>,
    pub depth: usize,
    pub subtree_weight: usize,
    pub removed_at_ms: Option<u64>,
    pub previous_incarnation: Option<IncarnationId>,
    pub incarnation_transitions: u32,
    pub stale_incarnation_events_ignored: u64,
    pub logs: VecDeque<SanitizedLogLine>,
    pub retained_log_bytes: usize,
    pub evicted_log_count: u64,
    pub evicted_log_bytes: u64,
    pub log_scroll_offset: usize,
    pub log_anchor_seq: Option<u64>,
    pub evicted_anchor_seq: Option<u64>,
    pub gap_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedRejectReason {
    SelfParent {
        node_id: NodeId,
    },
    CycleDetected {
        node_id: NodeId,
        parent_id: NodeId,
    },
    MaxDepthExceeded {
        node_id: NodeId,
        attempted_depth: usize,
        max_depth: usize,
    },
    StaleIncarnation {
        node_id: NodeId,
        event_incarnation: IncarnationId,
        active_incarnation: IncarnationId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsoleAction {
    Select(NodeId),
    SelectNextVisible,
    SelectPrevVisible,
    ZoomIn,
    ZoomTo(NodeId),
    Back,
    ResetZoom,
    ToggleFollowLogs,
    ScrollLogs(i32),
    Tick { now_ms: u64 },
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoiceIntent {
    SelectNode { target_id: String },
    NextNode,
    PreviousNode,
    ZoomBranch,
    ZoomToNode { target_id: String },
    GoParent,
    ResetRoot,
    FollowLogs,
    ScrollLogs { delta: i32 },
    TickClock { now_ms: u64 },
    QuitConsole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionParseError {
    InvalidJson(String),
    UnknownCommand(String),
    MissingField(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedEvent {
    UpsertNode(NodeSpec),
    RemoveNode {
        id: NodeId,
        timestamp_ms: u64,
    },
    RemoveNodeForIncarnation {
        id: NodeId,
        incarnation: IncarnationId,
        timestamp_ms: u64,
    },
    AppendLog {
        node_id: NodeId,
        seq: u64,
        timestamp_ms: u64,
        raw_text: String,
    },
    AppendLogForIncarnation {
        node_id: NodeId,
        incarnation: IncarnationId,
        seq: u64,
        timestamp_ms: u64,
        raw_text: String,
    },
    RecordGap {
        node_id: NodeId,
        missed_seq: u64,
        timestamp_ms: u64,
    },
    RecordGapForIncarnation {
        node_id: NodeId,
        incarnation: IncarnationId,
        missed_seq: u64,
        timestamp_ms: u64,
    },
    AdvanceClock {
        now_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoverageSummary {
    pub total_nodes: usize,
    pub observed_nodes: usize,
    pub fresh_nodes: usize,
    pub stale_nodes: usize,
    pub unknown_nodes: usize,
    pub removed_nodes: usize,
    pub gaps_detected: usize,
}

#[derive(Debug, Clone)]
pub struct ConsoleState {
    pub fixture_banner: String,
    pub nodes: BTreeMap<NodeId, NodeEntry>,
    pub roots: Vec<NodeId>,
    pub selected_id: Option<NodeId>,
    pub zoom_root_id: Option<NodeId>,
    pub follow_logs: bool,
    pub should_quit: bool,
    pub now_ms: u64,
    pub max_tree_depth: usize,
    pub max_logs_per_node: usize,
    pub max_log_line_bytes: usize,
    pub max_node_log_bytes: usize,
    pub max_total_log_bytes: usize,
    pub total_retained_log_bytes: usize,
    pub total_gaps_detected: usize,
    pub rejected_events: Vec<FeedRejectReason>,
}

impl ConsoleState {
    pub fn new(max_logs_per_node: usize) -> Self {
        Self {
            fixture_banner:
                "[SYNTHETIC DEMO FEED - NO LIVE AGENT / NO PROMPT DELIVERY / READ-ONLY PROTOTYPE]"
                    .to_string(),
            nodes: BTreeMap::new(),
            roots: Vec::new(),
            selected_id: None,
            zoom_root_id: None,
            follow_logs: false,
            should_quit: false,
            now_ms: 0,
            max_tree_depth: DEFAULT_MAX_TREE_DEPTH,
            max_logs_per_node: max_logs_per_node.max(1),
            max_log_line_bytes: DEFAULT_MAX_LOG_LINE_BYTES,
            max_node_log_bytes: DEFAULT_MAX_NODE_LOG_BYTES,
            max_total_log_bytes: DEFAULT_MAX_TOTAL_LOG_BYTES,
            total_retained_log_bytes: 0,
            total_gaps_detected: 0,
            rejected_events: Vec::new(),
        }
    }

    /// Apply a feed event without ever shifting `selected_id` or `zoom_root_id`
    /// once they have been established. Rejects cycles, excessive depth, and
    /// stale-incarnation mutations explicitly without corrupting valid state.
    pub fn apply_feed(&mut self, event: FeedEvent) {
        match event {
            FeedEvent::UpsertNode(mut spec) => {
                if let Some(p) = spec.progress_bp {
                    spec.progress_bp = Some(p.min(10_000));
                }
                if let Some(q) = spec.quality_bp {
                    spec.quality_bp = Some(q.min(10_000));
                }

                // 1. Validate self-parenting
                if spec.parent_id.as_deref() == Some(spec.id.as_str()) {
                    self.rejected_events
                        .push(FeedRejectReason::SelfParent { node_id: spec.id });
                    return;
                }

                // 2. Validate indirect cycles before mutating anything
                if let Some(ref p_id) = spec.parent_id {
                    if self.would_introduce_cycle(&spec.id, p_id) {
                        self.rejected_events.push(FeedRejectReason::CycleDetected {
                            node_id: spec.id,
                            parent_id: p_id.clone(),
                        });
                        return;
                    }
                }

                // 3. Validate maximum tree depth across `spec.id` and its attached/orphan descendants
                let proposed_depth = self.proposed_node_depth(spec.parent_id.as_deref());
                let max_descendant_span = self.max_descendant_span(&spec.id);
                let worst_depth = proposed_depth.saturating_add(max_descendant_span);
                if worst_depth > self.max_tree_depth {
                    self.rejected_events
                        .push(FeedRejectReason::MaxDepthExceeded {
                            node_id: spec.id,
                            attempted_depth: worst_depth,
                            max_depth: self.max_tree_depth,
                        });
                    return;
                }

                if let Some(hb) = spec.last_heartbeat_ms {
                    self.now_ms = self.now_ms.max(hb);
                }

                let node_id = spec.id.clone();
                let new_parent = spec.parent_id.clone();

                let mut old_parent: Option<NodeId> = None;
                let mut is_new_node = false;

                if let Some(existing) = self.nodes.get_mut(&node_id) {
                    old_parent = existing.spec.parent_id.clone();
                    if existing.spec.incarnation != spec.incarnation {
                        existing.previous_incarnation = Some(existing.spec.incarnation.clone());
                        existing.incarnation_transitions =
                            existing.incarnation_transitions.saturating_add(1);
                    }
                    existing.removed_at_ms = None;
                    existing.spec = spec;
                } else {
                    is_new_node = true;
                    let entry = NodeEntry {
                        spec,
                        children: Vec::new(),
                        depth: 0,
                        subtree_weight: 1,
                        removed_at_ms: None,
                        previous_incarnation: None,
                        incarnation_transitions: 0,
                        stale_incarnation_events_ignored: 0,
                        logs: VecDeque::new(),
                        retained_log_bytes: 0,
                        evicted_log_count: 0,
                        evicted_log_bytes: 0,
                        log_scroll_offset: 0,
                        log_anchor_seq: None,
                        evicted_anchor_seq: None,
                        gap_count: 0,
                    };
                    self.nodes.insert(node_id.clone(), entry);
                }

                if is_new_node || old_parent != new_parent {
                    // Detach from old parent or roots if needed
                    if let Some(ref prev_p) = old_parent {
                        if let Some(p_entry) = self.nodes.get_mut(prev_p) {
                            p_entry.children.retain(|c| c != &node_id);
                        }
                    } else if !is_new_node {
                        self.roots.retain(|r| r != &node_id);
                    }

                    // Attach to new parent if already present, else place in roots
                    let attached_to_parent = if let Some(ref p_id) = new_parent {
                        if self.nodes.contains_key(p_id) {
                            if let Some(p_entry) = self.nodes.get_mut(p_id) {
                                if !p_entry.children.contains(&node_id) {
                                    p_entry.children.push(node_id.clone());
                                    p_entry.children.sort();
                                }
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if !attached_to_parent {
                        if !self.roots.contains(&node_id) {
                            self.roots.push(node_id.clone());
                            self.roots.sort();
                        }
                    } else {
                        self.roots.retain(|r| r != &node_id);
                    }

                    // Reattach any existing orphans in `roots` that declared `node_id` as their parent
                    let orphan_roots: Vec<NodeId> = self
                        .roots
                        .iter()
                        .filter(|r| {
                            *r != &node_id
                                && self.nodes.get(*r).and_then(|n| n.spec.parent_id.as_ref())
                                    == Some(&node_id)
                        })
                        .cloned()
                        .collect();

                    let had_orphans = !orphan_roots.is_empty();
                    if had_orphans {
                        for orphan in orphan_roots {
                            self.roots.retain(|r| r != &orphan);
                            if let Some(parent_entry) = self.nodes.get_mut(&node_id) {
                                if !parent_entry.children.contains(&orphan) {
                                    parent_entry.children.push(orphan);
                                    parent_entry.children.sort();
                                }
                            }
                        }
                    }

                    if is_new_node && !had_orphans && old_parent.is_none() {
                        // Fast incremental path for normal top-down insertion (including 10k benchmark)
                        let d = self.proposed_node_depth(new_parent.as_deref());
                        if let Some(entry) = self.nodes.get_mut(&node_id) {
                            entry.depth = d;
                            entry.subtree_weight = 1;
                        }
                        self.add_weight_to_ancestors(&node_id, 1);
                    } else {
                        // Exact full topological recomputation whenever orphans reattach or a branch moves
                        self.recompute_all_depths_and_weights();
                    }
                }

                // Initialize default selection & zoom root only if none was ever set
                if self.zoom_root_id.is_none() {
                    self.zoom_root_id = self.roots.first().cloned();
                }
                if self.selected_id.is_none() {
                    self.selected_id = self.zoom_root_id.clone();
                }
            }
            FeedEvent::RemoveNode { id, timestamp_ms } => {
                self.handle_remove_node(&id, None, timestamp_ms);
            }
            FeedEvent::RemoveNodeForIncarnation {
                id,
                incarnation,
                timestamp_ms,
            } => {
                self.handle_remove_node(&id, Some(&incarnation), timestamp_ms);
            }
            FeedEvent::AppendLog {
                node_id,
                seq,
                timestamp_ms,
                raw_text,
            } => {
                self.handle_append_log(&node_id, None, seq, timestamp_ms, &raw_text);
            }
            FeedEvent::AppendLogForIncarnation {
                node_id,
                incarnation,
                seq,
                timestamp_ms,
                raw_text,
            } => {
                self.handle_append_log(&node_id, Some(&incarnation), seq, timestamp_ms, &raw_text);
            }
            FeedEvent::RecordGap {
                node_id,
                missed_seq,
                timestamp_ms,
            } => {
                self.handle_record_gap(&node_id, None, missed_seq, timestamp_ms);
            }
            FeedEvent::RecordGapForIncarnation {
                node_id,
                incarnation,
                missed_seq,
                timestamp_ms,
            } => {
                self.handle_record_gap(&node_id, Some(&incarnation), missed_seq, timestamp_ms);
            }
            FeedEvent::AdvanceClock { now_ms } => {
                if now_ms > self.now_ms {
                    self.now_ms = now_ms;
                }
            }
        }
    }

    fn handle_remove_node(
        &mut self,
        id: &str,
        expected_incarnation: Option<&str>,
        timestamp_ms: u64,
    ) {
        self.now_ms = self.now_ms.max(timestamp_ms);
        let Some(entry) = self.nodes.get_mut(id) else {
            return;
        };
        if let Some(exp_inc) = expected_incarnation {
            if exp_inc != entry.spec.incarnation {
                entry.stale_incarnation_events_ignored =
                    entry.stale_incarnation_events_ignored.saturating_add(1);
                self.rejected_events
                    .push(FeedRejectReason::StaleIncarnation {
                        node_id: id.to_string(),
                        event_incarnation: exp_inc.to_string(),
                        active_incarnation: entry.spec.incarnation.clone(),
                    });
                return;
            }
        }
        entry.removed_at_ms = Some(timestamp_ms);
    }

    fn handle_record_gap(
        &mut self,
        node_id: &str,
        expected_incarnation: Option<&str>,
        missed_seq: u64,
        timestamp_ms: u64,
    ) {
        self.now_ms = self.now_ms.max(timestamp_ms);
        let Some(entry) = self.nodes.get_mut(node_id) else {
            return;
        };
        if let Some(exp_inc) = expected_incarnation {
            if exp_inc != entry.spec.incarnation {
                entry.stale_incarnation_events_ignored =
                    entry.stale_incarnation_events_ignored.saturating_add(1);
                self.rejected_events
                    .push(FeedRejectReason::StaleIncarnation {
                        node_id: node_id.to_string(),
                        event_incarnation: exp_inc.to_string(),
                        active_incarnation: entry.spec.incarnation.clone(),
                    });
                return;
            }
        }
        entry.spec.reported_health = NodeHealth::Unknown(UnknownReason::GapDetected { missed_seq });
        entry.gap_count = entry.gap_count.saturating_add(1);
        self.total_gaps_detected = self.total_gaps_detected.saturating_add(1);
    }

    fn handle_append_log(
        &mut self,
        node_id: &str,
        log_incarnation: Option<&str>,
        seq: u64,
        timestamp_ms: u64,
        raw_text: &str,
    ) {
        self.now_ms = self.now_ms.max(timestamp_ms);
        let max_line_bytes = self.max_log_line_bytes.max(16);
        let (sanitized, was_escaped, was_truncated) =
            sanitize_log_line_bounded(raw_text, max_line_bytes);
        let line_bytes = sanitized.len();

        let max_cap = self.max_logs_per_node.max(1);
        let max_node_bytes = self.max_node_log_bytes.max(max_line_bytes);
        let max_total_bytes = self.max_total_log_bytes.max(max_node_bytes);
        let follow = self.follow_logs;

        let Some(entry) = self.nodes.get_mut(node_id) else {
            return;
        };

        let effective_inc = match log_incarnation {
            Some(inc) => {
                if inc != entry.spec.incarnation {
                    entry.stale_incarnation_events_ignored =
                        entry.stale_incarnation_events_ignored.saturating_add(1);
                }
                inc.to_string()
            }
            None => entry.spec.incarnation.clone(),
        };

        // Evict from this node while line count or node byte budget or total byte budget would be exceeded
        while !entry.logs.is_empty()
            && (entry.logs.len() >= max_cap
                || entry.retained_log_bytes.saturating_add(line_bytes) > max_node_bytes
                || self.total_retained_log_bytes.saturating_add(line_bytes) > max_total_bytes)
        {
            if let Some(evicted) = entry.logs.pop_front() {
                entry.retained_log_bytes =
                    entry.retained_log_bytes.saturating_sub(evicted.byte_len);
                self.total_retained_log_bytes = self
                    .total_retained_log_bytes
                    .saturating_sub(evicted.byte_len);
                entry.evicted_log_count = entry.evicted_log_count.saturating_add(1);
                entry.evicted_log_bytes = entry
                    .evicted_log_bytes
                    .saturating_add(evicted.byte_len as u64);

                if !follow && entry.log_anchor_seq == Some(evicted.seq) {
                    entry.evicted_anchor_seq = Some(evicted.seq);
                    entry.log_anchor_seq = entry.logs.front().map(|l| l.seq);
                }
            }
        }

        entry.logs.push_back(SanitizedLogLine {
            seq,
            timestamp_ms,
            incarnation: effective_inc,
            text: sanitized,
            byte_len: line_bytes,
            was_escaped,
            was_truncated,
        });
        entry.retained_log_bytes = entry.retained_log_bytes.saturating_add(line_bytes);
        self.total_retained_log_bytes = self.total_retained_log_bytes.saturating_add(line_bytes);

        if follow {
            entry.log_scroll_offset = entry.logs.len().saturating_sub(1);
            entry.log_anchor_seq = entry.logs.back().map(|l| l.seq);
            entry.evicted_anchor_seq = None;
        } else {
            if entry.log_anchor_seq.is_none() {
                entry.log_anchor_seq = entry.logs.front().map(|l| l.seq);
            }
            if let Some(anchor) = entry.log_anchor_seq {
                if let Some(pos) = entry.logs.iter().position(|l| l.seq == anchor) {
                    entry.log_scroll_offset = pos;
                }
            }
        }
    }

    pub fn apply_action(&mut self, action: ConsoleAction) {
        match action {
            ConsoleAction::Select(id) => {
                if self.nodes.contains_key(&id) {
                    self.selected_id = Some(id);
                }
            }
            ConsoleAction::SelectNextVisible => {
                if let Some(next_id) = self.step_visible_selection(true) {
                    self.selected_id = Some(next_id);
                }
            }
            ConsoleAction::SelectPrevVisible => {
                if let Some(prev_id) = self.step_visible_selection(false) {
                    self.selected_id = Some(prev_id);
                }
            }
            ConsoleAction::ZoomIn => {
                if let Some(sel) = self.selected_id.clone() {
                    if self.nodes.contains_key(&sel) {
                        self.zoom_root_id = Some(sel);
                    }
                }
            }
            ConsoleAction::ZoomTo(id) => {
                if self.nodes.contains_key(&id) {
                    self.zoom_root_id = Some(id);
                }
            }
            ConsoleAction::Back => {
                let Some(curr_zoom) = self.zoom_root_id.clone() else {
                    return;
                };
                let parent_opt = self
                    .nodes
                    .get(&curr_zoom)
                    .and_then(|n| n.spec.parent_id.clone())
                    .filter(|p| self.nodes.contains_key(p));

                if let Some(parent_id) = parent_opt {
                    self.zoom_root_id = Some(parent_id);
                }
            }
            ConsoleAction::ResetZoom => {
                if let Some(first_root) = self.roots.first().cloned() {
                    self.zoom_root_id = Some(first_root);
                }
            }
            ConsoleAction::ToggleFollowLogs => {
                self.follow_logs = !self.follow_logs;
                if self.follow_logs {
                    if let Some(sel) = self.selected_id.as_ref() {
                        if let Some(entry) = self.nodes.get_mut(sel) {
                            entry.log_scroll_offset = entry.logs.len().saturating_sub(1);
                            entry.log_anchor_seq = entry.logs.back().map(|l| l.seq);
                            entry.evicted_anchor_seq = None;
                        }
                    }
                }
            }
            ConsoleAction::ScrollLogs(delta) => {
                let Some(sel) = self.selected_id.as_ref() else {
                    return;
                };
                if let Some(entry) = self.nodes.get_mut(sel) {
                    let len = entry.logs.len();
                    if len == 0 {
                        entry.log_scroll_offset = 0;
                        entry.log_anchor_seq = None;
                    } else {
                        let base_idx = entry
                            .log_anchor_seq
                            .and_then(|anchor| entry.logs.iter().position(|l| l.seq == anchor))
                            .unwrap_or(entry.log_scroll_offset)
                            .min(len - 1);
                        let new_idx = if delta < 0 {
                            base_idx.saturating_sub(delta.unsigned_abs() as usize)
                        } else {
                            (base_idx + delta as usize).min(len - 1)
                        };
                        entry.log_scroll_offset = new_idx;
                        entry.log_anchor_seq = Some(entry.logs[new_idx].seq);
                        entry.evicted_anchor_seq = None;
                    }
                }
            }
            ConsoleAction::Tick { now_ms } => {
                if now_ms > self.now_ms {
                    self.now_ms = now_ms;
                }
            }
            ConsoleAction::Quit => {
                self.should_quit = true;
            }
        }
    }

    /// Compute the effective health of a node at `self.now_ms`.
    /// Silence never degrades into `Idle`: a node with expired heartbeat is `Stale`,
    /// and a node without any heartbeat is `Unknown(Unobserved)`.
    pub fn effective_health(&self, id: &str) -> NodeHealth {
        let Some(entry) = self.nodes.get(id) else {
            return NodeHealth::Unknown(UnknownReason::Unobserved);
        };

        if let NodeHealth::Unknown(reason) = &entry.spec.reported_health {
            return NodeHealth::Unknown(reason.clone());
        }

        match entry.spec.last_heartbeat_ms {
            None => NodeHealth::Unknown(UnknownReason::Unobserved),
            Some(last_hb) => {
                let elapsed = self.now_ms.saturating_sub(last_hb);
                if entry.spec.stale_after_ms > 0 && elapsed > entry.spec.stale_after_ms {
                    NodeHealth::Stale { silent_ms: elapsed }
                } else {
                    entry.spec.reported_health.clone()
                }
            }
        }
    }

    /// Breadcrumb chain from the top ancestor down to `zoom_root_id`.
    /// Invariant across arbitrary depths (no 256-level truncation).
    pub fn breadcrumbs(&self) -> Vec<NodeId> {
        let Some(mut curr) = self.zoom_root_id.clone() else {
            return Vec::new();
        };
        let mut chain = Vec::new();
        let mut visited = BTreeSet::new();
        while let Some(entry) = self.nodes.get(&curr) {
            if !visited.insert(curr.clone()) {
                break;
            }
            chain.push(curr.clone());
            match entry.spec.parent_id.as_ref() {
                Some(parent) if self.nodes.contains_key(parent) && parent != &curr => {
                    curr = parent.clone();
                }
                _ => break,
            }
        }
        chain.reverse();
        chain
    }

    /// Check whether `ancestor_candidate` is equal to `target` or is an ancestor of `target`.
    pub fn is_same_or_ancestor(&self, ancestor_candidate: &str, target: &str) -> bool {
        let mut curr = target;
        let mut visited = BTreeSet::new();
        while let Some(entry) = self.nodes.get(curr) {
            if curr == ancestor_candidate {
                return true;
            }
            if !visited.insert(curr) {
                break;
            }
            match entry.spec.parent_id.as_deref() {
                Some(p) if self.nodes.contains_key(p) => curr = p,
                _ => break,
            }
        }
        false
    }

    /// Full pre-order traversal under `zoom_root_id` without any 64-node cap.
    pub fn visible_navigation_order(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        let Some(root_id) = self.zoom_root_id.as_ref().or_else(|| self.roots.first()) else {
            return out;
        };
        let mut stack = vec![root_id.clone()];
        let mut visited = BTreeSet::new();
        while let Some(curr) = stack.pop() {
            if !visited.insert(curr.clone()) {
                continue;
            }
            let Some(entry) = self.nodes.get(&curr) else {
                continue;
            };
            if entry.removed_at_ms.is_some() && self.selected_id.as_ref() != Some(&curr) {
                continue;
            }
            out.push(curr);
            for child in entry.children.iter().rev() {
                stack.push(child.clone());
            }
        }
        out
    }

    fn step_visible_selection(&self, forward: bool) -> Option<NodeId> {
        let root_id = self.zoom_root_id.as_ref().or_else(|| self.roots.first())?;
        let target = self.selected_id.as_deref();
        let mut stack = vec![root_id.clone()];
        let mut visited = BTreeSet::new();
        let mut first_seen: Option<NodeId> = None;
        let mut prev_seen: Option<NodeId> = None;
        let mut return_next = false;

        while let Some(curr) = stack.pop() {
            if !visited.insert(curr.clone()) {
                continue;
            }
            let Some(entry) = self.nodes.get(&curr) else {
                continue;
            };
            if entry.removed_at_ms.is_some() && target != Some(curr.as_str()) {
                continue;
            }

            if first_seen.is_none() {
                first_seen = Some(curr.clone());
            }
            if return_next {
                return Some(curr);
            }
            if target.is_none() {
                return Some(curr);
            }
            if target == Some(curr.as_str()) {
                if forward {
                    return_next = true;
                } else {
                    return prev_seen.or(Some(curr));
                }
            }
            prev_seen = Some(curr);

            for child in entry.children.iter().rev() {
                stack.push(child.clone());
            }
        }

        if return_next {
            prev_seen
        } else {
            first_seen
        }
    }

    pub fn coverage_summary(&self) -> CoverageSummary {
        let mut summary = CoverageSummary {
            total_nodes: self.nodes.len(),
            gaps_detected: self.total_gaps_detected,
            ..Default::default()
        };

        for (id, entry) in &self.nodes {
            if entry.removed_at_ms.is_some() {
                summary.removed_nodes += 1;
                continue;
            }
            match self.effective_health(id) {
                NodeHealth::Unknown(_) => {
                    summary.unknown_nodes += 1;
                }
                NodeHealth::Stale { .. } => {
                    summary.observed_nodes += 1;
                    summary.stale_nodes += 1;
                }
                _ => {
                    summary.observed_nodes += 1;
                    summary.fresh_nodes += 1;
                }
            }
        }
        summary
    }

    /// Reference full recomputation of `(depth, subtree_weight)` for verification in tests.
    pub fn recompute_reference_metrics(&self) -> BTreeMap<NodeId, (usize, usize)> {
        let mut result = BTreeMap::new();
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut stack: Vec<(NodeId, usize)> =
            self.roots.iter().rev().map(|r| (r.clone(), 0)).collect();
        let mut visited = BTreeSet::new();

        while let Some((curr_id, depth)) = stack.pop() {
            if !visited.insert(curr_id.clone()) {
                continue;
            }
            let Some(entry) = self.nodes.get(&curr_id) else {
                continue;
            };
            result.insert(curr_id.clone(), (depth, 1));
            order.push(curr_id);
            for child in entry.children.iter().rev() {
                stack.push((child.clone(), depth + 1));
            }
        }

        for node_id in order.into_iter().rev() {
            let children = self
                .nodes
                .get(&node_id)
                .map(|e| e.children.clone())
                .unwrap_or_default();
            let mut weight = 1usize;
            for c in children {
                let cw = result.get(&c).map(|(_, w)| *w).unwrap_or(0);
                weight = weight.saturating_add(cw);
            }
            if let Some(entry_metric) = result.get_mut(&node_id) {
                entry_metric.1 = weight;
            }
        }
        result
    }

    pub fn recompute_all_depths_and_weights(&mut self) {
        let metrics = self.recompute_reference_metrics();
        for (id, (depth, weight)) in metrics {
            if let Some(entry) = self.nodes.get_mut(&id) {
                entry.depth = depth;
                entry.subtree_weight = weight;
            }
        }
    }

    fn would_introduce_cycle(&self, node_id: &str, proposed_parent_id: &str) -> bool {
        if node_id == proposed_parent_id {
            return true;
        }
        let mut curr = proposed_parent_id;
        let mut visited = BTreeSet::new();
        while let Some(entry) = self.nodes.get(curr) {
            if curr == node_id {
                return true;
            }
            if !visited.insert(curr) {
                return true;
            }
            match entry.spec.parent_id.as_deref() {
                Some(next_parent) => {
                    if next_parent == node_id {
                        return true;
                    }
                    curr = next_parent;
                }
                None => break,
            }
        }
        false
    }

    fn proposed_node_depth(&self, parent_id: Option<&str>) -> usize {
        let Some(mut curr) = parent_id else {
            return 0;
        };
        let mut depth = 0usize;
        let mut visited = BTreeSet::new();
        while let Some(entry) = self.nodes.get(curr) {
            depth = depth.saturating_add(1);
            if !visited.insert(curr) {
                break;
            }
            match entry.spec.parent_id.as_deref() {
                Some(p) if self.nodes.contains_key(p) => curr = p,
                _ => break,
            }
        }
        depth
    }

    fn max_descendant_span(&self, node_id: &str) -> usize {
        let existing_children_empty = self
            .nodes
            .get(node_id)
            .map(|e| e.children.is_empty())
            .unwrap_or(true);
        let has_orphan_waiting = self.roots.iter().any(|r| {
            r != node_id
                && self.nodes.get(r).and_then(|n| n.spec.parent_id.as_deref()) == Some(node_id)
        });
        if existing_children_empty && !has_orphan_waiting {
            return 0;
        }

        // Compute max relative depth beneath `node_id`, including any currently orphaned roots
        // that declare `node_id` as their parent.
        let mut max_span = 0usize;
        let mut stack: Vec<(NodeId, usize)> = Vec::new();

        if let Some(entry) = self.nodes.get(node_id) {
            for c in &entry.children {
                stack.push((c.clone(), 1));
            }
        }
        for r in &self.roots {
            if r != node_id
                && self.nodes.get(r).and_then(|n| n.spec.parent_id.as_deref()) == Some(node_id)
            {
                stack.push((r.clone(), 1));
            }
        }

        let mut visited = BTreeSet::new();
        while let Some((curr, rel_d)) = stack.pop() {
            if !visited.insert(curr.clone()) {
                continue;
            }
            max_span = max_span.max(rel_d);
            if let Some(entry) = self.nodes.get(&curr) {
                for c in &entry.children {
                    stack.push((c.clone(), rel_d + 1));
                }
            }
        }
        max_span
    }

    fn add_weight_to_ancestors(&mut self, node_id: &str, delta: usize) {
        let mut curr_parent = self
            .nodes
            .get(node_id)
            .and_then(|n| n.spec.parent_id.clone());
        let mut visited = BTreeSet::new();
        while let Some(p_id) = curr_parent {
            if !visited.insert(p_id.clone()) {
                break;
            }
            if let Some(p_entry) = self.nodes.get_mut(&p_id) {
                p_entry.subtree_weight = p_entry.subtree_weight.saturating_add(delta);
                curr_parent = p_entry.spec.parent_id.clone();
            } else {
                break;
            }
        }
    }
}

/// Escape any ANSI CSI/OSC sequences, control characters (`< 0x20`, `\r`, `\x7f`),
/// or non-ASCII codepoints into visible ASCII representations so logs can never
/// spoof or corrupt the operator terminal.
pub fn sanitize_log_line(raw: &str) -> (String, bool) {
    let (s, esc, _) = sanitize_log_line_bounded(raw, usize::MAX);
    (s, esc)
}

/// Escape dangerous characters and enforce a strict byte bound (`max_bytes`)
/// AFTER escape expansion so a payload of control characters cannot blow up memory.
pub fn sanitize_log_line_bounded(raw: &str, max_bytes: usize) -> (String, bool, bool) {
    let cap = max_bytes.min(raw.len().saturating_mul(2).max(64));
    let mut out = String::with_capacity(cap);
    let mut escaped = false;
    let mut truncated = false;

    for ch in raw.chars() {
        let piece: String = match ch {
            '\x1b' => {
                escaped = true;
                "^[".to_string()
            }
            '\r' => {
                escaped = true;
                "\\r".to_string()
            }
            '\n' => {
                escaped = true;
                "\\n".to_string()
            }
            '\t' => {
                escaped = true;
                "  ".to_string()
            }
            '\x00' => {
                escaped = true;
                "\\0".to_string()
            }
            c if c.is_ascii_control() => {
                escaped = true;
                format!("\\x{:02x}", c as u32)
            }
            c if c.is_ascii() => c.to_string(),
            c => {
                escaped = true;
                format!("\\u{{{:x}}}", c as u32)
            }
        };

        if out.len().saturating_add(piece.len()) > max_bytes {
            truncated = true;
            break;
        }
        out.push_str(&piece);
    }

    (out, escaped, truncated)
}

pub fn action_from_key(key: KeyEvent) -> Option<ConsoleAction> {
    // On Windows, Crossterm emits both KeyEventKind::Press and KeyEventKind::Release.
    // Never trigger actions on key release.
    if key.kind == KeyEventKind::Release {
        return None;
    }
    let is_press_only = key.kind == KeyEventKind::Press;

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return is_press_only.then_some(ConsoleAction::Quit);
    }
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => is_press_only.then_some(ConsoleAction::Quit),
        KeyCode::Down | KeyCode::Char('j') => Some(ConsoleAction::SelectNextVisible),
        KeyCode::Up | KeyCode::Char('k') => Some(ConsoleAction::SelectPrevVisible),
        KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('z') => {
            is_press_only.then_some(ConsoleAction::ZoomIn)
        }
        KeyCode::Backspace | KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('b') => {
            is_press_only.then_some(ConsoleAction::Back)
        }
        KeyCode::Home | KeyCode::Char('0') => is_press_only.then_some(ConsoleAction::ResetZoom),
        KeyCode::Char('f') => is_press_only.then_some(ConsoleAction::ToggleFollowLogs),
        KeyCode::PageDown => Some(ConsoleAction::ScrollLogs(5)),
        KeyCode::PageUp => Some(ConsoleAction::ScrollLogs(-5)),
        _ => None,
    }
}

pub fn action_from_voice(intent: &VoiceIntent) -> Result<ConsoleAction, ActionParseError> {
    match intent {
        VoiceIntent::SelectNode { target_id } => {
            if target_id.trim().is_empty() {
                Err(ActionParseError::MissingField("target_id"))
            } else {
                Ok(ConsoleAction::Select(target_id.clone()))
            }
        }
        VoiceIntent::NextNode => Ok(ConsoleAction::SelectNextVisible),
        VoiceIntent::PreviousNode => Ok(ConsoleAction::SelectPrevVisible),
        VoiceIntent::ZoomBranch => Ok(ConsoleAction::ZoomIn),
        VoiceIntent::ZoomToNode { target_id } => {
            if target_id.trim().is_empty() {
                Err(ActionParseError::MissingField("target_id"))
            } else {
                Ok(ConsoleAction::ZoomTo(target_id.clone()))
            }
        }
        VoiceIntent::GoParent => Ok(ConsoleAction::Back),
        VoiceIntent::ResetRoot => Ok(ConsoleAction::ResetZoom),
        VoiceIntent::FollowLogs => Ok(ConsoleAction::ToggleFollowLogs),
        VoiceIntent::ScrollLogs { delta } => Ok(ConsoleAction::ScrollLogs(*delta)),
        VoiceIntent::TickClock { now_ms } => Ok(ConsoleAction::Tick { now_ms: *now_ms }),
        VoiceIntent::QuitConsole => Ok(ConsoleAction::Quit),
    }
}

#[derive(Deserialize)]
struct JsonCommandEnvelope {
    action: String,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    delta: Option<i32>,
    #[serde(default)]
    now_ms: Option<u64>,
}

/// In-memory JSON action adapter.
/// Note: This is strictly an in-memory parser for testing action equivalence;
/// it is NOT a live WebSocket endpoint.
pub fn action_from_json(json: &str) -> Result<ConsoleAction, ActionParseError> {
    let env: JsonCommandEnvelope =
        serde_json::from_str(json).map_err(|e| ActionParseError::InvalidJson(e.to_string()))?;

    match env.action.as_str() {
        "select" => {
            let id = env
                .node_id
                .filter(|s| !s.trim().is_empty())
                .ok_or(ActionParseError::MissingField("node_id"))?;
            Ok(ConsoleAction::Select(id))
        }
        "select_next" | "next" => Ok(ConsoleAction::SelectNextVisible),
        "select_prev" | "prev" => Ok(ConsoleAction::SelectPrevVisible),
        "zoom_in" | "zoom" => Ok(ConsoleAction::ZoomIn),
        "zoom_to" => {
            let id = env
                .node_id
                .filter(|s| !s.trim().is_empty())
                .ok_or(ActionParseError::MissingField("node_id"))?;
            Ok(ConsoleAction::ZoomTo(id))
        }
        "back" | "parent" => Ok(ConsoleAction::Back),
        "reset_zoom" | "root" => Ok(ConsoleAction::ResetZoom),
        "toggle_follow" | "follow" => Ok(ConsoleAction::ToggleFollowLogs),
        "scroll_logs" => Ok(ConsoleAction::ScrollLogs(env.delta.unwrap_or(1))),
        "tick" => Ok(ConsoleAction::Tick {
            now_ms: env.now_ms.ok_or(ActionParseError::MissingField("now_ms"))?,
        }),
        "quit" => Ok(ConsoleAction::Quit),
        other => Err(ActionParseError::UnknownCommand(other.to_string())),
    }
}
