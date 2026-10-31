#![allow(dead_code)]
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// The most levels the icicle shows under its zoom root, the root included.
pub const VISIBLE_LEVELS: usize = 4;

pub type NodeId = String;
pub type IncarnationId = String;

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
    pub text: String,
    pub was_escaped: bool,
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
    pub logs: VecDeque<SanitizedLogLine>,
    pub evicted_log_count: u64,
    pub log_scroll_offset: usize,
    pub gap_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsoleAction {
    Select(NodeId),
    SelectNextVisible,
    SelectPrevVisible,
    SelectChild,
    SelectParent,
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
    ChildNode,
    ParentNode,
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
    AppendLog {
        node_id: NodeId,
        seq: u64,
        timestamp_ms: u64,
        raw_text: String,
    },
    RecordGap {
        node_id: NodeId,
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
    pub max_logs_per_node: usize,
    pub total_gaps_detected: usize,
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
            max_logs_per_node: max_logs_per_node.max(1),
            total_gaps_detected: 0,
        }
    }

    /// Apply a feed event without ever shifting `selected_id` or `zoom_root_id`
    /// once they have been established.
    pub fn apply_feed(&mut self, event: FeedEvent) {
        match event {
            FeedEvent::UpsertNode(mut spec) => {
                if let Some(p) = spec.progress_bp {
                    spec.progress_bp = Some(p.min(10_000));
                }
                if let Some(q) = spec.quality_bp {
                    spec.quality_bp = Some(q.min(10_000));
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
                        logs: VecDeque::new(),
                        evicted_log_count: 0,
                        log_scroll_offset: 0,
                        gap_count: 0,
                    };
                    self.nodes.insert(node_id.clone(), entry);
                }

                if is_new_node || old_parent != new_parent {
                    // Detach from old parent or roots if needed
                    if let Some(prev_p) = old_parent {
                        if let Some(p_entry) = self.nodes.get_mut(&prev_p) {
                            p_entry.children.retain(|c| c != &node_id);
                        }
                    } else if !is_new_node {
                        self.roots.retain(|r| r != &node_id);
                    }

                    // Attach to new parent or roots in deterministic sorted order by NodeId
                    let attached_to_parent = if let Some(ref p_id) = new_parent {
                        if p_id != &node_id && self.nodes.contains_key(p_id) {
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

                    if !attached_to_parent && !self.roots.contains(&node_id) {
                        self.roots.push(node_id.clone());
                        self.roots.sort();
                    }

                    // Reattach any existing orphans that declared `node_id` as their parent
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

                    if !orphan_roots.is_empty() {
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

                    self.refresh_depths_from(&node_id);
                    if is_new_node {
                        self.adjust_ancestor_weights(&node_id, 1);
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
                self.now_ms = self.now_ms.max(timestamp_ms);
                if let Some(entry) = self.nodes.get_mut(&id) {
                    entry.removed_at_ms = Some(timestamp_ms);
                }
                // Intentionally never move `selected_id` or `zoom_root_id`:
                // the operator's reading focus stays anchored on the tombstoned identity.
            }
            FeedEvent::AppendLog {
                node_id,
                seq,
                timestamp_ms,
                raw_text,
            } => {
                self.now_ms = self.now_ms.max(timestamp_ms);
                let (sanitized, was_escaped) = sanitize_log_line(&raw_text);
                let max_cap = self.max_logs_per_node;
                let follow = self.follow_logs;
                if let Some(entry) = self.nodes.get_mut(&node_id) {
                    if entry.logs.len() >= max_cap {
                        entry.logs.pop_front();
                        entry.evicted_log_count = entry.evicted_log_count.saturating_add(1);
                    }
                    entry.logs.push_back(SanitizedLogLine {
                        seq,
                        timestamp_ms,
                        text: sanitized,
                        was_escaped,
                    });
                    if follow {
                        entry.log_scroll_offset = entry.logs.len().saturating_sub(1);
                    }
                }
            }
            FeedEvent::RecordGap {
                node_id,
                missed_seq,
                timestamp_ms,
            } => {
                self.now_ms = self.now_ms.max(timestamp_ms);
                if let Some(entry) = self.nodes.get_mut(&node_id) {
                    entry.spec.reported_health =
                        NodeHealth::Unknown(UnknownReason::GapDetected { missed_seq });
                    entry.gap_count = entry.gap_count.saturating_add(1);
                    self.total_gaps_detected = self.total_gaps_detected.saturating_add(1);
                }
            }
            FeedEvent::AdvanceClock { now_ms } => {
                if now_ms > self.now_ms {
                    self.now_ms = now_ms;
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
                let order = self.visible_navigation_order(64);
                if order.is_empty() {
                    return;
                }
                let next_id = match self.selected_id.as_ref() {
                    Some(curr) => match order.iter().position(|x| x == curr) {
                        Some(idx) => order[(idx + 1).min(order.len() - 1)].clone(),
                        None => order[0].clone(),
                    },
                    None => order[0].clone(),
                };
                self.selected_id = Some(next_id);
            }
            ConsoleAction::SelectPrevVisible => {
                let order = self.visible_navigation_order(64);
                if order.is_empty() {
                    return;
                }
                let prev_id = match self.selected_id.as_ref() {
                    Some(curr) => match order.iter().position(|x| x == curr) {
                        Some(idx) => order[idx.saturating_sub(1)].clone(),
                        None => order[0].clone(),
                    },
                    None => order[0].clone(),
                };
                self.selected_id = Some(prev_id);
            }
            ConsoleAction::SelectChild => self.select_child(),
            ConsoleAction::SelectParent => self.select_parent(),
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
            }
            ConsoleAction::ScrollLogs(delta) => {
                let Some(sel) = self.selected_id.as_ref() else {
                    return;
                };
                if let Some(entry) = self.nodes.get_mut(sel) {
                    let len = entry.logs.len();
                    if len == 0 {
                        entry.log_scroll_offset = 0;
                    } else if delta < 0 {
                        entry.log_scroll_offset = entry
                            .log_scroll_offset
                            .saturating_sub(delta.unsigned_abs() as usize);
                    } else {
                        entry.log_scroll_offset =
                            (entry.log_scroll_offset + delta as usize).min(len - 1);
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

    /// Selects the selected node's first child still present. When that child would sit past
    /// the levels the icicle shows, the zoom moves down to the node just left, so the new
    /// selection stays in view. A leaf keeps the selection.
    fn select_child(&mut self) {
        let Some(sel) = self.selected_id.clone() else {
            return;
        };
        let Some(entry) = self.nodes.get(&sel) else {
            return;
        };
        let Some(child) = entry
            .children
            .iter()
            .find(|c| {
                self.nodes
                    .get(*c)
                    .is_some_and(|n| n.removed_at_ms.is_none())
            })
            .cloned()
        else {
            return;
        };
        let zoom_depth = self
            .zoom_root_id
            .as_ref()
            .and_then(|z| self.nodes.get(z))
            .map_or(0, |z| z.depth);
        let child_depth = self.nodes[&child].depth;
        if child_depth.saturating_sub(zoom_depth) >= VISIBLE_LEVELS {
            self.zoom_root_id = Some(sel);
        }
        self.selected_id = Some(child);
    }

    /// Selects the selected node's parent. Leaving the zoom root upward zooms out with it; a
    /// node without a parent keeps the selection.
    fn select_parent(&mut self) {
        let Some(sel) = self.selected_id.clone() else {
            return;
        };
        let Some(parent) = self
            .nodes
            .get(&sel)
            .and_then(|n| n.spec.parent_id.clone())
            .filter(|p| self.nodes.contains_key(p))
        else {
            return;
        };
        if self.zoom_root_id.as_ref() == Some(&sel) {
            self.zoom_root_id = Some(parent.clone());
        }
        self.selected_id = Some(parent);
    }

    /// Breadcrumb chain from the top ancestor down to `zoom_root_id`.
    /// Invariant across arbitrary depths.
    pub fn breadcrumbs(&self) -> Vec<NodeId> {
        let Some(mut curr) = self.zoom_root_id.clone() else {
            return Vec::new();
        };
        let mut chain = Vec::new();
        let mut guard = 0usize;
        while let Some(entry) = self.nodes.get(&curr) {
            chain.push(curr.clone());
            guard += 1;
            if guard > 256 {
                break;
            }
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

    /// Bounded pre-order traversal under `zoom_root_id` for keyboard navigation.
    pub fn visible_navigation_order(&self, max_nodes: usize) -> Vec<NodeId> {
        let mut out = Vec::new();
        let Some(root_id) = self.zoom_root_id.as_ref().or_else(|| self.roots.first()) else {
            return out;
        };
        let mut stack = vec![root_id.clone()];
        while let Some(curr) = stack.pop() {
            if out.len() >= max_nodes {
                break;
            }
            let Some(entry) = self.nodes.get(&curr) else {
                continue;
            };
            // Keep tombstoned selected node navigable if it is the current selection,
            // otherwise skip removed nodes in normal traversal.
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

    fn refresh_depths_from(&mut self, start_id: &str) {
        let base_depth = self
            .nodes
            .get(start_id)
            .and_then(|n| n.spec.parent_id.as_ref())
            .and_then(|p| self.nodes.get(p))
            .map(|p| p.depth + 1)
            .unwrap_or(0);

        let mut stack = vec![(start_id.to_string(), base_depth)];
        while let Some((curr_id, depth)) = stack.pop() {
            let children = if let Some(entry) = self.nodes.get_mut(&curr_id) {
                entry.depth = depth;
                entry.children.clone()
            } else {
                continue;
            };
            if depth < 256 {
                for child in children {
                    stack.push((child, depth + 1));
                }
            }
        }
    }

    fn adjust_ancestor_weights(&mut self, node_id: &str, delta: usize) {
        let mut curr_parent = self
            .nodes
            .get(node_id)
            .and_then(|n| n.spec.parent_id.clone());
        let mut guard = 0usize;
        while let Some(p_id) = curr_parent {
            guard += 1;
            if guard > 256 {
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
    let mut out = String::with_capacity(raw.len());
    let mut escaped = false;
    for ch in raw.chars() {
        match ch {
            '\x1b' => {
                out.push_str("^[");
                escaped = true;
            }
            '\r' => {
                out.push_str("\\r");
                escaped = true;
            }
            '\n' => {
                out.push_str("\\n");
                escaped = true;
            }
            '\t' => {
                out.push_str("  ");
                escaped = true;
            }
            '\x00' => {
                out.push_str("\\0");
                escaped = true;
            }
            c if c.is_ascii_control() => {
                out.push_str(&format!("\\x{:02x}", c as u32));
                escaped = true;
            }
            c if c.is_ascii() => {
                out.push(c);
            }
            c => {
                out.push_str(&format!("\\u{{{:x}}}", c as u32));
                escaped = true;
            }
        }
    }
    (out, escaped)
}

pub fn action_from_key(key: KeyEvent) -> Option<ConsoleAction> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(ConsoleAction::Quit);
    }
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => Some(ConsoleAction::Quit),
        KeyCode::Down | KeyCode::Char('j') => Some(ConsoleAction::SelectNextVisible),
        KeyCode::Up | KeyCode::Char('k') => Some(ConsoleAction::SelectPrevVisible),
        KeyCode::Right | KeyCode::Char('l') => Some(ConsoleAction::SelectChild),
        KeyCode::Left | KeyCode::Char('h') => Some(ConsoleAction::SelectParent),
        KeyCode::Enter | KeyCode::Char('z') => Some(ConsoleAction::ZoomIn),
        KeyCode::Backspace | KeyCode::Char('b') => Some(ConsoleAction::Back),
        KeyCode::Home | KeyCode::Char('0') => Some(ConsoleAction::ResetZoom),
        KeyCode::Char('f') => Some(ConsoleAction::ToggleFollowLogs),
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
        VoiceIntent::ChildNode => Ok(ConsoleAction::SelectChild),
        VoiceIntent::ParentNode => Ok(ConsoleAction::SelectParent),
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
        "select_child" | "child" => Ok(ConsoleAction::SelectChild),
        "select_parent" => Ok(ConsoleAction::SelectParent),
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
