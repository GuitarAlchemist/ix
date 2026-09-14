//! Work-history distillation: PR lifecycles as Markov chains (Demerzel#596).
//!
//! Turns PR timeline records (JSONL from `scripts/fetch-pr-lifecycle.sh`)
//! into lifecycle state sequences using the Seldon state vocabulary from
//! `Demerzel/schemas/seldon/markov-transition.schema.json`, then:
//!
//! - fits a transition table shaped like that schema ([`transition_table`]);
//! - runs absorbing-state analysis on the first-order chain ([`absorption`]);
//! - scores next-state prediction on a time-split held-out set, comparing
//!   the variable-order model ([`VariableOrderSelector`]) against order-0 and
//!   first-order baselines ([`held_out`]).
//!
//! Everything is advisory and deterministic: no clock (the caller passes
//! `as_of`), no randomness, no GitHub writes.
//!
//! A stuck state is inserted whenever a PR sits in a non-terminal state for
//! longer than `stale_after_secs`, before its next event, or before `as_of`
//! if it has none. It keeps the PR's origin: `pr.stuck_draft` (GitHub cannot
//! merge a draft) or `pr.stuck_ready`. A single `issue.stuck` forgot that
//! origin, and a variable-order model only looked better because it could
//! recover it from the previous state. Stuck is *not* absorbing: a stuck PR
//! can still be merged or closed later.

use crate::error::{MemristiveError, Result};
use crate::tensor::MarkovTensor;
use crate::vlmm::{FallbackStrategy, VariableOrderSelector};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Seldon lifecycle states: the schema's enum order, then the two
/// origin-aware stuck states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LifecycleState {
    IssueGrooming = 0,
    IssueReady = 1,
    IssueDelegated = 2,
    PrDraft = 3,
    PrReadyForReview = 4,
    PrMergeCandidate = 5,
    PrMerged = 6,
    PrRejected = 7,
    IssueStuck = 8,
    PrStuckDraft = 9,
    PrStuckReady = 10,
}

impl LifecycleState {
    pub const COUNT: usize = 11;
    pub const ALL: [LifecycleState; Self::COUNT] = [
        LifecycleState::IssueGrooming,
        LifecycleState::IssueReady,
        LifecycleState::IssueDelegated,
        LifecycleState::PrDraft,
        LifecycleState::PrReadyForReview,
        LifecycleState::PrMergeCandidate,
        LifecycleState::PrMerged,
        LifecycleState::PrRejected,
        LifecycleState::IssueStuck,
        LifecycleState::PrStuckDraft,
        LifecycleState::PrStuckReady,
    ];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn label(self) -> &'static str {
        match self {
            LifecycleState::IssueGrooming => "issue.grooming",
            LifecycleState::IssueReady => "issue.ready",
            LifecycleState::IssueDelegated => "issue.delegated",
            LifecycleState::PrDraft => "pr.draft",
            LifecycleState::PrReadyForReview => "pr.ready_for_review",
            LifecycleState::PrMergeCandidate => "pr.merge_candidate",
            LifecycleState::PrMerged => "pr.merged",
            LifecycleState::PrRejected => "pr.rejected",
            LifecycleState::IssueStuck => "issue.stuck",
            LifecycleState::PrStuckDraft => "pr.stuck_draft",
            LifecycleState::PrStuckReady => "pr.stuck_ready",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, LifecycleState::PrMerged | LifecycleState::PrRejected)
    }

    pub fn is_stuck(self) -> bool {
        matches!(
            self,
            LifecycleState::IssueStuck
                | LifecycleState::PrStuckDraft
                | LifecycleState::PrStuckReady
        )
    }

    /// The stuck state a PR enters after sitting too long in `self`.
    fn stuck_from(self) -> LifecycleState {
        match self {
            LifecycleState::PrDraft | LifecycleState::PrStuckDraft => LifecycleState::PrStuckDraft,
            LifecycleState::PrReadyForReview
            | LifecycleState::PrMergeCandidate
            | LifecycleState::PrStuckReady => LifecycleState::PrStuckReady,
            _ => LifecycleState::IssueStuck,
        }
    }
}

/// One lifecycle-relevant timeline item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrEvent {
    pub kind: String,
    pub at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// One PR as emitted by `scripts/fetch-pr-lifecycle.sh`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrRecord {
    pub repo: String,
    pub number: u64,
    pub created_at: String,
    pub state: String,
    pub is_draft: bool,
    pub head_ref: String,
    pub events: Vec<PrEvent>,
}

impl PrRecord {
    /// Worker inferred from the branch prefix (`codex/`, `claude/`, `jules…`).
    /// Most branches carry no worker marker; those are `unattributed`.
    pub fn worker(&self) -> &'static str {
        let r = self.head_ref.as_str();
        if r.starts_with("codex/") {
            "codex"
        } else if r.starts_with("claude/") {
            "claude"
        } else if r.starts_with("jules") {
            "jules"
        } else {
            "unattributed"
        }
    }
}

/// A state entered at `at` (Unix seconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    pub state: LifecycleState,
    pub at: i64,
}

/// Parse JSONL, skipping blank lines.
pub fn parse_jsonl(text: &str) -> Result<Vec<PrRecord>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(MemristiveError::from))
        .collect()
}

/// Parse GitHub's `YYYY-MM-DDTHH:MM:SSZ` into Unix seconds.
pub fn parse_timestamp(s: &str) -> Result<i64> {
    let bad = || MemristiveError::InvalidConfig(format!("bad timestamp: {s}"));
    let b = s.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[19] != b'Z' {
        return Err(bad());
    }
    let num = |r: std::ops::Range<usize>| s[r].parse::<i64>().map_err(|_| bad());
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (hh, mm, ss) = (num(11..13)?, num(14..16)?, num(17..19)?);
    // Howard Hinnant's days_from_civil.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Ok(days * 86_400 + hh * 3_600 + mm * 60 + ss)
}

const MERGE_CANDIDATE_LABELS: [&str; 2] = ["fleet:merge-ready", "agent-blackbox-reviewed"];

/// Lifecycle of one PR as seen at `as_of`: events after `as_of` are ignored,
/// consecutive repeats collapse, and a stuck state marks stale gaps.
///
/// A merge-candidate signal (label or approval) is sticky. On a draft it
/// creates no state: a draft cannot merge, so the PR becomes
/// `pr.merge_candidate` only when it is flipped ready.
pub fn lifecycle(pr: &PrRecord, as_of: i64, stale_after_secs: i64) -> Result<Vec<Step>> {
    let created = parse_timestamp(&pr.created_at)?;
    if created > as_of {
        return Ok(Vec::new());
    }
    let mut events = Vec::with_capacity(pr.events.len());
    for e in &pr.events {
        let at = parse_timestamp(&e.at)?;
        if at <= as_of {
            events.push((at, e));
        }
    }
    // Stable: equal timestamps keep GitHub's order.
    events.sort_by_key(|(at, _)| *at);

    // A PR's first draft flip tells how it was opened; with no flip the
    // current draft flag is the opening one.
    let opened_draft = pr
        .events
        .iter()
        .find(|e| e.kind == "ready_for_review" || e.kind == "convert_to_draft")
        .map_or(pr.is_draft, |e| e.kind == "ready_for_review");
    let opening = if opened_draft {
        LifecycleState::PrDraft
    } else {
        LifecycleState::PrReadyForReview
    };

    let mut raw = vec![Step {
        state: opening,
        at: created,
    }];
    let mut last_open = opening;
    let mut candidate = false;
    for (at, e) in events {
        let state = match e.kind.as_str() {
            "ready_for_review" if candidate => LifecycleState::PrMergeCandidate,
            "ready_for_review" => LifecycleState::PrReadyForReview,
            "convert_to_draft" => LifecycleState::PrDraft,
            "approved" | "labeled"
                if e.kind == "approved"
                    || e.label
                        .as_deref()
                        .is_some_and(|l| MERGE_CANDIDATE_LABELS.contains(&l)) =>
            {
                candidate = true;
                if last_open == LifecycleState::PrDraft {
                    continue;
                }
                LifecycleState::PrMergeCandidate
            }
            "merged" => LifecycleState::PrMerged,
            "closed" => LifecycleState::PrRejected,
            "reopened" => last_open,
            _ => continue,
        };
        if !state.is_terminal() {
            last_open = state;
        }
        raw.push(Step { state, at });
    }

    let mut steps: Vec<Step> = Vec::with_capacity(raw.len() + 1);
    for step in raw {
        if let Some(prev) = steps.last() {
            if prev.state == step.state {
                continue;
            }
            if !prev.state.is_terminal() && step.at - prev.at > stale_after_secs {
                steps.push(Step {
                    state: prev.state.stuck_from(),
                    at: prev.at + stale_after_secs,
                });
            }
        }
        steps.push(step);
    }
    if let Some(last) = steps.last().copied() {
        if !last.state.is_terminal() && !last.state.is_stuck() && as_of - last.at > stale_after_secs
        {
            steps.push(Step {
                state: last.state.stuck_from(),
                at: last.at + stale_after_secs,
            });
        }
    }
    Ok(steps)
}

/// One row of the Seldon `markov-transition` schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkovTransition {
    pub from_state: String,
    pub to_state: String,
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_duration_hours: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    pub sample_size: u64,
}

/// First-order transition table over `sequences`, sorted by (from, to).
/// `sample_size` is the count of that transition; `probability` divides by
/// all transitions leaving `from_state`. Rows into a stuck state carry no
/// median: the stuck step is placed at the stale threshold, so its duration
/// is the threshold, not a measurement.
pub fn transition_table(sequences: &[Vec<Step>], worker: Option<&str>) -> Vec<MarkovTransition> {
    let n = LifecycleState::COUNT;
    let mut hours: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n]; n];
    for seq in sequences {
        for w in seq.windows(2) {
            hours[w[0].state.index()][w[1].state.index()]
                .push((w[1].at - w[0].at) as f64 / 3_600.0);
        }
    }
    let mut rows = Vec::new();
    for from in LifecycleState::ALL {
        let out: usize = hours[from.index()].iter().map(Vec::len).sum();
        for to in LifecycleState::ALL {
            let durations = &mut hours[from.index()][to.index()];
            if durations.is_empty() {
                continue;
            }
            rows.push(MarkovTransition {
                from_state: from.label().to_string(),
                to_state: to.label().to_string(),
                probability: durations.len() as f64 / out as f64,
                median_duration_hours: (!to.is_stuck()).then(|| median(durations)),
                worker: worker.map(str::to_string),
                sample_size: durations.len() as u64,
            });
        }
    }
    rows
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Eventual-outcome probabilities from one state under the first-order chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Absorption {
    pub state: String,
    pub p_merged: f64,
    pub p_rejected: f64,
    /// Mass that neither merges nor closes: PRs still open at `as_of`. This
    /// is right-censoring, not an outcome.
    pub p_unresolved: f64,
    /// `p_merged / (p_merged + p_rejected)`: the merge rate among PRs that
    /// did resolve. Absent when none resolved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_merged_given_resolved: Option<f64>,
    /// Transitions observed leaving this state.
    pub sample_size: u64,
}

/// Absorbing-state analysis: `pr.merged` and `pr.rejected` absorb; a state
/// whose sequences end open keeps that mass as `p_unresolved`. Solved by
/// fixed-point iteration (the chain is small and substochastic).
pub fn absorption(sequences: &[Vec<Step>]) -> Vec<Absorption> {
    let n = LifecycleState::COUNT;
    let mut counts = vec![vec![0.0f64; n]; n];
    let mut visits = vec![0.0f64; n];
    for seq in sequences {
        for (i, s) in seq.iter().enumerate() {
            visits[s.state.index()] += 1.0;
            if let Some(next) = seq.get(i + 1) {
                counts[s.state.index()][next.state.index()] += 1.0;
            }
        }
    }
    let merged = LifecycleState::PrMerged.index();
    let rejected = LifecycleState::PrRejected.index();
    let mut pm = vec![0.0f64; n];
    let mut pr = vec![0.0f64; n];
    pm[merged] = 1.0;
    pr[rejected] = 1.0;
    for _ in 0..10_000 {
        let mut delta = 0.0f64;
        for s in 0..n {
            if s == merged || s == rejected || visits[s] == 0.0 {
                continue;
            }
            let (mut m, mut r) = (0.0, 0.0);
            for t in 0..n {
                let p = counts[s][t] / visits[s];
                m += p * pm[t];
                r += p * pr[t];
            }
            delta = delta.max((m - pm[s]).abs()).max((r - pr[s]).abs());
            pm[s] = m;
            pr[s] = r;
        }
        if delta < 1e-12 {
            break;
        }
    }
    LifecycleState::ALL
        .iter()
        .filter(|s| !s.is_terminal() && visits[s.index()] > 0.0)
        .map(|&s| {
            let i = s.index();
            let resolved = pm[i] + pr[i];
            Absorption {
                state: s.label().to_string(),
                p_merged: pm[i],
                p_rejected: pr[i],
                p_unresolved: (1.0 - resolved).max(0.0),
                p_merged_given_resolved: (resolved > 0.0).then(|| pm[i] / resolved),
                sample_size: counts[i].iter().sum::<f64>() as u64,
            }
        })
        .collect()
}

/// Uniform-mixture weights the held-out log-loss is reported at. Log-loss
/// on this data is decided by rare unseen transitions, so one ε would pick
/// the winner; accuracy does not depend on ε.
pub const SMOOTHING_SWEEP: [f64; 3] = [1e-6, 1e-3, 1e-2];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpsilonLoss {
    pub epsilon: f64,
    /// Mean negative log-likelihood (nats).
    pub log_loss: f64,
}

/// Next-state prediction quality on held-out transitions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelScore {
    pub model: String,
    /// Share of transitions whose argmax prediction was right. The primary
    /// metric: it does not depend on the smoothing constant.
    pub accuracy: f64,
    /// One entry per [`SMOOTHING_SWEEP`] value.
    pub log_loss: Vec<EpsilonLoss>,
}

/// VLMM vs first-order disagreements, and a sign test that does not treat
/// same-batch transitions as independent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Disagreement {
    /// Transitions only VLMM got right.
    pub vlmm_only_correct: usize,
    /// Transitions only first-order got right.
    pub first_order_only_correct: usize,
    /// Disagreements grouped by (repo, UTC day of the transition): PRs
    /// merged in one scripted batch are one decision, not many.
    pub clusters_favoring_vlmm: usize,
    pub clusters_favoring_first_order: usize,
    /// Exact two-sided sign test over the non-tied clusters (1.0 if none).
    pub cluster_sign_test_p: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeldOutReport {
    /// Train = PRs opened strictly before this instant, censored at it;
    /// test = PRs opened at or after it.
    pub cutoff: i64,
    pub train_sequences: usize,
    pub test_sequences: usize,
    pub test_transitions: usize,
    pub max_order: usize,
    pub min_observations: usize,
    pub scores: Vec<ModelScore>,
    pub disagreement: Disagreement,
    /// How often VLMM used each order (index = order) on the test set.
    pub vlmm_order_histogram: Vec<u64>,
}

/// Time-split held-out check. The cutoff is the creation time of the PR at
/// `train_fraction` of the creation order. Train PRs (opened before it) have
/// their later events hidden, so no future leaks in; test PRs are scored with
/// full history up to `as_of`.
pub fn held_out(
    prs: &[PrRecord],
    as_of: i64,
    stale_after_secs: i64,
    train_fraction: f64,
    max_order: usize,
    min_observations: usize,
) -> Result<HeldOutReport> {
    let mut dated = Vec::with_capacity(prs.len());
    for pr in prs {
        dated.push((parse_timestamp(&pr.created_at)?, pr));
    }
    dated.sort_by_key(|(t, _)| *t);
    let split = ((dated.len() as f64) * train_fraction) as usize;
    let empty = || {
        MemristiveError::InvalidConfig(format!(
            "train_fraction {train_fraction} leaves an empty side of {} PRs",
            dated.len()
        ))
    };
    if split >= dated.len() {
        return Err(empty());
    }
    let cutoff = dated[split].0;
    // Split on time, not index, so PRs tied with the cutoff land in test.
    let first_test = dated.partition_point(|(t, _)| *t < cutoff);
    if first_test == 0 {
        return Err(empty());
    }

    let lifecycle_at = |pr: &PrRecord, at: i64| lifecycle(pr, at, stale_after_secs);

    let mut order0 = MarkovTensor::new(0);
    let mut order1 = MarkovTensor::new(1);
    let mut vlmm_tensor = MarkovTensor::new(max_order);
    for (_, pr) in &dated[..first_test] {
        let seq: Vec<usize> = lifecycle_at(pr, cutoff)?
            .iter()
            .map(|s| s.state.index())
            .collect();
        for i in 1..seq.len() {
            order0.observe(&[], seq[i]);
            order1.observe(&seq[i - 1..i], seq[i]);
            vlmm_tensor.observe(&seq[..i], seq[i]);
        }
    }

    let mut vlmm = VariableOrderSelector::new(
        max_order,
        min_observations,
        FallbackStrategy::MarginalDistribution,
    );
    let mut nll = [[0.0f64; SMOOTHING_SWEEP.len()]; 3];
    let mut hits = [0usize; 3];
    let (mut first_order_only, mut vlmm_only) = (0usize, 0usize);
    let mut clusters: BTreeMap<(&str, i64), i64> = BTreeMap::new();
    let mut transitions = 0usize;
    for (_, pr) in &dated[first_test..] {
        let steps = lifecycle_at(pr, as_of)?;
        let seq: Vec<usize> = steps.iter().map(|s| s.state.index()).collect();
        for i in 1..seq.len() {
            let actual = seq[i];
            let first = {
                let d = order1.predict(&seq[i - 1..i]);
                if d.is_empty() {
                    order1.predict(&[])
                } else {
                    d
                }
            };
            let dists = [
                order0.predict(&[]),
                first,
                vlmm.predict(&vlmm_tensor, &seq[..i]),
            ];
            let mut right = [false; 3];
            for (m, d) in dists.iter().enumerate() {
                for (k, &eps) in SMOOTHING_SWEEP.iter().enumerate() {
                    nll[m][k] -= smooth(d, eps)[actual].ln();
                }
                right[m] = argmax(&smooth(d, SMOOTHING_SWEEP[0])) == actual;
                if right[m] {
                    hits[m] += 1;
                }
            }
            let day = steps[i].at.div_euclid(86_400);
            match (right[1], right[2]) {
                (true, false) => {
                    first_order_only += 1;
                    *clusters.entry((pr.repo.as_str(), day)).or_default() -= 1;
                }
                (false, true) => {
                    vlmm_only += 1;
                    *clusters.entry((pr.repo.as_str(), day)).or_default() += 1;
                }
                _ => {}
            }
            transitions += 1;
        }
    }
    let denom = transitions.max(1) as f64;
    let scores = ["order0_marginal", "first_order", "vlmm"]
        .iter()
        .enumerate()
        .map(|(m, name)| ModelScore {
            model: name.to_string(),
            accuracy: hits[m] as f64 / denom,
            log_loss: SMOOTHING_SWEEP
                .iter()
                .enumerate()
                .map(|(k, &epsilon)| EpsilonLoss {
                    epsilon,
                    log_loss: nll[m][k] / denom,
                })
                .collect(),
        })
        .collect();
    let favor_vlmm = clusters.values().filter(|&&v| v > 0).count();
    let favor_first = clusters.values().filter(|&&v| v < 0).count();
    Ok(HeldOutReport {
        cutoff,
        train_sequences: first_test,
        test_sequences: dated.len() - first_test,
        test_transitions: transitions,
        max_order,
        min_observations,
        scores,
        disagreement: Disagreement {
            vlmm_only_correct: vlmm_only,
            first_order_only_correct: first_order_only,
            clusters_favoring_vlmm: favor_vlmm,
            clusters_favoring_first_order: favor_first,
            cluster_sign_test_p: sign_test_p(favor_vlmm, favor_first),
        },
        vlmm_order_histogram: vlmm.order_histogram().to_vec(),
    })
}

/// Exact two-sided binomial sign test, p = 0.5 under the null.
fn sign_test_p(a: usize, b: usize) -> f64 {
    let n = a + b;
    if n == 0 {
        return 1.0;
    }
    let mut coef = 1.0f64; // C(n, 0)
    let mut tail = 0.0f64;
    for k in 0..=a.min(b) {
        if k > 0 {
            coef = coef * (n - k + 1) as f64 / k as f64;
        }
        tail += coef;
    }
    (2.0 * tail / 2f64.powi(n as i32)).min(1.0)
}

/// Draft→ready flips followed by a merge within a minute: a scripted flip,
/// not a review. Durations out of `pr.ready_for_review` should be read with
/// this share in mind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadyFlips {
    pub ready_flips: usize,
    pub merged_within_60s: usize,
}

fn ready_flips(prs: &[PrRecord]) -> Result<ReadyFlips> {
    let mut out = ReadyFlips {
        ready_flips: 0,
        merged_within_60s: 0,
    };
    for pr in prs {
        let mut merged = Vec::new();
        for e in pr.events.iter().filter(|e| e.kind == "merged") {
            merged.push(parse_timestamp(&e.at)?);
        }
        for e in pr.events.iter().filter(|e| e.kind == "ready_for_review") {
            let at = parse_timestamp(&e.at)?;
            out.ready_flips += 1;
            if merged.iter().any(|&m| (0..=60).contains(&(m - at))) {
                out.merged_within_60s += 1;
            }
        }
    }
    Ok(out)
}

/// Everything #596's Markov slice asks for, from one history snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkHistoryReport {
    /// Latest timestamp in the data; the snapshot instant.
    pub as_of: i64,
    pub stale_after_days: i64,
    pub prs: usize,
    pub transitions: Vec<MarkovTransition>,
    /// Same table per branch-inferred worker (`unattributed` omitted). A
    /// worker's `from_state` block is kept only if at least
    /// [`MIN_WORKER_OUTGOING`] transitions leave it, so no row is a
    /// probability of 1.0 read off a single PR.
    pub transitions_by_worker: Vec<MarkovTransition>,
    pub absorption: Vec<Absorption>,
    pub ready_flips: ReadyFlips,
    pub held_out: HeldOutReport,
}

pub const MIN_WORKER_OUTGOING: u64 = 5;

/// Latest `created_at` or event timestamp across `prs`.
pub fn latest_timestamp(prs: &[PrRecord]) -> Result<i64> {
    let mut latest = i64::MIN;
    for pr in prs {
        latest = latest.max(parse_timestamp(&pr.created_at)?);
        for e in &pr.events {
            latest = latest.max(parse_timestamp(&e.at)?);
        }
    }
    Ok(latest)
}

pub fn report(
    prs: &[PrRecord],
    stale_after_days: i64,
    train_fraction: f64,
    max_order: usize,
    min_observations: usize,
) -> Result<WorkHistoryReport> {
    let as_of = latest_timestamp(prs)?;
    let stale = stale_after_days * 86_400;
    let mut all = Vec::with_capacity(prs.len());
    let mut by_worker: Vec<(&str, Vec<Vec<Step>>)> = vec![
        ("claude", Vec::new()),
        ("codex", Vec::new()),
        ("jules", Vec::new()),
    ];
    for pr in prs {
        let seq = lifecycle(pr, as_of, stale)?;
        if let Some((_, seqs)) = by_worker.iter_mut().find(|(w, _)| *w == pr.worker()) {
            seqs.push(seq.clone());
        }
        all.push(seq);
    }
    let mut transitions_by_worker = Vec::new();
    for (w, seqs) in &by_worker {
        let rows = transition_table(seqs, Some(w));
        let mut outgoing: BTreeMap<&str, u64> = BTreeMap::new();
        for r in &rows {
            *outgoing.entry(r.from_state.as_str()).or_default() += r.sample_size;
        }
        let keep: Vec<bool> = rows
            .iter()
            .map(|r| outgoing[r.from_state.as_str()] >= MIN_WORKER_OUTGOING)
            .collect();
        transitions_by_worker.extend(
            rows.into_iter()
                .zip(keep)
                .filter_map(|(r, k)| k.then_some(r)),
        );
    }
    Ok(WorkHistoryReport {
        as_of,
        stale_after_days,
        prs: prs.len(),
        transitions: transition_table(&all, None),
        transitions_by_worker,
        absorption: absorption(&all),
        ready_flips: ready_flips(prs)?,
        held_out: held_out(
            prs,
            as_of,
            stale,
            train_fraction,
            max_order,
            min_observations,
        )?,
    })
}

fn smooth(dist: &[(usize, f64)], epsilon: f64) -> [f64; LifecycleState::COUNT] {
    let mut dense = [0.0f64; LifecycleState::COUNT];
    for &(s, p) in dist {
        if s < dense.len() {
            dense[s] += p;
        }
    }
    let total: f64 = dense.iter().sum();
    let uniform = 1.0 / LifecycleState::COUNT as f64;
    for p in &mut dense {
        let base = if total > 0.0 { *p / total } else { uniform };
        *p = (1.0 - epsilon) * base + epsilon * uniform;
    }
    dense
}

/// Lowest index wins ties, so HashMap iteration order never leaks in.
fn argmax(dense: &[f64]) -> usize {
    let mut best = 0;
    for (i, &p) in dense.iter().enumerate() {
        if p > dense[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400;

    fn pr(created: &str, is_draft: bool, events: &[(&str, &str)]) -> PrRecord {
        PrRecord {
            repo: "ix".into(),
            number: 1,
            created_at: created.into(),
            state: "OPEN".into(),
            is_draft,
            head_ref: "codex/x".into(),
            events: events
                .iter()
                .map(|(k, at)| {
                    // "labeled:<name>" carries a label.
                    let (kind, label) = match k.split_once(':') {
                        Some((kind, label)) => (kind, Some(label.to_string())),
                        None => (*k, None),
                    };
                    PrEvent {
                        kind: kind.into(),
                        at: (*at).into(),
                        label,
                    }
                })
                .collect(),
        }
    }

    fn states(steps: &[Step]) -> Vec<LifecycleState> {
        steps.iter().map(|s| s.state).collect()
    }

    fn at(s: &str) -> i64 {
        parse_timestamp(s).unwrap()
    }

    #[test]
    fn timestamp_matches_known_epoch() {
        assert_eq!(parse_timestamp("1970-01-01T00:00:00Z").unwrap(), 0);
        assert_eq!(
            parse_timestamp("2026-09-14T21:05:01Z").unwrap(),
            1_789_419_901
        );
        assert!(parse_timestamp("2026-09-14 21:05:01").is_err());
    }

    #[test]
    fn draft_opened_pr_flips_then_merges() {
        use LifecycleState::*;
        let p = pr(
            "2026-01-01T00:00:00Z",
            false,
            &[
                ("ready_for_review", "2026-01-01T02:00:00Z"),
                ("merged", "2026-01-01T03:00:00Z"),
            ],
        );
        let s = lifecycle(&p, at("2026-02-01T00:00:00Z"), 14 * DAY).unwrap();
        assert_eq!(states(&s), vec![PrDraft, PrReadyForReview, PrMerged]);
    }

    #[test]
    fn stale_gap_inserts_origin_aware_stuck_and_cutoff_hides_future() {
        use LifecycleState::*;
        let p = pr(
            "2026-01-01T00:00:00Z",
            false,
            &[("merged", "2026-03-01T00:00:00Z")],
        );
        let full = lifecycle(&p, at("2026-04-01T00:00:00Z"), 14 * DAY).unwrap();
        assert_eq!(
            states(&full),
            vec![PrReadyForReview, PrStuckReady, PrMerged]
        );
        let censored = lifecycle(&p, at("2026-02-01T00:00:00Z"), 14 * DAY).unwrap();
        assert_eq!(states(&censored), vec![PrReadyForReview, PrStuckReady]);

        let d = pr("2026-01-01T00:00:00Z", true, &[]);
        let s = lifecycle(&d, at("2026-02-01T00:00:00Z"), 14 * DAY).unwrap();
        assert_eq!(states(&s), vec![PrDraft, PrStuckDraft]);
    }

    #[test]
    fn merge_candidate_label_on_draft_waits_for_ready_flip() {
        use LifecycleState::*;
        let p = pr(
            "2026-01-01T00:00:00Z",
            false,
            &[
                ("labeled:agent-blackbox-reviewed", "2026-01-01T01:00:00Z"),
                ("ready_for_review", "2026-01-01T01:10:00Z"),
                ("merged", "2026-01-01T01:20:00Z"),
            ],
        );
        let s = lifecycle(&p, at("2026-01-02T00:00:00Z"), 14 * DAY).unwrap();
        assert_eq!(states(&s), vec![PrDraft, PrMergeCandidate, PrMerged]);
        // Promoted when the draft became mergeable, not when it was labelled.
        assert_eq!(s[1].at, at("2026-01-01T01:10:00Z"));

        // On a ready PR the label promotes at once; other labels do nothing.
        let r = pr(
            "2026-01-01T00:00:00Z",
            false,
            &[
                ("labeled:worker:codex", "2026-01-01T00:30:00Z"),
                ("labeled:fleet:merge-ready", "2026-01-01T01:00:00Z"),
                ("merged", "2026-01-01T01:20:00Z"),
            ],
        );
        let s = lifecycle(&r, at("2026-01-02T00:00:00Z"), 14 * DAY).unwrap();
        assert_eq!(
            states(&s),
            vec![PrReadyForReview, PrMergeCandidate, PrMerged]
        );
    }

    #[test]
    fn reopen_returns_to_last_open_state() {
        use LifecycleState::*;
        let p = pr(
            "2026-01-01T00:00:00Z",
            true,
            &[
                ("closed", "2026-01-01T01:00:00Z"),
                ("reopened", "2026-01-01T02:00:00Z"),
            ],
        );
        let s = lifecycle(&p, at("2026-01-02T00:00:00Z"), 14 * DAY);
        assert_eq!(states(&s.unwrap()), vec![PrDraft, PrRejected, PrDraft]);
    }

    #[test]
    fn transition_probabilities_sum_to_one_per_state() {
        let as_of = at("2026-02-01T00:00:00Z");
        let seqs: Vec<Vec<Step>> = [
            pr(
                "2026-01-01T00:00:00Z",
                false,
                &[("merged", "2026-01-01T01:00:00Z")],
            ),
            pr(
                "2026-01-01T00:00:00Z",
                false,
                &[("closed", "2026-01-01T03:00:00Z")],
            ),
            pr(
                "2026-01-01T00:00:00Z",
                false,
                &[("merged", "2026-01-01T05:00:00Z")],
            ),
        ]
        .iter()
        .map(|p| lifecycle(p, as_of, 14 * DAY).unwrap())
        .collect();
        let table = transition_table(&seqs, None);
        let from_ready: Vec<_> = table
            .iter()
            .filter(|r| r.from_state == "pr.ready_for_review")
            .collect();
        let total: f64 = from_ready.iter().map(|r| r.probability).sum();
        assert!((total - 1.0).abs() < 1e-12);
        let merged = from_ready
            .iter()
            .find(|r| r.to_state == "pr.merged")
            .unwrap();
        assert_eq!(merged.sample_size, 2);
        assert_eq!(merged.median_duration_hours, Some(3.0));

        let abs = absorption(&seqs);
        let ready = abs
            .iter()
            .find(|a| a.state == "pr.ready_for_review")
            .unwrap();
        assert!((ready.p_merged - 2.0 / 3.0).abs() < 1e-9);
        assert!((ready.p_rejected - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn stuck_rows_have_no_median() {
        let d = pr("2026-01-01T00:00:00Z", true, &[]);
        let seqs = vec![lifecycle(&d, at("2026-02-01T00:00:00Z"), 14 * DAY).unwrap()];
        let table = transition_table(&seqs, None);
        assert_eq!(table.len(), 1);
        assert_eq!(table[0].to_state, "pr.stuck_draft");
        assert_eq!(table[0].median_duration_hours, None);
    }

    #[test]
    fn cutoff_ties_go_to_test_not_nowhere() {
        let prs: Vec<PrRecord> = [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ]
        .iter()
        .map(|c| pr(c, false, &[]))
        .collect();
        // split index 2 lands on the second of two tied PRs.
        let h = held_out(&prs, at("2026-01-04T00:00:00Z"), 14 * DAY, 0.5, 2, 1).unwrap();
        assert_eq!((h.train_sequences, h.test_sequences), (1, 3));
    }

    #[test]
    fn sign_test_matches_binomial() {
        assert_eq!(sign_test_p(0, 0), 1.0);
        assert!((sign_test_p(3, 0) - 0.25).abs() < 1e-12);
        assert!((sign_test_p(13, 0) - 2.0 / 8192.0).abs() < 1e-12);
        assert_eq!(sign_test_p(2, 2), 1.0);
    }
}
