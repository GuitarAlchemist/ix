//! Methodology Guard usefulness + anti-process metrics
//! (`GuitarAlchemist.IX.Metrics.MethodologyGuard`).
//!
//! ## Why this module has the shape it has
//!
//! A metric about *process* is the easiest thing in the world to make
//! unfalsifiable. The test this module is built to pass is concrete: three
//! guard runs that actually happened in this repository must land in three
//! different buckets.
//!
//! 1. **A gate correctly stopped an agent editing `.github/workflows/**` and
//!    required a human label.** Working as designed. Nothing to fix.
//! 2. **A risk-report gate produced a *correct verdict* with a *false
//!    reason*** — `GuitarAlchemist/ix#316`. It blocked, and blocking was the
//!    right call, but the paths it cited as its evidence have zero commits in
//!    this repository's history.
//! 3. **A check gated on a JSON artifact describing a different changeset
//!    than the report humans read** — same incident: `risk-report.md` said 25
//!    files, `risk-report.json` (the artifact `enforce --report` consumes)
//!    said 12.
//!
//! Cases 1 and 2 are *identical* on the verdict axis: both fired, and in both
//! the block was right. The draft metric in
//! `docs/metrics/methodology-guard-usefulness-metrics.md` —
//! `false_positive_rate = not-useful findings / findings reviewed` — therefore
//! scores case 2 as a **success**, which is why that draft is not measuring
//! anything. So the assessment here is deliberately **three orthogonal axes**,
//! not one verdict:
//!
//! | axis | question | computable? |
//! |---|---|---|
//! | [`Verdict`] | was blocking (or passing) the right call? | **no** — human adjudication, supplied as input |
//! | [`ReasonFidelity`] | does every path the reason cites exist in the changeset the gate enforced against? | **yes**, mechanically |
//! | [`Provenance`] | is the artifact the gate enforced the same changeset a human reviewed? | **yes**, mechanically |
//!
//! Two of the three being mechanical matters: a usefulness programme that
//! depends entirely on humans grading findings decays the moment nobody grades
//! them. [`GuardMetrics::adjudication_coverage`] reports how much of a window
//! carries a human judgement at all, so an ungraded window reads as ungraded
//! rather than silently scoring well.
//!
//! ## Anti-process
//!
//! Two failure modes, both measurable, both fail-closed here:
//!
//! * **A gate that always fires has stopped discriminating.** Measured by
//!   [`GuardMetrics::fire_rate`]. A gate that *never* fires across a window
//!   large enough to have caught something is the same non-signal from the
//!   other side, and is reported the same way.
//! * **A gate nobody has watched fail is not a gate.** Measured by
//!   [`GuardMetrics::seeded_violation_caught`], which defaults to `None`, and
//!   whose `None` yields [`AntiProcessClass::Unwatched`]. This is the module's
//!   strongest assertion: *every gate is presumed not to be a gate until
//!   someone has seeded a violation and watched it caught.* Green CI on an
//!   unwatched gate is not evidence of anything.
//!
//! Bypass is split in two, because on real history the two numbers are wildly
//! different and mean opposite things. Over 120 sampled `ix` pull requests,
//! 40 Agent Blackbox runs were red; 29 of those merged anyway, but only 4 ever
//! carried the `agent-blackbox-reviewed` label. Folding them together into one
//! "override rate" reports a governed-looking 0.725. Splitting them reports a
//! *sanctioned* 0.100 and a *silent* 0.625 — see
//! [`GuardMetrics::sanctioned_override_rate`] and
//! [`GuardMetrics::silent_bypass_rate`].
//!
//! ## Boundaries, not totals
//!
//! [`delta`] compares two windows. A single cumulative number cannot say
//! whether a guard is improving; only the difference between two windows can.
//!
//! ## What this module deliberately does not do
//!
//! It computes no single global usefulness score, sets no policy gate, and
//! returns `None` rather than `0.0` for any rate whose denominator is empty.
//! A rate over zero runs is unknown, not zero — reporting it as zero is how a
//! measurement programme ends up green and dead.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::gate_ledger::{GateDecision, GateLedgerEntry, LedgerLine, OperatorAck};

/// Human adjudication of whether the gate's decision was the right call.
///
/// Deliberately **not** computable. Any metric claiming to derive this
/// mechanically is asserting it can tell a true block from a false one without
/// looking at the change, which nothing in this repository can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Verdict {
    /// Nobody has judged this run. The honest default.
    #[default]
    Unadjudicated,
    /// A human agreed the gate's decision — fire or pass — was right.
    Correct,
    /// A human judged the gate's decision wrong.
    Incorrect,
}

/// What happened to the change after the gate spoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Outcome {
    /// The change did not land while the gate was red.
    Blocked,
    /// The change landed with a recorded acknowledgement — an override label
    /// or an [`OperatorAck`] row. Sanctioned: someone put their name on it.
    OverriddenWithAck,
    /// The change landed while the gate was red with no acknowledgement
    /// anywhere. Nobody put their name on it.
    MergedWithoutAck,
    /// The gate was green and the change landed normally.
    #[default]
    Clean,
}

/// One guard run under evaluation.
///
/// `gated_changeset` and `reviewed_changeset` are separate fields on purpose.
/// Collapsing them assumes the thing being enforced and the thing being read
/// are the same document, and case 3 above is exactly the incident where they
/// were not.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardRun {
    pub id: String,
    /// Which gate produced this run, e.g. `agent-blackbox/risk-report`.
    pub gate: String,
    pub decision: GateDecision,
    #[serde(default)]
    pub verdict: Verdict,
    #[serde(default)]
    pub outcome: Outcome,
    /// Paths the gate's stated *reason* cites as its evidence.
    #[serde(default)]
    pub cited_evidence: Vec<String>,
    /// Files listed by the artifact the gate actually enforced against — for
    /// Agent Blackbox, `dist/risk-report.json`.
    #[serde(default)]
    pub gated_changeset: Vec<String>,
    /// Files in the changeset a human actually reviewed — for Agent Blackbox,
    /// the pull-request diff and `risk-report.md`.
    #[serde(default)]
    pub reviewed_changeset: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ack: Option<OperatorAck>,
}

impl GuardRun {
    /// Did the gate speak against the change? `Warn` counts: a gate that only
    /// ever warns still consumes reviewer attention, and that attention is the
    /// cost anti-process metrics exist to price.
    pub fn fired(&self) -> bool {
        matches!(self.decision, GateDecision::Fail | GateDecision::Warn)
    }
}

/// Does the gate's stated reason describe the changeset it enforced against?
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "state")]
pub enum ReasonFidelity {
    /// The reason cites no paths, or no gated changeset was recorded, so there
    /// is nothing to check. **Not** evidence of soundness.
    Unknown,
    /// Every cited path appears in the gated changeset.
    Sound,
    /// The reason cites paths absent from the changeset it enforced against.
    /// This is ix#316's shape.
    Fabricated { paths: Vec<String> },
}

/// Is the artifact the gate enforced the same changeset a human reviewed?
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "state")]
pub enum Provenance {
    /// One or both sides were not recorded; no comparison is possible.
    Unknown,
    /// The two changesets agree as sets.
    Agreed,
    /// The gate and the human were looking at different changes.
    Diverged {
        only_gated: Vec<String>,
        only_reviewed: Vec<String>,
    },
}

/// The single label for a run, for reporting. The three axes stay available on
/// [`RunAssessment`] — the label is a projection, never a replacement.
///
/// Precedence is declaration order, and it is not arbitrary: a provenance
/// break invalidates the reason *and* the verdict alike, because you cannot
/// call a verdict correct about a changeset the gate never saw. A fabricated
/// reason outranks the verdict for the same argument in miniature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunClass {
    /// Case 3. The gate enforced a different changeset than the one reviewed.
    ProvenanceBreak,
    /// Case 2. Right call, fabricated evidence — ix#316.
    CorrectVerdictFalseReason,
    /// Fabricated evidence, verdict not yet adjudicated.
    FabricatedReason,
    /// Fired, and a human says the call was wrong.
    NoisyBlock,
    /// Case 1. Fired, the call was right, the reason checks out, provenance
    /// agrees.
    UsefulBlock,
    /// Fired, mechanical axes clean, nobody has judged it.
    UnadjudicatedFire,
    /// Passed, and a human says it should not have.
    MissedViolation,
    /// Passed, nothing wrong found.
    CleanPass,
}

/// A run's three axes plus the derived label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAssessment {
    pub id: String,
    pub gate: String,
    pub fired: bool,
    pub class: RunClass,
    pub verdict: Verdict,
    pub fidelity: ReasonFidelity,
    pub provenance: Provenance,
}

/// Mechanical axis 1: every path the reason cites must appear in the changeset
/// the gate enforced against.
pub fn reason_fidelity(run: &GuardRun) -> ReasonFidelity {
    if run.cited_evidence.is_empty() || run.gated_changeset.is_empty() {
        return ReasonFidelity::Unknown;
    }
    let gated: BTreeSet<&str> = run.gated_changeset.iter().map(String::as_str).collect();
    let mut missing: Vec<String> = run
        .cited_evidence
        .iter()
        .filter(|p| !gated.contains(p.as_str()))
        .cloned()
        .collect();
    if missing.is_empty() {
        ReasonFidelity::Sound
    } else {
        missing.sort();
        missing.dedup();
        ReasonFidelity::Fabricated { paths: missing }
    }
}

/// Mechanical axis 2: the enforced artifact and the reviewed diff must
/// describe the same changeset.
pub fn provenance(run: &GuardRun) -> Provenance {
    if run.gated_changeset.is_empty() || run.reviewed_changeset.is_empty() {
        return Provenance::Unknown;
    }
    let gated: BTreeSet<&str> = run.gated_changeset.iter().map(String::as_str).collect();
    let reviewed: BTreeSet<&str> = run.reviewed_changeset.iter().map(String::as_str).collect();
    let only_gated: Vec<String> = gated
        .difference(&reviewed)
        .map(|s| (*s).to_string())
        .collect();
    let only_reviewed: Vec<String> = reviewed
        .difference(&gated)
        .map(|s| (*s).to_string())
        .collect();
    if only_gated.is_empty() && only_reviewed.is_empty() {
        Provenance::Agreed
    } else {
        Provenance::Diverged {
            only_gated,
            only_reviewed,
        }
    }
}

/// Assess one run on all three axes and derive its label.
pub fn assess(run: &GuardRun) -> RunAssessment {
    let fidelity = reason_fidelity(run);
    let prov = provenance(run);
    let fired = run.fired();

    let class = if matches!(prov, Provenance::Diverged { .. }) {
        RunClass::ProvenanceBreak
    } else if matches!(fidelity, ReasonFidelity::Fabricated { .. }) {
        match run.verdict {
            Verdict::Correct => RunClass::CorrectVerdictFalseReason,
            Verdict::Incorrect => RunClass::NoisyBlock,
            Verdict::Unadjudicated => RunClass::FabricatedReason,
        }
    } else {
        match (fired, run.verdict) {
            (true, Verdict::Correct) => RunClass::UsefulBlock,
            (true, Verdict::Incorrect) => RunClass::NoisyBlock,
            (true, Verdict::Unadjudicated) => RunClass::UnadjudicatedFire,
            (false, Verdict::Incorrect) => RunClass::MissedViolation,
            (false, _) => RunClass::CleanPass,
        }
    };

    RunAssessment {
        id: run.id.clone(),
        gate: run.gate.clone(),
        fired,
        class,
        verdict: run.verdict,
        fidelity,
        provenance: prov,
    }
}

/// A rate whose denominator may be empty. `None` means *unknown*, which is not
/// the same as `0.0` and must not be rendered as such.
fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

/// Aggregate measurement over one window of runs.
///
/// Every field is a rate over a named, countable denominator. There is no
/// composite score: the issue's non-goals forbid one, and the three-case test
/// is precisely what a composite would fail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardMetrics {
    pub runs: usize,
    pub fired: usize,
    /// `fired / runs`. Approaching 1.0 means the gate has stopped
    /// discriminating; sitting at 0.0 across a long window means the same from
    /// the other side.
    pub fire_rate: Option<f64>,
    /// `overridden-with-ack / fired`. Someone put their name on it.
    pub sanctioned_override_rate: Option<f64>,
    /// `merged-without-ack / fired`. Nobody did.
    pub silent_bypass_rate: Option<f64>,
    /// `sound / (sound + fabricated)`. Mechanical.
    pub reason_fidelity_rate: Option<f64>,
    /// `agreed / (agreed + diverged)`. Mechanical.
    pub provenance_agreement_rate: Option<f64>,
    /// `useful-block / adjudicated fires`. Human-dependent, so it must be read
    /// alongside `adjudication_coverage`.
    pub useful_block_rate: Option<f64>,
    /// `adjudicated fires / fires`. How much of the window a human judged.
    pub adjudication_coverage: Option<f64>,
    pub class_counts: BTreeMap<RunClass, usize>,
    /// Has anyone seeded a violation and watched this gate catch it? `None` —
    /// the default — means no, and yields [`AntiProcessClass::Unwatched`].
    pub seeded_violation_caught: Option<bool>,
}

/// Measure a window. `seeded_violation_caught` starts `None`; supply it with
/// [`GuardMetrics::with_seed_probe`] once a probe has actually been run.
pub fn measure(runs: &[GuardRun]) -> GuardMetrics {
    let assessments: Vec<RunAssessment> = runs.iter().map(assess).collect();

    let mut class_counts: BTreeMap<RunClass, usize> = BTreeMap::new();
    for a in &assessments {
        *class_counts.entry(a.class).or_insert(0) += 1;
    }

    let fired = assessments.iter().filter(|a| a.fired).count();
    let sanctioned = runs
        .iter()
        .filter(|r| r.fired() && r.outcome == Outcome::OverriddenWithAck)
        .count();
    let silent = runs
        .iter()
        .filter(|r| r.fired() && r.outcome == Outcome::MergedWithoutAck)
        .count();

    let sound = assessments
        .iter()
        .filter(|a| a.fidelity == ReasonFidelity::Sound)
        .count();
    let fabricated = assessments
        .iter()
        .filter(|a| matches!(a.fidelity, ReasonFidelity::Fabricated { .. }))
        .count();
    let agreed = assessments
        .iter()
        .filter(|a| a.provenance == Provenance::Agreed)
        .count();
    let diverged = assessments
        .iter()
        .filter(|a| matches!(a.provenance, Provenance::Diverged { .. }))
        .count();

    let adjudicated_fires = assessments
        .iter()
        .filter(|a| a.fired && a.verdict != Verdict::Unadjudicated)
        .count();
    let useful = class_counts
        .get(&RunClass::UsefulBlock)
        .copied()
        .unwrap_or(0);

    GuardMetrics {
        runs: runs.len(),
        fired,
        fire_rate: rate(fired, runs.len()),
        sanctioned_override_rate: rate(sanctioned, fired),
        silent_bypass_rate: rate(silent, fired),
        reason_fidelity_rate: rate(sound, sound + fabricated),
        provenance_agreement_rate: rate(agreed, agreed + diverged),
        useful_block_rate: rate(useful, adjudicated_fires),
        adjudication_coverage: rate(adjudicated_fires, fired),
        class_counts,
        seeded_violation_caught: None,
    }
}

impl GuardMetrics {
    /// Record that someone seeded a violation and observed whether the gate
    /// caught it. Until this is called, [`Self::anti_process`] answers
    /// [`AntiProcessClass::Unwatched`].
    pub fn with_seed_probe(mut self, caught: bool) -> Self {
        self.seeded_violation_caught = Some(caught);
        self
    }

    /// Classify the gate itself, not its individual runs.
    pub fn anti_process(&self, t: &AntiProcessThresholds) -> AntiProcess {
        if self.seeded_violation_caught != Some(true) {
            let why = match self.seeded_violation_caught {
                None => "no seeded violation has been run against this gate",
                Some(false) => "a seeded violation was run and the gate did not catch it",
                Some(true) => unreachable!("guarded by the enclosing condition"),
            };
            return AntiProcess {
                class: AntiProcessClass::Unwatched,
                reason: format!("{why}; a gate nobody has watched fail is not a gate"),
            };
        }
        if self.runs < t.min_runs {
            return AntiProcess {
                class: AntiProcessClass::Insufficient,
                reason: format!(
                    "{} run(s) in window, fewer than the {} needed to read a rate",
                    self.runs, t.min_runs
                ),
            };
        }
        let Some(fr) = self.fire_rate else {
            return AntiProcess {
                class: AntiProcessClass::Insufficient,
                reason: "no runs in window".to_string(),
            };
        };
        if fr >= t.always_fires_at {
            return AntiProcess {
                class: AntiProcessClass::NotDiscriminating,
                reason: format!(
                    "fires on {:.1}% of runs (>= {:.1}%): a gate that always fires has \
                     stopped separating anything",
                    fr * 100.0,
                    t.always_fires_at * 100.0
                ),
            };
        }
        if fr <= t.never_fires_at {
            return AntiProcess {
                class: AntiProcessClass::NotDiscriminating,
                reason: format!(
                    "fires on {:.1}% of runs (<= {:.1}%) across {} runs: silent, and \
                     silence over a window this size is not the same as clean",
                    fr * 100.0,
                    t.never_fires_at * 100.0,
                    self.runs
                ),
            };
        }
        if let Some(sb) = self.silent_bypass_rate {
            if sb >= t.ceremonial_bypass_at {
                return AntiProcess {
                    class: AntiProcessClass::Ceremonial,
                    reason: format!(
                        "{:.1}% of fires merged with no acknowledgement (>= {:.1}%): the \
                         gate speaks and the work ships regardless",
                        sb * 100.0,
                        t.ceremonial_bypass_at * 100.0
                    ),
                };
            }
        }
        AntiProcess {
            class: AntiProcessClass::Discriminating,
            reason: format!(
                "fires on {:.1}% of runs, seeded violation caught, bypass within bounds",
                fr * 100.0
            ),
        }
    }
}

/// How a gate scores as a *gate*, independent of any individual run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AntiProcessClass {
    /// Nobody has seeded a violation and watched this gate catch it.
    Unwatched,
    /// Fires on everything, or on nothing.
    NotDiscriminating,
    /// Fires, and is routinely bypassed without acknowledgement.
    Ceremonial,
    /// Too few runs to read a rate.
    Insufficient,
    /// Fires selectively, has been watched failing, is not routinely ignored.
    Discriminating,
}

/// An anti-process classification with the number that drove it. A verdict
/// with no reason is the failure mode this module exists to name, so the
/// reason is not optional.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AntiProcess {
    pub class: AntiProcessClass,
    pub reason: String,
}

/// Provisional thresholds. The issue requires these be explicitly provisional:
/// they are calibration starting points, not policy, and nothing in `ix` gates
/// on them.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AntiProcessThresholds {
    /// Below this many runs, report [`AntiProcessClass::Insufficient`] rather
    /// than a rate.
    pub min_runs: usize,
    /// At or above this fire rate, the gate is not separating anything.
    pub always_fires_at: f64,
    /// At or below this fire rate across `min_runs`, likewise.
    pub never_fires_at: f64,
    /// At or above this silent-bypass rate, the gate is ceremony.
    pub ceremonial_bypass_at: f64,
}

impl Default for AntiProcessThresholds {
    fn default() -> Self {
        Self {
            min_runs: 20,
            always_fires_at: 0.95,
            never_fires_at: 0.0,
            ceremonial_bypass_at: 0.50,
        }
    }
}

/// The difference between two windows.
///
/// A cumulative total cannot say whether a guard is getting better. This
/// repository has already reported a wrong number once by measuring a single
/// run instead of the difference between two, so the boundary form is the one
/// the module offers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowDelta {
    pub before: GuardMetrics,
    pub after: GuardMetrics,
    pub fire_rate_delta: Option<f64>,
    pub silent_bypass_delta: Option<f64>,
    pub sanctioned_override_delta: Option<f64>,
    pub reason_fidelity_delta: Option<f64>,
    pub adjudication_coverage_delta: Option<f64>,
}

/// A delta between two rates is unknown if either side is unknown.
fn diff(after: Option<f64>, before: Option<f64>) -> Option<f64> {
    Some(after? - before?)
}

/// Measure two windows and report their boundary.
pub fn delta(before_runs: &[GuardRun], after_runs: &[GuardRun]) -> WindowDelta {
    let before = measure(before_runs);
    let after = measure(after_runs);
    WindowDelta {
        fire_rate_delta: diff(after.fire_rate, before.fire_rate),
        silent_bypass_delta: diff(after.silent_bypass_rate, before.silent_bypass_rate),
        sanctioned_override_delta: diff(
            after.sanctioned_override_rate,
            before.sanctioned_override_rate,
        ),
        reason_fidelity_delta: diff(after.reason_fidelity_rate, before.reason_fidelity_rate),
        adjudication_coverage_delta: diff(after.adjudication_coverage, before.adjudication_coverage),
        before,
        after,
    }
}

// ---------------------------------------------------------------------------
// Self-probe: the checkers have to be watched failing too
// ---------------------------------------------------------------------------

/// Result of running the two mechanical checkers against fixtures built to
/// violate them.
///
/// The module asserts that an unwatched gate is not a gate. That claim applies
/// to this module's own checkers, so they ship with a probe a consumer can run
/// in its own CI rather than trusting that some unit test was once green.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfProbe {
    /// A run whose reason cites a path absent from the gated changeset was
    /// classified [`ReasonFidelity::Fabricated`].
    pub fidelity_check_bites: bool,
    /// A run whose gated and reviewed changesets differ was classified
    /// [`Provenance::Diverged`].
    pub provenance_check_bites: bool,
    /// A clean run was flagged by neither checker.
    pub clean_run_stays_clean: bool,
}

impl SelfProbe {
    pub fn all_pass(&self) -> bool {
        self.fidelity_check_bites && self.provenance_check_bites && self.clean_run_stays_clean
    }
}

fn probe_run(id: &str, cited: &[&str], gated: &[&str], reviewed: &[&str]) -> GuardRun {
    GuardRun {
        id: id.to_string(),
        gate: "self-probe".to_string(),
        decision: GateDecision::Fail,
        verdict: Verdict::Unadjudicated,
        outcome: Outcome::Blocked,
        cited_evidence: cited.iter().map(|s| (*s).to_string()).collect(),
        gated_changeset: gated.iter().map(|s| (*s).to_string()).collect(),
        reviewed_changeset: reviewed.iter().map(|s| (*s).to_string()).collect(),
        ack: None,
    }
}

/// Seed a violation of each mechanical check and report whether it was caught.
pub fn self_probe() -> SelfProbe {
    let fabricated = probe_run(
        "probe-fidelity",
        &["src/never-existed.rs"],
        &["src/real.rs"],
        &["src/real.rs"],
    );
    let diverged = probe_run(
        "probe-provenance",
        &["src/real.rs"],
        &["src/real.rs"],
        &["src/real.rs", "src/other.rs"],
    );
    let clean = probe_run(
        "probe-clean",
        &["src/real.rs"],
        &["src/real.rs"],
        &["src/real.rs"],
    );
    SelfProbe {
        fidelity_check_bites: matches!(
            reason_fidelity(&fabricated),
            ReasonFidelity::Fabricated { .. }
        ),
        provenance_check_bites: matches!(provenance(&diverged), Provenance::Diverged { .. }),
        clean_run_stays_clean: reason_fidelity(&clean) == ReasonFidelity::Sound
            && provenance(&clean) == Provenance::Agreed,
    }
}

// ---------------------------------------------------------------------------
// Ledger bridge
// ---------------------------------------------------------------------------

/// The `extra.guard` object a producer attaches to a [`GateLedgerEntry`].
///
/// The ledger already ships — `crates/ix-quality-trend/src/gate_ledger.rs`,
/// contract `docs/contracts/2026-05-24-quality-gate-ledger.contract.md` — with
/// `decision`, `evidence` and `operator_ack`. Rather than add a second store,
/// guard-specific fields ride in the existing open `extra` slot, so nothing in
/// the frozen v1 schema changes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GuardExtra {
    #[serde(default)]
    pub cited_evidence: Vec<String>,
    #[serde(default)]
    pub gated_changeset: Vec<String>,
    #[serde(default)]
    pub reviewed_changeset: Vec<String>,
    #[serde(default)]
    pub verdict: Verdict,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<Outcome>,
}

/// Lift one ledger entry into a [`GuardRun`].
///
/// If `extra.guard.outcome` is absent the outcome is inferred conservatively
/// from the decision and the presence of an [`OperatorAck`]: a fire with an ack
/// is sanctioned, a fire without one is `Blocked` — *not* `MergedWithoutAck`,
/// because the ledger alone cannot see whether the change landed. Silent
/// bypass must be asserted by a producer that knows, never guessed here.
pub fn run_from_entry(entry: &GateLedgerEntry) -> GuardRun {
    let guard: GuardExtra = entry
        .extra
        .as_ref()
        .and_then(|v| v.get("guard").cloned())
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();

    let fired = matches!(entry.decision, GateDecision::Fail | GateDecision::Warn);
    let outcome = guard.outcome.unwrap_or(if !fired {
        Outcome::Clean
    } else if entry.operator_ack.is_some() {
        Outcome::OverriddenWithAck
    } else {
        Outcome::Blocked
    });

    let mut cited = guard.cited_evidence;
    if cited.is_empty() {
        if let Some(ev) = &entry.evidence {
            cited.push(ev.ref_.clone());
        }
    }

    GuardRun {
        id: entry.id.clone(),
        gate: format!("{}/{}", entry.source, entry.domain),
        decision: entry.decision,
        verdict: guard.verdict,
        outcome,
        cited_evidence: cited,
        gated_changeset: guard.gated_changeset,
        reviewed_changeset: guard.reviewed_changeset,
        ack: entry.operator_ack.clone(),
    }
}

/// Lift the v1 lines of a ledger into runs. Legacy v0 rows are skipped: they
/// predate the guard fields entirely, and silently defaulting them would
/// inflate `runs` with rows carrying no signal.
pub fn runs_from_ledger(lines: &[LedgerLine]) -> Vec<GuardRun> {
    lines
        .iter()
        .filter_map(|l| match l {
            LedgerLine::V1(e) => Some(run_from_entry(e)),
            LedgerLine::LegacyV0(_) => None,
        })
        .collect()
}
