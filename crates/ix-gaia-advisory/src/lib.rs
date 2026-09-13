//! Read-only M3 advisory artifact over copied immutable Gaia evidence.
//!
//! The crate has exactly two public entry points, [`advise`] and [`evaluate`].
//! It reads a copied evidence directory and its declared inventory manifest and
//! returns one typed deterministic advisory artifact; or it reads a locked
//! corpus of staged cases and returns one Class-`Development` characterization
//! report of count vectors. It routes no work, mutates nothing, fits no model,
//! contacts no bus, selects no parameter, and carries no score, threshold,
//! margin, verdict, freshness, or authority value.
//!
//! Three disciplines are worth stating at the top, because the artifact is
//! worthless without them:
//!
//! * **units and nulls are typed, not narrated.** Every number declares its
//!   unit, and a quantity that is not measurable is `null` — never `0`, which
//!   would read as a measurement that happened to be zero.
//! * **cost is computed by the harness, never self-reported by a rule.** Two
//!   rules accounting for themselves are not comparable like for like.
//! * **the out-of-sample clause is a typed `Unknown` with a machine-readable
//!   reason**, not a hedge in prose and not an omitted field.

use std::path::PathBuf;

use serde::Serialize;
use thiserror::Error;

mod digest;
mod evaluation;
mod manifest;
mod root;
mod rules;

pub mod rate;

pub use evaluation::{evaluate, Cell, EvalRequest, EvaluationReport, PooledCell, RuleWidth};
/// The workspace's canonical six-valued truth type. There is exactly one
/// hexavalent truth table in the workspace and this crate does not fork it.
pub use ix_types::Hexavalent;
pub use rate::{CountVector, Ratio};

/// This crate's own schema version, independent of `ix-gaia-census`'s.
/// Bumped on **any** change to the emitted key set.
pub const SCHEMA_VERSION: u32 = 1;

/// `IX-AGG-1` (§3.1) over the 26 files of the reviewed M2 subject.
///
/// A declared constant, not a runtime measurement: the reviewed subject is a
/// frozen snapshot this crate does not hold and must not read. It is carried so
/// every artifact is content-addressed to the exact code that produced the
/// exact reference rule, and NC-13 re-measures it out of band.
pub const SUBJECT_AGGREGATE_IX_AGG_1: &str =
    "a3e6895759d007c45d7311c9b05387befdf22212cc64671c6a94cd731dc4c235";

/// The rule's own source text — the input to `rule_source_digest`.
const RULE_SOURCE_TEXT: &str = include_str!("rules/window_probe.rs");

/// The closed set of rules under characterization.
///
/// Closed by construction: every rule runs through the same dispatch, the same
/// harness-computed cost accounting, and the same scoring path, so no rule can
/// look cheap or confident by accounting differently from the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum RuleId {
    /// Always agree. The trivial floor, not a rival: on any corpus whose
    /// truth-`F` cases all bind, its `false_agreement` is exactly `1.0`.
    AlwaysAgree,
    /// Disagree iff the root's name set differs from the declared name set.
    NameSetOnly,
    /// Disagree iff any declared length differs from the measured length.
    LengthOnly,
    /// `NameSetOnly` or `LengthOnly` — the content-free ablation. Dominated by
    /// construction by neither of its parts alone; stated rather than hidden.
    StructuralOnly,
    /// `StructuralOnly`, plus head/tail window digests against the Window
    /// Reference Table. Swept over its width, never selected.
    WindowProbe,
    /// The exact reference: `ix_gaia_census::census`. A ceiling, not a rival.
    FullDigest,
}

impl RuleId {
    /// Declaration order, which is the ordinal order every report uses.
    pub const fn all() -> [RuleId; 6] {
        [
            RuleId::AlwaysAgree,
            RuleId::NameSetOnly,
            RuleId::LengthOnly,
            RuleId::StructuralOnly,
            RuleId::WindowProbe,
            RuleId::FullDigest,
        ]
    }
}

/// A frozen provenance the caller requires the run to reproduce.
///
/// When present, any mismatch is a refusal rather than a note in the artifact:
/// a tampered Window Reference Table or a tampered manifest must not be
/// silently used, because every window outcome and every declared value in the
/// artifact would then rest on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedProvenance {
    pub window_reference_digest: String,
    pub manifest_sha256: String,
}

/// The declared inputs of one advisory run.
#[derive(Debug, Clone)]
pub struct AdvisoryRequest {
    /// Directory holding the copied immutable evidence.
    pub evidence_root: PathBuf,
    /// File name, inside `evidence_root`, of the manifest declaring the inventory.
    pub manifest_file_name: String,
    /// Which rule to run. Swept, never selected.
    pub rule: RuleId,
    /// Window width in bytes. `0` means "structural only", not "unset".
    /// Must be `0` for every rule other than [`RuleId::WindowProbe`].
    pub window_bytes: u32,
    /// Root holding the **pristine** bytes the Window Reference Table is
    /// derived from. Required when the rule is [`RuleId::WindowProbe`] at a
    /// non-zero width; a windowed rule that derived its reference from the root
    /// under test would certify that root against itself.
    pub window_reference_root: Option<PathBuf>,
    /// When present, the run refuses unless it reproduces this provenance.
    pub expected_provenance: Option<ExpectedProvenance>,
}

/// One typed deterministic advisory artifact.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AdvisoryArtifact {
    pub schema_version: u32,
    pub rule: RuleId,
    /// unit: bytes; `0` means "structural only", not "unset".
    pub window_bytes: u32,
    pub provenance: Provenance,
    /// One per declared row, ordinal by name.
    pub rows: Vec<AdvisoryRow>,
    pub root: RootFindings,
    pub cost: CostAccount,
    /// Fold of the row states and the root finding through the canonical
    /// hexavalent `and`. Range is exactly `{T, F, U, C}`; `P` and `D` are never
    /// produced, because a gradient is precisely what an advisory must not
    /// carry.
    pub reconciles: Hexavalent,
    /// The out-of-sample clause, typed rather than narrated.
    pub out_of_sample: OutOfSampleStatus,
}

impl AdvisoryArtifact {
    /// The artifact's canonical serialization: one setting, no variance.
    ///
    /// Every field is a named object member and **no map is ever serialized**,
    /// so key order is struct declaration order rather than hash order and the
    /// byte sequence is a function of the measured evidence alone.
    pub fn to_canonical_json(&self) -> String {
        serde_json::to_string(self).expect("the artifact holds only strings, integers, and enums")
    }
}

/// What every number in the artifact is content-addressed to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Provenance {
    pub manifest_file_name: String,
    /// SHA-256 of the manifest as read — the declaration of record this run used.
    pub manifest_sha256: String,
    /// `IX-AGG-1` (§3.1) over the evidence root.
    pub evidence_aggregate_ix_agg_1: String,
    /// `GAIA-AGG-1` (§3.2) over the evidence root — a quoted foreign
    /// construction, declared by approved evidence and reproduced, not invented.
    pub declared_bundle_aggregate_gaia_agg_1: String,
    /// `IX-AGG-1` over the 26 files of the reviewed M2 subject; a declared
    /// constant, see [`SUBJECT_AGGREGATE_IX_AGG_1`].
    pub subject_aggregate_ix_agg_1: String,
    /// SHA-256 of the Window Reference Table text. A single-artifact digest,
    /// not a multi-file record aggregate.
    pub window_reference_digest: String,
    /// SHA-256 of the rule's own source text.
    pub rule_source_digest: String,
    /// `Some` only inside an evaluation run.
    pub corpus_digest_ix_agg_1: Option<String>,
    /// `Some` only inside a Class-`Holdout` run. No such run exists in M3.
    pub ledger_digest_ix_agg_1: Option<String>,
    /// `Some` only inside an evaluation run.
    pub case_id: Option<String>,
    /// `Some` only inside an evaluation run. The key is always present: a
    /// missing key and a null are different facts.
    pub evidence_class: Option<EvidenceClass>,
}

/// Development, selection, holdout — kept apart so a number computed from one
/// can never be reported as if it came from another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum EvidenceClass {
    /// Descriptive counts and rates only. No threshold, no margin, no ordering
    /// claim, no verdict. The entire synthetic corpus is this, permanently.
    Development,
    /// Choosing a parameter value. Never populated: M3 selects no parameter.
    Selection,
    /// One scored comparison, once per fingerprint. Never populated: no
    /// eligible unseen population exists.
    Holdout,
}

/// One declared file, as declared, as corroborated, and as measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdvisoryRow {
    pub name: String,
    /// unit: bytes — the declaration of record. Never null.
    pub declared_bytes: u64,
    /// unit: bytes — a second claim site's figure for the same quantity.
    /// `null` when the document carries no second site. When it is present and
    /// differs from `declared_bytes`, **both are retained** and `state` is `C`.
    pub corroborated_bytes: Option<u64>,
    /// unit: bytes — `null` iff not measurable (absent, or not a plain file).
    /// **Never** defaulted to `0`: an empty file is a different fact.
    pub measured_bytes: Option<u64>,
    pub head_window_matches: Hexavalent,
    pub tail_window_matches: Hexavalent,
    pub state: Hexavalent,
}

/// What the root holds, against what the manifest declares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RootFindings {
    /// unit: count
    pub declared_files: u64,
    /// unit: count — declared names the root holds as plain files.
    pub present_files: u64,
    /// Present names the manifest declares neither as a row nor as itself,
    /// ordinal. Reported, never ignored.
    pub unlisted_names: Vec<String>,
    /// Declared names the root does not hold, ordinal.
    pub missing_names: Vec<String>,
    /// Entries that are not plain files, ordinal, each with its class.
    pub non_file_names: Vec<NonFileEntry>,
}

/// What the run cost, measured by the harness rather than reported by the rule.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CostAccount {
    /// unit: bytes — declared before the run, from the manifest's own figures.
    pub budget_bytes: u64,
    /// unit: bytes — measured. Invariant: `bytes_read <= budget_bytes`.
    pub bytes_read: u64,
    /// unit: count
    pub files_opened: u64,
    /// unit: bytes — the exact reference's cost over the same root.
    pub reference_bytes_read: u64,
    /// unit: bytes — the one full pristine pass the Window Reference Table
    /// costs. Zero at width zero, where the windows are empty.
    pub wrt_construction_bytes: u64,
    /// unit: bytes — the one pass the two provenance aggregates cost.
    /// Disclosed separately for the same reason the WRT pass is: a rule's byte
    /// figure describes the rule, and folding the provenance pass into it would
    /// make every rule look identical.
    pub provenance_bytes_read: u64,
    /// dimensionless — `bytes_read` over `reference_bytes_read`, both in bytes.
    pub cost_ratio: Ratio,
}

/// One entry in the evidence root that this crate cannot measure as raw bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NonFileEntry {
    pub name: String,
    pub entry_class: EntryClass,
}

/// The class of an entry that is not a plain file. No class is elided: an entry
/// that is none of the named classes is reported as `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum EntryClass {
    Directory,
    Symlink,
    Other,
}

/// The out-of-sample clause of the specification, typed rather than narrated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status")]
#[non_exhaustive]
pub enum OutOfSampleStatus {
    /// No out-of-sample measurement exists. Carries the machine-readable
    /// primary reason and every other reason that also applies, in declaration
    /// order.
    Unknown {
        reason: OosUnknownReason,
        also_applicable: Vec<OosUnknownReason>,
    },
    /// **Reserved.** Constructible only from a Class-`Holdout` population
    /// admitted under the preregistered trigger and consumed under the
    /// consumption ledger.
    ///
    /// There is exactly one construction site **in this crate** —
    /// [`measured_from_earned_holdout`] — and reaching it requires an
    /// [`earned::EarnedHoldout`], which [`earned::EarnedHoldout::admit`] issues
    /// only when every earned condition is *measured*. Since R2 that is enforced
    /// by the compiler and not only by a control: `EarnedHoldout` lives in the
    /// private [`earned`] module with child-private fields, so no other code in
    /// this crate can forge one. NC-20 still fails the suite if a second site
    /// appears or the one site moves out of that function, and NC-20d fails it
    /// if the seal is loosened.
    ///
    /// The qualifier "in this crate" is exact and deliberate. `#[non_exhaustive]`
    /// on an enum prevents exhaustive *matching*, not variant *construction*, so
    /// a downstream crate can still write this variant into a value of its own.
    /// That is inert here: it forges nothing this crate emits, and `AdvisoryArtifact`'s
    /// fields are public anyway, so a downstream crate can already build any
    /// artifact it likes. What no caller can do is make **this crate** emit it.
    ///
    /// **No path M3-as-selected can reach constructs it.** `advise` binds no
    /// admission and `evaluate` refuses Class-`Holdout` outright, so the earned
    /// path is prepared and closed, not open: nothing but an independently
    /// approved S12/S13 preregistration supplying a real admitted population
    /// and ledger can open it.
    Measured {
        corpus_digest_ix_agg_1: String,
        report_digest: String,
    },
}

/// Why no out-of-sample measurement exists.
///
/// **Declaration order is emission-priority order.** Do not reorder without a
/// schema bump: the emitted primary reason is the first variant whose condition
/// holds, and the ordering is what keeps the primary reason independent of any
/// contested reading of the specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum OosUnknownReason {
    /// No population exists that is simultaneously available, entitled and
    /// unexposed. A measured fact about the population.
    NoEligibleUnseenPopulation,
    /// The synthetic corpus is Class-`Development` permanently and the lineage
    /// roots are exposed. Every population reachable here is one or the other.
    SyntheticOrExposedPopulationOnly,
    /// Every real-label population in the approved evidence is uniformly
    /// positive, so an always-agree baseline scores 100% on all of them.
    NoRealLabelPopulation,
    /// The rule is parameter-frozen; there is nothing to fit. This is the one
    /// reason that depends on a contested reading, so it is **last** and is
    /// never emitted as the primary reason.
    NoModelFitted,
}

impl OosUnknownReason {
    const DECLARATION_ORDER: [OosUnknownReason; 4] = [
        OosUnknownReason::NoEligibleUnseenPopulation,
        OosUnknownReason::SyntheticOrExposedPopulationOnly,
        OosUnknownReason::NoRealLabelPopulation,
        OosUnknownReason::NoModelFitted,
    ];

    /// The reasons that may be emitted as the **primary** reason (NC-25a).
    ///
    /// `NoModelFitted` is deliberately absent. It is permanently true under C6
    /// — the rule is parameter-frozen and this crate fits nothing, ever — so a
    /// reading on which it *explains* the absence of a measurement would make
    /// the out-of-sample gate unsatisfiable by any evidence whatsoever, which
    /// is the weaker construction. It stays in [`Self::DECLARATION_ORDER`] and
    /// so still appears in `also_applicable`: it is a disclosure, not a bar.
    const PRIMARY_ORDER: [OosUnknownReason; 3] = [
        OosUnknownReason::NoEligibleUnseenPopulation,
        OosUnknownReason::SyntheticOrExposedPopulationOnly,
        OosUnknownReason::NoRealLabelPopulation,
    ];
}

// --------------------------------------------- the earned-only `Measured` path

/// The canonical relative path §11.3 pins `report_digest` to.
///
/// `IX-AGG-1` embeds each row's path, so two lanes digesting the same report at
/// two filesystem locations would diverge and trip `ReplayDivergence`
/// spuriously. The digest is therefore always taken at this one relative path,
/// and any other location is a **refusal**, not a divergence.
const CANONICAL_REPORT_RELATIVE_PATH: &str = "corpus/report/evaluation-report.json";

/// One §11.3 ledger receipt, as a value.
///
/// The fields are private and there is no public constructor, so nothing
/// outside this crate can forge one. **Nothing inside it constructs one
/// either**: reading and appending `holdout-ledger.jsonl` is S12's work and no
/// part of this repair — no ledger exists, none is created, and none is read.
/// The type exists so [`earned::EarnedHoldout::admit`] can require a receipt
/// *by type* rather than by promise.
///
/// It is deliberately **not** sealed into [`earned`]. A receipt models evidence
/// originating *outside* this crate, in the consumption ledger; sealing it would
/// only mean this crate mints its own receipts, which is the opposite of what
/// §11.3 asks. That a receipt genuinely originates in the ledger rather than in
/// the lane presenting it is S12's obligation and its reviewer's check.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
struct LedgerReceipt {
    corpus_digest_ix_agg_1: String,
    candidate_fingerprint: String,
    report_digest: String,
}

/// The sealed proof-token boundary.
///
/// Every type declared here is a **proof token**: a value whose existence is
/// itself the evidence that a condition was *measured* rather than claimed. The
/// module exists so that the compiler enforces that, instead of a doc comment
/// asserting it.
///
/// Rust field privacy is **module**-scoped, not type-scoped. A struct with
/// private fields declared at crate root can still be built by struct literal
/// from anywhere else in `lib.rs` — which is exactly where S12 will work, and
/// exactly the bypass the R1 Spec review found: a forged `EarnedHoldout` reached
/// `OutOfSampleStatus::Measured` carrying attacker-chosen digests with every
/// control green, and a forged pair of measured digests did the same *through*
/// `admit`. Declaring these three types in a **private child module** whose
/// fields carry **no** visibility qualifier moves the guarantee from convention
/// to the compiler: the parent module and every sibling can name these types and
/// call their constructors, and none of them — nor any future S12 edit to
/// `lib.rs` — can write a literal of one.
///
/// The parent-facing surface is the minimum the emission path and the boundary
/// controls need: three constructors that *measure*, and four read-only
/// accessors that cannot construct. NC-20d pins the invariant in source text, so
/// that a later `pub` on any field here fails the suite instead of passing
/// silently.
mod earned {
    use super::{
        candidate_fingerprint, digest, AdvisoryRefusal, EvidenceClass, LedgerReceipt,
        CANONICAL_REPORT_RELATIVE_PATH,
    };

    /// A corpus digest **this crate measured itself**.
    ///
    /// The field is private to this module and the only constructor measures the
    /// corpus root, so no caller — and no sibling module — can hand one in. That
    /// is the whole content of NC-20a's third condition: a digest supplied by
    /// the lane under audit certifies nothing, because that lane could supply
    /// the digest of any corpus it liked.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct MeasuredCorpusDigest(String);

    impl MeasuredCorpusDigest {
        /// The only constructor, using the same `digest::measure_root` every
        /// other aggregate in this crate is built from.
        // Reached by S12 when a holdout corpus is admitted. No path in
        // M3-as-selected calls it; the colocated boundary tests measure their
        // own scratch roots, so the constructor is exercised, not asserted.
        #[allow(dead_code)]
        pub(super) fn measure(corpus_root: &std::path::Path) -> Result<Self, AdvisoryRefusal> {
            Ok(Self(digest::measure_root(corpus_root)?.ix_agg_1))
        }

        /// Read-only. An accessor can disclose a measurement; it can never
        /// launder a claim into one.
        #[allow(dead_code)]
        pub(super) fn as_str(&self) -> &str {
            &self.0
        }
    }

    /// A report digest **recomputed from the emitted report's own bytes**.
    ///
    /// Private field, single constructor, same reasoning as
    /// [`MeasuredCorpusDigest`]: a `report_digest` a caller states is a claim
    /// about a report, and a claim cannot certify the report it is a claim
    /// about.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct MeasuredReportDigest(String);

    impl MeasuredReportDigest {
        /// The only constructor. Refuses any relative path but the canonical
        /// one.
        // Reached by S12; no path in M3-as-selected calls it.
        #[allow(dead_code)]
        pub(super) fn recompute(
            relative_path: &str,
            report_bytes: &[u8],
        ) -> Result<Self, AdvisoryRefusal> {
            if relative_path != CANONICAL_REPORT_RELATIVE_PATH {
                return Err(AdvisoryRefusal::ProvenanceMismatch {
                    field: "report_path".to_string(),
                    expected: CANONICAL_REPORT_RELATIVE_PATH.to_string(),
                    measured: relative_path.to_string(),
                });
            }
            Ok(Self(digest::ix_agg_1_single(relative_path, report_bytes)))
        }

        /// Read-only, for the same reason as [`MeasuredCorpusDigest::as_str`].
        #[allow(dead_code)]
        pub(super) fn as_str(&self) -> &str {
            &self.0
        }
    }

    /// Proof that every earned condition of §10.3 held for one run.
    ///
    /// A value of this type **is** the proof — now by construction rather than
    /// by assertion. Its fields are private to this module, so the only code in
    /// the crate that can write them is [`EarnedHoldout::admit`] below, and
    /// holding one is the only thing that entitles a run to a `Measured`
    /// out-of-sample status.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct EarnedHoldout {
        corpus_digest_ix_agg_1: String,
        report_digest: String,
    }

    impl EarnedHoldout {
        /// The admission gate, and the **only construction site of this type**.
        /// Total, and with no discretion in it.
        ///
        /// * `Ok(None)` — an earning condition is absent, so the run is not
        ///   admitted and emits `Unknown` (NC-20a, NC-20b). Fail-closed.
        /// * `Err(ReplayDivergence)` — the receipt's `report_digest` does not
        ///   equal the digest recomputed from the report actually emitted
        ///   (NC-20c). The run is `Void` and produces **no artifact**.
        /// * `Ok(Some(_))` — every condition measured and present.
        // Called by S12 once a population is admitted. No path in M3-as-selected
        // calls it, which is why `Measured` is unreachable today by construction
        // rather than by a grep forbidding the variant from being written at all.
        #[allow(dead_code)]
        pub(super) fn admit(
            evidence_class: Option<EvidenceClass>,
            window_bytes: u32,
            corpus: &MeasuredCorpusDigest,
            report: &MeasuredReportDigest,
            receipt: Option<&LedgerReceipt>,
        ) -> Result<Option<Self>, AdvisoryRefusal> {
            // NC-20a(i) / NC-20b — the class firewall. `Development`,
            // `Selection` and a class-less run can never reach `Measured`.
            if evidence_class != Some(EvidenceClass::Holdout) {
                return Ok(None);
            }
            // NC-20a(ii) — a §11.3 receipt must exist for this run.
            let Some(receipt) = receipt else {
                return Ok(None);
            };
            // NC-20a(iii) — and it must be a receipt for the corpus this crate
            // measured itself, at this candidate's fingerprint and swept width.
            // A caller-supplied digest never reaches this comparison: `corpus`
            // can only have come from `MeasuredCorpusDigest::measure`, whose
            // field no code outside this module can write.
            if receipt.corpus_digest_ix_agg_1 != corpus.0 {
                return Ok(None);
            }
            if receipt.candidate_fingerprint != candidate_fingerprint(window_bytes) {
                return Ok(None);
            }
            // NC-20c — no self-certification. The digest the receipt carries
            // must equal the one recomputed from the report actually emitted.
            if receipt.report_digest != report.0 {
                return Err(AdvisoryRefusal::ReplayDivergence {
                    expected: receipt.report_digest.clone(),
                    measured: report.0.clone(),
                });
            }
            Ok(Some(Self {
                corpus_digest_ix_agg_1: corpus.0.clone(),
                report_digest: report.0.clone(),
            }))
        }

        /// The measured corpus digest this admission carries. Read-only.
        pub(super) fn corpus_digest_ix_agg_1(&self) -> &str {
            &self.corpus_digest_ix_agg_1
        }

        /// The recomputed report digest this admission carries. Read-only.
        pub(super) fn report_digest(&self) -> &str {
            &self.report_digest
        }
    }
}

use earned::EarnedHoldout;

/// `candidate_fingerprint = sha256( rule_source_digest ‖ NUL ‖ "window_bytes=<w>" )`.
///
/// §17's recipe, reproduced here so the admission gate can check that a receipt
/// names *this* candidate at *this* swept width. It is a function of the rule's
/// own source text and the width alone, and reads no caller-supplied value.
fn candidate_fingerprint(window_bytes: u32) -> String {
    let mut input = digest::sha256_hex(RULE_SOURCE_TEXT.as_bytes()).into_bytes();
    input.push(0);
    input.extend_from_slice(format!("window_bytes={window_bytes}").as_bytes());
    digest::sha256_hex(&input)
}

/// **The one and only construction site of [`OutOfSampleStatus::Measured`].**
///
/// NC-20 asserts over the crate's whole source text that exactly one such site
/// exists and that it is this function. Reaching it requires an
/// [`EarnedHoldout`], which requires [`earned::EarnedHoldout::admit`], which
/// measures every §10.3 condition — and since R2 that requirement is enforced by
/// the compiler rather than by this sentence: [`earned`] is a private child
/// module whose fields no code here can write, so this function cannot be
/// reached with a token it forged itself. It reads the admission through
/// read-only accessors, which is all the parent module is given.
fn measured_from_earned_holdout(earned: &EarnedHoldout) -> OutOfSampleStatus {
    OutOfSampleStatus::Measured {
        corpus_digest_ix_agg_1: earned.corpus_digest_ix_agg_1().to_string(),
        report_digest: earned.report_digest().to_string(),
    }
}

/// Which out-of-sample conditions hold for one run, each measured rather than
/// assumed.
struct OosConditions<'a> {
    /// The run's earned §10.3 admission, when it has one. `None` on every path
    /// M3-as-selected can reach: `advise` binds none, and `evaluate` refuses
    /// Class-`Holdout` before it ever builds a binding.
    earned: Option<&'a EarnedHoldout>,
    /// True iff no population is bound that is available, entitled and unexposed.
    no_eligible_unseen_population: bool,
    /// True iff every population this run can reach is synthetic or exposed.
    synthetic_or_exposed_population_only: bool,
    /// True iff no population with both truth directions populated is bound.
    no_real_label_population: bool,
    /// True iff nothing was fitted. Structural: the crate fits nothing.
    no_model_fitted: bool,
}

impl<'a> OosConditions<'a> {
    /// Measured from the run's own **admission**, not from its declared class.
    ///
    /// Only an earned admission can falsify the first three conditions. A
    /// caller's bare `EvidenceClass::Holdout` tag is a claim about a
    /// population, not a measurement of one; treating it as though it falsified
    /// them is exactly what left `NoModelFitted` as the sole surviving reason
    /// on the holdout path, and so made the first holdout run emit an artifact
    /// violating NC-25 (blocker B2).
    fn measure(earned: Option<&'a EarnedHoldout>) -> Self {
        let admitted = earned.is_some();
        Self {
            earned,
            no_eligible_unseen_population: !admitted,
            synthetic_or_exposed_population_only: !admitted,
            no_real_label_population: !admitted,
            // C6 forbids fitting permanently, so this is true forever. It is a
            // disclosure and never a bar — see `OosUnknownReason::PRIMARY_ORDER`.
            no_model_fitted: true,
        }
    }

    fn holds(&self, reason: OosUnknownReason) -> bool {
        match reason {
            OosUnknownReason::NoEligibleUnseenPopulation => self.no_eligible_unseen_population,
            OosUnknownReason::SyntheticOrExposedPopulationOnly => {
                self.synthetic_or_exposed_population_only
            }
            OosUnknownReason::NoRealLabelPopulation => self.no_real_label_population,
            OosUnknownReason::NoModelFitted => self.no_model_fitted,
        }
    }

    /// The emission rule, binding and deterministic.
    ///
    /// A run holding an earned §10.3 admission emits `Measured`. Otherwise
    /// `reason` is the **first** variant in [`OosUnknownReason::PRIMARY_ORDER`]
    /// whose condition holds, and `also_applicable` lists every **later**
    /// variant in declaration order whose condition also holds. A total
    /// function of measured conditions — no discretion.
    fn emit(&self) -> OutOfSampleStatus {
        if let Some(earned) = self.earned {
            return measured_from_earned_holdout(earned);
        }
        // An unadmitted run has `no_eligible_unseen_population` true by
        // construction — `measure` sets it from `!admitted` — so a primary
        // reason always exists and `NoModelFitted` is never reached for it.
        let reason = OosUnknownReason::PRIMARY_ORDER
            .into_iter()
            .find(|reason| self.holds(*reason))
            .expect(
                "an unadmitted run holds NoEligibleUnseenPopulation, which is a primary reason",
            );
        let also_applicable = OosUnknownReason::DECLARATION_ORDER
            .into_iter()
            .skip_while(|candidate| *candidate != reason)
            .skip(1)
            .filter(|candidate| self.holds(*candidate))
            .collect();
        OutOfSampleStatus::Unknown {
            reason,
            also_applicable,
        }
    }
}

/// A refusal. When the run refuses, no artifact is produced at all.
///
/// `#[non_exhaustive]` from day one: a consumer must not assume the set is
/// closed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum AdvisoryRefusal {
    #[error("evidence root unreadable: {detail}")]
    EvidenceRootUnreadable { detail: String },
    #[error("evidence root holds an entry whose name is not valid UTF-8: {lossy:?}")]
    NonUtf8EntryName { lossy: String },
    #[error("manifest unparseable at line {line}: {detail}")]
    ManifestUnparseable { line: usize, detail: String },
    #[error("manifest declares no rows")]
    ManifestDeclaresNoRows,
    #[error("binder cannot resolve {field}: {detail}")]
    BinderIncomplete { field: String, detail: String },
    #[error("manifest declares {name} more than once")]
    DuplicateManifestRow { name: String },
    #[error("manifest declares an inventory section at line {first} and again at line {second}")]
    DuplicateInventorySection { first: usize, second: usize },
    #[error("manifest declares a name that is not a plain file name: {name}")]
    UnsafeManifestName { name: String },
    #[error("provenance mismatch on {field}: expected {expected}, measured {measured}")]
    ProvenanceMismatch {
        field: String,
        expected: String,
        measured: String,
    },
    #[error("corpus digest mismatch: expected {expected}, measured {measured}")]
    CorpusDigestMismatch { expected: String, measured: String },
    #[error("corpus is unreadable or malformed: {detail}")]
    CorpusUnreadable { detail: String },
    #[error("evidence class {class:?} is reserved and unpopulated: {detail}")]
    ReservedEvidenceClass {
        class: EvidenceClass,
        detail: String,
    },
    /// **Reserved**, for the dormant Class-`Holdout` path.
    #[error("replay diverged: expected report digest {expected}, produced {measured}")]
    ReplayDivergence { expected: String, measured: String },
    /// **Reserved**, for the dormant Class-`Holdout` path.
    #[error("holdout fingerprint budget exhausted; already recorded: {recorded:?}")]
    HoldoutBudgetExhausted { recorded: Vec<String> },
}

/// Produce one advisory artifact over copied immutable evidence.
///
/// Returns one typed deterministic advisory artifact, or a refusal. A refusal
/// emits no artifact: the binder refuses before it emits, it never degrades and
/// emits.
pub fn advise(request: &AdvisoryRequest) -> Result<AdvisoryArtifact, AdvisoryRefusal> {
    advise_with(request, &EvaluationBinding::none())
}

/// The three provenance fields an evaluation run binds and a bare `advise` does
/// not. Kept as one value so `advise` cannot forget one of them.
struct EvaluationBinding {
    corpus_digest_ix_agg_1: Option<String>,
    case_id: Option<String>,
    evidence_class: Option<EvidenceClass>,
    /// The run's earned §10.3 admission. `None` on every path M3-as-selected
    /// can reach: `advise` binds none, and `evaluate` refuses Class-`Holdout`
    /// before it builds a binding at all. The field is what makes the earned
    /// path *prepared* rather than *open* — S12 is the only thing that could
    /// ever populate it, and S12 does not exist.
    earned_holdout: Option<EarnedHoldout>,
}

impl EvaluationBinding {
    fn none() -> Self {
        Self {
            corpus_digest_ix_agg_1: None,
            case_id: None,
            evidence_class: None,
            earned_holdout: None,
        }
    }
}

fn advise_with(
    request: &AdvisoryRequest,
    binding: &EvaluationBinding,
) -> Result<AdvisoryArtifact, AdvisoryRefusal> {
    // `window_bytes` is a width, not a flag. A non-zero width on a rule that
    // reads no window is an unresolvable declared field, not a value to ignore.
    if request.rule != RuleId::WindowProbe && request.window_bytes != 0 {
        return Err(AdvisoryRefusal::BinderIncomplete {
            field: "window_bytes".to_string(),
            detail: format!(
                "{:?} reads no window, so a width of {} cannot be honoured",
                request.rule, request.window_bytes
            ),
        });
    }
    if request.rule == RuleId::WindowProbe
        && request.window_bytes > 0
        && request.window_reference_root.is_none()
    {
        return Err(AdvisoryRefusal::BinderIncomplete {
            field: "window_reference_root".to_string(),
            detail: "a windowed rule needs pristine reference bytes it does not have".to_string(),
        });
    }

    let listing = root::list(&request.evidence_root)?;

    let manifest_path = request.evidence_root.join(&request.manifest_file_name);
    if !listing.files.contains_key(&request.manifest_file_name) {
        return Err(AdvisoryRefusal::BinderIncomplete {
            field: request.manifest_file_name.clone(),
            detail: "declared manifest is not present in the evidence root".to_string(),
        });
    }
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
            detail: format!("{}: {e}", request.manifest_file_name),
        })?;
    let manifest_sha256 = digest::sha256_hex(&manifest_bytes);
    let declaration = manifest::parse(&manifest_bytes)?;

    let wrt = rules::window_probe::build(
        request.window_reference_root.as_deref(),
        request.window_bytes,
    )?;

    if let Some(expected) = &request.expected_provenance {
        if expected.window_reference_digest != wrt.digest {
            return Err(AdvisoryRefusal::ProvenanceMismatch {
                field: "window_reference_digest".to_string(),
                expected: expected.window_reference_digest.clone(),
                measured: wrt.digest,
            });
        }
        if expected.manifest_sha256 != manifest_sha256 {
            return Err(AdvisoryRefusal::ProvenanceMismatch {
                field: "manifest_sha256".to_string(),
                expected: expected.manifest_sha256.clone(),
                measured: manifest_sha256,
            });
        }
    }

    let measured = digest::measure_root(&request.evidence_root)?;
    let reference_bytes_read: u64 = measured.files.values().map(|file| file.bytes).sum();

    let outcome = rules::run(&rules::RuleCtx {
        rule: request.rule,
        window_bytes: request.window_bytes,
        evidence_root: &request.evidence_root,
        manifest_file_name: &request.manifest_file_name,
        declared: &declaration.rows,
        listing: &listing,
        wrt: &wrt,
    })?;

    let rows: Vec<AdvisoryRow> = declaration
        .rows
        .iter()
        .zip(outcome.row_verdicts.iter())
        .map(|(declared, verdict)| {
            let corroborated_bytes = declaration.corroborated_bytes.get(&declared.name).copied();
            // Two claim sites in the same document stating different figures
            // for the same quantity is a fact about the *document*, not about
            // any rule, so it applies whatever the rule concluded — the trivial
            // floor cannot agree its way out of it either. Both figures are
            // retained: choosing one would be agreement bought by re-stamping,
            // which the approved evidence considered and refused.
            let contradicted = corroborated_bytes.is_some_and(|claimed| claimed != declared.bytes);
            AdvisoryRow {
                name: declared.name.clone(),
                declared_bytes: declared.bytes,
                corroborated_bytes,
                measured_bytes: listing.files.get(&declared.name).map(|file| file.bytes),
                head_window_matches: verdict.head,
                tail_window_matches: verdict.tail,
                state: if contradicted {
                    Hexavalent::Contradictory
                } else {
                    verdict.state
                },
            }
        })
        .collect();

    let declared_names: std::collections::BTreeSet<&str> = declaration
        .rows
        .iter()
        .map(|row| row.name.as_str())
        .collect();
    let unlisted_names: Vec<String> = listing
        .files
        .keys()
        .filter(|name| {
            name.as_str() != request.manifest_file_name && !declared_names.contains(name.as_str())
        })
        .cloned()
        .collect();
    let missing_names: Vec<String> = declaration
        .rows
        .iter()
        .filter(|row| !listing.files.contains_key(&row.name))
        .map(|row| row.name.clone())
        .collect();

    let reconciles = rows
        .iter()
        .fold(Hexavalent::True, |acc, row| acc.and(row.state))
        .and(outcome.root_state);

    Ok(AdvisoryArtifact {
        schema_version: SCHEMA_VERSION,
        rule: request.rule,
        window_bytes: request.window_bytes,
        rows,
        provenance: Provenance {
            manifest_file_name: request.manifest_file_name.clone(),
            manifest_sha256,
            evidence_aggregate_ix_agg_1: measured.ix_agg_1,
            declared_bundle_aggregate_gaia_agg_1: measured.gaia_agg_1,
            subject_aggregate_ix_agg_1: SUBJECT_AGGREGATE_IX_AGG_1.to_string(),
            window_reference_digest: wrt.digest,
            rule_source_digest: digest::sha256_hex(RULE_SOURCE_TEXT.as_bytes()),
            corpus_digest_ix_agg_1: binding.corpus_digest_ix_agg_1.clone(),
            ledger_digest_ix_agg_1: None,
            case_id: binding.case_id.clone(),
            evidence_class: binding.evidence_class,
        },
        root: RootFindings {
            declared_files: declaration.rows.len() as u64,
            present_files: declaration
                .rows
                .iter()
                .filter(|row| listing.files.contains_key(&row.name))
                .count() as u64,
            unlisted_names,
            missing_names,
            non_file_names: listing.non_files.clone(),
        },
        cost: CostAccount {
            budget_bytes: outcome.budget_bytes,
            bytes_read: outcome.bytes_read,
            files_opened: outcome.files_opened,
            reference_bytes_read,
            wrt_construction_bytes: wrt.construction_bytes,
            provenance_bytes_read: measured.bytes_read,
            cost_ratio: Ratio::new(outcome.bytes_read, reference_bytes_read),
        },
        reconciles,
        out_of_sample: OosConditions::measure(binding.earned_holdout.as_ref()).emit(),
    })
}

// ------------------------------------------------- the out-of-sample boundary

/// Colocated because the behaviour under test is **not reachable through the
/// public seam**. `evaluate` refuses Class-`Holdout` outright, and must keep
/// refusing it until an independently approved S12/S13 preregistration supplies
/// a real admitted population and ledger — so no integration test can drive the
/// holdout emission path without opening exactly the door that must stay shut.
/// These tests drive it one layer in, and construct nothing on disk.
#[cfg(test)]
mod oos_emission {
    use super::earned::{MeasuredCorpusDigest, MeasuredReportDigest};
    use super::*;

    const CLASSES: [Option<EvidenceClass>; 4] = [
        None,
        Some(EvidenceClass::Development),
        Some(EvidenceClass::Selection),
        Some(EvidenceClass::Holdout),
    ];

    const SWEPT_WIDTHS: [u32; 3] = [0, 1024, 4096];

    /// A scratch root outside the worktree, honouring the same override the
    /// Lane-W suite uses. Two ordinary files: this is a directory to *measure*,
    /// never a population, never a corpus, and never a holdout.
    fn scratch(slice: &str) -> std::path::PathBuf {
        let base = match std::env::var_os("IX_GAIA_M3_SCRATCH_DIR") {
            Some(dir) => PathBuf::from(dir),
            None => std::env::temp_dir().join("ix-gaia-m3-scratch"),
        };
        let dir = base.join("b2-oos-boundary").join(slice);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("scratch is removable");
        }
        std::fs::create_dir_all(&dir).expect("scratch is creatable");
        std::fs::write(dir.join("a.txt"), b"alpha").expect("writable");
        std::fs::write(dir.join("b.txt"), b"beta").expect("writable");
        dir
    }

    /// A second scratch root whose **contents** differ, so it measures to a
    /// genuinely different `IX-AGG-1`.
    ///
    /// `ix_agg_1` embeds each row's relative path and bytes and nothing else, so
    /// two roots holding the same names with the same bytes measure identically
    /// — a "foreign" corpus built that way would not be foreign at all. This one
    /// carries an extra file, and the caller asserts the two digests differ
    /// before relying on the distinction.
    fn foreign_scratch(slice: &str) -> std::path::PathBuf {
        let dir = scratch(slice);
        std::fs::write(dir.join("c.txt"), b"gamma").expect("writable");
        dir
    }

    /// The report bytes the receipt in these tests is a receipt *for*.
    const REPORT_BYTES: &[u8] = br#"{"schema_version":1,"cells":[]}"#;

    fn canonical_report() -> MeasuredReportDigest {
        MeasuredReportDigest::recompute(CANONICAL_REPORT_RELATIVE_PATH, REPORT_BYTES)
            .expect("the canonical path is accepted")
    }

    /// A receipt that earns admission for `corpus` at `window_bytes`.
    fn matching_receipt(corpus: &MeasuredCorpusDigest, window_bytes: u32) -> LedgerReceipt {
        LedgerReceipt {
            corpus_digest_ix_agg_1: corpus.as_str().to_string(),
            candidate_fingerprint: candidate_fingerprint(window_bytes),
            report_digest: canonical_report().as_str().to_string(),
        }
    }

    fn emitted(admission: Option<&EarnedHoldout>) -> String {
        serde_json::to_string(&OosConditions::measure(admission).emit())
            .expect("the status serializes")
    }

    // ------------------------------------------------------------ NC-20a, positive

    #[test]
    fn nc_20a_all_earned_conditions_present_yields_measured() {
        let corpus = MeasuredCorpusDigest::measure(&scratch("earned")).expect("scratch measures");
        let report = canonical_report();

        for window_bytes in SWEPT_WIDTHS {
            let receipt = matching_receipt(&corpus, window_bytes);
            let admission = EarnedHoldout::admit(
                Some(EvidenceClass::Holdout),
                window_bytes,
                &corpus,
                &report,
                Some(&receipt),
            )
            .expect("a matching receipt is not a divergence")
            .expect("every earned condition is present, so the run is admitted");

            let json = emitted(Some(&admission));
            assert!(
                json.contains(r#""status":"Measured""#),
                "an earned holdout must reach the Measured variant; got {json}"
            );
            // The two carried values are the *measured* ones, not any value a
            // caller could have named.
            assert!(
                json.contains(corpus.as_str()),
                "the measured corpus digest is carried"
            );
            assert!(
                json.contains(report.as_str()),
                "the recomputed report digest is carried"
            );
        }
    }

    #[test]
    fn nc_20a_admission_is_exhaustive_over_its_conditions() {
        // Every combination of the three earning conditions. `Measured` iff all
        // three hold — asserted in both directions, so a gate that dropped a
        // conjunct fails here as surely as one that added a false green.
        let corpus = MeasuredCorpusDigest::measure(&scratch("exhaustive")).expect("measures");
        // A corpus this run genuinely measured, and which genuinely is not the
        // bound one. Since R2 it cannot be a hand-written digest: the newtype is
        // sealed, so the only way to hold one is to measure a real root — which
        // is a strictly stronger stand-in for "a corpus the caller named".
        let other = MeasuredCorpusDigest::measure(&foreign_scratch("exhaustive-other"))
            .expect("the foreign scratch measures");
        assert_ne!(
            other.as_str(),
            corpus.as_str(),
            "the foreign corpus must actually differ, or this control is vacuous"
        );
        let report = canonical_report();

        for holdout_class in [false, true] {
            for receipt_present in [false, true] {
                for corpus_matches in [false, true] {
                    let class = if holdout_class {
                        Some(EvidenceClass::Holdout)
                    } else {
                        Some(EvidenceClass::Development)
                    };
                    let bound = if corpus_matches { &corpus } else { &other };
                    let receipt = matching_receipt(&corpus, 0);
                    let admission = EarnedHoldout::admit(
                        class,
                        0,
                        bound,
                        &report,
                        receipt_present.then_some(&receipt),
                    )
                    .expect("no report-digest divergence is set up here");

                    let expected = holdout_class && receipt_present && corpus_matches;
                    assert_eq!(
                        admission.is_some(),
                        expected,
                        "class_holdout={holdout_class} receipt={receipt_present} \
                         corpus_matches={corpus_matches}: admission must be {expected}"
                    );
                }
            }
        }
    }

    // ---------------------------------------------- NC-20a / NC-20b, negatives

    #[test]
    fn nc_20b_no_non_holdout_class_can_ever_be_admitted() {
        let corpus = MeasuredCorpusDigest::measure(&scratch("class-firewall")).expect("measures");
        let report = canonical_report();
        let receipt = matching_receipt(&corpus, 0);

        for class in CLASSES {
            let admission = EarnedHoldout::admit(class, 0, &corpus, &report, Some(&receipt))
                .expect("no divergence");
            assert_eq!(
                admission.is_some(),
                class == Some(EvidenceClass::Holdout),
                "NC-20b: only Class-Holdout may be admitted, and {class:?} was"
            );
            if class != Some(EvidenceClass::Holdout) {
                assert!(
                    emitted(admission.as_ref()).contains(r#""status":"Unknown""#),
                    "NC-20b: {class:?} must emit Unknown"
                );
            }
        }
    }

    #[test]
    fn nc_20a_a_missing_receipt_is_never_admitted() {
        let corpus = MeasuredCorpusDigest::measure(&scratch("no-receipt")).expect("measures");
        let report = canonical_report();
        let admission =
            EarnedHoldout::admit(Some(EvidenceClass::Holdout), 0, &corpus, &report, None)
                .expect("an absent receipt is not a divergence");
        assert!(
            admission.is_none(),
            "NC-20a: a holdout run without a §11.3 receipt is not admitted"
        );
        assert!(emitted(None).contains(r#""reason":"NoEligibleUnseenPopulation""#));
    }

    #[test]
    fn nc_20a_a_caller_supplied_corpus_digest_never_self_certifies() {
        // The receipt names a corpus digest this run did not measure. The
        // caller cannot close the gap: `MeasuredCorpusDigest` has one
        // constructor and it measures the root.
        let corpus = MeasuredCorpusDigest::measure(&scratch("self-cert")).expect("measures");
        let report = canonical_report();
        let forged = LedgerReceipt {
            corpus_digest_ix_agg_1: digest::sha256_hex(b"a digest the caller simply asserted"),
            candidate_fingerprint: candidate_fingerprint(0),
            report_digest: canonical_report().as_str().to_string(),
        };
        assert_ne!(forged.corpus_digest_ix_agg_1, corpus.as_str());

        let admission = EarnedHoldout::admit(
            Some(EvidenceClass::Holdout),
            0,
            &corpus,
            &report,
            Some(&forged),
        )
        .expect("a corpus mismatch is not a report divergence");
        assert!(
            admission.is_none(),
            "NC-20a: a digest the caller supplied must never certify the corpus it names"
        );
    }

    #[test]
    fn nc_20a_a_receipt_for_another_candidate_width_is_never_admitted() {
        let corpus = MeasuredCorpusDigest::measure(&scratch("wrong-width")).expect("measures");
        let report = canonical_report();
        // A receipt earned at w=0, presented by a run sweeping w=1024.
        let receipt = matching_receipt(&corpus, 0);
        let admission = EarnedHoldout::admit(
            Some(EvidenceClass::Holdout),
            1024,
            &corpus,
            &report,
            Some(&receipt),
        )
        .expect("a fingerprint mismatch is not a report divergence");
        assert!(
            admission.is_none(),
            "NC-20a: a receipt binds one candidate fingerprint, so one swept width"
        );
        // And the three swept widths really are three distinct fingerprints.
        let mut prints: Vec<String> = SWEPT_WIDTHS
            .iter()
            .map(|w| candidate_fingerprint(*w))
            .collect();
        prints.sort();
        prints.dedup();
        assert_eq!(prints.len(), 3);
    }

    // ------------------------------------------------------------------ NC-20c

    #[test]
    fn nc_20c_a_report_digest_mismatch_is_a_replay_divergence_and_no_status() {
        let corpus = MeasuredCorpusDigest::measure(&scratch("divergence")).expect("measures");
        let report = canonical_report();
        let stale = LedgerReceipt {
            corpus_digest_ix_agg_1: corpus.as_str().to_string(),
            candidate_fingerprint: candidate_fingerprint(0),
            report_digest: digest::sha256_hex(b"the digest of a report this run did not produce"),
        };

        let refusal = EarnedHoldout::admit(
            Some(EvidenceClass::Holdout),
            0,
            &corpus,
            &report,
            Some(&stale),
        )
        .expect_err("NC-20c: a report-digest mismatch is a refusal, not a quiet Unknown");
        match refusal {
            AdvisoryRefusal::ReplayDivergence { expected, measured } => {
                assert_eq!(expected, stale.report_digest);
                assert_eq!(measured, report.as_str());
            }
            other => panic!("NC-20c: the refusal must be ReplayDivergence, got {other:?}"),
        }
    }

    #[test]
    fn nc_20c_the_report_digest_is_pinned_to_the_canonical_path() {
        // §11.3: any other location is a refusal, not a divergence — otherwise
        // a lane digesting the same bytes elsewhere would trip ReplayDivergence
        // indistinguishably from a real one.
        let refusal =
            MeasuredReportDigest::recompute("report/evaluation-report.json", REPORT_BYTES)
                .expect_err("a non-canonical path is refused");
        match refusal {
            AdvisoryRefusal::ProvenanceMismatch {
                field, expected, ..
            } => {
                assert_eq!(field, "report_path");
                assert_eq!(expected, CANONICAL_REPORT_RELATIVE_PATH);
            }
            other => panic!("a non-canonical report path must be a refusal, got {other:?}"),
        }
        // The digest is path-pinned, so the same bytes at the canonical path
        // are a different value from the same bytes at any other path.
        assert_ne!(
            canonical_report().as_str(),
            digest::ix_agg_1_single("evaluation-report.json", REPORT_BYTES)
        );
    }

    // ------------------------------------------------------------------ NC-25a

    /// NC-25a (preflight §10.3): `NoModelFitted` is never the emitted primary
    /// reason on **any** path, including the holdout path. It is permanently
    /// true under C6 — the rule is parameter-frozen and nothing is ever fitted
    /// — so a reading on which it explains the absence of a measurement makes
    /// the out-of-sample gate unsatisfiable by any evidence whatsoever. It is
    /// carried as a disclosure in `also_applicable`, never as `reason`.
    #[test]
    fn nc_25a_no_model_fitted_is_never_the_primary_reason_on_any_path() {
        let corpus = MeasuredCorpusDigest::measure(&scratch("nc25a")).expect("measures");
        let report = canonical_report();
        let matching = matching_receipt(&corpus, 0);
        let mismatched = LedgerReceipt {
            corpus_digest_ix_agg_1: digest::sha256_hex(b"another corpus"),
            candidate_fingerprint: candidate_fingerprint(0),
            report_digest: canonical_report().as_str().to_string(),
        };

        // The whole product of reachable admission states, including the
        // holdout path that used to have `NoModelFitted` as its only survivor.
        for class in CLASSES {
            for receipt in [None, Some(&matching), Some(&mismatched)] {
                let admission = EarnedHoldout::admit(class, 0, &corpus, &report, receipt)
                    .expect("no divergence is set up here");
                let json = emitted(admission.as_ref());
                assert!(
                    !json.contains(r#""reason":"NoModelFitted""#),
                    "NC-25a: NoModelFitted is a disclosure, never the primary reason; \
                     class {class:?} emitted {json}"
                );
            }
        }
    }

    // ------------------------------------------- §17 candidate fingerprint pin

    /// The frozen §17 rule identity: `SHA-256(src/rules/window_probe.rs)`, as
    /// declared in the plan's §2 block and preflight §13.1.
    const FROZEN_RULE_SOURCE_DIGEST: &str =
        "668cc15761dc7c3924c425ce55d31be0fc73855d0d9f62a50b8e4eec6173a1dc";

    /// The three frozen `candidate_fingerprint` values, one per swept width.
    const FROZEN_CANDIDATE_FINGERPRINTS: [(u32, &str); 3] = [
        (
            0,
            "24a0c60e4a189ad12c50eb7db729ec619d125c1bf156d7d1732906c324fcb712",
        ),
        (
            1024,
            "af886b20232ec000ca7ac4c4aacd15a905253ffb20d70dcdb361cb563d0c880c",
        ),
        (
            4096,
            "e9273462d2a41d20c29abf8d1a8f7eceab7bc6c02737b8e58859cffa4e729f25",
        ),
    ];

    /// R-2 (R1 Spec review): the crate's `candidate_fingerprint` was pinned by
    /// **nothing**.
    ///
    /// `tests/s11_freeze.rs` reimplements §17's recipe locally, so it pins its
    /// own copy and not this function; and NC-20a's width control builds its
    /// receipts with the very function it tests, so both sides move together.
    /// Decoupling `candidate_fingerprint` from [`RULE_SOURCE_TEXT`] entirely
    /// therefore failed no test in any of the fourteen binaries — the receipt
    /// names *this candidate* only up to **width**, never up to **rule
    /// identity**.
    ///
    /// This pins all three, and pins them twice over: against the frozen literal
    /// values, and against §17's recipe re-derived here from the frozen rule
    /// digest **constant** rather than from the crate's own source text — so a
    /// change to either the rule or the recipe fails, and the two checks cannot
    /// drift together.
    #[test]
    fn r2_candidate_fingerprint_is_pinned_to_the_three_frozen_widths() {
        // (i) rule identity — the input half of §17's recipe.
        assert_eq!(
            digest::sha256_hex(RULE_SOURCE_TEXT.as_bytes()),
            FROZEN_RULE_SOURCE_DIGEST,
            "R-2: the candidate rule's source digest moved; §17's fingerprints rest on it"
        );

        // The pin table must cover exactly the swept widths, or a width could
        // be added to the sweep and silently escape the pin.
        let pinned_widths: Vec<u32> = FROZEN_CANDIDATE_FINGERPRINTS
            .iter()
            .map(|(width, _)| *width)
            .collect();
        assert_eq!(
            pinned_widths.as_slice(),
            SWEPT_WIDTHS.as_slice(),
            "R-2: every swept width must be pinned"
        );

        for (window_bytes, frozen) in FROZEN_CANDIDATE_FINGERPRINTS {
            // (ii) the crate function against the frozen value.
            assert_eq!(
                candidate_fingerprint(window_bytes),
                frozen,
                "R-2: candidate_fingerprint({window_bytes}) moved off its frozen §17 value"
            );

            // (iii) the crate function against §17's recipe, re-derived from the
            // frozen rule digest constant:
            //   sha256( ascii(rule_source_digest) ‖ 0x00 ‖ ascii("window_bytes=<w>") )
            let mut recipe = FROZEN_RULE_SOURCE_DIGEST.as_bytes().to_vec();
            recipe.push(0);
            recipe.extend_from_slice(format!("window_bytes={window_bytes}").as_bytes());
            assert_eq!(
                digest::sha256_hex(&recipe),
                frozen,
                "R-2: §17's recipe re-derived from the frozen rule digest must give the \
                 frozen fingerprint at w={window_bytes}"
            );
        }

        // Non-vacuity: three widths, three distinct fingerprints.
        let mut distinct: Vec<&str> = FROZEN_CANDIDATE_FINGERPRINTS
            .iter()
            .map(|(_, frozen)| *frozen)
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 3, "R-2: the three pins must be distinct");
    }

    /// NC-25 preserved verbatim on the other side: `NoModelFitted` must still
    /// be *carried*. Demoting it from a bar must not delete the disclosure.
    #[test]
    fn nc_25_no_model_fitted_is_still_disclosed_on_every_unadmitted_run() {
        let json = emitted(None);
        assert!(json.contains(r#""reason":"NoEligibleUnseenPopulation""#));
        assert!(
            json.contains("NoModelFitted"),
            "the disclosure must survive the demotion; got {json}"
        );
        assert_eq!(
            json,
            r#"{"status":"Unknown","reason":"NoEligibleUnseenPopulation","also_applicable":["SyntheticOrExposedPopulationOnly","NoRealLabelPopulation","NoModelFitted"]}"#,
            "the unadmitted emission is byte-identical to the frozen S11 shape"
        );
    }
}
