//! The Class-`Development` characterization: count vectors, per cell.
//!
//! **The output is a table of counts.** No ratio crosses a threshold, no rule
//! is declared a winner, and nothing here is out of sample. The corpus is
//! synthetic and its family support is citation-derived, which licenses a
//! description of *which documented drift shapes a bounded read catches, at
//! what byte cost* — and licenses nothing about how often real evidence drifts
//! or whether any window would catch it.
//!
//! Pooling is permitted only with the decomposition printed alongside: a pooled
//! number without its per-family counts is a weighted number with the weights
//! hidden, and there is no frequency source that could justify a weight.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;

use crate::rate::{CountVector, Ratio};
use crate::{
    advise_with, AdvisoryRefusal, AdvisoryRequest, EvaluationBinding, EvidenceClass, Hexavalent,
    Provenance, RuleId, SCHEMA_VERSION,
};

/// One `(rule, window_bytes)` pair to characterize. Swept, never selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleWidth {
    pub rule: RuleId,
    /// unit: bytes
    pub window_bytes: u32,
}

/// The declared inputs of one evaluation run.
#[derive(Debug, Clone)]
pub struct EvalRequest {
    /// Root of the locked corpus.
    pub corpus_root: PathBuf,
    /// The run refuses unless the on-disk corpus re-measures to this value.
    pub expected_corpus_digest_ix_agg_1: String,
    pub manifest_file_name: String,
    /// The cells to characterize, ordinal in the emitted report.
    pub cells: Vec<RuleWidth>,
    /// The pristine reference the Window Reference Table is derived from.
    pub window_reference_root: PathBuf,
    /// Only `Development` is populated. The other two are reserved and refused.
    pub evidence_class: EvidenceClass,
}

/// One `(rule, window_bytes, family, evidence_class)` cell.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cell {
    pub rule: RuleId,
    /// unit: bytes
    pub window_bytes: u32,
    pub family: String,
    pub evidence_class: EvidenceClass,
    /// The **primary** reported output. Every ratio below is derived from it.
    pub counts: CountVector,
    pub false_agreement: Ratio,
    pub detection: Ratio,
    pub false_drift: Ratio,
    pub unknown_rate: Ratio,
    pub coverage: Ratio,
    /// unit: bytes — summed over the cell's cases.
    pub bytes_read: u64,
    /// unit: bytes — the exact reference's cost over the same cases.
    pub reference_bytes_read: u64,
    /// dimensionless, both terms in bytes.
    pub cost_ratio: Ratio,
    /// SHA-256 of the Window Reference Table this cell's width used.
    pub window_reference_digest: String,
}

/// One pooled `(rule, window_bytes)` cell, always carrying the names of the
/// families it summed so the decomposition is never lost.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PooledCell {
    pub rule: RuleId,
    /// unit: bytes
    pub window_bytes: u32,
    pub evidence_class: EvidenceClass,
    /// Exactly the families summed into `counts`, ordinal.
    pub families: Vec<String>,
    pub counts: CountVector,
    pub false_agreement: Ratio,
    pub detection: Ratio,
    pub false_drift: Ratio,
    pub unknown_rate: Ratio,
    pub coverage: Ratio,
}

/// One Class-`Development` characterization report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub provenance: Provenance,
    /// unit: count
    pub cases_admitted: u64,
    /// Ordinal by rule declaration order, then width, then family.
    pub cells: Vec<Cell>,
    pub pooled: Vec<PooledCell>,
}

impl EvaluationReport {
    /// The report's canonical serialization. No map is ever serialized.
    pub fn to_canonical_json(&self) -> String {
        serde_json::to_string(self).expect("the report holds only strings, integers, and enums")
    }
}

/// One sealed case, as the corpus recorded it at lock time.
struct SealedCase {
    case_id: String,
    family: String,
    /// The oracle's label, computed over the staged bytes and sealed at lock
    /// time. `T` or `F`; there is no third label and no `L = U`.
    label: Hexavalent,
}

/// Characterize a locked corpus.
pub fn evaluate(request: &EvalRequest) -> Result<EvaluationReport, AdvisoryRefusal> {
    // The reserved classes are refused outright rather than quietly producing a
    // Development-shaped answer under another name.
    match request.evidence_class {
        EvidenceClass::Development => {}
        EvidenceClass::Holdout => {
            return Err(AdvisoryRefusal::ReservedEvidenceClass {
                class: EvidenceClass::Holdout,
                detail: "no population exists that is simultaneously available, entitled and \
                         unexposed, so no holdout is constructible"
                    .to_string(),
            })
        }
        EvidenceClass::Selection => {
            return Err(AdvisoryRefusal::ReservedEvidenceClass {
                class: EvidenceClass::Selection,
                detail: "no parameter is selected: the window width is swept, never chosen"
                    .to_string(),
            })
        }
    }

    // Invariant 2: the corpus must re-measure to the digest it was locked at.
    let measured = crate::digest::measure_root(&request.corpus_root)?;
    if measured.ix_agg_1 != request.expected_corpus_digest_ix_agg_1 {
        return Err(AdvisoryRefusal::CorpusDigestMismatch {
            expected: request.expected_corpus_digest_ix_agg_1.clone(),
            measured: measured.ix_agg_1,
        });
    }

    let cases = read_case_index(&request.corpus_root)?;

    // One Window Reference Table per swept width, and a single-artifact digest
    // over the whole sweep so the report binds every table it rested on.
    let mut widths: Vec<u32> = request.cells.iter().map(|cell| cell.window_bytes).collect();
    widths.sort_unstable();
    widths.dedup();
    let mut wrt_digests: BTreeMap<u32, String> = BTreeMap::new();
    for width in &widths {
        let table =
            crate::rules::window_probe::build(Some(&request.window_reference_root), *width)?;
        wrt_digests.insert(*width, table.digest);
    }

    let mut counts: BTreeMap<(usize, u32, String), CountVector> = BTreeMap::new();
    let mut bytes: BTreeMap<(usize, u32, String), (u64, u64)> = BTreeMap::new();

    for cell in &request.cells {
        let rule_index = rule_index(cell.rule);
        for case in &cases {
            let artifact = advise_with(
                &AdvisoryRequest {
                    evidence_root: request.corpus_root.join("dev").join(&case.case_id),
                    manifest_file_name: request.manifest_file_name.clone(),
                    rule: cell.rule,
                    window_bytes: cell.window_bytes,
                    window_reference_root: Some(request.window_reference_root.clone()),
                    expected_provenance: None,
                },
                &EvaluationBinding {
                    corpus_digest_ix_agg_1: Some(measured.ix_agg_1.clone()),
                    case_id: Some(case.case_id.clone()),
                    evidence_class: Some(EvidenceClass::Development),
                    // A Class-`Development` run earns no admission, and the
                    // two reserved classes never reach this line: they are
                    // refused at the top of `evaluate`.
                    earned_holdout: None,
                },
            )?;

            let key = (rule_index, cell.window_bytes, case.family.clone());
            let entry = counts.entry(key.clone()).or_default();
            tally(entry, case.label, artifact.reconciles);
            let cost = bytes.entry(key).or_insert((0, 0));
            cost.0 += artifact.cost.bytes_read;
            cost.1 += artifact.cost.reference_bytes_read;
        }
    }

    let cells: Vec<Cell> = counts
        .iter()
        .map(|((rule_index, window_bytes, family), counts)| {
            let (read, reference) = bytes[&(*rule_index, *window_bytes, family.clone())];
            Cell {
                rule: RuleId::all()[*rule_index],
                window_bytes: *window_bytes,
                family: family.clone(),
                evidence_class: EvidenceClass::Development,
                counts: *counts,
                false_agreement: counts.false_agreement(),
                detection: counts.detection(),
                false_drift: counts.false_drift(),
                unknown_rate: counts.unknown_rate(),
                coverage: counts.coverage(),
                bytes_read: read,
                reference_bytes_read: reference,
                cost_ratio: Ratio::new(read, reference),
                window_reference_digest: wrt_digests[window_bytes].clone(),
            }
        })
        .collect();

    let mut pooled_counts: BTreeMap<(usize, u32), (CountVector, Vec<String>)> = BTreeMap::new();
    for cell in &cells {
        let entry = pooled_counts
            .entry((rule_index(cell.rule), cell.window_bytes))
            .or_insert_with(|| (CountVector::default(), Vec::new()));
        entry.0 = entry.0.plus(&cell.counts);
        entry.1.push(cell.family.clone());
    }
    let pooled: Vec<PooledCell> = pooled_counts
        .into_iter()
        .map(
            |((rule_index, window_bytes), (counts, families))| PooledCell {
                rule: RuleId::all()[rule_index],
                window_bytes,
                evidence_class: EvidenceClass::Development,
                families,
                counts,
                false_agreement: counts.false_agreement(),
                detection: counts.detection(),
                false_drift: counts.false_drift(),
                unknown_rate: counts.unknown_rate(),
                coverage: counts.coverage(),
            },
        )
        .collect();

    let reference = crate::digest::measure_root(&request.window_reference_root)?;
    let manifest_bytes = std::fs::read(
        request
            .window_reference_root
            .join(&request.manifest_file_name),
    )
    .map_err(|e| AdvisoryRefusal::EvidenceRootUnreadable {
        detail: format!("{}: {e}", request.manifest_file_name),
    })?;

    Ok(EvaluationReport {
        schema_version: SCHEMA_VERSION,
        provenance: Provenance {
            manifest_file_name: request.manifest_file_name.clone(),
            manifest_sha256: crate::digest::sha256_hex(&manifest_bytes),
            evidence_aggregate_ix_agg_1: reference.ix_agg_1,
            declared_bundle_aggregate_gaia_agg_1: reference.gaia_agg_1,
            subject_aggregate_ix_agg_1: crate::SUBJECT_AGGREGATE_IX_AGG_1.to_string(),
            window_reference_digest: sweep_digest(&wrt_digests),
            rule_source_digest: crate::digest::sha256_hex(crate::RULE_SOURCE_TEXT.as_bytes()),
            corpus_digest_ix_agg_1: Some(measured.ix_agg_1),
            ledger_digest_ix_agg_1: None,
            case_id: None,
            evidence_class: Some(EvidenceClass::Development),
        },
        cases_admitted: cases.len() as u64,
        cells,
        pooled,
    })
}

/// A single-artifact digest over the whole sweep: one `width|digest` row per
/// swept width, ascending, LF-joined, no trailing newline. Not a multi-file
/// record aggregate, so §3.3's `*_agg_1` naming rule does not reach it.
fn sweep_digest(wrt_digests: &BTreeMap<u32, String>) -> String {
    let text = wrt_digests
        .iter()
        .map(|(width, digest)| format!("{width}|{digest}"))
        .collect::<Vec<_>>()
        .join("\n");
    crate::digest::sha256_hex(text.as_bytes())
}

/// Declaration order, which is the ordinal order the report uses.
fn rule_index(rule: RuleId) -> usize {
    RuleId::all()
        .iter()
        .position(|candidate| *candidate == rule)
        .expect("the rule set is closed")
}

/// Fold one case into its cell.
///
/// The truth label is `T` or `F` — a case whose label could not be established
/// never entered the corpus. The prediction is `T`, `F`, or **not bound**:
/// `Unknown` is not bound, and neither is `Contradictory`, which says two claim
/// sites disagree rather than that the root reconciles. `Probable` and
/// `Doubtful` are never produced.
fn tally(counts: &mut CountVector, label: Hexavalent, prediction: Hexavalent) {
    match (label, prediction) {
        (Hexavalent::True, Hexavalent::True) => counts.n_tt += 1,
        (Hexavalent::True, Hexavalent::False) => counts.n_tf += 1,
        (Hexavalent::True, _) => counts.n_tu += 1,
        (_, Hexavalent::True) => counts.n_ft += 1,
        (_, Hexavalent::False) => counts.n_ff += 1,
        (_, _) => counts.n_fu += 1,
    }
}

/// Read the labels the corpus sealed at lock time.
fn read_case_index(corpus_root: &std::path::Path) -> Result<Vec<SealedCase>, AdvisoryRefusal> {
    let path = corpus_root.join("cases.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| AdvisoryRefusal::CorpusUnreadable {
        detail: format!("{}: {e}", path.display()),
    })?;

    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|e| AdvisoryRefusal::CorpusUnreadable {
                detail: format!("cases.jsonl line {}: {e}", index + 1),
            })?;
        let field = |name: &str| -> Result<String, AdvisoryRefusal> {
            value
                .get(name)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| AdvisoryRefusal::CorpusUnreadable {
                    detail: format!("cases.jsonl line {} lacks {name}", index + 1),
                })
        };
        let label = match field("label")?.as_str() {
            "T" => Hexavalent::True,
            "F" => Hexavalent::False,
            other => {
                return Err(AdvisoryRefusal::CorpusUnreadable {
                    detail: format!(
                        "cases.jsonl line {} carries label {other:?}; a sealed label is T or F",
                        index + 1
                    ),
                })
            }
        };
        out.push(SealedCase {
            case_id: field("case_id")?,
            family: field("family")?,
            label,
        });
    }

    if out.is_empty() {
        return Err(AdvisoryRefusal::CorpusUnreadable {
            detail: "the corpus admits no case".to_string(),
        });
    }
    Ok(out)
}
