//! Coverage matrix + optimality analysis.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use thiserror::Error;

use crate::invariant::{Invariant, InvariantStatus};

/// Canonical dashboard envelope read by the GA dev dashboard
/// (`/test#dev/summary`) and any other consumer of structural-quality
/// snapshots. The envelope wraps the underlying [`Firings`] data
/// additively — the existing `exemplars` and `fired` fields are flattened
/// at the top level so legacy readers still parse correctly.
///
/// See `ga/state/quality/ga-harness/last.json` for the reference shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiringsEnvelope {
    /// Logical domain — always `"invariants"` for this producer.
    pub domain: String,
    /// RFC3339 UTC timestamp of when the snapshot was emitted.
    pub emitted_at: String,
    /// Headline metric name — always `"invariant_pass_ratio"`.
    pub metric_name: String,
    /// Headline metric value in `[0.0, 1.0]`: fraction of invariants
    /// that fired on at least one exemplar (i.e. non-orphan invariants).
    pub metric_value: f64,
    /// `"ok"` | `"warn"` | `"error"`. See [`OracleStatus`] mapping.
    pub oracle_status: String,
    /// Short human-readable summary.
    pub summary: String,
    /// Failing-invariant details, capped at 50 entries. Empty when all
    /// invariants fired at least once.
    pub problems: Vec<Problem>,
    /// Underlying firings data — preserved verbatim for the existing
    /// `ix-invariant-coverage` consumer that ingests `{exemplars, fired}`.
    #[serde(flatten)]
    pub firings: Firings,
}

/// One failing invariant surfaced in the envelope's `problems` array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Problem {
    /// Stable identifier (e.g. `"invariant-18"`).
    pub name: String,
    /// Human-readable explanation of why the invariant is failing.
    pub message: String,
}

impl FiringsEnvelope {
    /// Wrap a [`Firings`] payload with the canonical envelope.
    ///
    /// Status mapping:
    /// - all invariants fired on ≥1 exemplar (no orphans) → `"ok"`
    /// - any invariant has zero firings (orphaned) → `"warn"`
    /// - producer-detected catastrophe (currently unused — callers map
    ///   their own error paths to `"error"` via [`FiringsEnvelope::error`])
    ///   → `"error"`
    pub fn wrap(firings: Firings, emitted_at: String) -> Self {
        let total = firings.fired.len() as f64;
        let (passed, problems) = if total == 0.0 {
            (0.0, Vec::new())
        } else {
            let mut problems: Vec<Problem> = Vec::new();
            let mut passed = 0u64;
            for (id, hits) in &firings.fired {
                if hits.is_empty() {
                    if problems.len() < 50 {
                        problems.push(Problem {
                            name: format!("invariant-{id}"),
                            message: format!(
                                "invariant {id} did not fire on any exemplar (orphan)"
                            ),
                        });
                    }
                } else {
                    passed += 1;
                }
            }
            (passed as f64, problems)
        };

        let metric_value = if total == 0.0 { 0.0 } else { passed / total };
        let (oracle_status, summary) = if total == 0.0 {
            ("warn".to_string(), "no invariants evaluated".to_string())
        } else if problems.is_empty() {
            (
                "ok".to_string(),
                format!("{} of {} invariants passing", passed as u64, total as u64),
            )
        } else {
            (
                "warn".to_string(),
                format!(
                    "{} of {} invariants passing ({} orphaned)",
                    passed as u64,
                    total as u64,
                    problems.len()
                ),
            )
        };

        Self {
            domain: "invariants".to_string(),
            emitted_at,
            metric_name: "invariant_pass_ratio".to_string(),
            metric_value,
            oracle_status,
            summary,
            problems,
            firings,
        }
    }

    /// Build a catastrophic-error envelope when the producer can't run.
    /// `metric_value` is set to `0.0` and the `firings` field is the
    /// empty default so the JSON shape is stable for downstream readers.
    pub fn error(message: impl Into<String>, emitted_at: String) -> Self {
        let msg = message.into();
        Self {
            domain: "invariants".to_string(),
            emitted_at,
            metric_name: "invariant_pass_ratio".to_string(),
            metric_value: 0.0,
            oracle_status: "error".to_string(),
            summary: msg.clone(),
            problems: vec![Problem {
                name: "producer-error".to_string(),
                message: msg,
            }],
            firings: Firings::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("firings json: {0}")]
    Firings(String),
}

/// External firings: which invariants fired on which exemplars during a test run.
///
/// The producer of this JSON is a test suite (GA unit tests, ix-embedding-diagnostics,
/// voicing-audit, etc.) that knows about specific exemplars and invariant IDs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Firings {
    /// All exemplars the test run considered. Declared explicitly so a column can be all-zero
    /// (coverage gap) without the exemplar silently disappearing.
    pub exemplars: Vec<Exemplar>,
    /// Map of invariant id → list of exemplar ids that triggered it.
    pub fired: BTreeMap<u32, BTreeSet<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Exemplar {
    pub id: String,
    pub description: String,
    pub kind: String,
}

impl Firings {
    pub fn from_path(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("reading {}: {e}", path.display())))?;
        serde_json::from_str(&text).map_err(|e| Error::Firings(e.to_string()))
    }
}

/// Binary invariant × exemplar matrix.
pub struct CoverageMatrix {
    pub invariants: Vec<Invariant>,
    pub exemplars: Vec<Exemplar>,
    /// Row-major; rows.len() == invariants.len(), each row has exemplars.len() entries.
    rows: Vec<Vec<u8>>,
}

impl CoverageMatrix {
    pub fn new(invariants: Vec<Invariant>, firings: Firings) -> Self {
        let exemplar_index: BTreeMap<String, usize> = firings
            .exemplars
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id.clone(), i))
            .collect();

        let rows = invariants
            .iter()
            .map(|inv| {
                let mut row = vec![0u8; firings.exemplars.len()];
                if let Some(hits) = firings.fired.get(&inv.id) {
                    for hit in hits {
                        if let Some(&col) = exemplar_index.get(hit) {
                            row[col] = 1;
                        }
                    }
                }
                row
            })
            .collect();

        Self {
            invariants,
            exemplars: firings.exemplars,
            rows,
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.invariants.len(), self.exemplars.len())
    }

    /// GF(2) rank via Gaussian elimination. O(rows × cols × min(rows, cols)).
    #[allow(clippy::needless_range_loop)] // pivot algorithm is inherently index-addressed
    pub fn rank(&self) -> usize {
        if self.exemplars.is_empty() {
            return 0;
        }
        let mut mat: Vec<Vec<u8>> = self.rows.clone();
        let rows = mat.len();
        let cols = self.exemplars.len();

        let mut rank = 0;
        let mut col = 0;
        let mut r = 0;
        while r < rows && col < cols {
            // Find pivot in column `col` at or below row r
            let mut pivot = None;
            for i in r..rows {
                if mat[i][col] == 1 {
                    pivot = Some(i);
                    break;
                }
            }
            match pivot {
                Some(p) => {
                    mat.swap(r, p);
                    // Eliminate this column from every other row
                    for i in 0..rows {
                        if i != r && mat[i][col] == 1 {
                            for j in col..cols {
                                mat[i][j] ^= mat[r][j];
                            }
                        }
                    }
                    rank += 1;
                    r += 1;
                }
                None => {
                    // No pivot in this column; move on
                }
            }
            col += 1;
        }
        rank
    }

    /// Pairs of invariants with identical *non-empty* firing signatures — strict duplicates.
    /// Returned as (lower_id, higher_id) tuples, sorted.
    /// All-zero rows are excluded: with no exemplars hitting either one, "duplicate" has no
    /// meaningful content and would flood the report when firings data is sparse or absent.
    pub fn redundancy_pairs(&self) -> Vec<(u32, u32)> {
        let mut pairs = Vec::new();
        let n = self.invariants.len();
        for i in 0..n {
            if self.rows[i].iter().all(|&v| v == 0) {
                continue;
            }
            for j in (i + 1)..n {
                if self.rows[i] == self.rows[j] {
                    let a = self.invariants[i].id;
                    let b = self.invariants[j].id;
                    pairs.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
        pairs.sort();
        pairs
    }

    /// Invariants with zero firings — either untested or unreachable.
    pub fn orphan_invariants(&self) -> Vec<&Invariant> {
        self.invariants
            .iter()
            .zip(self.rows.iter())
            .filter(|(_, row)| row.iter().all(|&v| v == 0))
            .map(|(inv, _)| inv)
            .collect()
    }

    /// Exemplars no invariant fires on — gaps the catalog fails to catch.
    pub fn coverage_gaps(&self) -> Vec<&Exemplar> {
        (0..self.exemplars.len())
            .filter(|&col| self.rows.iter().all(|row| row[col] == 0))
            .map(|col| &self.exemplars[col])
            .collect()
    }

    pub fn report(&self) -> Report {
        let rank = self.rank();
        let redundancy_pairs = self.redundancy_pairs();
        let orphans: Vec<u32> = self.orphan_invariants().iter().map(|i| i.id).collect();
        let gaps: Vec<String> = self.coverage_gaps().iter().map(|e| e.id.clone()).collect();

        let orphans_tested = self
            .invariants
            .iter()
            .zip(self.rows.iter())
            .filter(|(inv, row)| {
                inv.status == InvariantStatus::Tested && row.iter().all(|&v| v == 0)
            })
            .map(|(inv, _)| inv.id)
            .collect();

        let verdict = if self.exemplars.is_empty() {
            OptimalityVerdict::NoEvidence
        } else if rank == self.invariants.len()
            && orphans.is_empty()
            && gaps.is_empty()
            && redundancy_pairs.is_empty()
        {
            OptimalityVerdict::Optimal
        } else {
            OptimalityVerdict::Suboptimal
        };

        Report {
            shape: self.shape(),
            rank,
            redundancy_pairs,
            orphan_invariants: orphans,
            orphan_invariants_claimed_tested: orphans_tested,
            coverage_gaps: gaps,
            verdict,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub shape: (usize, usize),
    pub rank: usize,
    pub redundancy_pairs: Vec<(u32, u32)>,
    pub orphan_invariants: Vec<u32>,
    /// Orphans with status `T` — the worst kind: asserted to be tested but no firing observed.
    /// Usually means the test is vacuous or the exemplar suite doesn't exercise it.
    pub orphan_invariants_claimed_tested: Vec<u32>,
    pub coverage_gaps: Vec<String>,
    pub verdict: OptimalityVerdict,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimalityVerdict {
    /// rank = n, no orphans, no gaps, no duplicate rows.
    Optimal,
    /// Any of the above fails.
    Suboptimal,
    /// No exemplars in firings — can't judge.
    NoEvidence,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv(id: u32) -> Invariant {
        Invariant {
            id,
            domain: "test".into(),
            description: format!("inv {id}"),
            artifact: "artifact".into(),
            status: InvariantStatus::Tested,
        }
    }

    fn ex(id: &str) -> Exemplar {
        Exemplar {
            id: id.into(),
            description: id.into(),
            kind: "Synthetic".into(),
        }
    }

    #[test]
    fn rank_of_identity_matrix_equals_dimension() {
        let invs = vec![inv(1), inv(2), inv(3)];
        let mut fired = BTreeMap::new();
        fired.insert(1, ["a"].into_iter().map(String::from).collect());
        fired.insert(2, ["b"].into_iter().map(String::from).collect());
        fired.insert(3, ["c"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("a"), ex("b"), ex("c")],
            fired,
        };
        let m = CoverageMatrix::new(invs, firings);
        assert_eq!(m.rank(), 3);
        assert!(m.redundancy_pairs().is_empty());
        assert!(m.orphan_invariants().is_empty());
        assert!(m.coverage_gaps().is_empty());
    }

    #[test]
    fn duplicate_rows_collapse_rank_and_show_as_redundant() {
        let invs = vec![inv(1), inv(2), inv(3)];
        let mut fired = BTreeMap::new();
        fired.insert(1, ["a"].into_iter().map(String::from).collect());
        fired.insert(2, ["a"].into_iter().map(String::from).collect()); // duplicate of 1
        fired.insert(3, ["b"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("a"), ex("b")],
            fired,
        };
        let m = CoverageMatrix::new(invs, firings);
        assert_eq!(m.rank(), 2);
        assert_eq!(m.redundancy_pairs(), vec![(1, 2)]);
    }

    #[test]
    fn orphan_invariant_has_no_firings_and_verdict_is_suboptimal() {
        let invs = vec![inv(1), inv(2)];
        let mut fired = BTreeMap::new();
        fired.insert(1, ["a"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("a")],
            fired,
        };
        let m = CoverageMatrix::new(invs, firings);
        let report = m.report();
        assert_eq!(report.orphan_invariants, vec![2]);
        assert_eq!(report.verdict, OptimalityVerdict::Suboptimal);
        // Inv 2 is status=Tested with zero firings — flagged as the worst orphan kind
        assert_eq!(report.orphan_invariants_claimed_tested, vec![2]);
    }

    #[test]
    fn coverage_gap_surfaces_exemplars_nothing_fires_on() {
        let invs = vec![inv(1)];
        let mut fired = BTreeMap::new();
        fired.insert(1, ["a"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("a"), ex("untested-case")],
            fired,
        };
        let m = CoverageMatrix::new(invs, firings);
        let report = m.report();
        assert_eq!(report.coverage_gaps, vec!["untested-case"]);
        assert_eq!(report.verdict, OptimalityVerdict::Suboptimal);
    }

    #[test]
    fn linear_combination_collapses_rank_below_row_count() {
        // Rows (GF(2)): a=[1,1,0], b=[0,1,1], c=a⊕b=[1,0,1]. Rank = 2, not 3.
        let invs = vec![inv(1), inv(2), inv(3)];
        let mut fired = BTreeMap::new();
        fired.insert(1, ["x", "y"].into_iter().map(String::from).collect());
        fired.insert(2, ["y", "z"].into_iter().map(String::from).collect());
        fired.insert(3, ["x", "z"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("x"), ex("y"), ex("z")],
            fired,
        };
        let m = CoverageMatrix::new(invs, firings);
        // These three rows are linearly dependent over GF(2) — rank is 2.
        assert_eq!(m.rank(), 2);
        // But they aren't strict duplicates, so no redundancy pair is returned.
        assert!(m.redundancy_pairs().is_empty());
    }

    #[test]
    fn empty_exemplars_yields_no_evidence_verdict() {
        let invs = vec![inv(1)];
        let firings = Firings::default();
        let m = CoverageMatrix::new(invs, firings);
        assert_eq!(m.report().verdict, OptimalityVerdict::NoEvidence);
    }

    #[test]
    fn envelope_wrap_all_passing_is_ok() {
        let mut fired = BTreeMap::new();
        fired.insert(1, ["a"].into_iter().map(String::from).collect());
        fired.insert(2, ["a", "b"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("a"), ex("b")],
            fired,
        };
        let env = FiringsEnvelope::wrap(firings, "2026-05-24T08:00:00Z".to_string());
        assert_eq!(env.domain, "invariants");
        assert_eq!(env.metric_name, "invariant_pass_ratio");
        assert_eq!(env.metric_value, 1.0);
        assert_eq!(env.oracle_status, "ok");
        assert!(env.problems.is_empty());
        assert_eq!(env.summary, "2 of 2 invariants passing");
        // Underlying firings preserved.
        assert_eq!(env.firings.exemplars.len(), 2);
        assert_eq!(env.firings.fired.len(), 2);
    }

    #[test]
    fn envelope_wrap_orphan_invariant_is_warn() {
        let mut fired = BTreeMap::new();
        fired.insert(1, ["a"].into_iter().map(String::from).collect());
        fired.insert(2, BTreeSet::new()); // orphan
        let firings = Firings {
            exemplars: vec![ex("a")],
            fired,
        };
        let env = FiringsEnvelope::wrap(firings, "2026-05-24T08:00:00Z".to_string());
        assert_eq!(env.oracle_status, "warn");
        assert_eq!(env.metric_value, 0.5);
        assert_eq!(env.problems.len(), 1);
        assert_eq!(env.problems[0].name, "invariant-2");
    }

    #[test]
    fn envelope_serializes_envelope_fields_at_root_with_firings_flattened() {
        let mut fired = BTreeMap::new();
        fired.insert(7, ["x"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("x")],
            fired,
        };
        let env = FiringsEnvelope::wrap(firings, "2026-05-24T08:00:00Z".to_string());
        let json = serde_json::to_value(&env).expect("serializes");
        let obj = json.as_object().expect("object");
        // Envelope fields at root.
        assert_eq!(
            obj.get("domain").and_then(|v| v.as_str()),
            Some("invariants")
        );
        assert_eq!(
            obj.get("metric_name").and_then(|v| v.as_str()),
            Some("invariant_pass_ratio")
        );
        assert!(obj.contains_key("metric_value"));
        assert_eq!(
            obj.get("oracle_status").and_then(|v| v.as_str()),
            Some("ok")
        );
        assert!(obj.contains_key("summary"));
        assert!(obj.contains_key("problems"));
        assert!(obj.contains_key("emitted_at"));
        // Original firings fields flattened to root — back-compat with legacy
        // consumers that read `{exemplars, fired}` directly.
        assert!(obj.contains_key("exemplars"));
        assert!(obj.contains_key("fired"));
    }

    #[test]
    fn envelope_round_trip_preserves_firings_for_legacy_consumer() {
        let mut fired = BTreeMap::new();
        fired.insert(3, ["a", "b"].into_iter().map(String::from).collect());
        let firings = Firings {
            exemplars: vec![ex("a"), ex("b")],
            fired,
        };
        let env = FiringsEnvelope::wrap(firings, "2026-05-24T08:00:00Z".to_string());
        let json = serde_json::to_string(&env).expect("ser");
        // Legacy consumer (ix-invariant-coverage) deserializes into `Firings`,
        // ignoring the additional envelope fields.
        let parsed: Firings = serde_json::from_str(&json).expect("de");
        assert_eq!(parsed.exemplars.len(), 2);
        assert_eq!(parsed.fired.len(), 1);
        assert_eq!(parsed.fired[&3].len(), 2);
    }

    #[test]
    fn envelope_problem_list_caps_at_50() {
        let mut fired = BTreeMap::new();
        for i in 0..60u32 {
            fired.insert(i, BTreeSet::new()); // every one is orphaned
        }
        let firings = Firings {
            exemplars: vec![ex("a")],
            fired,
        };
        let env = FiringsEnvelope::wrap(firings, "2026-05-24T08:00:00Z".to_string());
        assert_eq!(env.problems.len(), 50);
        assert_eq!(env.oracle_status, "warn");
    }

    #[test]
    fn envelope_error_constructor_yields_error_status() {
        let env = FiringsEnvelope::error(
            "producer crashed".to_string(),
            "2026-05-24T08:00:00Z".to_string(),
        );
        assert_eq!(env.oracle_status, "error");
        assert_eq!(env.metric_value, 0.0);
        assert_eq!(env.problems.len(), 1);
        assert_eq!(env.problems[0].name, "producer-error");
    }
}
