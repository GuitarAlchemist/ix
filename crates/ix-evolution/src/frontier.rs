//! Deterministic Pareto frontier over a **long-form** objective table.
//!
//! This is the layer around [`crate::pareto`], not a second copy of it. The
//! dominance rule itself is [`crate::pareto::rank`] and is not re-derived here;
//! what this module adds is the relational contract that a warehouse row set
//! arrives in:
//!
//! ```text
//! (subject_revision, task_class, candidate_id, metric, direction, value)
//! ```
//!
//! validated fail-closed, grouped per `(subject_revision, task_class)`, pivoted
//! into a wide objective vector under a canonical metric order, ranked by
//! [`crate::pareto::rank`], and emitted as front 0 only.
//!
//! # Determinism
//!
//! The same input multiset produces byte-identical output regardless of the
//! order rows arrive in. That rests on one claim, spelled out because
//! "deterministic" is otherwise unfalsifiable:
//!
//! **Output rows are totally ordered by `(subject_revision, task_class,
//! candidate_id)` compared as byte sequences, and that triple is a key.**
//!
//! It is a key because [`FrontierError::DuplicateMetric`] rejects any input
//! that repeats `(revision, class, candidate, metric)`, so a candidate appears
//! at most once per group and therefore contributes at most one output row.
//! Lexicographic order over a key is *total*: two distinct output rows differ
//! in at least one component of the triple, and byte comparison of two distinct
//! byte strings is antisymmetric and never answers "equal". No residual tie
//! survives for a secondary rule to break, which is what makes the ordering a
//! function of the input set rather than of the input sequence.
//!
//! **Ties in objective values are a separate thing and do not threaten this.**
//! Two candidates with identical objective vectors are mutually non-dominated —
//! neither is strictly better on any objective — so both stay on the frontier.
//! A value tie never merges, drops, or reorders rows; it just yields two rows
//! that the triple above then separates by `candidate_id`.
//!
//! Within a row, metric names are sorted ascending before pivoting, so the
//! objective vector is positionally identical no matter what order the metric
//! rows arrived in. Grouping and per-group candidate iteration go through
//! `BTreeMap`/`BTreeSet`, never a hash container, so no iteration order is
//! left to a random seed.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::pareto::{rank, Candidate, Objective, ParetoError};

/// Text spelling of [`Objective::Minimize`] accepted in the `direction` column.
pub const DIRECTION_MIN: &str = "MIN";
/// Text spelling of [`Objective::Maximize`] accepted in the `direction` column.
pub const DIRECTION_MAX: &str = "MAX";

/// One cell of the long-form objective table: what one candidate scored on one
/// metric, and which way that metric improves.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveRow {
    pub subject_revision: String,
    pub task_class: String,
    pub candidate_id: String,
    pub metric: String,
    /// `"MIN"` or `"MAX"`; anything else is rejected, never guessed.
    pub direction: String,
    pub value: f64,
}

impl ObjectiveRow {
    pub fn new(
        subject_revision: impl Into<String>,
        task_class: impl Into<String>,
        candidate_id: impl Into<String>,
        metric: impl Into<String>,
        direction: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            subject_revision: subject_revision.into(),
            task_class: task_class.into(),
            candidate_id: candidate_id.into(),
            metric: metric.into(),
            direction: direction.into(),
            value,
        }
    }
}

/// One objective of an emitted frontier row, in canonical (metric-ascending)
/// position.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontierObjective {
    pub metric: String,
    pub direction: Objective,
    pub value: f64,
}

/// A non-dominated candidate, echoing the exact revision and task class it came
/// from. Advisory and read-only: nothing here writes state or gates anything.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontierRow {
    pub subject_revision: String,
    pub task_class: String,
    pub candidate_id: String,
    /// Objective values in canonical metric-ascending order — the same order
    /// for every row of the same `(subject_revision, task_class)` group.
    pub objectives: Vec<FrontierObjective>,
}

/// Every way the input table can be rejected. There is no partial or
/// best-effort mode: one violation aborts the whole run.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FrontierError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error(
        "direction {direction:?} for metric {metric} of candidate {candidate_id} is not MIN or MAX"
    )]
    UnknownDirection {
        candidate_id: String,
        metric: String,
        direction: String,
    },
    #[error("candidate {candidate_id} in {subject_revision}/{task_class} repeats metric {metric}")]
    DuplicateMetric {
        subject_revision: String,
        task_class: String,
        candidate_id: String,
        metric: String,
    },
    #[error(
        "metric {metric} in {subject_revision}/{task_class} declares both {first} and {second}"
    )]
    MixedDirection {
        subject_revision: String,
        task_class: String,
        metric: String,
        first: String,
        second: String,
    },
    #[error(
        "candidate {candidate_id} in {subject_revision}/{task_class} is missing metric {metric}"
    )]
    MissingMetric {
        subject_revision: String,
        task_class: String,
        candidate_id: String,
        metric: String,
    },
    #[error("value of metric {metric} for candidate {candidate_id} is not finite")]
    NonFiniteValue {
        candidate_id: String,
        metric: String,
    },
    #[error("at least one objective row is required")]
    EmptyInput,
    /// The dominance kernel rejected an input this module thought it had
    /// already made legal. Surfaced rather than unwrapped so a contract gap
    /// shows up as an error instead of a panic.
    #[error("dominance kernel rejected a validated group: {0}")]
    Kernel(#[from] ParetoError),
}

/// Group key. A `BTreeMap` over this tuple iterates in exactly the
/// `(subject_revision, task_class)` byte order the output contract promises.
type GroupKey = (String, String);

/// One candidate's objectives, keyed by metric name. `BTreeMap`, so the metric
/// order is byte-lexicographic rather than insertion-dependent.
type CandidateObjectives = BTreeMap<String, (Objective, f64)>;

/// Every candidate of one `(subject_revision, task_class)` group, keyed by
/// candidate id — again a `BTreeMap` so iteration order is a function of the
/// ids, not of the order rows arrived in.
type GroupCandidates = BTreeMap<String, CandidateObjectives>;

/// Compute the Pareto frontier of a long-form objective table.
///
/// Returns front 0 only, ordered by `(subject_revision, task_class,
/// candidate_id)`. Dominance is delegated to [`crate::pareto::rank`].
///
/// # Errors
///
/// Fails closed on every violation in [`FrontierError`]. Checks run in a fixed
/// sequence and each scans the input in canonical sorted order, so the *error*
/// a bad table produces is as order-independent as the frontier a good one
/// produces.
// @ai:invariant frontier output is byte-identical under any permutation of the input rows [T:test conf:0.99 src:tests/frontier.rs]
// @ai:invariant a candidate whose objective vector equals another's stays on the frontier (dominance is strict) [T:test conf:0.99 src:tests/frontier.rs]
pub fn frontier(rows: &[ObjectiveRow]) -> Result<Vec<FrontierRow>, FrontierError> {
    // Canonical scan order for validation. Sorting indices (not rows) keeps the
    // caller's slice untouched while making every error message a function of
    // the input *set*, not the input *sequence*.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&left, &right| sort_key(&rows[left]).cmp(&sort_key(&rows[right])));

    validate(rows, &order)?;

    // Pivot. BTreeMap at every level: group order, candidate order, and metric
    // order are all byte-lexicographic and none of them depends on insertion.
    let mut groups: BTreeMap<GroupKey, GroupCandidates> = BTreeMap::new();
    for row in rows {
        let direction = parse_direction(&row.direction).expect("validated above");
        groups
            .entry((row.subject_revision.clone(), row.task_class.clone()))
            .or_default()
            .entry(row.candidate_id.clone())
            .or_default()
            .insert(row.metric.clone(), (direction, row.value));
    }

    let mut out = Vec::new();
    for ((subject_revision, task_class), candidates) in groups {
        // Canonical metric order for this group: ascending metric name. Every
        // candidate in the group is already known to carry exactly this set
        // (MissingMetric / DuplicateMetric ran above), so one sorted list fixes
        // the vector layout for the whole group.
        let metrics: Vec<String> = candidates
            .values()
            .flat_map(|by_metric| by_metric.keys().cloned())
            .collect::<BTreeSet<String>>()
            .into_iter()
            .collect();
        let objectives: Vec<Objective> = metrics
            .iter()
            .map(|metric| {
                candidates
                    .values()
                    .next()
                    .and_then(|by_metric| by_metric.get(metric))
                    .map(|(direction, _)| *direction)
                    .expect("validated above")
            })
            .collect();

        let ranked: Vec<Candidate> = candidates
            .iter()
            .map(|(candidate_id, by_metric)| {
                let values: Vec<f64> = metrics.iter().map(|metric| by_metric[metric].1).collect();
                Candidate::new(candidate_id.clone(), values)
            })
            .collect();

        let archive = rank(&ranked, &objectives)?;
        // `rank` already sorts each front by candidate id, so front 0 arrives in
        // the order the output contract wants.
        for candidate_id in archive.front(0).unwrap_or_default() {
            let by_metric = &candidates[candidate_id];
            out.push(FrontierRow {
                subject_revision: subject_revision.clone(),
                task_class: task_class.clone(),
                candidate_id: candidate_id.clone(),
                objectives: metrics
                    .iter()
                    .map(|metric| {
                        let (direction, value) = by_metric[metric];
                        FrontierObjective {
                            metric: metric.clone(),
                            direction,
                            value,
                        }
                    })
                    .collect(),
            });
        }
    }

    Ok(out)
}

/// Render a frontier as the canonical CSV that the DuckDB SQL surface
/// (`crates/ix-duck/sql/pareto_frontier.sql`) must reproduce byte for byte.
///
/// Values are formatted with six fixed decimals, which Rust's `{:.6}` and C's
/// `%.6f` agree on; dominance itself always used the full `f64`. Lines are
/// separated by `\n` on every platform — a `\r` would be a byte difference, and
/// the point of this function is that there are none.
pub fn to_csv(rows: &[FrontierRow]) -> String {
    let mut out = String::from("subject_revision,task_class,candidate_id,objectives\n");
    for row in rows {
        let objectives = row
            .objectives
            .iter()
            .map(|objective| {
                format!(
                    "{}:{}={:.6}",
                    objective.metric,
                    direction_text(objective.direction),
                    objective.value
                )
            })
            .collect::<Vec<_>>()
            .join(";");
        out.push_str(&format!(
            "{},{},{},{}\n",
            row.subject_revision, row.task_class, row.candidate_id, objectives
        ));
    }
    out
}

fn direction_text(objective: Objective) -> &'static str {
    match objective {
        Objective::Minimize => DIRECTION_MIN,
        Objective::Maximize => DIRECTION_MAX,
    }
}

fn parse_direction(direction: &str) -> Option<Objective> {
    match direction {
        DIRECTION_MIN => Some(Objective::Minimize),
        DIRECTION_MAX => Some(Objective::Maximize),
        _ => None,
    }
}

fn sort_key(row: &ObjectiveRow) -> (&str, &str, &str, &str) {
    (
        &row.subject_revision,
        &row.task_class,
        &row.candidate_id,
        &row.metric,
    )
}

/// Fixed-sequence validation. The passes below run in the order documented in
/// `docs/plans/2026-09-07-feat-deterministic-pareto-frontier-pipeline.md`, and
/// each walks `order` (the canonical sort) rather than the caller's sequence,
/// so a table with several defects always reports the same one.
fn validate(rows: &[ObjectiveRow], order: &[usize]) -> Result<(), FrontierError> {
    // 1. empty identifiers.
    for &index in order {
        let row = &rows[index];
        for (field, value) in [
            ("subject_revision", &row.subject_revision),
            ("task_class", &row.task_class),
            ("candidate_id", &row.candidate_id),
            ("metric", &row.metric),
        ] {
            if value.trim().is_empty() {
                return Err(FrontierError::EmptyField { field });
            }
        }
    }

    // 2. direction spelling.
    for &index in order {
        let row = &rows[index];
        if parse_direction(&row.direction).is_none() {
            return Err(FrontierError::UnknownDirection {
                candidate_id: row.candidate_id.clone(),
                metric: row.metric.clone(),
                direction: row.direction.clone(),
            });
        }
    }

    // 3. duplicate (revision, class, candidate, metric). In canonical order a
    //    duplicate is always an adjacent pair, so one linear scan suffices.
    for window in order.windows(2) {
        let previous = &rows[window[0]];
        let current = &rows[window[1]];
        if sort_key(previous) == sort_key(current) {
            return Err(FrontierError::DuplicateMetric {
                subject_revision: current.subject_revision.clone(),
                task_class: current.task_class.clone(),
                candidate_id: current.candidate_id.clone(),
                metric: current.metric.clone(),
            });
        }
    }

    // 4. one direction per (revision, class, metric).
    let mut directions: BTreeMap<(&str, &str, &str), &str> = BTreeMap::new();
    for &index in order {
        let row = &rows[index];
        let key = (
            row.subject_revision.as_str(),
            row.task_class.as_str(),
            row.metric.as_str(),
        );
        match directions.get(&key) {
            Some(first) if *first != row.direction => {
                return Err(FrontierError::MixedDirection {
                    subject_revision: row.subject_revision.clone(),
                    task_class: row.task_class.clone(),
                    metric: row.metric.clone(),
                    first: (*first).to_owned(),
                    second: row.direction.clone(),
                });
            }
            Some(_) => {}
            None => {
                directions.insert(key, &row.direction);
            }
        }
    }

    // 5. every candidate exposes the whole metric set its task class declares.
    let mut declared: BTreeMap<(&str, &str), BTreeSet<&str>> = BTreeMap::new();
    let mut present: BTreeMap<(&str, &str, &str), BTreeSet<&str>> = BTreeMap::new();
    for &index in order {
        let row = &rows[index];
        declared
            .entry((&row.subject_revision, &row.task_class))
            .or_default()
            .insert(&row.metric);
        present
            .entry((&row.subject_revision, &row.task_class, &row.candidate_id))
            .or_default()
            .insert(&row.metric);
    }
    for ((subject_revision, task_class, candidate_id), metrics) in &present {
        let expected = &declared[&(*subject_revision, *task_class)];
        if let Some(missing) = expected.difference(metrics).next() {
            return Err(FrontierError::MissingMetric {
                subject_revision: (*subject_revision).to_owned(),
                task_class: (*task_class).to_owned(),
                candidate_id: (*candidate_id).to_owned(),
                metric: (*missing).to_owned(),
            });
        }
    }

    // 6. finite values.
    for &index in order {
        let row = &rows[index];
        if !row.value.is_finite() {
            return Err(FrontierError::NonFiniteValue {
                candidate_id: row.candidate_id.clone(),
                metric: row.metric.clone(),
            });
        }
    }

    // 7. an empty table clears every pass above, so it is caught last.
    if rows.is_empty() {
        return Err(FrontierError::EmptyInput);
    }

    Ok(())
}
