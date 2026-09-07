//! Deterministic non-dominated sorting for multi-objective candidates.
//!
//! This module ranks an already evaluated candidate table. It does not run a
//! physics solver, search a design space, or turn a Nash equilibrium into a
//! Pareto front.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

/// Direction in which one objective improves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    Minimize,
    Maximize,
}

/// One evaluated candidate and its ordered objective values.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub values: Vec<f64>,
}

impl Candidate {
    pub fn new(id: impl Into<String>, values: impl Into<Vec<f64>>) -> Self {
        Self {
            id: id.into(),
            values: values.into(),
        }
    }
}

/// Deterministic Pareto ranks. Front zero is the non-dominated archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoArchive {
    fronts: Vec<Vec<String>>,
    ranks: BTreeMap<String, usize>,
}

impl ParetoArchive {
    pub fn front(&self, rank: usize) -> Option<&[String]> {
        self.fronts.get(rank).map(Vec::as_slice)
    }

    pub fn rank_of(&self, id: &str) -> Option<usize> {
        self.ranks.get(id).copied()
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParetoError {
    #[error("at least one candidate is required")]
    EmptyCandidates,
    #[error("at least one objective is required")]
    EmptyObjectives,
    #[error("candidate id must not be empty")]
    EmptyCandidateId,
    #[error("duplicate candidate id: {0}")]
    DuplicateCandidateId(String),
    #[error("candidate {id} has {actual} values, expected {expected}")]
    ObjectiveCount {
        id: String,
        expected: usize,
        actual: usize,
    },
    #[error("candidate {id} objective {objective} is not finite")]
    NonFiniteValue { id: String, objective: usize },
}

/// Rank evaluated candidates by Pareto dominance.
///
/// Every front is sorted by candidate id, making the result independent of
/// input row order. Equal objective vectors do not dominate one another.
// @ai:invariant Pareto ranks are independent of candidate row order and equal vectors remain co-nondominated [T:test conf:0.99 src:tests/pareto.rs]
pub fn rank(
    candidates: &[Candidate],
    objectives: &[Objective],
) -> Result<ParetoArchive, ParetoError> {
    validate(candidates, objectives)?;

    let mut dominates = vec![Vec::new(); candidates.len()];
    let mut dominated_by = vec![0usize; candidates.len()];

    for left in 0..candidates.len() {
        for right in (left + 1)..candidates.len() {
            if candidate_dominates(&candidates[left], &candidates[right], objectives) {
                dominates[left].push(right);
                dominated_by[right] += 1;
            } else if candidate_dominates(&candidates[right], &candidates[left], objectives) {
                dominates[right].push(left);
                dominated_by[left] += 1;
            }
        }
    }

    let mut current: Vec<usize> = dominated_by
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count == 0).then_some(index))
        .collect();
    let mut fronts = Vec::new();
    let mut ranks = BTreeMap::new();

    while !current.is_empty() {
        current.sort_by(|left, right| candidates[*left].id.cmp(&candidates[*right].id));
        let rank = fronts.len();
        let ids: Vec<String> = current
            .iter()
            .map(|index| candidates[*index].id.clone())
            .collect();
        for id in &ids {
            ranks.insert(id.clone(), rank);
        }
        fronts.push(ids);

        let mut next = Vec::new();
        for index in current {
            for &dominated in &dominates[index] {
                dominated_by[dominated] -= 1;
                if dominated_by[dominated] == 0 {
                    next.push(dominated);
                }
            }
        }
        current = next;
    }

    Ok(ParetoArchive { fronts, ranks })
}

fn validate(candidates: &[Candidate], objectives: &[Objective]) -> Result<(), ParetoError> {
    if candidates.is_empty() {
        return Err(ParetoError::EmptyCandidates);
    }
    if objectives.is_empty() {
        return Err(ParetoError::EmptyObjectives);
    }

    let mut ids = BTreeSet::new();
    for candidate in candidates {
        if candidate.id.trim().is_empty() {
            return Err(ParetoError::EmptyCandidateId);
        }
        if !ids.insert(candidate.id.clone()) {
            return Err(ParetoError::DuplicateCandidateId(candidate.id.clone()));
        }
        if candidate.values.len() != objectives.len() {
            return Err(ParetoError::ObjectiveCount {
                id: candidate.id.clone(),
                expected: objectives.len(),
                actual: candidate.values.len(),
            });
        }
        if let Some(objective) = candidate.values.iter().position(|value| !value.is_finite()) {
            return Err(ParetoError::NonFiniteValue {
                id: candidate.id.clone(),
                objective,
            });
        }
    }
    Ok(())
}

fn candidate_dominates(left: &Candidate, right: &Candidate, objectives: &[Objective]) -> bool {
    let mut strictly_better = false;
    for ((left, right), objective) in left.values.iter().zip(&right.values).zip(objectives) {
        let no_worse = match objective {
            Objective::Minimize => left <= right,
            Objective::Maximize => left >= right,
        };
        if !no_worse {
            return false;
        }
        strictly_better |= match objective {
            Objective::Minimize => left < right,
            Objective::Maximize => left > right,
        };
    }
    strictly_better
}
