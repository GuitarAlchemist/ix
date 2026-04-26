//! G-Set CRDT merge for hexavalent observations.
//!
//! Implements the merge rules specified in
//! `governance/demerzel/logic/hex-merge.md`. The public API surface
//! is intentionally small: one struct ([`HexObservation`]), one
//! merge function ([`merge`]), one result type ([`MergedState`]),
//! and a handful of pure helpers.
//!
//! # What this module does
//!
//! - Deduplicates observations by `(source, diagnosis_id, round, ordinal)`
//! - Groups observations by `claim_key`
//! - Synthesizes contradiction observations for same-claim opposite-
//!   polarity pairs using the Belnap-extended weight table from
//!   `hex-merge.md`
//! - Synthesizes meta-conflict observations for cross-aspect
//!   disagreements on the same action
//! - Applies the staleness budget (default K=5 rounds)
//! - Derives a [`HexavalentDistribution`] from the final observation
//!   set by summing weights per variant and normalizing
//!
//! # Proof obligations
//!
//! The test module verifies all six CRDT correctness obligations
//! from `hex-merge.md §CRDT Correctness Proof Obligations`:
//!
//! 1. Commutativity: `merge(A, B) == merge(B, A)`
//! 2. Associativity: `merge(merge(A, B), C) == merge(A, merge(B, C))`
//! 3. Idempotence: `merge(A, A) == A`
//! 4. Monotonicity: `|merge(A, B)| >= max(|A|, |B|)`
//! 5. Dedup by key: two observations with the same dedup key merge to one
//! 6. Belnap symmetry: synthesized C for `(T, F)` is the same
//!    regardless of which observation was added first
//!
//! # What this module does NOT do
//!
//! - Parse JSON (that's the SessionEvent layer's job)
//! - Authenticate sources (trusted by field value)
//! - Cross-claim reasoning (the LLM's job, not the merge's)
//! - Normalize claim_key strings (canonicalization is the agent's job)

use std::collections::BTreeMap;

use ix_types::Hexavalent;

use crate::error::FuzzyError;
use crate::hexavalent::{hexavalent_from_tpudfc, HexavalentDistribution};

/// Default staleness budget: observations from rounds older than
/// `current_round - K` are dropped before merging. Matches the
/// default in `hex-merge.md §Staleness Policy`.
pub const DEFAULT_STALENESS_K: u32 = 5;

/// Reserved aspect name for synthesized cross-aspect conflicts.
/// The merge function emits observations with this aspect; agents
/// must never emit it directly.
pub const META_CONFLICT_ASPECT: &str = "meta_conflict";

/// Reserved source name for synthesized contradictions produced by
/// this merge function. Observations from this source are derived,
/// not primary.
pub const MERGE_SOURCE: &str = "demerzel-merge";

/// A single hexavalent observation contributed by one source about
/// one claim. The shape matches `SessionEvent::ObservationAdded` in
/// `ix-agent-core` and the JSON Schema in
/// `demerzel/schemas/session-event.schema.json`.
///
/// `ix-fuzzy` doesn't depend on `ix-agent-core` (the dependency
/// direction is the other way), so this type is defined locally.
/// Callers converting from a `SessionEvent::ObservationAdded` use
/// a simple field-by-field construction.
#[derive(Debug, Clone, PartialEq)]
pub struct HexObservation {
    /// Emitting agent identifier (e.g. `"tars"`, `"ix"`,
    /// `"demerzel-merge"`).
    pub source: String,
    /// Content hash of the originating diagnosis. Used for
    /// correlation and dedup.
    pub diagnosis_id: String,
    /// Remediation round number. Used by the staleness filter.
    pub round: u32,
    /// Monotone position within `(source, diagnosis_id, round)`.
    pub ordinal: u32,
    /// The claim this observation takes a position on. See
    /// `hex-merge.md §Claim Key Grammar` — format is
    /// `action_key::aspect`.
    pub claim_key: String,
    /// Hexavalent value the source is asserting.
    pub variant: Hexavalent,
    /// Confidence weight in `(0.0, 1.0]`.
    pub weight: f64,
    /// Optional audit-trail evidence string. Not used by the merge.
    pub evidence: Option<String>,
}

impl HexObservation {
    /// Deduplication key: `(source, diagnosis_id, round, ordinal)`.
    /// Two observations with the same key are the same observation.
    pub fn dedup_key(&self) -> (String, String, u32, u32) {
        (
            self.source.clone(),
            self.diagnosis_id.clone(),
            self.round,
            self.ordinal,
        )
    }

    /// Split the `claim_key` into `(action_key, aspect)`. Returns
    /// the full claim_key as action_key if there's no `::` separator.
    ///
    /// Uses `rfind` so claim keys whose action_key contains `::`
    /// (common for Rust test names like `test:my_module::test_fn`)
    /// split correctly: the aspect is the text after the LAST `::`,
    /// not the first. See `demerzel/logic/harness-cargo.md` §"Worked
    /// example" for the motivating case.
    pub fn action_and_aspect(&self) -> (&str, &str) {
        match self.claim_key.rfind("::") {
            Some(idx) => (&self.claim_key[..idx], &self.claim_key[idx + 2..]),
            None => (self.claim_key.as_str(), "valuable"),
        }
    }

    /// `true` iff the variant is on the positive side of the truth
    /// axis (T or P). Used by the meta-conflict detection rule.
    pub fn is_positive(&self) -> bool {
        matches!(self.variant, Hexavalent::True | Hexavalent::Probable)
    }

    /// `true` iff the variant is on the negative side (D or F).
    pub fn is_negative(&self) -> bool {
        matches!(self.variant, Hexavalent::Doubtful | Hexavalent::False)
    }
}

/// Result of merging a set of observations. Carries both the raw
/// observation set (after deduplication and staleness filtering)
/// AND the synthesized contradictions, plus the derived
/// distribution.
#[derive(Debug, Clone)]
pub struct MergedState {
    /// Deduplicated, staleness-filtered observations from all
    /// sources, plus any synthesized contradiction observations.
    pub observations: Vec<HexObservation>,
    /// Synthesized contradiction observations only (subset of
    /// `observations` with `source == "demerzel-merge"`). Useful
    /// for audit display without rescanning.
    pub contradictions: Vec<HexObservation>,
    /// Derived hexavalent distribution: `sum(weights per variant) /
    /// total`. Callers pass this to
    /// [`crate::hexavalent::escalation_triggered`] for the plan-
    /// level escalation gate.
    pub distribution: HexavalentDistribution,
}

/// The Belnap-extended weight table from `hex-merge.md §Belnap-
/// extended Contradiction Table`. Returns `Some(multiplier)` if the
/// pair should synthesize a `C` observation, or `None` if not.
///
/// The multiplier is applied to `min(weight_a, weight_b)` to
/// compute the synthesized observation's weight.
pub fn belnap_weight(a: Hexavalent, b: Hexavalent) -> Option<f64> {
    use Hexavalent::*;
    // Normalize order — the table is symmetric across the diagonal.
    let (lo, hi) = if variant_rank(a) <= variant_rank(b) {
        (a, b)
    } else {
        (b, a)
    };
    match (lo, hi) {
        // T + F = full contradiction
        (True, False) => Some(1.0),
        // T + D = strong (definite vs leaning)
        (True, Doubtful) => Some(0.8),
        // P + F = strong (leaning vs definite)
        (Probable, False) => Some(0.8),
        // P + D = soft (both sides leaning)
        (Probable, Doubtful) => Some(0.5),
        // All other pairs: no synthesis.
        // - Same-side (T+P, D+F) = agreement with different confidence
        // - U + anything = unknown preserves
        // - C + anything = already contradictory
        _ => None,
    }
}

/// Stable ordering over hexavalent variants so `belnap_weight` can
/// normalize `(a, b)` → canonical `(lo, hi)`.
fn variant_rank(v: Hexavalent) -> u8 {
    match v {
        Hexavalent::True => 0,
        Hexavalent::Probable => 1,
        Hexavalent::Unknown => 2,
        Hexavalent::Doubtful => 3,
        Hexavalent::False => 4,
        Hexavalent::Contradictory => 5,
    }
}

/// Content-derived `diagnosis_id` for a synthesized observation.
/// Produced from the sorted dedup keys of the contributing
/// observations plus a `kind` discriminator and the target
/// `claim_key`. Two calls with the same inputs — even across
/// separate merge invocations — produce the same id, so dedup
/// collapses re-merged synthesis output. This is the property that
/// restores associativity.
fn synthesis_diagnosis_id(
    kind: &str,
    claim_key: &str,
    a: &HexObservation,
    b: &HexObservation,
) -> String {
    let ka = format!("{}|{}|{}|{}", a.source, a.diagnosis_id, a.round, a.ordinal);
    let kb = format!("{}|{}|{}|{}", b.source, b.diagnosis_id, b.round, b.ordinal);
    // Canonicalize order so (a, b) and (b, a) produce the same id.
    let (lo, hi) = if ka <= kb { (ka, kb) } else { (kb, ka) };
    format!("merge:{kind}:{claim_key}:{lo}+{hi}")
}

/// Merge a set of observations into a [`MergedState`]. Implements
/// the full pipeline specified in `hex-merge.md`:
///
/// 1. Deduplicate by `(source, diagnosis_id, round, ordinal)`
/// 2. Apply staleness filter: drop obs with `round < current_round - K`
/// 3. Group by claim_key, synthesize direct contradictions per the
///    Belnap-extended table
/// 4. Group by action_key (dropping aspect), synthesize meta-
///    conflicts for cross-aspect disagreements
/// 5. Derive hexavalent distribution by summing per-variant weights
///    and normalizing
///
/// `current_round` and `staleness_k` may be `None` to skip the
/// staleness step (useful in tests and for full-history merges).
pub fn merge(
    observations: &[HexObservation],
    current_round: Option<u32>,
    staleness_k: Option<u32>,
) -> Result<MergedState, FuzzyError> {
    // ── Step 1: deduplicate by dedup key ──────────────────────────
    // BTreeMap keeps order deterministic so the merge output is
    // reproducible across runs regardless of input order. This is
    // load-bearing for the CRDT correctness properties.
    let mut by_key: BTreeMap<(String, String, u32, u32), HexObservation> = BTreeMap::new();
    for obs in observations {
        by_key.entry(obs.dedup_key()).or_insert_with(|| obs.clone());
    }

    // ── Step 2: staleness filter ──────────────────────────────────
    if let (Some(current), Some(k)) = (current_round, staleness_k) {
        let cutoff = current.saturating_sub(k);
        by_key.retain(|_, obs| obs.round >= cutoff);
    }

    let deduped: Vec<HexObservation> = by_key.into_values().collect();

    // ── Step 3: direct contradictions by claim_key ───────────────
    //
    // Synthesized observations use **content-derived dedup keys** so
    // that re-merging produces bit-identical contradictions. Without
    // this, a running counter would give each call different
    // ordinals, breaking associativity: merge(merge(A,B),C) would
    // produce C with ordinal=0 in the outer merge, and
    // merge(A,merge(B,C)) would too, but their dedup keys would
    // collide with the nested call's output.
    let mut synthesized: Vec<HexObservation> = Vec::new();
    let mut by_claim: BTreeMap<String, Vec<&HexObservation>> = BTreeMap::new();
    for obs in &deduped {
        by_claim.entry(obs.claim_key.clone()).or_default().push(obs);
    }

    for (claim_key, obs_list) in &by_claim {
        // Skip already-synthesized meta_conflict entries — don't
        // double-derive from our own output.
        if claim_key.ends_with(&format!("::{META_CONFLICT_ASPECT}")) {
            continue;
        }
        for (i, a) in obs_list.iter().enumerate() {
            for b in obs_list.iter().skip(i + 1) {
                if a.source == b.source {
                    // Same source disagreeing with itself is not a
                    // cross-source contradiction — skip.
                    continue;
                }
                // Skip pairs where either side is already a merge
                // synthesis — contradictions don't contradict their
                // own products.
                if a.source == MERGE_SOURCE || b.source == MERGE_SOURCE {
                    continue;
                }
                if let Some(mult) = belnap_weight(a.variant, b.variant) {
                    let weight = mult * a.weight.min(b.weight);
                    synthesized.push(HexObservation {
                        source: MERGE_SOURCE.to_string(),
                        diagnosis_id: synthesis_diagnosis_id("direct", claim_key, a, b),
                        round: a.round.max(b.round),
                        ordinal: 0,
                        claim_key: claim_key.clone(),
                        variant: Hexavalent::Contradictory,
                        weight,
                        evidence: Some(format!(
                            "{}:{:?} vs {}:{:?}",
                            a.source, a.variant, b.source, b.variant
                        )),
                    });
                }
            }
        }
    }

    // ── Step 4: meta-conflicts (cross-aspect) ────────────────────
    let mut by_action: BTreeMap<String, Vec<&HexObservation>> = BTreeMap::new();
    for obs in &deduped {
        let (action, _aspect) = obs.action_and_aspect();
        by_action.entry(action.to_string()).or_default().push(obs);
    }

    for (action, obs_list) in &by_action {
        let positives: Vec<&HexObservation> = obs_list
            .iter()
            .filter(|o| o.is_positive())
            .copied()
            .collect();
        let negatives: Vec<&HexObservation> = obs_list
            .iter()
            .filter(|o| o.is_negative())
            .copied()
            .collect();

        for pos in &positives {
            for neg in &negatives {
                // Skip if same aspect (already handled by direct
                // contradictions above).
                let (_, pos_aspect) = pos.action_and_aspect();
                let (_, neg_aspect) = neg.action_and_aspect();
                if pos_aspect == neg_aspect {
                    continue;
                }
                if pos.source == neg.source {
                    // Single source disagreeing with itself across
                    // aspects — not a cross-source conflict.
                    continue;
                }
                if pos.source == MERGE_SOURCE || neg.source == MERGE_SOURCE {
                    // Don't re-derive from our own output.
                    continue;
                }
                let meta_claim = format!("{action}::{META_CONFLICT_ASPECT}");
                synthesized.push(HexObservation {
                    source: MERGE_SOURCE.to_string(),
                    diagnosis_id: synthesis_diagnosis_id("meta", &meta_claim, pos, neg),
                    round: pos.round.max(neg.round),
                    ordinal: 0,
                    claim_key: meta_claim,
                    variant: Hexavalent::Contradictory,
                    weight: pos.weight.min(neg.weight),
                    evidence: Some(format!(
                        "cross-aspect: {}:{}:{:?} vs {}:{}:{:?}",
                        pos.source, pos_aspect, pos.variant, neg.source, neg_aspect, neg.variant
                    )),
                });
            }
        }
    }

    // Deduplicate synthesized observations by their content-derived
    // dedup key. Two passes of the same algorithm on the same inputs
    // produce identical diagnosis_ids, so dedup collapses re-merges
    // cleanly. This is half of the associativity fix.
    //
    // The other half: when re-merging input that already contains
    // merge-synthesized observations, those carried-over entries
    // have the SAME content-derived dedup key as the newly-
    // synthesized ones. We must collapse them together, not count
    // them twice. That's done below when we build `all`.
    let mut synth_by_key: BTreeMap<(String, String, u32, u32), HexObservation> = BTreeMap::new();
    for s in synthesized {
        synth_by_key.entry(s.dedup_key()).or_insert(s);
    }
    let synthesized: Vec<HexObservation> = synth_by_key.into_values().collect();

    // ── Step 5: derive distribution ──────────────────────────────
    //
    // Combine primary observations (which may include carried-over
    // merge-synthesized entries from a previous merge call) with the
    // newly-synthesized observations from this call. Deduplicate by
    // content-derived key so carried-over contradictions don't get
    // counted twice. This is the other half of the associativity
    // fix.
    let mut all_by_key: BTreeMap<(String, String, u32, u32), HexObservation> = BTreeMap::new();
    for obs in deduped.iter().chain(synthesized.iter()) {
        all_by_key
            .entry(obs.dedup_key())
            .or_insert_with(|| obs.clone());
    }
    let all: Vec<HexObservation> = all_by_key.into_values().collect();

    // Rebuild synthesized (for the caller's `contradictions` field)
    // from the deduplicated set so it reflects only observations
    // that were actually counted.
    let synthesized: Vec<HexObservation> = all
        .iter()
        .filter(|o| o.source == MERGE_SOURCE)
        .cloned()
        .collect();

    let mut weights = [0.0_f64; 6];
    for obs in &all {
        let idx = variant_rank(obs.variant) as usize;
        weights[idx] += obs.weight;
    }
    let total: f64 = weights.iter().sum();
    let distribution = if total == 0.0 {
        // No observations or all zero-weighted — fall back to
        // uniform so escalation_triggered sees a well-formed input.
        HexavalentDistribution::uniform(vec![
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ])?
    } else {
        hexavalent_from_tpudfc(
            weights[0] / total, // T
            weights[1] / total, // P
            weights[2] / total, // U
            weights[3] / total, // D
            weights[4] / total, // F
            weights[5] / total, // C
        )?
    };

    Ok(MergedState {
        observations: all,
        contradictions: synthesized,
        distribution,
    })
}

/// Convenience: merge with the default staleness budget and a
/// current round of 0 (disables staleness filtering — equivalent to
/// passing `None, None`).
pub fn merge_all(observations: &[HexObservation]) -> Result<MergedState, FuzzyError> {
    merge(observations, None, None)
}

/// Convenience: merge applying the default staleness budget
/// (`DEFAULT_STALENESS_K = 5`) against a given current round.
pub fn merge_with_default_staleness(
    observations: &[HexObservation],
    current_round: u32,
) -> Result<MergedState, FuzzyError> {
    merge(observations, Some(current_round), Some(DEFAULT_STALENESS_K))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical ordering of `MergedState.observations` so tests comparing
    /// two MergedStates from different merge orderings can use equality
    /// modulo ordering. Sorts by the dedup key first, then claim_key and
    /// variant to stabilize merge-synthesized entries which all share
    /// source "demerzel-merge".
    fn canonicalize_observations(obs: &mut [HexObservation]) {
        obs.sort_by(|a, b| {
            a.source
                .cmp(&b.source)
                .then(a.diagnosis_id.cmp(&b.diagnosis_id))
                .then(a.round.cmp(&b.round))
                .then(a.ordinal.cmp(&b.ordinal))
                .then(a.claim_key.cmp(&b.claim_key))
                .then(variant_rank(a.variant).cmp(&variant_rank(b.variant)))
        });
    }

    /// Test fixture: build an observation with minimal ceremony.
    fn obs(
        source: &str,
        diagnosis_id: &str,
        round: u32,
        ordinal: u32,
        claim_key: &str,
        variant: Hexavalent,
        weight: f64,
    ) -> HexObservation {
        HexObservation {
            source: source.to_string(),
            diagnosis_id: diagnosis_id.to_string(),
            round,
            ordinal,
            claim_key: claim_key.to_string(),
            variant,
            weight,
            evidence: None,
        }
    }

    /// Compare two MergedStates for equality modulo observation
    /// ordering. Distributions must be bit-exactly equal (they're
    /// derived from the same sums, so there's no reason they
    /// wouldn't be).
    fn states_equal(a: &MergedState, b: &MergedState) -> bool {
        let mut a_obs = a.observations.clone();
        let mut b_obs = b.observations.clone();
        canonicalize_observations(&mut a_obs);
        canonicalize_observations(&mut b_obs);
        if a_obs.len() != b_obs.len() {
            return false;
        }
        for (x, y) in a_obs.iter().zip(b_obs.iter()) {
            if x != y {
                return false;
            }
        }
        // Distributions: compare by component within 1e-9.
        let variants = [
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ];
        for v in variants {
            let da = a.distribution.get(&v);
            let db = b.distribution.get(&v);
            if (da - db).abs() > 1e-9 {
                return false;
            }
        }
        true
    }

    // ────────────────────────────────────────────────────────────
    // Proof obligation 1: Commutativity
    // ────────────────────────────────────────────────────────────

    #[test]
    fn proof_commutativity() {
        let a = [
            obs(
                "tars",
                "dx1",
                0,
                0,
                "ix_stats::valuable",
                Hexavalent::True,
                0.8,
            ),
            obs(
                "ix",
                "dx1",
                0,
                1,
                "ix_stats::valuable",
                Hexavalent::Probable,
                0.6,
            ),
        ];
        let b = [obs(
            "tars",
            "dx2",
            0,
            0,
            "ix_fft::valuable",
            Hexavalent::Doubtful,
            0.7,
        )];

        let ab: Vec<HexObservation> = a.iter().cloned().chain(b.iter().cloned()).collect();
        let ba: Vec<HexObservation> = b.iter().cloned().chain(a.iter().cloned()).collect();

        let s_ab = merge_all(&ab).unwrap();
        let s_ba = merge_all(&ba).unwrap();

        assert!(
            states_equal(&s_ab, &s_ba),
            "commutativity violated:\n  A then B: {:#?}\n  B then A: {:#?}",
            s_ab.observations,
            s_ba.observations
        );
    }

    // ────────────────────────────────────────────────────────────
    // Proof obligation 2: Associativity
    // ────────────────────────────────────────────────────────────

    #[test]
    fn proof_associativity() {
        let a = [obs("tars", "d", 0, 0, "k::valuable", Hexavalent::True, 1.0)];
        let b = [obs("ix", "d", 0, 1, "k::valuable", Hexavalent::False, 1.0)];
        let c = [obs("tars", "d", 0, 2, "k::safe", Hexavalent::Probable, 0.5)];

        // merge(merge(A, B), C)
        let ab_all: Vec<HexObservation> = a.iter().cloned().chain(b.iter().cloned()).collect();
        let ab_state = merge_all(&ab_all).unwrap();
        // Take ab_state's observations (includes synthesized) and merge
        // them with C. This tests whether merging twice produces the
        // same result as merging once.
        let ab_then_c: Vec<HexObservation> = ab_state
            .observations
            .iter()
            .cloned()
            .chain(c.iter().cloned())
            .collect();
        let left_assoc = merge_all(&ab_then_c).unwrap();

        // merge(A, merge(B, C))
        let bc_all: Vec<HexObservation> = b.iter().cloned().chain(c.iter().cloned()).collect();
        let bc_state = merge_all(&bc_all).unwrap();
        let a_then_bc: Vec<HexObservation> = a
            .iter()
            .cloned()
            .chain(bc_state.observations.iter().cloned())
            .collect();
        let right_assoc = merge_all(&a_then_bc).unwrap();

        assert!(
            states_equal(&left_assoc, &right_assoc),
            "associativity violated"
        );
    }

    // ────────────────────────────────────────────────────────────
    // Proof obligation 3: Idempotence
    // ────────────────────────────────────────────────────────────

    #[test]
    fn proof_idempotence() {
        let a = vec![
            obs("tars", "d", 0, 0, "k::valuable", Hexavalent::True, 0.8),
            obs("ix", "d", 0, 1, "k::valuable", Hexavalent::False, 0.9),
        ];
        let doubled: Vec<HexObservation> = a.iter().cloned().chain(a.iter().cloned()).collect();

        let once = merge_all(&a).unwrap();
        let twice = merge_all(&doubled).unwrap();

        assert!(
            states_equal(&once, &twice),
            "idempotence violated: merging A ∪ A should equal merging A"
        );
    }

    // ────────────────────────────────────────────────────────────
    // Proof obligation 4: Monotonicity
    // ────────────────────────────────────────────────────────────

    #[test]
    fn proof_monotonicity() {
        let a = vec![
            obs("tars", "d1", 0, 0, "k::valuable", Hexavalent::True, 0.8),
            obs("ix", "d1", 0, 1, "k::valuable", Hexavalent::Probable, 0.6),
        ];
        let b = vec![obs(
            "tars",
            "d2",
            0,
            0,
            "k::safe",
            Hexavalent::Doubtful,
            0.5,
        )];
        let ab: Vec<HexObservation> = a.iter().cloned().chain(b.iter().cloned()).collect();

        let sa = merge_all(&a).unwrap();
        let sb = merge_all(&b).unwrap();
        let sab = merge_all(&ab).unwrap();

        assert!(
            sab.observations.len() >= sa.observations.len(),
            "monotonicity violated: |merge(A∪B)|={} < |merge(A)|={}",
            sab.observations.len(),
            sa.observations.len()
        );
        assert!(
            sab.observations.len() >= sb.observations.len(),
            "monotonicity violated: |merge(A∪B)|={} < |merge(B)|={}",
            sab.observations.len(),
            sb.observations.len()
        );
    }

    // ────────────────────────────────────────────────────────────
    // Proof obligation 5: Deduplication by key
    // ────────────────────────────────────────────────────────────

    #[test]
    fn proof_dedup_by_key() {
        // Same dedup key = same observation. The second copy (with
        // DIFFERENT weight) should be ignored — first-write-wins on
        // tied keys.
        let a = obs("tars", "d", 0, 0, "k::valuable", Hexavalent::True, 0.8);
        let a_dup = HexObservation {
            weight: 0.3, // different, should be ignored
            ..a.clone()
        };
        let result = merge_all(&[a.clone(), a_dup]).unwrap();

        // Only one primary observation (plus any synthesized).
        let primary_count = result
            .observations
            .iter()
            .filter(|o| o.source == "tars")
            .count();
        assert_eq!(primary_count, 1, "dedup should collapse to one");
        // The remaining copy should be the first one.
        let remaining = result
            .observations
            .iter()
            .find(|o| o.source == "tars")
            .unwrap();
        assert!((remaining.weight - 0.8).abs() < 1e-9);
    }

    // ────────────────────────────────────────────────────────────
    // Proof obligation 6: Belnap symmetry
    // ────────────────────────────────────────────────────────────

    #[test]
    fn proof_belnap_symmetry() {
        let a = obs("tars", "d", 0, 0, "k::valuable", Hexavalent::True, 1.0);
        let b = obs("ix", "d", 0, 1, "k::valuable", Hexavalent::False, 1.0);

        let ab = merge_all(&[a.clone(), b.clone()]).unwrap();
        let ba = merge_all(&[b.clone(), a.clone()]).unwrap();

        // Both should produce exactly one synthesized C observation
        // with weight 1.0 (T+F=1.0 multiplier × min(1.0, 1.0)).
        assert_eq!(ab.contradictions.len(), 1);
        assert_eq!(ba.contradictions.len(), 1);
        assert_eq!(
            ab.contradictions[0].weight, ba.contradictions[0].weight,
            "Belnap symmetry violated"
        );
        assert!((ab.contradictions[0].weight - 1.0).abs() < 1e-9);
    }

    // ────────────────────────────────────────────────────────────
    // Functional tests
    // ────────────────────────────────────────────────────────────

    #[test]
    fn belnap_table_matches_spec() {
        use Hexavalent::*;

        // Strong contradictions
        assert_eq!(belnap_weight(True, False), Some(1.0));
        assert_eq!(belnap_weight(False, True), Some(1.0));

        assert_eq!(belnap_weight(True, Doubtful), Some(0.8));
        assert_eq!(belnap_weight(Doubtful, True), Some(0.8));

        assert_eq!(belnap_weight(Probable, False), Some(0.8));
        assert_eq!(belnap_weight(False, Probable), Some(0.8));

        assert_eq!(belnap_weight(Probable, Doubtful), Some(0.5));
        assert_eq!(belnap_weight(Doubtful, Probable), Some(0.5));

        // Same-side pairs (agreement, NOT contradiction)
        assert_eq!(belnap_weight(True, Probable), None);
        assert_eq!(belnap_weight(Probable, True), None);
        assert_eq!(belnap_weight(Doubtful, False), None);
        assert_eq!(belnap_weight(False, Doubtful), None);

        // Unknown preserves
        for v in [True, Probable, Unknown, Doubtful, False, Contradictory] {
            assert_eq!(
                belnap_weight(Unknown, v),
                None,
                "U + {:?} should not synthesize",
                v
            );
            assert_eq!(
                belnap_weight(v, Unknown),
                None,
                "{:?} + U should not synthesize",
                v
            );
        }

        // Contradictory is terminal
        for v in [True, Probable, Unknown, Doubtful, False, Contradictory] {
            assert_eq!(belnap_weight(Contradictory, v), None);
            assert_eq!(belnap_weight(v, Contradictory), None);
        }

        // Same-variant pairs — no self-contradiction
        for v in [True, Probable, Unknown, Doubtful, False, Contradictory] {
            assert_eq!(belnap_weight(v, v), None);
        }
    }

    #[test]
    fn tars_ix_agreement_produces_no_contradiction() {
        // tars says T, ix says P — both positive, no synthesis.
        let obs_list = vec![
            obs(
                "tars",
                "d",
                0,
                0,
                "ix_stats::valuable",
                Hexavalent::True,
                0.9,
            ),
            obs(
                "ix",
                "d",
                0,
                1,
                "ix_stats::valuable",
                Hexavalent::Probable,
                0.7,
            ),
        ];
        let state = merge_all(&obs_list).unwrap();
        assert_eq!(state.contradictions.len(), 0);
        assert!(state.distribution.get(&Hexavalent::Contradictory) < 1e-9);
    }

    #[test]
    fn tars_ix_disagreement_escalates() {
        // tars says T (verified helpful), ix trace says F (refuted).
        // Should synthesize C with full weight.
        let obs_list = vec![
            obs(
                "tars",
                "d",
                0,
                0,
                "ix_git_gc::valuable",
                Hexavalent::True,
                1.0,
            ),
            obs(
                "ix",
                "d",
                0,
                1,
                "ix_git_gc::valuable",
                Hexavalent::False,
                1.0,
            ),
        ];
        let state = merge_all(&obs_list).unwrap();
        assert_eq!(state.contradictions.len(), 1);
        assert_eq!(state.contradictions[0].variant, Hexavalent::Contradictory);
        assert_eq!(state.contradictions[0].weight, 1.0);
        // Distribution: 3 observations (T, F, C) with weights 1+1+1=3
        // → C mass = 1/3 ≈ 0.33, above the escalation threshold 0.3.
        let c_mass = state.distribution.get(&Hexavalent::Contradictory);
        assert!(c_mass > 0.33 - 1e-9, "expected C mass ~0.333, got {c_mass}");
        assert!(crate::hexavalent::escalation_triggered(&state.distribution));
    }

    #[test]
    fn meta_conflict_cross_aspect_same_action() {
        // tars says restart_gpu is valuable=T, ix says it's safe=F.
        // Different claim_keys → no direct contradiction, but SAME
        // action_key → meta_conflict should fire.
        let obs_list = vec![
            obs(
                "tars",
                "d",
                0,
                0,
                "restart_gpu::valuable",
                Hexavalent::True,
                0.9,
            ),
            obs("ix", "d", 0, 1, "restart_gpu::safe", Hexavalent::False, 1.0),
        ];
        let state = merge_all(&obs_list).unwrap();
        let meta_conflicts: Vec<&HexObservation> = state
            .contradictions
            .iter()
            .filter(|o| o.claim_key.ends_with("::meta_conflict"))
            .collect();
        assert_eq!(
            meta_conflicts.len(),
            1,
            "expected one meta_conflict, got contradictions: {:#?}",
            state.contradictions
        );
        assert_eq!(meta_conflicts[0].variant, Hexavalent::Contradictory);
        // Weight = min(0.9, 1.0) = 0.9
        assert!((meta_conflicts[0].weight - 0.9).abs() < 1e-9);
    }

    #[test]
    fn same_source_cross_aspect_is_not_meta_conflict() {
        // A single source saying its own valuable=T and safe=F is
        // self-inconsistent, not a cross-source conflict. The
        // meta-conflict rule explicitly skips same-source pairs.
        let obs_list = vec![
            obs(
                "tars",
                "d",
                0,
                0,
                "restart_gpu::valuable",
                Hexavalent::True,
                0.9,
            ),
            obs(
                "tars",
                "d",
                0,
                1,
                "restart_gpu::safe",
                Hexavalent::False,
                1.0,
            ),
        ];
        let state = merge_all(&obs_list).unwrap();
        assert_eq!(state.contradictions.len(), 0);
    }

    #[test]
    fn staleness_filter_drops_old_rounds() {
        let obs_list = vec![
            obs("tars", "d", 0, 0, "k::valuable", Hexavalent::True, 1.0), // round 0
            obs("tars", "d", 3, 0, "k::valuable", Hexavalent::True, 1.0), // round 3
            obs("tars", "d", 10, 0, "k::valuable", Hexavalent::True, 1.0), // round 10
        ];
        // current_round=10, K=5 → cutoff=5, keep only round >= 5
        let state = merge(&obs_list, Some(10), Some(5)).unwrap();
        assert_eq!(state.observations.len(), 1);
        assert_eq!(state.observations[0].round, 10);
    }

    #[test]
    fn empty_input_yields_uniform_distribution() {
        let state = merge_all(&[]).unwrap();
        assert_eq!(state.observations.len(), 0);
        // Uniform: each variant ≈ 1/6
        let variants = [
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ];
        for v in variants {
            let mass = state.distribution.get(&v);
            assert!(
                (mass - 1.0 / 6.0).abs() < 1e-9,
                "expected uniform 1/6, got {mass} for {:?}",
                v
            );
        }
    }

    #[test]
    fn dedup_preserves_first_write() {
        // Per BTreeMap::entry::or_insert_with semantics, the first
        // observation with a given key wins. This is deterministic
        // across runs because inputs are processed in slice order,
        // and the test exercises the guarantee so a future
        // refactor can't silently change it without tripping.
        let a = obs("tars", "d", 0, 0, "k::valuable", Hexavalent::True, 0.5);
        let b = HexObservation {
            weight: 0.9,
            ..a.clone()
        };
        let state = merge_all(&[a, b]).unwrap();
        let primary = state
            .observations
            .iter()
            .find(|o| o.source == "tars")
            .unwrap();
        assert!((primary.weight - 0.5).abs() < 1e-9);
    }

    #[test]
    fn action_and_aspect_splits_correctly() {
        let o = obs("s", "d", 0, 0, "ix_stats::valuable", Hexavalent::True, 1.0);
        let (action, aspect) = o.action_and_aspect();
        assert_eq!(action, "ix_stats");
        assert_eq!(aspect, "valuable");

        // Rust test names contain `::` in their module paths. With
        // `rfind`, the LAST `::` is the aspect separator, so the
        // full test path becomes the action_key and the trailing
        // aspect parses cleanly.
        let o2 = obs(
            "s",
            "d",
            0,
            0,
            "test:ix_math::eigen::jacobi::valuable",
            Hexavalent::True,
            1.0,
        );
        let (action, aspect) = o2.action_and_aspect();
        assert_eq!(action, "test:ix_math::eigen::jacobi");
        assert_eq!(aspect, "valuable");
    }

    #[test]
    fn action_and_aspect_handles_deep_module_paths_in_test_names() {
        // Regression guard for the cargo adapter case: test names
        // with multiple `::` must parse so the aspect is always the
        // final segment.
        let o = obs(
            "cargo",
            "d",
            0,
            0,
            "test:foo::bar::baz::qux::timely",
            Hexavalent::Doubtful,
            0.6,
        );
        let (action, aspect) = o.action_and_aspect();
        assert_eq!(action, "test:foo::bar::baz::qux");
        assert_eq!(aspect, "timely");
    }

    #[test]
    fn action_and_aspect_default_aspect_when_no_delimiter() {
        let o = obs("s", "d", 0, 0, "ix_stats", Hexavalent::True, 1.0);
        let (action, aspect) = o.action_and_aspect();
        assert_eq!(action, "ix_stats");
        assert_eq!(aspect, "valuable");
    }
}
