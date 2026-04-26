//! Hexavalent majority-vote aggregation for judge verdicts.
//!
//! Aggregates verdicts from multiple judges using the six-valued logic:
//! T (True), P (Probable), U (Unknown), D (Disputed), F (False), C (Contradictory).
//!
//! # Aggregation rules
//!
//! | Input pattern             | Aggregate |
//! |---------------------------|-----------|
//! | 3x T                      | T         |
//! | 2x T + 1x P              | T         |
//! | Any P, no F/D             | P         |
//! | 2+ agree F                | F         |
//! | Any D                     | D         |
//! | Judges contradict on facts| C         |
//! | No majority               | U         |

use serde::{Deserialize, Serialize};

/// A single judge's structured verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeVerdict {
    /// Identifier of the judge persona (e.g., "code-reviewer").
    pub judge: String,
    /// Hexavalent verdict: T, P, U, D, F, or C.
    pub verdict: char,
    /// Does every voicing ID in the response exist in the corpus?
    pub grounded: bool,
    /// Are the musical claims correct?
    pub accurate: bool,
    /// No injection, hallucination, or persona break?
    pub safe: bool,
    /// Human-readable reasoning (one sentence).
    pub reasoning: String,
    /// Free-form flags for downstream triage.
    pub flags: Vec<String>,
}

/// The aggregated QA result for a single prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaResult {
    /// Unique prompt identifier (e.g., "grounding-001").
    pub prompt_id: String,
    /// Verdict from deterministic checks (Layers 0-2), if run.
    pub deterministic_verdict: Option<char>,
    /// Verdicts from LLM judge panel (Layer 3).
    pub judge_verdicts: Vec<JudgeVerdict>,
    /// Final aggregated hexavalent verdict.
    pub aggregate: char,
}

/// Aggregate multiple judge verdicts into a single hexavalent verdict.
///
/// # Rules (applied in priority order)
///
/// 1. Any D in input -> D (disputed evidence propagates)
/// 2. Judges contradict on facts (one says T, another says F on `grounded`
///    or `safe`) -> C
/// 3. 2+ judges agree F -> F
/// 4. All T (3x T) -> T
/// 5. 2x T + 1x P -> T
/// 6. Any P and no F/D -> P
/// 7. No majority -> U
///
/// Empty input returns U.
pub fn aggregate_verdicts(verdicts: &[JudgeVerdict]) -> char {
    if verdicts.is_empty() {
        return 'U';
    }

    let chars: Vec<char> = verdicts.iter().map(|v| v.verdict).collect();

    // Rule 1: Any D -> D
    if chars.contains(&'D') {
        return 'D';
    }

    // Rule 2: Contradiction detection on boolean facts
    if has_fact_contradiction(verdicts) {
        return 'C';
    }

    // Rule 3: 2+ judges agree F -> F
    let f_count = chars.iter().filter(|&&c| c == 'F').count();
    if f_count >= 2 {
        return 'F';
    }

    // Rule 4: All T -> T
    let t_count = chars.iter().filter(|&&c| c == 'T').count();
    if t_count == chars.len() {
        return 'T';
    }

    // Rule 5: 2x T + 1x P -> T
    let p_count = chars.iter().filter(|&&c| c == 'P').count();
    if t_count >= 2 && p_count >= 1 && t_count + p_count == chars.len() {
        return 'T';
    }

    // Rule 6: Any P and no F/D -> P
    let has_f = f_count > 0;
    if p_count > 0 && !has_f {
        return 'P';
    }

    // Rule 7: No majority -> U
    'U'
}

/// Check if judges contradict on factual boolean dimensions.
///
/// A contradiction occurs when one judge says `grounded: true` and another
/// says `grounded: false`, or similarly for `safe`.
fn has_fact_contradiction(verdicts: &[JudgeVerdict]) -> bool {
    if verdicts.len() < 2 {
        return false;
    }

    let grounded_true = verdicts.iter().any(|v| v.grounded);
    let grounded_false = verdicts.iter().any(|v| !v.grounded);
    if grounded_true && grounded_false {
        return true;
    }

    let safe_true = verdicts.iter().any(|v| v.safe);
    let safe_false = verdicts.iter().any(|v| !v.safe);
    if safe_true && safe_false {
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judge(
        name: &str,
        verdict: char,
        grounded: bool,
        accurate: bool,
        safe: bool,
    ) -> JudgeVerdict {
        JudgeVerdict {
            judge: name.to_string(),
            verdict,
            grounded,
            accurate,
            safe,
            reasoning: format!("Test judge {}", name),
            flags: vec![],
        }
    }

    #[test]
    fn three_true_yields_true() {
        let verdicts = vec![
            judge("code-reviewer", 'T', true, true, true),
            judge("security-auditor", 'T', true, true, true),
            judge("music-theory", 'T', true, true, true),
        ];
        assert_eq!(aggregate_verdicts(&verdicts), 'T');
    }

    #[test]
    fn two_true_one_probable_yields_true() {
        let verdicts = vec![
            judge("code-reviewer", 'T', true, true, true),
            judge("security-auditor", 'T', true, true, true),
            judge("music-theory", 'P', true, true, true),
        ];
        assert_eq!(aggregate_verdicts(&verdicts), 'T');
    }

    #[test]
    fn any_p_no_f_d_yields_probable() {
        let verdicts = vec![
            judge("code-reviewer", 'P', true, true, true),
            judge("security-auditor", 'T', true, true, true),
            judge("music-theory", 'P', true, true, true),
        ];
        assert_eq!(aggregate_verdicts(&verdicts), 'P');
    }

    #[test]
    fn two_false_yields_false() {
        let verdicts = vec![
            judge("code-reviewer", 'F', true, true, true),
            judge("security-auditor", 'F', true, true, true),
            judge("music-theory", 'T', true, true, true),
        ];
        assert_eq!(aggregate_verdicts(&verdicts), 'F');
    }

    #[test]
    fn any_disputed_yields_disputed() {
        let verdicts = vec![
            judge("code-reviewer", 'T', true, true, true),
            judge("security-auditor", 'D', true, true, true),
            judge("music-theory", 'T', true, true, true),
        ];
        assert_eq!(aggregate_verdicts(&verdicts), 'D');
    }

    #[test]
    fn fact_contradiction_yields_contradictory() {
        // One judge says grounded, another says not grounded
        let verdicts = vec![
            judge("code-reviewer", 'T', true, true, true),
            judge("security-auditor", 'T', false, true, true),
            judge("music-theory", 'T', true, true, true),
        ];
        assert_eq!(aggregate_verdicts(&verdicts), 'C');
    }

    #[test]
    fn no_majority_yields_unknown() {
        // T + F + U with no 2+ agreement on F
        let verdicts = vec![
            judge("code-reviewer", 'T', true, true, true),
            judge("security-auditor", 'F', true, true, true),
            judge("music-theory", 'U', true, true, true),
        ];
        assert_eq!(aggregate_verdicts(&verdicts), 'U');
    }

    #[test]
    fn empty_verdicts_yields_unknown() {
        assert_eq!(aggregate_verdicts(&[]), 'U');
    }

    #[test]
    fn single_deterministic_verdict_true() {
        // When used with deterministic-only (single "judge"), a T verdict passes
        let verdicts = vec![judge("deterministic", 'T', true, true, true)];
        assert_eq!(aggregate_verdicts(&verdicts), 'T');
    }

    #[test]
    fn single_deterministic_verdict_false() {
        let verdicts = vec![judge("deterministic", 'F', true, true, true)];
        // Single F is not 2+ F, so no majority -> U
        // This is correct: a single failing judge is inconclusive
        assert_eq!(aggregate_verdicts(&verdicts), 'U');
    }
}
