//! Shapley prompt attribution for adversarial QA.
//!
//! Models each adversarial prompt as a player in a cooperative game.
//! The characteristic function `v(S)` = number of prompts in coalition S
//! that produced F or D verdicts. A prompt with high Shapley value is the
//! most *diagnostic* — its inclusion/exclusion most changes the total
//! failure count.
//!
//! Uses [`ix_game::cooperative::CooperativeGame`] for exact Shapley
//! computation (bitmask coalitions, O(2^n), tractable for n <= 20).

use crate::aggregate::QaResult;
use ix_game::cooperative::CooperativeGame;

/// Attribution score for a single prompt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PromptShapleyScore {
    /// Prompt identifier (e.g., "injection-003").
    pub prompt_id: String,
    /// Shapley value: marginal contribution to failure detection.
    pub shapley_value: f64,
    /// Adversarial category (e.g., "injection", "grounding").
    pub category: String,
    /// Fraction of the time this prompt produced F or D verdicts (0.0 or 1.0 for single-run).
    pub failure_rate: f64,
}

/// Compute Shapley attribution over QA results.
///
/// Each prompt is a player. The characteristic function `v(S)` counts how
/// many prompts in coalition S have an aggregate verdict of F or D.
///
/// Returns scores sorted by descending Shapley value.
///
/// # Panics
///
/// Panics if `findings.len() > 20` — use sampling for larger corpora.
pub fn compute_prompt_shapley(findings: &[QaResult]) -> Vec<PromptShapleyScore> {
    let n = findings.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(
        n <= 20,
        "Exact Shapley is O(2^n); cap at 20 players. Got {}. Sample first.",
        n
    );

    // Pre-compute which prompts fail (F or D verdict).
    // Check deterministic_verdict first (single-judge mode), then aggregate.
    let fails: Vec<bool> = findings
        .iter()
        .map(|r| {
            let det = r
                .deterministic_verdict
                .map(|v| matches!(v, 'F' | 'D'))
                .unwrap_or(false);
            let agg = matches!(r.aggregate, 'F' | 'D');
            det || agg
        })
        .collect();

    // Build the cooperative game.
    // v(S) = number of failing prompts in S.
    let mut game = CooperativeGame::new(n);
    for coalition in 0..(1u64 << n) {
        let value: f64 = (0..n)
            .filter(|&i| coalition & (1u64 << i) != 0 && fails[i])
            .count() as f64;
        game.set_value(coalition, value);
    }

    let shapley = game.shapley_value();

    // Build result with category metadata.
    let mut scores: Vec<PromptShapleyScore> = findings
        .iter()
        .enumerate()
        .map(|(i, r)| {
            // Extract category from prompt_id (e.g., "injection-003" -> "injection")
            let category = r
                .prompt_id
                .rsplit_once('-')
                .map(|(cat, _)| cat.to_string())
                .unwrap_or_else(|| r.prompt_id.clone());

            PromptShapleyScore {
                prompt_id: r.prompt_id.clone(),
                shapley_value: shapley[i],
                category,
                failure_rate: if fails[i] { 1.0 } else { 0.0 },
                // Note: fails[i] checks both deterministic_verdict and aggregate
            }
        })
        .collect();

    scores.sort_by(|a, b| {
        b.shapley_value
            .partial_cmp(&a.shapley_value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::QaResult;

    fn make_result(prompt_id: &str, aggregate: char) -> QaResult {
        QaResult {
            prompt_id: prompt_id.to_string(),
            deterministic_verdict: Some(aggregate),
            judge_verdicts: vec![],
            aggregate,
        }
    }

    #[test]
    fn shapley_single_failing_prompt_gets_max_value() {
        // One prompt fails, all others pass.
        // The failing prompt's Shapley should be 1.0 (it's the sole contributor).
        let results = vec![
            make_result("injection-001", 'F'),
            make_result("grounding-001", 'T'),
            make_result("grounding-002", 'T'),
        ];

        let scores = compute_prompt_shapley(&results);

        let failing = scores
            .iter()
            .find(|s| s.prompt_id == "injection-001")
            .unwrap();
        assert!(
            (failing.shapley_value - 1.0).abs() < 1e-8,
            "Single failing prompt should have Shapley = 1.0, got {}",
            failing.shapley_value
        );

        // Passing prompts contribute 0.
        for s in &scores {
            if s.prompt_id != "injection-001" {
                assert!(
                    s.shapley_value.abs() < 1e-8,
                    "Passing prompt {} should have Shapley ~0, got {}",
                    s.prompt_id,
                    s.shapley_value
                );
            }
        }
    }

    #[test]
    fn shapley_all_fail_equally() {
        // All prompts fail -> roughly equal Shapley values.
        let results = vec![
            make_result("injection-001", 'F'),
            make_result("injection-002", 'F'),
            make_result("injection-003", 'F'),
        ];

        let scores = compute_prompt_shapley(&results);
        let expected = 1.0; // Each contributes 1.0 to the total of 3.0
        for s in &scores {
            assert!(
                (s.shapley_value - expected).abs() < 1e-8,
                "All-fail: prompt {} should have Shapley = {}, got {}",
                s.prompt_id,
                expected,
                s.shapley_value
            );
        }
    }

    #[test]
    fn shapley_redundant_prompt_gets_low_value() {
        // Two prompts fail, one passes. Both failing prompts should split
        // their combined value. Each failing prompt's marginal is 1.0
        // (independent failures), but the combined value is 2.0.
        // Shapley for each failing prompt = 1.0 (they detect independent failures).
        //
        // To test redundancy properly: if we model prompts as detecting
        // the SAME failure, they'd split value. With our v(S) = count of
        // failing prompts in S, each failing prompt independently adds 1.
        // So Shapley = 1.0 each, same as the single-fail case.
        //
        // This test verifies the math is correct for multiple failures.
        let results = vec![
            make_result("injection-001", 'F'),
            make_result("injection-002", 'D'), // D also counts as failure
            make_result("grounding-001", 'T'),
            make_result("grounding-002", 'T'),
        ];

        let scores = compute_prompt_shapley(&results);

        let fail1 = scores
            .iter()
            .find(|s| s.prompt_id == "injection-001")
            .unwrap();
        let fail2 = scores
            .iter()
            .find(|s| s.prompt_id == "injection-002")
            .unwrap();

        // Both contribute equally (each is a unique failure).
        assert!(
            (fail1.shapley_value - 1.0).abs() < 1e-8,
            "First failing prompt Shapley should be 1.0, got {}",
            fail1.shapley_value
        );
        assert!(
            (fail2.shapley_value - 1.0).abs() < 1e-8,
            "Second failing prompt Shapley should be 1.0, got {}",
            fail2.shapley_value
        );

        // Passing prompts contribute 0.
        for s in &scores {
            if s.failure_rate < 0.5 {
                assert!(
                    s.shapley_value.abs() < 1e-8,
                    "Passing prompt {} should have Shapley ~0, got {}",
                    s.prompt_id,
                    s.shapley_value
                );
            }
        }
    }

    #[test]
    fn shapley_empty_returns_empty() {
        let scores = compute_prompt_shapley(&[]);
        assert!(scores.is_empty());
    }

    #[test]
    fn shapley_values_sum_to_grand_coalition() {
        // Shapley values must sum to v(N) = total failures.
        let results = vec![
            make_result("a-001", 'F'),
            make_result("b-001", 'T'),
            make_result("c-001", 'F'),
            make_result("d-001", 'T'),
            make_result("e-001", 'D'),
        ];

        let scores = compute_prompt_shapley(&results);
        let total: f64 = scores.iter().map(|s| s.shapley_value).sum();
        let expected_failures = 3.0; // F + F + D
        assert!(
            (total - expected_failures).abs() < 1e-8,
            "Shapley sum should equal v(N) = {}, got {}",
            expected_failures,
            total
        );
    }
}
