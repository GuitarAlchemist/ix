//! Bayesian (Beta-Binomial) weighted grammar rules.
//!
//! Ports TARS `WeightedGrammar.fs`: Beta prior, softmax selection,
//! JSON persistence to `~/.machin/grammar_weights.json`.

use rand::Rng;
use serde::{Deserialize, Serialize};

pub type RuleId = String;

/// A grammar rule with a Beta-Binomial Bayesian weight.
///
/// The weight is the mean of the Beta distribution: `alpha / (alpha + beta)`.
/// Starts at 0.5 (uniform prior: alpha=1, beta=1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedRule {
    pub id: RuleId,
    /// Beta prior success count (pseudo-counts start at 1).
    pub alpha: f64,
    /// Beta prior failure count (pseudo-counts start at 1).
    pub beta: f64,
    /// Current weight = alpha / (alpha + beta).
    pub weight: f64,
    /// Grammar level / depth hint for the rule.
    pub level: usize,
    /// Source label (e.g. crate name, module, or grammar file).
    pub source: String,
}

impl WeightedRule {
    /// Create a new rule with uniform Beta(1,1) prior (weight = 0.5).
    pub fn new(id: impl Into<String>, level: usize, source: impl Into<String>) -> Self {
        WeightedRule {
            id: id.into(),
            alpha: 1.0,
            beta: 1.0,
            weight: 0.5,
            level,
            source: source.into(),
        }
    }
}

/// Bayesian update via Beta-Binomial conjugacy.
///
/// `success=true`  → alpha += 1 (rule produced a good derivation)
/// `success=false` → beta  += 1 (rule produced a bad derivation)
///
/// Returns the updated rule with refreshed `weight = alpha / (alpha + beta)`.
///
/// ```
/// use ix_grammar::weighted::{WeightedRule, bayesian_update};
/// let rule = WeightedRule::new("r1", 0, "test");
/// let updated = bayesian_update(&rule, true);
/// assert!((updated.weight - 2.0 / 3.0).abs() < 1e-10);
/// ```
pub fn bayesian_update(rule: &WeightedRule, success: bool) -> WeightedRule {
    let mut updated = rule.clone();
    if success {
        updated.alpha += 1.0;
    } else {
        updated.beta += 1.0;
    }
    updated.weight = updated.alpha / (updated.alpha + updated.beta);
    updated
}

/// Temperature-scaled softmax over rule weights.
///
/// Returns `(rule_id, probability)` pairs summing to 1.0.
/// Higher `temperature` → more uniform; lower → sharper peak at highest weight.
///
/// ```
/// use ix_grammar::weighted::{WeightedRule, softmax};
/// let rules = vec![
///     WeightedRule::new("a", 0, ""),
///     WeightedRule::new("b", 0, ""),
/// ];
/// let probs = softmax(&rules, 1.0);
/// let total: f64 = probs.iter().map(|(_, p)| p).sum();
/// assert!((total - 1.0).abs() < 1e-10);
/// ```
pub fn softmax(rules: &[WeightedRule], temperature: f64) -> Vec<(RuleId, f64)> {
    if rules.is_empty() {
        return vec![];
    }
    let temp = temperature.max(1e-10);
    let scaled: Vec<f64> = rules.iter().map(|r| r.weight / temp).collect();
    // Numerically stable: subtract max before exp
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    rules
        .iter()
        .zip(exp_vals.iter())
        .map(|(r, &e)| (r.id.clone(), e / sum))
        .collect()
}

/// Sample a rule id using softmax probabilities at temperature 1.0.
///
/// Returns `None` only if `rules` is empty.
///
/// ```
/// use ix_grammar::weighted::{WeightedRule, select_weighted};
/// use rand::SeedableRng;
/// let rules = vec![WeightedRule::new("x", 0, "")];
/// let mut rng = rand::rngs::StdRng::seed_from_u64(1);
/// assert_eq!(select_weighted(&rules, &mut rng), Some("x".to_string()));
/// ```
pub fn select_weighted(rules: &[WeightedRule], rng: &mut impl Rng) -> Option<RuleId> {
    let probs = softmax(rules, 1.0);
    if probs.is_empty() {
        return None;
    }
    let r: f64 = rng.random();
    let mut cumsum = 0.0;
    for (id, p) in &probs {
        cumsum += p;
        if r <= cumsum {
            return Some(id.clone());
        }
    }
    // Floating-point safety: return last
    probs.last().map(|(id, _)| id.clone())
}

/// Persist rules to a JSON file (e.g. `~/.machin/grammar_weights.json`).
pub fn save(rules: &[WeightedRule], path: &str) -> Result<(), String> {
    let json = serde_json::to_string_pretty(rules).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

/// Load rules from a JSON file.
pub fn load(path: &str) -> Result<Vec<WeightedRule>, String> {
    let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&json).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_update_success() {
        let rule = WeightedRule::new("r1", 0, "test");
        let updated = bayesian_update(&rule, true);
        assert_eq!(updated.alpha, 2.0);
        assert_eq!(updated.beta, 1.0);
        assert!((updated.weight - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bayesian_update_failure() {
        let rule = WeightedRule::new("r1", 0, "test");
        let updated = bayesian_update(&rule, false);
        assert_eq!(updated.alpha, 1.0);
        assert_eq!(updated.beta, 2.0);
        assert!((updated.weight - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let rules: Vec<WeightedRule> = (0..5)
            .map(|i| WeightedRule::new(format!("r{}", i), 0, ""))
            .collect();
        let probs = softmax(&rules, 1.0);
        let total: f64 = probs.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_high_temperature_uniform() {
        let mut rules = vec![
            WeightedRule::new("low", 0, ""),
            WeightedRule::new("high", 0, ""),
        ];
        rules[1].weight = 0.9;
        // High temperature → near uniform
        let probs = softmax(&rules, 100.0);
        let diff = (probs[0].1 - probs[1].1).abs();
        assert!(
            diff < 0.05,
            "Expected near-uniform at high temperature, diff={}",
            diff
        );
    }

    #[test]
    fn test_softmax_low_temperature_sharp() {
        let mut rules = vec![
            WeightedRule::new("low", 0, ""),
            WeightedRule::new("high", 0, ""),
        ];
        rules[1].weight = 0.9;
        // Low temperature → sharp peak on highest weight
        let probs = softmax(&rules, 0.01);
        assert!(probs[1].1 > 0.99, "Expected sharp peak, got {}", probs[1].1);
    }

    #[test]
    fn test_select_weighted_single_rule() {
        use rand::SeedableRng;
        let rules = vec![WeightedRule::new("only", 0, "")];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        assert_eq!(select_weighted(&rules, &mut rng), Some("only".to_string()));
    }

    #[test]
    fn test_select_weighted_favors_high_weight() {
        use rand::SeedableRng;
        let mut rules = vec![
            WeightedRule::new("low", 0, ""),
            WeightedRule::new("high", 0, ""),
        ];
        rules[1].weight = 0.99;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        // At temperature=1.0 with weights [0.5, 0.99], softmax gives ~62% to "high".
        // Use 500 samples and require >55% (well below the 62% expected).
        let selections: Vec<RuleId> = (0..500)
            .map(|_| select_weighted(&rules, &mut rng).unwrap())
            .collect();
        let high_count = selections.iter().filter(|s| s.as_str() == "high").count();
        assert!(
            high_count > 275,
            "High-weight rule should be selected more often (>55%): {}",
            high_count
        );
    }
}
