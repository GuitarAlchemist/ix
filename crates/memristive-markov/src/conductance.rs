use std::collections::HashMap;
use serde::{Serialize, Deserialize, Serializer, Deserializer};

fn serialize_conductances<S: Serializer>(
    map: &HashMap<(Vec<usize>, usize), f64>, serializer: S
) -> std::result::Result<S::Ok, S::Error> {
    let string_map: HashMap<String, f64> = map.iter()
        .map(|((ctx, next), &g)| (format!("{:?}->{}", ctx, next), g))
        .collect();
    string_map.serialize(serializer)
}

#[allow(clippy::type_complexity)]
fn deserialize_conductances<'de, D: Deserializer<'de>>(
    deserializer: D
) -> std::result::Result<HashMap<(Vec<usize>, usize), f64>, D::Error> {
    let string_map: HashMap<String, f64> = HashMap::deserialize(deserializer)?;
    let mut result = HashMap::new();
    for (key, g) in string_map {
        if let Some((ctx_str, next_str)) = key.rsplit_once("->") {
            let ctx: Vec<usize> = ctx_str.trim_matches(|c| c == '[' || c == ']')
                .split(", ").filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok()).collect();
            if let Ok(next) = next_str.parse() {
                result.insert((ctx, next), g);
            }
        }
    }
    Ok(result)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductanceMatrix {
    #[serde(serialize_with = "serialize_conductances", deserialize_with = "deserialize_conductances")]
    conductances: HashMap<(Vec<usize>, usize), f64>,
    alpha: f64,
    beta: f64,
    g_min: f64,
}

impl ConductanceMatrix {
    pub fn new(alpha: f64, beta: f64, g_min: f64) -> Self {
        Self { conductances: HashMap::new(), alpha, beta, g_min }
    }

    pub fn get(&self, context: &[usize], next: usize) -> f64 {
        self.conductances.get(&(context.to_vec(), next)).copied().unwrap_or(self.g_min)
    }

    /// Hebbian strengthening: g += alpha * (1 - g)
    pub fn strengthen(&mut self, context: &[usize], next: usize) {
        let key = (context.to_vec(), next);
        let g = self.conductances.entry(key).or_insert(self.g_min);
        *g += self.alpha * (1.0 - *g);
    }

    /// Decay all conductances: g = max(g_min, g * (1 - beta))
    pub fn decay_all(&mut self) {
        let g_min = self.g_min;
        let beta = self.beta;
        self.conductances.values_mut().for_each(|g| {
            *g = (*g * (1.0 - beta)).max(g_min);
        });
    }

    /// Modulate base probabilities by conductance, re-normalize.
    pub fn modulate(&self, context: &[usize], base_probs: &[(usize, f64)]) -> Vec<(usize, f64)> {
        let weighted: Vec<(usize, f64)> = base_probs.iter()
            .map(|&(state, prob)| (state, prob * self.get(context, state)))
            .collect();
        let total: f64 = weighted.iter().map(|(_, w)| w).sum();
        if total == 0.0 { return base_probs.to_vec(); }
        weighted.into_iter().map(|(s, w)| (s, w / total)).collect()
    }

    /// Iterate over all conductance entries
    pub fn for_each(&self, mut f: impl FnMut(&[usize], usize, f64)) {
        for ((context, next), &g) in &self.conductances {
            f(context, *next, g);
        }
    }

    /// Set conductance directly (used by consolidator)
    pub fn set(&mut self, context: &[usize], next: usize, g: f64) {
        self.conductances.insert((context.to_vec(), next), g.max(self.g_min));
    }

    pub fn alpha(&self) -> f64 { self.alpha }
    pub fn beta(&self) -> f64 { self.beta }
    pub fn g_min(&self) -> f64 { self.g_min }
    pub fn len(&self) -> usize { self.conductances.len() }
    pub fn is_empty(&self) -> bool { self.conductances.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_conductance_is_g_min() {
        let cm = ConductanceMatrix::new(0.1, 0.01, 0.01);
        assert_eq!(cm.get(&[0, 1], 2), 0.01);
    }

    #[test]
    fn test_hebbian_strengthens() {
        let mut cm = ConductanceMatrix::new(0.1, 0.01, 0.01);
        cm.strengthen(&[0, 1], 2);
        assert!(cm.get(&[0, 1], 2) > 0.01);
    }

    #[test]
    fn test_hebbian_asymptotes_at_one() {
        let mut cm = ConductanceMatrix::new(0.5, 0.01, 0.01);
        for _ in 0..100 { cm.strengthen(&[0], 1); }
        let g = cm.get(&[0], 1);
        assert!(g > 0.99 && g <= 1.0, "Should approach 1.0: {}", g);
    }

    #[test]
    fn test_decay_weakens() {
        let mut cm = ConductanceMatrix::new(0.1, 0.1, 0.01);
        cm.strengthen(&[0], 1);
        let before = cm.get(&[0], 1);
        cm.decay_all();
        assert!(cm.get(&[0], 1) < before);
    }

    #[test]
    fn test_decay_floors_at_g_min() {
        let mut cm = ConductanceMatrix::new(0.1, 0.99, 0.05);
        cm.strengthen(&[0], 1);
        for _ in 0..100 { cm.decay_all(); }
        assert!(cm.get(&[0], 1) >= 0.05);
    }

    #[test]
    fn test_modulate_probabilities() {
        let mut cm = ConductanceMatrix::new(0.5, 0.01, 0.01);
        for _ in 0..20 { cm.strengthen(&[0], 2); }
        let base = vec![(1, 0.5), (2, 0.5)];
        let modulated = cm.modulate(&[0], &base);
        let p2 = modulated.iter().find(|(s, _)| *s == 2).unwrap().1;
        let p1 = modulated.iter().find(|(s, _)| *s == 1).unwrap().1;
        assert!(p2 > p1);
        let total: f64 = modulated.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
}
