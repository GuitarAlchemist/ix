use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Wrapper to serialize Vec<usize> keys as strings for JSON compatibility
mod serde_helpers {
    use super::*;
    use serde::{Serializer, Deserializer};

    pub fn ser_ctx_map<S: Serializer>(
        map: &HashMap<Vec<usize>, HashMap<usize, f64>>, s: S,
    ) -> Result<S::Ok, S::Error> {
        let m: HashMap<String, HashMap<String, f64>> = map.iter()
            .map(|(k, v)| (format!("{:?}", k), v.iter().map(|(k2, v2)| (k2.to_string(), *v2)).collect()))
            .collect();
        m.serialize(s)
    }

    #[allow(clippy::type_complexity)]
    pub fn de_ctx_map<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<Vec<usize>, HashMap<usize, f64>>, D::Error> {
        let m: HashMap<String, HashMap<String, f64>> = HashMap::deserialize(d)?;
        Ok(m.into_iter().map(|(k, v)| {
            let ctx: Vec<usize> = k.trim_matches(|c| c == '[' || c == ']')
                .split(", ").filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok()).collect();
            let next_map = v.into_iter().filter_map(|(k2, v2)| k2.parse().ok().map(|k2| (k2, v2))).collect();
            (ctx, next_map)
        }).collect())
    }

    pub fn ser_totals<S: Serializer>(
        map: &HashMap<Vec<usize>, f64>, s: S,
    ) -> Result<S::Ok, S::Error> {
        let m: HashMap<String, f64> = map.iter().map(|(k, v)| (format!("{:?}", k), *v)).collect();
        m.serialize(s)
    }

    pub fn de_totals<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<Vec<usize>, f64>, D::Error> {
        let m: HashMap<String, f64> = HashMap::deserialize(d)?;
        Ok(m.into_iter().map(|(k, v)| {
            let ctx: Vec<usize> = k.trim_matches(|c| c == '[' || c == ']')
                .split(", ").filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok()).collect();
            (ctx, v)
        }).collect())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovTensor {
    max_order: usize,
    #[serde(serialize_with = "serde_helpers::ser_ctx_map", deserialize_with = "serde_helpers::de_ctx_map")]
    transitions: HashMap<Vec<usize>, HashMap<usize, f64>>,
    #[serde(serialize_with = "serde_helpers::ser_totals", deserialize_with = "serde_helpers::de_totals")]
    context_totals: HashMap<Vec<usize>, f64>,
    state_count: usize,
}

impl MarkovTensor {
    pub fn new(max_order: usize) -> Self {
        Self {
            max_order,
            transitions: HashMap::new(),
            context_totals: HashMap::new(),
            state_count: 0,
        }
    }

    pub fn max_order(&self) -> usize { self.max_order }
    pub fn state_count(&self) -> usize { self.state_count }

    pub fn sparsity(&self) -> f64 {
        if self.state_count == 0 { return 1.0; }
        let total_possible = self.context_totals.len() as f64 * self.state_count as f64;
        if total_possible == 0.0 { return 1.0; }
        let total_nonzero: f64 = self.transitions.values().map(|m| m.len() as f64).sum();
        1.0 - (total_nonzero / total_possible)
    }

    pub fn observe(&mut self, context: &[usize], next: usize) {
        for &s in context.iter().chain(std::iter::once(&next)) {
            if s >= self.state_count { self.state_count = s + 1; }
        }
        let len = context.len().min(self.max_order);
        for start in 0..len {
            let ctx = context[start..].to_vec();
            *self.transitions.entry(ctx.clone()).or_default().entry(next).or_insert(0.0) += 1.0;
            *self.context_totals.entry(ctx).or_insert(0.0) += 1.0;
        }
        *self.transitions.entry(vec![]).or_default().entry(next).or_insert(0.0) += 1.0;
        *self.context_totals.entry(vec![]).or_insert(0.0) += 1.0;
    }

    pub fn predict(&self, context: &[usize]) -> Vec<(usize, f64)> {
        let ctx = context.to_vec();
        match (self.transitions.get(&ctx), self.context_totals.get(&ctx)) {
            (Some(next_map), Some(&total)) if total > 0.0 => {
                next_map.iter().map(|(&state, &count)| (state, count / total)).collect()
            }
            _ => Vec::new(),
        }
    }

    pub fn context_count(&self, context: &[usize]) -> f64 {
        self.context_totals.get(context).copied().unwrap_or(0.0)
    }

    pub fn merge(&mut self, other: &MarkovTensor) {
        for (ctx, next_map) in &other.transitions {
            for (&state, &count) in next_map {
                *self.transitions.entry(ctx.clone()).or_default().entry(state).or_insert(0.0) += count;
            }
        }
        for (ctx, &total) in &other.context_totals {
            *self.context_totals.entry(ctx.clone()).or_insert(0.0) += total;
        }
        self.state_count = self.state_count.max(other.state_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor_is_empty() {
        let t = MarkovTensor::new(3);
        assert_eq!(t.max_order(), 3);
        assert_eq!(t.state_count(), 0);
    }

    #[test]
    fn test_observe_single_transition() {
        let mut t = MarkovTensor::new(2);
        t.observe(&[0, 1], 2);
        let dist = t.predict(&[0, 1]);
        assert_eq!(dist.len(), 1);
        assert_eq!(dist[0], (2, 1.0));
    }

    #[test]
    fn test_observe_multi_order_storage() {
        let mut t = MarkovTensor::new(3);
        t.observe(&[0, 1, 2], 3);
        assert!(!t.predict(&[0, 1, 2]).is_empty());
        assert!(!t.predict(&[1, 2]).is_empty());
        assert!(!t.predict(&[2]).is_empty());
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let mut t = MarkovTensor::new(2);
        t.observe(&[0, 1], 2);
        t.observe(&[0, 1], 3);
        t.observe(&[0, 1], 2);
        let dist = t.predict(&[0, 1]);
        let total: f64 = dist.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_unknown_context_returns_empty() {
        let t = MarkovTensor::new(2);
        assert!(t.predict(&[99, 100]).is_empty());
    }

    #[test]
    fn test_merge_tensors() {
        let mut t1 = MarkovTensor::new(2);
        t1.observe(&[0, 1], 2);
        let mut t2 = MarkovTensor::new(2);
        t2.observe(&[0, 1], 3);
        t1.merge(&t2);
        assert_eq!(t1.predict(&[0, 1]).len(), 2);
    }

    #[test]
    fn test_state_count_tracks_unique_states() {
        let mut t = MarkovTensor::new(2);
        t.observe(&[0, 1], 2);
        t.observe(&[3, 4], 5);
        assert_eq!(t.state_count(), 6);
    }
}
