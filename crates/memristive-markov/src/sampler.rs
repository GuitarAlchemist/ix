use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    Greedy,
    TopK { k: usize },
    Nucleus { p: f64 },
    Temperature { t: f64 },
    TemperatureTopK { t: f64, k: usize },
}

impl SamplingStrategy {
    pub fn sample(&self, dist: &[(usize, f64)], rng: &mut impl Rng) -> Option<usize> {
        if dist.is_empty() {
            return None;
        }
        match self {
            SamplingStrategy::Greedy => dist
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|&(s, _)| s),
            SamplingStrategy::TopK { k } => {
                let mut sorted: Vec<_> = dist.to_vec();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                sorted.truncate(*k);
                Self::weighted_sample(&sorted, rng)
            }
            SamplingStrategy::Nucleus { p } => {
                let mut sorted: Vec<_> = dist.to_vec();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut cumulative = 0.0;
                let truncated: Vec<_> = sorted
                    .into_iter()
                    .take_while(|&(_, prob)| {
                        let include = cumulative < *p;
                        cumulative += prob;
                        include
                    })
                    .collect();
                Self::weighted_sample(&truncated, rng)
            }
            SamplingStrategy::Temperature { t } => {
                let scaled = Self::apply_temperature(dist, *t);
                Self::weighted_sample(&scaled, rng)
            }
            SamplingStrategy::TemperatureTopK { t, k } => {
                let mut scaled = Self::apply_temperature(dist, *t);
                scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scaled.truncate(*k);
                Self::weighted_sample(&scaled, rng)
            }
        }
    }

    fn apply_temperature(dist: &[(usize, f64)], t: f64) -> Vec<(usize, f64)> {
        let t = t.max(1e-10);
        let scaled: Vec<f64> = dist.iter().map(|(_, p)| (p.ln() / t).exp()).collect();
        let total: f64 = scaled.iter().sum();
        dist.iter()
            .zip(scaled.iter())
            .map(|(&(s, _), &w)| (s, w / total))
            .collect()
    }

    fn weighted_sample(dist: &[(usize, f64)], rng: &mut impl Rng) -> Option<usize> {
        if dist.is_empty() {
            return None;
        }
        let total: f64 = dist.iter().map(|(_, p)| p).sum();
        let mut r: f64 = rng.random::<f64>() * total;
        for &(state, prob) in dist {
            r -= prob;
            if r <= 0.0 {
                return Some(state);
            }
        }
        Some(dist.last().unwrap().0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_greedy_picks_highest() {
        let dist = vec![(0, 0.1), (1, 0.7), (2, 0.2)];
        assert_eq!(
            SamplingStrategy::Greedy.sample(&dist, &mut test_rng()),
            Some(1)
        );
    }

    #[test]
    fn test_top_k_limits_candidates() {
        let dist = vec![(0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05)];
        let mut rng = test_rng();
        for _ in 0..100 {
            let s = SamplingStrategy::TopK { k: 2 }
                .sample(&dist, &mut rng)
                .unwrap();
            assert!(s <= 1, "TopK(2) produced state {}", s);
        }
    }

    #[test]
    fn test_empty_dist_returns_none() {
        assert_eq!(SamplingStrategy::Greedy.sample(&[], &mut test_rng()), None);
    }

    #[test]
    fn test_temperature_low_concentrates() {
        let dist = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        let scaled = SamplingStrategy::apply_temperature(&dist, 0.1);
        assert!(
            scaled[0].1 > 0.9,
            "Low temp should concentrate: {:?}",
            scaled
        );
    }
}
