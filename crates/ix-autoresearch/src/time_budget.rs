//! Per-iteration time budget — a soft deadline (always present, hint to the
//! eval) plus an optional hard timeout (kernel watchdog kills the eval).
//!
//! In v1 the hard-timeout watchdog is implemented per-target inside each
//! shell-out adapter via `wait-timeout`, not centrally — the kernel's only
//! responsibility is to *carry* the deadline values into `evaluate()`.

use std::time::{Duration, Instant};

/// Time budget passed to `run_experiment`. The soft deadline is converted
/// to an `Instant` at the start of every iteration and handed to the eval.
#[derive(Debug, Clone, Copy)]
pub struct TimeBudget {
    /// Hint to the evaluator: "self-terminate by this point or return
    /// `AutoresearchError::TimedOut`." Always present.
    pub soft_deadline_per_iter: Duration,

    /// If `Some`, the kernel watchdog terminates evaluators that exceed
    /// this duration, returning `AutoresearchError::HardKilled`. v1
    /// enforcement is per-adapter (the kernel passes the value through
    /// for the adapter to use with `wait-timeout`).
    pub hard_timeout_per_iter: Option<Duration>,
}

impl TimeBudget {
    /// Convenience: soft-only budget. Default for in-process targets where
    /// killing a thread isn't safe on Windows anyway.
    pub fn soft(soft: Duration) -> Self {
        Self {
            soft_deadline_per_iter: soft,
            hard_timeout_per_iter: None,
        }
    }

    /// Both deadlines. Use for shell-out adapters (Targets A and B).
    pub fn soft_and_hard(soft: Duration, hard: Duration) -> Self {
        Self {
            soft_deadline_per_iter: soft,
            hard_timeout_per_iter: Some(hard),
        }
    }

    /// Build the soft deadline `Instant` for an iteration starting now.
    pub fn soft_deadline_from_now(&self) -> Instant {
        Instant::now() + self.soft_deadline_per_iter
    }
}

impl Default for TimeBudget {
    /// 5-minute soft deadline, no hard kill. Karpathy-style.
    fn default() -> Self {
        Self::soft(Duration::from_secs(300))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_only_budget_has_no_hard_timeout() {
        let b = TimeBudget::soft(Duration::from_secs(60));
        assert_eq!(b.soft_deadline_per_iter, Duration::from_secs(60));
        assert!(b.hard_timeout_per_iter.is_none());
    }

    #[test]
    fn soft_and_hard_budget_carries_both() {
        let b = TimeBudget::soft_and_hard(Duration::from_secs(60), Duration::from_secs(120));
        assert_eq!(b.soft_deadline_per_iter, Duration::from_secs(60));
        assert_eq!(b.hard_timeout_per_iter, Some(Duration::from_secs(120)));
    }

    #[test]
    fn soft_deadline_from_now_is_in_the_future() {
        let b = TimeBudget::soft(Duration::from_secs(10));
        let deadline = b.soft_deadline_from_now();
        assert!(deadline > Instant::now());
    }

    #[test]
    fn default_is_soft_5min() {
        let b = TimeBudget::default();
        assert_eq!(b.soft_deadline_per_iter, Duration::from_secs(300));
        assert!(b.hard_timeout_per_iter.is_none());
    }
}
