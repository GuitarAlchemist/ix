//! # ix-loop-detect — sliding-window loop detector for agent tool calls
//!
//! First concrete primitive of the `ix-middleware` stack (see
//! `docs/brainstorms/2026-04-10-ix-harness-primitives.md`, item 2).
//!
//! ## What it does
//!
//! Tracks how many times a given `AgentAction` (keyed by
//! [`ix_agent_core::AgentAction::loop_key`]) has been recorded within a
//! sliding time window. When the count exceeds a configured threshold,
//! subsequent [`LoopDetector::record`] calls return
//! [`LoopVerdict::TooManyEdits`] until enough events age out of the
//! window.
//!
//! This is the **circuit breaker** half of LangChain's
//! `LoopDetectionMiddleware` — the primitive that drives their +13.7
//! pp Terminal Bench 2.0 improvement attributed to harness-level
//! changes alone. The brainstorm's target behavior:
//!
//! > "10 edits to same file in 5 min → reconsider"
//!
//! ## Breaking change in v2
//!
//! The v1 API took `&str` keys directly. v2 takes `&AgentAction` and
//! lets the action's own `loop_key()` decide granularity — bare tool
//! name for coarse loop detection, `tool:target_hint` composite for
//! fine-grained. This migration keeps the detector stateless wrt
//! action semantics while letting consumers upgrade granularity by
//! populating `target_hint` in their `AgentAction::InvokeTool` rather
//! than teaching the detector new tricks.
//!
//! Non-invoke actions (observations, returns, approvals) have
//! `loop_key() == None` and are always [`LoopVerdict::Ok`] — they
//! cannot loop.
//!
//! ## Scope
//!
//! - Pure library — no async, no global state, no side effects
//! - `Send + Sync` via a single `Mutex<HashMap<String, VecDeque<Instant>>>`
//! - Keyed by [`AgentAction::loop_key`] output
//! - Window is wall-clock (`std::time::Instant`), not logical time
//! - No persistence across process restarts
//!
//! ## Non-goals
//!
//! - Not a generic rate limiter (no token bucket semantics)
//! - Not a request counter (no total-call accounting across windows)
//! - Not a middleware framework (see `ix-middleware` when it lands)
//! - Not belief-aware — verdicts are plain structural counts, not
//!   hexavalent truth values

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ix_agent_core::AgentAction;
use serde::{Deserialize, Serialize};

pub mod middleware;
pub use middleware::LoopDetectMiddleware;

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// The outcome of a [`LoopDetector::record`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopVerdict {
    /// The record was accepted and the key is below threshold.
    Ok,
    /// The threshold was exceeded — the caller should halt or
    /// escalate rather than continue the loop.
    TooManyEdits {
        /// How many events are currently inside the window for this
        /// key (including the one just recorded).
        count: usize,
        /// The window that was being enforced when the verdict fired.
        window: Duration,
        /// The threshold that was tripped.
        threshold: usize,
    },
}

impl LoopVerdict {
    /// `true` iff the verdict says the caller should halt.
    pub fn is_blocked(&self) -> bool {
        matches!(self, LoopVerdict::TooManyEdits { .. })
    }
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Configuration for a [`LoopDetector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopDetectorConfig {
    /// Maximum number of events allowed inside the window before
    /// subsequent events trip the circuit breaker.
    pub threshold: usize,
    /// The rolling time window.
    pub window: Duration,
}

impl Default for LoopDetectorConfig {
    /// Brainstorm default: 10 events in 5 minutes.
    fn default() -> Self {
        Self {
            threshold: 10,
            window: Duration::from_secs(5 * 60),
        }
    }
}

/// Sliding-window counter for agent tool-call loops.
///
/// Clone-safe via `Arc` if you need shared ownership; the inner
/// `Mutex` handles cross-thread access. For single-owner use in a
/// dispatcher, you can store it behind a `OnceLock<LoopDetector>`.
#[derive(Debug)]
pub struct LoopDetector {
    config: LoopDetectorConfig,
    windows: Mutex<HashMap<String, VecDeque<Instant>>>,
}

impl LoopDetector {
    /// Construct a detector with the given configuration.
    pub fn new(config: LoopDetectorConfig) -> Self {
        Self {
            config,
            windows: Mutex::new(HashMap::new()),
        }
    }

    /// Construct a detector with the default configuration
    /// (10 events / 5 min).
    pub fn with_defaults() -> Self {
        Self::new(LoopDetectorConfig::default())
    }

    /// Borrow the live configuration.
    pub fn config(&self) -> LoopDetectorConfig {
        self.config
    }

    /// Record an action and return the verdict.
    ///
    /// The detector calls [`AgentAction::loop_key`] to derive the
    /// window key. Actions whose `loop_key()` returns `None`
    /// (observations, returns, approvals) are always
    /// [`LoopVerdict::Ok`] — only tool invocations can loop.
    ///
    /// Internally:
    /// 1. Compute the key via `action.loop_key()`. If `None`, return
    ///    `Ok` without mutating state.
    /// 2. Acquire the mutex.
    /// 3. Drop events older than `now - window` from the key's deque.
    /// 4. Append `now` to the deque.
    /// 5. Return `TooManyEdits` if the resulting length exceeds
    ///    `threshold`, otherwise `Ok`.
    pub fn record(&self, action: &AgentAction) -> LoopVerdict {
        self.record_at(action, Instant::now())
    }

    /// Like [`Self::record`] but uses a caller-supplied `now` — used
    /// by tests to exercise window expiration deterministically.
    pub fn record_at(&self, action: &AgentAction, now: Instant) -> LoopVerdict {
        let Some(key) = action.loop_key() else {
            return LoopVerdict::Ok;
        };
        self.record_key_at(&key, now)
    }

    /// Internal helper that runs the sliding-window counter logic for
    /// an already-computed string key. Private so the public surface
    /// stays limited to [`AgentAction`] inputs — callers cannot
    /// smuggle in arbitrary keys.
    fn record_key_at(&self, key: &str, now: Instant) -> LoopVerdict {
        let mut guard = self.windows.lock().expect("loop-detector mutex poisoned");
        let cutoff = now.checked_sub(self.config.window);
        let deque = guard.entry(key.to_string()).or_default();

        if let Some(cutoff) = cutoff {
            while let Some(front) = deque.front() {
                if *front < cutoff {
                    deque.pop_front();
                } else {
                    break;
                }
            }
        }

        deque.push_back(now);
        let count = deque.len();

        if count > self.config.threshold {
            LoopVerdict::TooManyEdits {
                count,
                window: self.config.window,
                threshold: self.config.threshold,
            }
        } else {
            LoopVerdict::Ok
        }
    }

    /// Drop all state for `key` — useful when the caller knows the
    /// loop has been legitimately resolved and wants a fresh window.
    pub fn clear_key(&self, key: &str) {
        let mut guard = self.windows.lock().expect("loop-detector mutex poisoned");
        guard.remove(key);
    }

    /// Drop all state for all keys.
    pub fn clear_all(&self) {
        let mut guard = self.windows.lock().expect("loop-detector mutex poisoned");
        guard.clear();
    }

    /// How many keys are currently being tracked.
    pub fn tracked_keys(&self) -> usize {
        self.windows
            .lock()
            .expect("loop-detector mutex poisoned")
            .len()
    }

    /// Current count for a specific key (0 if untracked). Intended
    /// for observability / tests; production code should use
    /// `record` and inspect the verdict.
    pub fn count(&self, key: &str) -> usize {
        self.windows
            .lock()
            .expect("loop-detector mutex poisoned")
            .get(key)
            .map(|d| d.len())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn short() -> LoopDetector {
        LoopDetector::new(LoopDetectorConfig {
            threshold: 3,
            window: Duration::from_secs(60),
        })
    }

    /// Test helper: construct an AgentAction::InvokeTool with the given
    /// tool name. Uses an empty params object and ordinal 0 — loop
    /// detection ignores both, so defaults keep the test body readable.
    fn action(tool: &str) -> AgentAction {
        AgentAction::InvokeTool {
            tool_name: tool.to_string(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: None,
        }
    }

    // ── Basic counting ─────────────────────────────────────────────────

    #[test]
    fn ok_below_threshold() {
        let d = short();
        for _ in 0..3 {
            assert_eq!(d.record(&action("x")), LoopVerdict::Ok);
        }
        assert_eq!(d.count("x"), 3);
    }

    #[test]
    fn fires_on_exceeding_threshold() {
        let d = short();
        for _ in 0..3 {
            assert_eq!(d.record(&action("x")), LoopVerdict::Ok);
        }
        match d.record(&action("x")) {
            LoopVerdict::TooManyEdits {
                count,
                threshold,
                window: _,
            } => {
                assert_eq!(count, 4);
                assert_eq!(threshold, 3);
            }
            other => panic!("expected TooManyEdits, got {:?}", other),
        }
    }

    #[test]
    fn verdict_is_blocked_helper() {
        let ok = LoopVerdict::Ok;
        let blocked = LoopVerdict::TooManyEdits {
            count: 11,
            window: Duration::from_secs(300),
            threshold: 10,
        };
        assert!(!ok.is_blocked());
        assert!(blocked.is_blocked());
    }

    #[test]
    fn non_invoke_actions_never_loop() {
        // Observations, returns, and approvals have loop_key() == None
        // and should always be Ok, no matter how many times they fire.
        let d = short();
        let obs = AgentAction::EmitObservation {
            stream: "progress".into(),
            payload: serde_json::json!({}),
            ordinal: 0,
        };
        for _ in 0..100 {
            assert_eq!(d.record(&obs), LoopVerdict::Ok);
        }
        assert_eq!(d.tracked_keys(), 0);
    }

    #[test]
    fn target_hint_creates_distinct_keys() {
        let d = short();
        let bare = AgentAction::InvokeTool {
            tool_name: "ix_context_walk".into(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: None,
        };
        let targeted_a = AgentAction::InvokeTool {
            tool_name: "ix_context_walk".into(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: Some("fn_a".into()),
        };
        let targeted_b = AgentAction::InvokeTool {
            tool_name: "ix_context_walk".into(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: Some("fn_b".into()),
        };
        // Bare, targeted_a, and targeted_b all key differently.
        d.record(&bare);
        d.record(&targeted_a);
        d.record(&targeted_b);
        assert_eq!(d.tracked_keys(), 3);
        assert_eq!(d.count("ix_context_walk"), 1);
        assert_eq!(d.count("ix_context_walk:fn_a"), 1);
        assert_eq!(d.count("ix_context_walk:fn_b"), 1);
    }

    // ── Key isolation ──────────────────────────────────────────────────

    #[test]
    fn different_keys_are_independent() {
        let d = short();
        for _ in 0..3 {
            assert_eq!(d.record(&action("a")), LoopVerdict::Ok);
        }
        // Key `a` is at threshold; key `b` is untouched.
        assert_eq!(d.count("a"), 3);
        assert_eq!(d.count("b"), 0);
        // Recording `b` should still be Ok — its window is empty.
        assert_eq!(d.record(&action("b")), LoopVerdict::Ok);
        // And `a` is still at 3, not bumped.
        assert_eq!(d.count("a"), 3);
    }

    // ── Window expiration ─────────────────────────────────────────────

    #[test]
    fn old_events_age_out_of_window() {
        let d = short();
        let t0 = Instant::now();
        for i in 0..3 {
            assert_eq!(
                d.record_at(&action("x"), t0 + Duration::from_secs(i)),
                LoopVerdict::Ok
            );
        }
        assert_eq!(d.count("x"), 3);

        // Jump 120 seconds ahead — past the 60s window.
        let later = t0 + Duration::from_secs(120);
        assert_eq!(d.record_at(&action("x"), later), LoopVerdict::Ok);
        // Only the new event should remain.
        assert_eq!(d.count("x"), 1);
    }

    #[test]
    fn partial_window_expiration_keeps_live_events() {
        let d = short();
        let t0 = Instant::now();
        d.record_at(&action("x"), t0);
        d.record_at(&action("x"), t0 + Duration::from_secs(1));
        assert_eq!(d.count("x"), 2);

        // At t0+65, the first two events are past the 60s window.
        let v = d.record_at(&action("x"), t0 + Duration::from_secs(65));
        assert_eq!(v, LoopVerdict::Ok);
        assert_eq!(d.count("x"), 1);
    }

    // ── Clear ─────────────────────────────────────────────────────────

    #[test]
    fn clear_key_resets_one_key() {
        let d = short();
        d.record(&action("a"));
        d.record(&action("a"));
        d.record(&action("b"));
        assert_eq!(d.count("a"), 2);
        assert_eq!(d.count("b"), 1);

        d.clear_key("a");
        assert_eq!(d.count("a"), 0);
        assert_eq!(d.count("b"), 1);
    }

    #[test]
    fn clear_all_resets_everything() {
        let d = short();
        d.record(&action("a"));
        d.record(&action("b"));
        d.record(&action("c"));
        assert_eq!(d.tracked_keys(), 3);

        d.clear_all();
        assert_eq!(d.tracked_keys(), 0);
        assert_eq!(d.count("a"), 0);
    }

    // ── Repeated firing ───────────────────────────────────────────────

    #[test]
    fn keeps_firing_while_over_threshold() {
        let d = short();
        for _ in 0..3 {
            d.record(&action("x"));
        }
        // All subsequent records inside the window should report
        // TooManyEdits — not just the first crossing.
        for expected_count in 4..=6 {
            match d.record(&action("x")) {
                LoopVerdict::TooManyEdits { count, .. } => {
                    assert_eq!(count, expected_count);
                }
                other => panic!(
                    "expected TooManyEdits at count {expected_count}, got {:?}",
                    other
                ),
            }
        }
    }

    // ── Defaults ──────────────────────────────────────────────────────

    #[test]
    fn default_config_is_brainstorm_values() {
        let c = LoopDetectorConfig::default();
        assert_eq!(c.threshold, 10);
        assert_eq!(c.window, Duration::from_secs(5 * 60));
    }

    #[test]
    fn with_defaults_constructor_matches() {
        let d = LoopDetector::with_defaults();
        assert_eq!(d.config(), LoopDetectorConfig::default());
    }

    // ── Thread safety sanity ──────────────────────────────────────────

    #[test]
    fn parallel_records_do_not_panic() {
        use std::sync::Arc;
        use std::thread;

        let d = Arc::new(LoopDetector::new(LoopDetectorConfig {
            threshold: 1000,
            window: Duration::from_secs(60),
        }));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let d = Arc::clone(&d);
                thread::spawn(move || {
                    for j in 0..100 {
                        let tool = format!("t{}", j % 3);
                        let _ = d.record(&action(&tool));
                        if i == 0 && j == 50 {
                            d.clear_key("t0");
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        // Exact counts are timing-dependent; just verify we're still
        // tracking some keys and the Mutex isn't poisoned.
        assert!(d.tracked_keys() >= 1);
        // Sanity: new records still work after the contention.
        assert_eq!(d.record(&action("sanity")), LoopVerdict::Ok);
    }

    // ── Verdict serde ─────────────────────────────────────────────────

    #[test]
    fn verdict_serde_roundtrip() {
        let v = LoopVerdict::TooManyEdits {
            count: 11,
            window: Duration::from_secs(300),
            threshold: 10,
        };
        let json = serde_json::to_string(&v).expect("serialize");
        let back: LoopVerdict = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, v);
    }
}
