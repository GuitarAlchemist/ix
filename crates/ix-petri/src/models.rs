//! Canonical nets whose behaviour is a documented fact, so they can serve as
//! oracles for the analyses rather than as decoration.
//!
//! The dining philosophers is the standard example precisely because the
//! textbook answer is known both ways: taking the two forks one at a time
//! deadlocks, taking them atomically does not. An analysis that gets both
//! right on a net it did not author is evidence; an analysis checked only
//! against nets written to match it is not.

use crate::net::{PetriError, PetriNet};

/// How a philosopher acquires the two forks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForkProtocol {
    /// Take the left fork, then the right. The classic deadlock: if everyone
    /// takes their left fork, every right fork is already held.
    LeftThenRight,
    /// Take both forks in one atomic step. Deadlock-free — no philosopher can
    /// hold one fork while waiting for the other.
    Atomic,
}

/// The dining philosophers as a P/T net, for `n >= 2` philosophers.
///
/// Places are `FORK_i` (one token each initially) and `THINK_i` (one token
/// each), plus `WAIT_i` under [`ForkProtocol::LeftThenRight`]. Philosopher `i`
/// uses fork `i` on the left and fork `(i + 1) % n` on the right.
///
/// Ids are zero-padded (`FORK_03`) so that ascending id order matches
/// ascending philosopher number for any `n` — cosmetic for correctness, but it
/// keeps witness sequences readable.
pub fn dining_philosophers(n: usize, protocol: ForkProtocol) -> Result<PetriNet, PetriError> {
    assert!(n >= 2, "the dining philosophers needs at least 2 diners");
    let width = n.to_string().len();
    let id = |prefix: &str, i: usize| format!("{prefix}_{i:0width$}", width = width);

    let mut b = PetriNet::builder().name(match protocol {
        ForkProtocol::LeftThenRight => "dining-philosophers/left-then-right",
        ForkProtocol::Atomic => "dining-philosophers/atomic",
    });

    for i in 0..n {
        b = b.place(id("FORK", i), 1).place(id("THINK", i), 1);
        if protocol == ForkProtocol::LeftThenRight {
            b = b.place(id("WAIT", i), 0);
        }
    }

    for i in 0..n {
        let left = id("FORK", i);
        let right = id("FORK", (i + 1) % n);
        let think = id("THINK", i);

        match protocol {
            ForkProtocol::LeftThenRight => {
                let wait = id("WAIT", i);
                let take_left = id("TAKE_LEFT", i);
                let take_right = id("TAKE_RIGHT", i);
                let release = id("RELEASE", i);

                b = b
                    .transition(take_left.clone())
                    .arc(think, take_left.clone())
                    .arc(left.clone(), take_left.clone())
                    .arc(take_left, wait.clone())
                    // Holding the left fork, now blocked on the right one.
                    .transition(take_right.clone())
                    .arc(wait, take_right.clone())
                    .arc(right.clone(), take_right.clone())
                    .arc(take_right, id("EAT", i))
                    .place(id("EAT", i), 0)
                    .transition(release.clone())
                    .arc(id("EAT", i), release.clone())
                    .arc(release.clone(), left)
                    .arc(release.clone(), right)
                    .arc(release, id("THINK", i));
            }
            ForkProtocol::Atomic => {
                let take = id("TAKE_BOTH", i);
                let release = id("RELEASE", i);

                b = b
                    .place(id("EAT", i), 0)
                    .transition(take.clone())
                    .arc(think, take.clone())
                    .arc(left.clone(), take.clone())
                    .arc(right.clone(), take.clone())
                    .arc(take, id("EAT", i))
                    .transition(release.clone())
                    .arc(id("EAT", i), release.clone())
                    .arc(release.clone(), left)
                    .arc(release.clone(), right)
                    .arc(release, id("THINK", i));
            }
        }
    }

    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{analyze, Limits, Verdict};

    /// The textbook fact, both ways round. If the analyser disagrees with this
    /// it is the analyser that is wrong.
    #[test]
    fn left_then_right_deadlocks_and_atomic_does_not() {
        for n in 2..=5 {
            let bad = analyze(
                &dining_philosophers(n, ForkProtocol::LeftThenRight).unwrap(),
                Limits::default(),
            );
            assert!(
                bad.deadlock_free.fails(),
                "{n} philosophers taking forks one at a time must deadlock"
            );
            assert!(bad.live.fails(), "a net with a deadlock is not live");

            let good = analyze(
                &dining_philosophers(n, ForkProtocol::Atomic).unwrap(),
                Limits::default(),
            );
            assert!(
                good.deadlock_free.holds(),
                "{n} philosophers taking both forks atomically cannot deadlock"
            );
            assert!(good.live.holds(), "and every transition stays available");
            assert!(
                good.reversible.holds(),
                "everyone can always return to thinking"
            );
        }
    }

    /// The deadlock witness must be the *specific* everybody-holds-their-left
    /// state, not merely "some marking with no successor".
    #[test]
    fn the_deadlock_witness_is_everyone_holding_their_left_fork() {
        let net = dining_philosophers(3, ForkProtocol::LeftThenRight).unwrap();
        let a = analyze(&net, Limits::default());

        let Verdict::Fails(deadlocks) = &a.deadlock_free else {
            panic!("expected a deadlock");
        };
        assert_eq!(
            a.deadlock_count, 1,
            "exactly one way to wedge 3 philosophers"
        );

        let d = &deadlocks[0];
        assert_eq!(d.marking, "WAIT_0=1 WAIT_1=1 WAIT_2=1");
        assert_eq!(d.witness.len(), 3, "one TAKE_LEFT per philosopher");
        assert!(d.witness.iter().all(|t| t.starts_with("TAKE_LEFT")));

        // Every fork is gone and nobody is eating or thinking.
        for p in net.places() {
            if p.id.starts_with("FORK") || p.id.starts_with("EAT") || p.id.starts_with("THINK") {
                assert!(!d.marking.contains(&p.id), "{} should hold no token", p.id);
            }
        }
    }

    /// Both nets are safe (1-bounded): a fork is held or free, never doubled.
    #[test]
    fn both_protocols_are_one_bounded() {
        for protocol in [ForkProtocol::LeftThenRight, ForkProtocol::Atomic] {
            let a = analyze(
                &dining_philosophers(4, protocol).unwrap(),
                Limits::default(),
            );
            let Verdict::Holds(bounds) = &a.bounded else {
                panic!("{protocol:?}: boundedness should be decided for 4 philosophers");
            };
            assert_eq!(bounds.k, 1, "{protocol:?} must be safe (1-bounded)");
        }
    }
}
