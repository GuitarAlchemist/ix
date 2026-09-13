//! The net that motivated the crate: a work pump whose lanes contend for a
//! shared worktree pool and the repository-wide git stash stack.
//!
//! # The question this answers
//!
//! `CLAUDE.md` warns, in the environment preamble every session loads:
//!
//! > The git stash stack is shared with the main checkout and all other
//! > worktrees, and other Claude sessions may push or pop it concurrently.
//!
//! So a lane in a multi-agent pump holds two things at once: a working tree,
//! and the one shared stash stack. Two resources, several lanes, and a cycle
//! back to `READY` when a lane finishes — that is not something
//! `ix_pipeline::dag::Dag` can express, because a DAG rejects the cycle, and
//! not something `ix_graph::markov` answers, because the question is not
//! "how likely" but "is it possible at all".
//!
//! The question is whether the pump can wedge — one lane holding the tree and
//! waiting for the stash, another holding the stash and waiting for a tree —
//! and whether a canonical acquisition order removes the possibility. Both
//! answers are computed here rather than argued.

use ix_petri::analysis::{analyze, Limits, Verdict};
use ix_petri::{PetriNet, PetriNetBuilder};

/// The order in which a lane takes its two resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    /// Working tree first, then the stash stack.
    TreeThenStash,
    /// Stash stack first, then a working tree.
    StashThenTree,
}

/// A pump of `lanes` lanes sharing `trees` working trees and one stash stack.
///
/// `orders[i]` is how lane `i` acquires. A lane cycles
/// `READY -> holding one -> holding both -> READY`, releasing both resources
/// atomically at the end of its unit of work.
fn work_pump(lanes: usize, trees: u64, orders: &[Order]) -> PetriNet {
    assert_eq!(orders.len(), lanes, "one acquisition order per lane");

    let mut b: PetriNetBuilder = PetriNet::builder()
        .name("agent-work-pump")
        .named_place("RES_stash", "stash-stack", 1)
        .named_place("RES_tree", "free-worktrees", trees);

    for (i, order) in orders.iter().enumerate() {
        let ready = format!("L{i}_a_ready");
        let held_one = format!("L{i}_b_holds_first");
        let held_both = format!("L{i}_c_holds_both");
        let (first, second) = match order {
            Order::TreeThenStash => ("RES_tree", "RES_stash"),
            Order::StashThenTree => ("RES_stash", "RES_tree"),
        };
        let take_first = format!("L{i}_1_take_{}", first.trim_start_matches("RES_"));
        let take_second = format!("L{i}_2_take_{}", second.trim_start_matches("RES_"));
        let finish = format!("L{i}_3_release_both");

        b = b
            .place(ready.clone(), 1)
            .place(held_one.clone(), 0)
            .place(held_both.clone(), 0)
            // Grab the first resource and keep holding it while waiting.
            .transition(take_first.clone())
            .arc(ready.clone(), take_first.clone())
            .arc(first, take_first.clone())
            .arc(take_first, held_one.clone())
            // Grab the second. If it is taken, the lane is stuck holding the
            // first — which is exactly how a pump wedges.
            .transition(take_second.clone())
            .arc(held_one, take_second.clone())
            .arc(second, take_second.clone())
            .arc(take_second, held_both.clone())
            // Do the work, hand both back.
            .transition(finish.clone())
            .arc(held_both, finish.clone())
            .arc(finish.clone(), first)
            .arc(finish.clone(), second)
            .arc(finish, ready);
    }

    b.build().expect("the pump net is well formed")
}

/// The seeded deadlock. Two lanes, one shared tree, one stash stack, and the
/// two lanes acquiring in opposite orders: lane 0 can be holding the tree while
/// lane 1 holds the stash, and neither can proceed or release.
///
/// This is the test that must fail if deadlock detection regresses.
#[test]
fn opposite_acquisition_orders_wedge_the_pump() {
    let net = work_pump(2, 1, &[Order::TreeThenStash, Order::StashThenTree]);
    let report = analyze(&net, Limits::default());

    let Verdict::Fails(deadlocks) = &report.deadlock_free else {
        panic!("two lanes taking two resources in opposite orders must be able to wedge");
    };
    assert_eq!(report.deadlock_count, 1, "exactly one wedged marking");

    let d = &deadlocks[0];
    // Each lane is parked holding its first resource, and both resources are
    // gone from the shared pools.
    assert_eq!(d.marking, "L0_b_holds_first=1 L1_b_holds_first=1");
    assert_eq!(
        net.label_sequence(&d.witness),
        ["L0_1_take_tree", "L1_1_take_stash"],
        "the shortest way to wedge is one grab each"
    );

    // A wedged pump is by definition not live and cannot return to its start.
    assert!(report.live.fails());
    assert!(report.reversible.fails());

    // Replaying the witness must land on the marking that was reported —
    // the witness is a reproduction recipe, not a label.
    let mut m = net.initial_marking().clone();
    for id in &d.witness {
        m = net
            .fire(
                &m,
                net.transition_index(id)
                    .expect("witness id is a transition"),
            )
            .expect("witness sequence is fireable");
    }
    assert_eq!(net.describe_marking(&m), d.marking);
    assert!(
        net.enabled(&m).is_empty(),
        "and nothing can fire from there"
    );
}

/// The fix, checked rather than asserted: make every lane acquire in the same
/// order and the wedge becomes unreachable — for any lane count in range and
/// with the tree pool scarcer than the lanes.
#[test]
fn a_canonical_acquisition_order_removes_the_deadlock() {
    for lanes in 2..=4 {
        for order in [Order::TreeThenStash, Order::StashThenTree] {
            let orders = vec![order; lanes];
            let net = work_pump(lanes, 1, &orders);
            let report = analyze(&net, Limits::default());

            assert!(
                report.deadlock_free.holds(),
                "{lanes} lanes all acquiring {order:?} must not wedge (got {:?})",
                report.deadlock_free
            );
            assert!(report.live.holds(), "every lane keeps making progress");
            assert!(
                report.reversible.holds(),
                "the pump can always drain back to all-ready"
            );
        }
    }
}

/// Mixing orders is what breaks it, not lane count: one contrarian lane is
/// enough, and adding conforming lanes does not repair it.
#[test]
fn one_lane_acquiring_out_of_order_is_enough_to_wedge_the_whole_pump() {
    for lanes in 2..=4 {
        let mut orders = vec![Order::TreeThenStash; lanes];
        orders[lanes - 1] = Order::StashThenTree;

        let net = work_pump(lanes, 1, &orders);
        let report = analyze(&net, Limits::default());
        assert!(
            report.deadlock_free.fails(),
            "{lanes} lanes with one contrarian must be able to wedge"
        );
    }
}

/// Enough trees to go round removes the contention rather than the cycle, so
/// the pump is deadlock-free even with mixed orders. Recorded because it is
/// the *other* available fix, and it is worth knowing it also works.
#[test]
fn giving_every_lane_its_own_tree_also_removes_the_deadlock() {
    let lanes = 3;
    let net = work_pump(
        lanes,
        lanes as u64,
        &[
            Order::TreeThenStash,
            Order::StashThenTree,
            Order::TreeThenStash,
        ],
    );
    let report = analyze(&net, Limits::default());
    assert!(
        report.deadlock_free.holds(),
        "with a tree per lane only the stash stack is contended, and one \
         resource cannot deadlock"
    );
    assert!(report.live.holds());
}

/// The pump never invents resources: exactly one stash stack and `trees` trees
/// exist at every reachable marking, whatever the lanes do.
#[test]
fn resources_are_conserved_across_the_whole_state_space() {
    let net = work_pump(3, 2, &[Order::TreeThenStash; 3]);
    let report = analyze(&net, Limits::default());

    let Verdict::Holds(bounds) = &report.bounded else {
        panic!("a fixed pool of resources is bounded");
    };
    let bound = |id: &str| {
        bounds
            .per_place
            .iter()
            .find(|(p, _)| p == id)
            .map(|(_, k)| *k)
            .unwrap_or_else(|| panic!("no bound recorded for {id}"))
    };
    assert_eq!(bound("RES_stash"), 1, "the stash stack is never duplicated");
    assert_eq!(bound("RES_tree"), 2, "nor are the worktrees");
    for lane in 0..3 {
        assert_eq!(bound(&format!("L{lane}_c_holds_both")), 1);
    }
}
