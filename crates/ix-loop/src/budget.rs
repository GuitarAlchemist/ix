//! What a loop may still spend, and how a child's share is carved out of its parent's.
//!
//! A budget counts four things: **fuel** (Lua instructions), **calls** to the semantic node,
//! a **deadline**, and the **depth** of the loop in the tree (the root is 0). A child starts
//! with what it asks for or what its parent has left, whichever is less, and the parent loses
//! that share at once. When the child ends, what it did not spend goes back to the parent. So
//! no loop ever holds more than its parent had, and the root's budget bounds the whole tree.

use std::cell::Cell;
use std::time::{Duration, Instant};

/// A loop deeper than this cannot start a child.
pub const MAX_DEPTH: u32 = 3;

/// What a loop asks for, for itself or for a child.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grant {
    pub calls: i64,
    pub fuel: i64,
    pub seconds: f64,
}

impl Grant {
    /// Everything: a child asking for this gets all its parent has left.
    pub const ALL: Grant = Grant {
        calls: i64::MAX,
        fuel: i64::MAX,
        seconds: f64::INFINITY,
    };
}

/// One loop's budget. Interior mutability: the Lua hooks and capabilities of one loop share it.
#[derive(Debug)]
pub struct Budget {
    fuel: Cell<i64>,
    calls: Cell<i64>,
    deadline: Instant,
    depth: u32,
    /// Calls this loop and its children spent.
    calls_used: Cell<i64>,
    /// What the semantic node reported, at list price; not what was billed.
    notional_usd: Cell<f64>,
}

fn deadline_after(seconds: f64) -> Instant {
    let now = Instant::now();
    if seconds.is_finite() && seconds >= 0.0 {
        // A year is "no deadline" for a loop; it keeps `Instant` arithmetic in range.
        now + Duration::from_secs_f64(seconds.min(365.0 * 86_400.0))
    } else {
        now + Duration::from_secs(365 * 86_400)
    }
}

impl Budget {
    /// The root of a tree of loops.
    pub fn root(grant: &Grant) -> Self {
        Self {
            fuel: Cell::new(grant.fuel.max(0)),
            calls: Cell::new(grant.calls.max(0)),
            deadline: deadline_after(grant.seconds),
            depth: 0,
            calls_used: Cell::new(0),
            notional_usd: Cell::new(0.0),
        }
    }

    /// Carves a child's budget out of this one: the least of what the child asks for and what
    /// is left, taken from this budget now. `None` at [`MAX_DEPTH`].
    pub fn carve(&self, want: &Grant) -> Option<Budget> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        let calls = want.calls.min(self.calls.get()).max(0);
        let fuel = want.fuel.min(self.fuel.get()).max(0);
        self.calls.set(self.calls.get() - calls);
        self.fuel.set(self.fuel.get() - fuel);
        Some(Budget {
            fuel: Cell::new(fuel),
            calls: Cell::new(calls),
            deadline: self.deadline.min(deadline_after(want.seconds)),
            depth: self.depth + 1,
            calls_used: Cell::new(0),
            notional_usd: Cell::new(0.0),
        })
    }

    /// Gives back what `child` did not spend, and charges this budget with what it did.
    pub fn refund(&self, child: &Budget) {
        self.calls.set(self.calls.get() + child.calls.get().max(0));
        self.fuel.set(self.fuel.get() + child.fuel.get().max(0));
        self.calls_used
            .set(self.calls_used.get() + child.calls_used.get());
        self.notional_usd
            .set(self.notional_usd.get() + child.notional_usd.get());
    }

    /// Spends `n` instructions; an error once the fuel or the time is gone.
    pub fn burn(&self, n: i64) -> Result<(), &'static str> {
        self.fuel.set(self.fuel.get().saturating_sub(n));
        match self.exhausted() {
            Some(why) => Err(why),
            None => Ok(()),
        }
    }

    /// Takes one call to the semantic node; an error when none is left or the time is gone.
    pub fn take_call(&self) -> Result<(), &'static str> {
        if self.calls.get() <= 0 {
            return Err("budget: no semantic-node calls left");
        }
        if Instant::now() >= self.deadline {
            return Err("budget: out of time");
        }
        self.calls.set(self.calls.get() - 1);
        self.calls_used.set(self.calls_used.get() + 1);
        Ok(())
    }

    /// Why this budget is spent, once its fuel or its time is gone.
    pub fn exhausted(&self) -> Option<&'static str> {
        if self.fuel.get() < 0 {
            Some("budget: out of fuel (Lua instructions)")
        } else if Instant::now() >= self.deadline {
            Some("budget: out of time")
        } else {
            None
        }
    }

    /// Records what one call cost, at list price.
    pub fn add_notional(&self, usd: f64) {
        if usd.is_finite() && usd > 0.0 {
            self.notional_usd.set(self.notional_usd.get() + usd);
        }
    }

    pub fn fuel(&self) -> i64 {
        self.fuel.get().max(0)
    }

    pub fn calls(&self) -> i64 {
        self.calls.get().max(0)
    }

    pub fn calls_used(&self) -> i64 {
        self.calls_used.get()
    }

    pub fn notional_usd(&self) -> f64 {
        self.notional_usd.get()
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    pub fn seconds_left(&self) -> f64 {
        self.deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_child_gets_no_more_than_its_parent_has_and_gives_back_the_rest() {
        let root = Budget::root(&Grant {
            calls: 3,
            fuel: 1_000,
            seconds: 60.0,
        });
        let child = root
            .carve(&Grant {
                calls: 10,
                fuel: 400,
                seconds: 60.0,
            })
            .unwrap();
        assert_eq!((child.calls(), child.fuel()), (3, 400));
        assert_eq!((root.calls(), root.fuel()), (0, 600));
        child.take_call().unwrap();
        child.burn(100).unwrap();
        root.refund(&child);
        assert_eq!((root.calls(), root.fuel(), root.calls_used()), (2, 900, 1));
    }

    #[test]
    fn the_depth_limit_stops_carving() {
        let mut b = Budget::root(&Grant::ALL);
        for _ in 0..MAX_DEPTH {
            b = b.carve(&Grant::ALL).unwrap();
        }
        assert_eq!(b.depth(), MAX_DEPTH);
        assert!(b.carve(&Grant::ALL).is_none());
    }
}
