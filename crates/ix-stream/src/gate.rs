//! A gate that sends each value to one expert, as the router of a mixture of experts does,
//! under budgets.
//!
//! For each context (a bucket the caller computes, such as how sure a cheap expert was) the
//! gate keeps an [`ix_rl::bandit::UCB1`] of how good each arm's answers turned out. It ranks
//! the arms by that estimate plus UCB1's exploration bonus, minus the arm's price.
//!
//! An arm with a budget has a shadow price that follows its pace: the share of the budget
//! spent, minus the share of the expected stream seen. Spending ahead of pace raises the
//! price, which keeps the arm for the contexts where it gains the most; falling behind lowers
//! it. An arm whose budget cannot pay one more call is never ranked. The gate learns from what
//! the judge says about each item, which reaches it on the bus, so it never waits for it.
//!
//! The [`Ledger`] holds the budgets. The lanes spend from it, and refuse a call it cannot pay:
//! the gate's price steers, the ledger's cap is the hard stop.

use ix_rl::bandit::UCB1;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// One budget: `cap` of `unit` (dollars, calls, tokens).
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    pub name: String,
    pub unit: String,
    pub cap: f64,
    pub spent: f64,
}

impl Budget {
    pub fn new(name: &str, unit: &str, cap: f64) -> Self {
        Self {
            name: name.to_string(),
            unit: unit.to_string(),
            cap,
            spent: 0.0,
        }
    }
}

/// The budgets, shared by the gate and the lanes that spend them.
#[derive(Debug)]
pub struct Ledger(Mutex<Vec<Budget>>);

impl Ledger {
    pub fn new(budgets: Vec<Budget>) -> Arc<Self> {
        Arc::new(Self(Mutex::new(budgets)))
    }

    fn with<T>(&self, name: &str, f: impl FnOnce(&mut Budget) -> T) -> Option<T> {
        let mut all = self.0.lock().unwrap_or_else(|e| e.into_inner());
        all.iter_mut().find(|b| b.name == name).map(f)
    }

    /// Spends `amount` if the budget can pay it, and says whether it did. An unknown budget
    /// pays nothing.
    pub fn spend(&self, name: &str, amount: f64) -> bool {
        self.with(name, |b| {
            if b.spent + amount <= b.cap + 1e-12 {
                b.spent += amount;
                true
            } else {
                false
            }
        })
        .unwrap_or(false)
    }

    /// Gives back what a call reserved and did not use.
    pub fn refund(&self, name: &str, amount: f64) {
        self.with(name, |b| b.spent = (b.spent - amount).max(0.0));
    }

    pub fn get(&self, name: &str) -> Option<Budget> {
        self.with(name, |b| b.clone())
    }

    pub fn json(&self) -> Value {
        let all = self.0.lock().unwrap_or_else(|e| e.into_inner());
        Value::Object(
            all.iter()
                .map(|b| {
                    (
                        b.name.clone(),
                        json!({ "unit": b.unit, "cap": b.cap, "spent": b.spent }),
                    )
                })
                .collect(),
        )
    }
}

/// An expert the gate can choose.
#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub name: String,
    /// What a call costs on top of its budget's pacing, in the units of quality (0 to 1).
    pub price: f64,
    /// The budget it spends, and what one call takes from it.
    pub budget: Option<(String, f64)>,
}

impl Arm {
    /// An arm that costs nothing and has no budget: the cheap expert, accepted.
    pub fn free(name: &str) -> Self {
        Self {
            name: name.to_string(),
            price: 0.0,
            budget: None,
        }
    }

    pub fn paid(name: &str, price: f64, budget: &str, per_call: f64) -> Self {
        Self {
            name: name.to_string(),
            price,
            budget: Some((budget.to_string(), per_call)),
        }
    }
}

/// The gate's ranking for one value.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// Arms to try, best first. Arms that cannot be paid are left out.
    pub ranked: Vec<usize>,
    pub prices: Vec<f64>,
    pub scores: Vec<f64>,
    /// The share of the expected stream seen, which the prices pace against.
    pub pace: f64,
}

/// The gate. See the module documentation.
pub struct Gate {
    arms: Vec<Arm>,
    ledger: Arc<Ledger>,
    estimates: Vec<UCB1>,
    horizon: f64,
    seen: u64,
    /// How much a budget's lead over its pace adds to its price.
    pub gain: f64,
    /// The weight of UCB1's exploration bonus.
    pub explore: f64,
    pending: HashMap<String, (usize, usize)>,
}

impl Gate {
    /// A gate over `arms` in `contexts` contexts, pacing its budgets over `horizon` values.
    pub fn new(contexts: usize, arms: Vec<Arm>, ledger: Arc<Ledger>, horizon: u64) -> Self {
        let n = arms.len();
        Self {
            arms,
            ledger,
            estimates: (0..contexts.max(1)).map(|_| UCB1::new(n)).collect(),
            horizon: horizon.max(1) as f64,
            seen: 0,
            gain: 4.0,
            explore: 0.3,
            pending: HashMap::new(),
        }
    }

    pub fn arms(&self) -> &[Arm] {
        &self.arms
    }

    pub fn ledger(&self) -> &Arc<Ledger> {
        &self.ledger
    }

    /// The share of the expected stream seen so far.
    pub fn pace(&self) -> f64 {
        (self.seen as f64 / self.horizon).min(1.0)
    }

    /// What `arm` costs now; infinite when its budget cannot pay one more call.
    pub fn price(&self, arm: usize) -> f64 {
        let a = &self.arms[arm];
        let Some((name, per_call)) = &a.budget else {
            return a.price;
        };
        let Some(b) = self.ledger.get(name) else {
            return f64::INFINITY;
        };
        if b.spent + per_call > b.cap + 1e-12 || b.cap <= 0.0 {
            return f64::INFINITY;
        }
        (a.price + self.gain * (b.spent / b.cap - self.pace())).max(0.0)
    }

    /// Ranks the arms for one value in `ctx`, and counts it toward the pace.
    pub fn choose(&mut self, ctx: usize) -> Choice {
        let ctx = ctx.min(self.estimates.len() - 1);
        let prices: Vec<f64> = (0..self.arms.len()).map(|a| self.price(a)).collect();
        let e = &self.estimates[ctx];
        let total = (e.total_count.max(1) as f64).ln();
        let scores: Vec<f64> = (0..self.arms.len())
            .map(|a| {
                if prices[a].is_infinite() {
                    f64::NEG_INFINITY
                } else if e.counts[a] == 0 {
                    f64::INFINITY
                } else {
                    let bonus = (2.0 * total / e.counts[a] as f64).sqrt();
                    e.q_values[a] + self.explore * bonus - prices[a]
                }
            })
            .collect();
        let mut ranked: Vec<usize> = (0..self.arms.len())
            .filter(|&a| scores[a] > f64::NEG_INFINITY)
            .collect();
        // Untried arms first, in arm order; then by score; ties keep arm order.
        ranked.sort_by(|&x, &y| scores[y].total_cmp(&scores[x]).then(x.cmp(&y)));
        let pace = self.pace();
        self.seen += 1;
        Choice {
            ranked,
            prices,
            scores,
            pace,
        }
    }

    /// Records that `item`, in `ctx`, went to `arm`.
    pub fn commit(&mut self, item: &str, ctx: usize, arm: usize) {
        let ctx = ctx.min(self.estimates.len() - 1);
        self.pending.insert(item.to_string(), (ctx, arm));
    }

    /// Learns how good the answer for `item` was (0 to 1). False when the gate never sent it.
    pub fn learn(&mut self, item: &str, quality: f64) -> bool {
        match self.pending.remove(item) {
            Some((ctx, arm)) => {
                self.estimates[ctx].update(arm, quality);
                true
            }
            None => false,
        }
    }

    /// What the gate believes in `ctx`: each arm's mean quality and how often it was judged.
    pub fn beliefs(&self, ctx: usize) -> (Vec<f64>, Vec<usize>) {
        let e = &self.estimates[ctx.min(self.estimates.len() - 1)];
        (e.q_values.clone(), e.counts.clone())
    }

    /// Values sent and not yet judged.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }
}
