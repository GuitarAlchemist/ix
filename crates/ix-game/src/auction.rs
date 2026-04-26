//! Auction mechanisms — first/second price, English, Dutch, all-pay.

use rand::prelude::*;

/// Bid in an auction.
#[derive(Debug, Clone, Copy)]
pub struct Bid {
    pub bidder: usize,
    pub amount: f64,
}

/// Auction result.
#[derive(Debug, Clone)]
pub struct AuctionResult {
    pub winner: usize,
    pub payment: f64,
    pub all_bids: Vec<Bid>,
}

/// First-price sealed-bid auction.
/// Winner pays their own bid.
pub fn first_price_auction(bids: &[Bid]) -> Option<AuctionResult> {
    if bids.is_empty() {
        return None;
    }

    let winner = bids
        .iter()
        .max_by(|a, b| a.amount.partial_cmp(&b.amount).unwrap())?;

    Some(AuctionResult {
        winner: winner.bidder,
        payment: winner.amount,
        all_bids: bids.to_vec(),
    })
}

/// Second-price sealed-bid (Vickrey) auction.
/// Winner pays the second-highest bid. Truthful bidding is dominant.
pub fn second_price_auction(bids: &[Bid]) -> Option<AuctionResult> {
    if bids.len() < 2 {
        return first_price_auction(bids);
    }

    let mut sorted: Vec<&Bid> = bids.iter().collect();
    sorted.sort_by(|a, b| b.amount.partial_cmp(&a.amount).unwrap());

    Some(AuctionResult {
        winner: sorted[0].bidder,
        payment: sorted[1].amount, // Pay second-highest
        all_bids: bids.to_vec(),
    })
}

/// All-pay auction: everyone pays, only highest bidder wins.
/// Returns (winner, payments_by_bidder).
pub fn all_pay_auction(bids: &[Bid]) -> Option<(usize, Vec<f64>)> {
    if bids.is_empty() {
        return None;
    }

    let winner = bids
        .iter()
        .max_by(|a, b| a.amount.partial_cmp(&b.amount).unwrap())?;

    let max_bidder = bids.iter().map(|b| b.bidder).max().unwrap_or(0);
    let mut payments = vec![0.0; max_bidder + 1];
    for bid in bids {
        payments[bid.bidder] = bid.amount;
    }

    Some((winner.bidder, payments))
}

/// Simulate an English (ascending) auction.
///
/// Each round, the price increases by `increment`.
/// Bidders drop out when price exceeds their value.
/// Returns the final price and winner.
pub fn english_auction(values: &[f64], start_price: f64, increment: f64) -> AuctionResult {
    let n = values.len();
    let mut active = vec![true; n];
    let mut price = start_price;

    loop {
        let num_active: usize = active.iter().filter(|&&a| a).count();
        if num_active <= 1 {
            break;
        }

        price += increment;

        // Drop bidders whose value is below current price
        for i in 0..n {
            if active[i] && values[i] < price {
                active[i] = false;
            }
        }
    }

    let winner = active.iter().position(|&a| a).unwrap_or(0);
    let bids: Vec<Bid> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| Bid {
            bidder: i,
            amount: v,
        })
        .collect();

    AuctionResult {
        winner,
        payment: price,
        all_bids: bids,
    }
}

/// Simulate a Dutch (descending) auction.
///
/// Price starts high and decreases. First bidder to accept wins.
pub fn dutch_auction(values: &[f64], start_price: f64, decrement: f64) -> AuctionResult {
    let mut price = start_price;

    loop {
        // Check if any bidder accepts
        for (i, &v) in values.iter().enumerate() {
            if v >= price {
                let bids: Vec<Bid> = values
                    .iter()
                    .enumerate()
                    .map(|(j, &val)| Bid {
                        bidder: j,
                        amount: val,
                    })
                    .collect();
                return AuctionResult {
                    winner: i,
                    payment: price,
                    all_bids: bids,
                };
            }
        }

        price -= decrement;
        if price <= 0.0 {
            break;
        }
    }

    // No one bid
    let bids: Vec<Bid> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| Bid {
            bidder: i,
            amount: v,
        })
        .collect();
    AuctionResult {
        winner: 0,
        payment: 0.0,
        all_bids: bids,
    }
}

/// Revenue equivalence theorem check: compare average revenues.
///
/// Runs many auction simulations with random values and compares
/// first-price and second-price auction revenues.
pub fn revenue_equivalence_test(num_bidders: usize, num_trials: usize, seed: u64) -> (f64, f64) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut fp_total = 0.0;
    let mut sp_total = 0.0;

    for _ in 0..num_trials {
        let values: Vec<f64> = (0..num_bidders).map(|_| rng.random::<f64>()).collect();

        // Second-price: bid truthfully
        let sp_bids: Vec<Bid> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| Bid {
                bidder: i,
                amount: v,
            })
            .collect();
        if let Some(sp_result) = second_price_auction(&sp_bids) {
            sp_total += sp_result.payment;
        }

        // First-price: optimal bid = (n-1)/n * value (for uniform distribution)
        let factor = (num_bidders - 1) as f64 / num_bidders as f64;
        let fp_bids: Vec<Bid> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| Bid {
                bidder: i,
                amount: v * factor,
            })
            .collect();
        if let Some(fp_result) = first_price_auction(&fp_bids) {
            fp_total += fp_result.payment;
        }
    }

    (fp_total / num_trials as f64, sp_total / num_trials as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_second_price_truthful() {
        let bids = vec![
            Bid {
                bidder: 0,
                amount: 10.0,
            },
            Bid {
                bidder: 1,
                amount: 7.0,
            },
            Bid {
                bidder: 2,
                amount: 5.0,
            },
        ];

        let result = second_price_auction(&bids).unwrap();
        assert_eq!(result.winner, 0);
        assert!(
            (result.payment - 7.0).abs() < 1e-10,
            "Should pay second-highest bid"
        );
    }

    #[test]
    fn test_english_auction() {
        let values = vec![10.0, 7.0, 5.0];
        let result = english_auction(&values, 0.0, 0.5);
        assert_eq!(result.winner, 0);
        // Price should be near the second-highest value
        assert!(result.payment >= 7.0 && result.payment <= 8.0);
    }

    #[test]
    fn test_revenue_equivalence() {
        let (fp_rev, sp_rev) = revenue_equivalence_test(3, 10_000, 42);
        // Revenue equivalence: both should yield similar average revenue
        let ratio = fp_rev / sp_rev;
        assert!(
            ratio > 0.85 && ratio < 1.15,
            "Revenue equivalence: FP={:.4}, SP={:.4}, ratio={:.4}",
            fp_rev,
            sp_rev,
            ratio
        );
    }
}
