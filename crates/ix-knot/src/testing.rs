//! Fixtures the tests of several modules share.

use crate::{layout, Braid, Rope};
use std::f64::consts::PI;

/// A braid closure drawn as a ring: braid position k at radius 2 + k, the
/// braid's length once around, z as the layout lifts it. The map keeps the
/// plane's orientation, so the crossings keep their signs.
pub(crate) fn ring(b: &Braid) -> Vec<Rope> {
    // An odd count: no sample lands on a crossing, where two strands meet.
    let samples = 3;
    let paths = layout(b, samples).unwrap();
    let turn = 2.0 * PI / b.crossings() as f64;
    let perm = b.permutation();
    let mut seen = vec![false; b.strands()];
    let mut ropes = Vec::new();
    for first in 0..b.strands() {
        let mut points = Vec::new();
        let mut s = first;
        while !seen[s] {
            seen[s] = true;
            let path = &paths[s].points;
            for p in &path[..path.len() - 1] {
                let (a, r) = (turn * p[1], 2.0 + p[0]);
                points.push([r * a.cos(), r * a.sin(), p[2]]);
            }
            s = perm[s];
        }
        if !points.is_empty() {
            ropes.push(Rope {
                points,
                closed: true,
            });
        }
    }
    ropes
}

/// `count` fixed pseudo-random braid words on 2 to 4 strands.
pub(crate) fn words(count: usize) -> Vec<Braid> {
    let mut x: u64 = 0x2026_1004_0d1a;
    let mut next = |m: u64| {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) % m
    };
    (0..count)
        .map(|_| {
            let strands = 2 + next(3) as usize;
            let len = 1 + next(7) as usize;
            let word = (0..len)
                .map(|_| {
                    let k = 1 + next(strands as u64 - 1) as i32;
                    if next(2) == 0 {
                        k
                    } else {
                        -k
                    }
                })
                .collect();
            Braid::new(strands, word).unwrap()
        })
        .collect()
}
