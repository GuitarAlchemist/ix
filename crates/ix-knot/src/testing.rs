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

/// Two interlocked bights, six crossings: the sailor-knot series' drawing of the
/// reef knot. Each rope's tail is its first point, its standing part its last.
/// Its heights make it neither the reef nor the granny: joining the two tails
/// and the two standing parts closes it into the unknot. The crossings that make
/// it either are in `mechanics`'s tests.
pub(crate) fn bights() -> Vec<Rope> {
    let rope = |pts: &[[f64; 3]]| Rope {
        points: pts.to_vec(),
        closed: false,
    };
    let left = rope(&[
        [-3.7, 0.4, 0.0],
        [-3.0, 0.4, -0.5],
        [-2.35, 0.4, -1.0],
        [-1.3, 0.42, -0.6],
        [0.0, 0.8, -1.0],
        [0.9, 1.12, -0.2],
        [1.6, 1.1, 0.5],
        [2.2, 0.75, 1.0],
        [2.42, 0.0, 1.0],
        [2.2, -0.75, 1.0],
        [1.6, -1.1, 0.5],
        [0.9, -1.12, 0.6],
        [0.0, -0.8, 1.0],
        [-1.3, -0.42, 0.3],
        [-2.35, -0.4, -1.0],
        [-3.2, -0.4, -0.5],
        [-4.8, -0.4, 0.0],
    ]);
    let right = rope(&[
        [3.7, 0.4, 0.0],
        [3.0, 0.4, -0.5],
        [2.35, 0.4, -1.0],
        [1.3, 0.42, -0.3],
        [0.04, 0.83, 1.0],
        [-0.9, 1.12, 0.6],
        [-1.6, 1.1, 0.5],
        [-2.2, 0.75, 1.0],
        [-2.42, 0.0, 1.0],
        [-2.2, -0.75, 1.0],
        [-1.6, -1.1, 0.5],
        [-0.9, -1.12, -0.2],
        [0.04, -0.83, -1.0],
        [1.3, -0.42, -0.6],
        [2.35, -0.4, -1.0],
        [3.2, -0.4, -0.5],
        [4.8, -0.4, 0.0],
    ]);
    vec![left, right]
}

/// A clove hitch on a spar, as the sailor-knot series draws it: rope 0 taken
/// twice round rope 1, a straight spar, read by height. Its rope crosses itself
/// at about 42 degrees just after passing over the spar.
pub(crate) fn clove_hitch() -> Vec<Rope> {
    let turns = [
        [1.0, -4.2, 0.0],
        [1.0, -3.0, 0.0],
        [1.0, -2.0, -0.3],
        [1.0, -1.18, -0.6],
        [1.0, -0.5, 0.2],
        [1.0, 0.07, 0.6],
        [1.0, 0.8, 0.4],
        [1.08, 1.6, 0.2],
        [1.35, 2.05, 0.0],
        [1.8, 2.1, -0.1],
        [2.15, 1.65, -0.3],
        [2.27, 0.8, -0.5],
        [2.27, -0.06, -0.6],
        [2.2, -1.0, -0.5],
        [2.05, -1.75, -0.2],
        [1.8, -2.08, 0.1],
        [1.45, -1.56, 0.5],
        [0.98, -1.04, 0.6],
        [0.5, -0.5, 0.6],
        [0.06, 0.03, 0.7],
        [-0.5, 0.6, 0.6],
        [-1.02, 1.18, 0.6],
        [-1.42, 1.62, 0.4],
        [-1.8, 2.08, 0.1],
        [-2.15, 1.65, -0.3],
        [-2.27, 0.8, -0.5],
        [-2.27, -0.06, -0.6],
        [-2.2, -1.0, -0.5],
        [-2.0, -1.75, -0.2],
        [-1.6, -2.05, 0.1],
        [-1.15, -1.6, 0.3],
        [-1.0, -0.8, 0.5],
        [-1.0, 0.04, 0.6],
        [-1.0, 0.6, 0.0],
        [-1.02, 1.27, -0.6],
        [-0.97, 2.1, -0.2],
        [-0.88, 3.0, 0.0],
        [-0.8, 3.7, 0.0],
    ];
    vec![
        Rope {
            points: turns.to_vec(),
            closed: false,
        },
        Rope {
            points: vec![[-3.6, 0.0, 0.0], [3.6, 0.0, 0.0]],
            closed: false,
        },
    ]
}

/// A constrictor knot on a spar, as the sailor-knot series draws it: rope 0
/// taken twice round rope 1, read by height, passing over the spar and under it
/// again a few diameters on.
pub(crate) fn constrictor() -> Vec<Rope> {
    let turns = [
        [0.0, -3.6, 0.0],
        [0.0, -2.3, 0.0],
        [0.0, -1.45, -1.0],
        [0.0, -0.95, -1.0],
        [0.0, -0.62, 1.0],
        [0.0, -0.2, 1.0],
        [0.12, 0.32, 1.0],
        [0.42, 0.85, 0.0],
        [0.76, 0.32, -1.0],
        [0.84, -0.3, -1.0],
        [0.88, -0.95, 0.0],
        [0.8, -1.5, 0.0],
        [0.55, -1.62, 0.5],
        [0.3, -1.35, 1.0],
        [-0.3, -1.05, 1.0],
        [-0.62, -0.62, 1.0],
        [-0.8, -0.15, 1.0],
        [-0.92, 0.3, 1.0],
        [-1.35, 0.78, 0.0],
        [-1.56, 0.3, -1.0],
        [-1.63, -0.3, -1.0],
        [-1.6, -0.82, 0.0],
        [-1.3, -1.05, 0.0],
        [-0.95, -0.72, -1.0],
        [-0.25, -0.46, -1.0],
        [0.25, -0.43, -1.0],
        [0.6, -0.45, 1.0],
        [1.05, -0.48, 1.0],
        [1.6, -0.56, 0.0],
        [2.6, -0.72, 0.0],
    ];
    vec![
        Rope {
            points: turns.to_vec(),
            closed: false,
        },
        Rope {
            points: vec![
                [-3.5, 0.0, 0.0],
                [-2.37, 0.0, 0.0],
                [2.33, 0.0, 0.0],
                [3.5, 0.0, 0.0],
            ],
            closed: false,
        },
    ]
}
