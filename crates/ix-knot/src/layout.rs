//! Strand paths in 3D, for drawing a braid: one crossing per unit of y.

use crate::braid::Braid;
use std::f64::consts::PI;

/// The most points a layout returns, over all strands.
pub const MAX_POINTS: usize = 100_000;

/// One strand's path: the position it starts in (0-based) and its points
/// `[x, y, z]`.
#[derive(Debug, Clone, PartialEq)]
pub struct StrandPath {
    pub start: usize,
    pub points: Vec<[f64; 3]>,
}

/// Why a layout was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayoutError {
    #[error("samples per crossing must be at least 2, got {0}")]
    Samples(usize),
    #[error("the layout would have {0} points; the limit is {MAX_POINTS}")]
    Points(usize),
}

/// Lay a braid out in a right-handed frame: x across the strands (position k at
/// x = k - 1), y along the braid (crossing j between y = j and y = j + 1), z
/// toward the viewer.
///
/// At σₖ the strand in position k moves to position k + 1 and rises to z = 1
/// halfway, in front, while the other strand moves the other way and dips to
/// z = -1; σₖ⁻¹ swaps which one is in front. Strands that do not cross stay
/// put at z = 0. Each crossing is sampled at `samples` points, and the last row
/// is added at y = crossings.
///
/// At every whole y each strand sits at a whole x with z = 0, so the layout of a
/// repeated word repeats with it, and a picture of it tiles. A raster whose rows
/// grow downward shows this picture reflected, which is the mirror braid: lay
/// out [`Braid::mirror`], or draw y upward, to picture the braid itself.
pub fn layout(braid: &Braid, samples: usize) -> Result<Vec<StrandPath>, LayoutError> {
    if samples < 2 {
        return Err(LayoutError::Samples(samples));
    }
    let n = braid.strands();
    let per_strand = braid.crossings() * samples + 1;
    if n * per_strand > MAX_POINTS {
        return Err(LayoutError::Points(n * per_strand));
    }
    let mut paths: Vec<StrandPath> = (0..n)
        .map(|start| StrandPath {
            start,
            points: Vec::with_capacity(per_strand),
        })
        .collect();
    let mut at: Vec<usize> = (0..n).collect(); // at[position] = strand
    for (j, &g) in braid.word().iter().enumerate() {
        let k = g.unsigned_abs() as usize - 1;
        let (left, right) = (at[k], at[k + 1]);
        let front = if g > 0 { left } else { right };
        for i in 0..samples {
            let u = i as f64 / samples as f64;
            let ease = (1.0 - (PI * u).cos()) / 2.0;
            let lift = (PI * u).sin();
            for (position, &strand) in at.iter().enumerate() {
                // `0.0 - lift`, not `-lift`: no -0.0 where the crossing starts.
                let z = if strand == front { lift } else { 0.0 - lift };
                let point = if strand == left {
                    [k as f64 + ease, j as f64 + u, z]
                } else if strand == right {
                    [(k + 1) as f64 - ease, j as f64 + u, z]
                } else {
                    [position as f64, j as f64 + u, 0.0]
                };
                paths[strand].points.push(point);
            }
        }
        at.swap(k, k + 1);
    }
    let y = braid.crossings() as f64;
    for (position, &strand) in at.iter().enumerate() {
        paths[strand].points.push([position as f64, y, 0.0]);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The word as the picture shows it: wherever two neighbouring strands swap
    /// sides, the one nearer the viewer just before the swap is in front.
    fn read_word(paths: &[StrandPath]) -> Vec<i32> {
        let mut order: Vec<usize> = (0..paths.len()).collect();
        order.sort_by(|a, b| paths[*a].points[0][0].total_cmp(&paths[*b].points[0][0]));
        let mut word = Vec::new();
        for i in 1..paths[0].points.len() {
            for p in 0..order.len().saturating_sub(1) {
                let (a, b) = (order[p], order[p + 1]);
                if paths[b].points[i][0] < paths[a].points[i][0] - 1e-9 {
                    let k = p as i32 + 1;
                    let a_in_front = paths[a].points[i - 1][2] > paths[b].points[i - 1][2];
                    word.push(if a_in_front { k } else { -k });
                    order.swap(p, p + 1);
                }
            }
        }
        word
    }

    #[test]
    fn the_word_reads_back_from_the_picture() {
        for text in [
            "s1 s2^-1",
            "s1 s2",
            "s1^3 s2^-3",
            "s1^3 s2^3",
            "s2 s1^-1 s3 s2^-2 s1",
        ] {
            let b = Braid::parse(None, text).unwrap().repeat(3).unwrap();
            for samples in [2, 3, 16] {
                let paths = layout(&b, samples).unwrap();
                assert_eq!(read_word(&paths), b.word(), "{text} at {samples} samples");
                assert!(paths
                    .iter()
                    .all(|p| p.points.len() == b.crossings() * samples + 1));
            }
        }
    }

    #[test]
    fn a_repeated_word_repeats_so_its_picture_tiles() {
        let samples = 8;
        let plait = Braid::parse(None, "s1 s2^-1").unwrap().repeat(6).unwrap();
        let paths = layout(&plait, samples).unwrap();
        // One plait period, σ₁σ₂⁻¹ three times, brings every strand home.
        let period = 6 * samples;
        let rows = |i: usize| {
            let mut row: Vec<[f64; 2]> = paths
                .iter()
                .map(|p| [p.points[i][0], p.points[i][2]])
                .collect();
            row.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            row
        };
        for i in 0..paths[0].points.len() - period {
            assert_eq!(rows(i), rows(i + period), "row {i}");
        }
        for p in &paths {
            for point in p.points.iter().step_by(samples) {
                assert_eq!(point[0].fract(), 0.0);
                assert_eq!(point[2], 0.0);
            }
        }
    }

    #[test]
    fn refuses_too_few_samples_and_too_many_points() {
        let b = Braid::parse(None, "s1 s2^-1").unwrap();
        assert_eq!(layout(&b, 1), Err(LayoutError::Samples(1)));
        let long = Braid::new(8, (1..8).cycle().take(64).collect()).unwrap();
        assert_eq!(
            layout(&long, 200),
            Err(LayoutError::Points(8 * (64 * 200 + 1)))
        );
    }
}
