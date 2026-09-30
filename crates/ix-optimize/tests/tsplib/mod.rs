//! Minimal TSPLIB reader for the `EUC_2D` instances under `tests/fixtures`.
//!
//! `berlin52.tsp` and `berlin52.opt.tour` are the TSPLIB95 files from
//! comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/, with line endings
//! normalised to LF. Shared by `tests/aco_berlin52.rs` and
//! `examples/aco_berlin52.rs`.

use ndarray::Array2;

pub const BERLIN52_TSP: &str = include_str!("../fixtures/berlin52.tsp");
pub const BERLIN52_OPT_TOUR: &str = include_str!("../fixtures/berlin52.opt.tour");

/// Proven optimal tour length of berlin52 (TSPLIB).
pub const BERLIN52_OPTIMUM: f64 = 7542.0;

/// The distance matrix of an `EUC_2D` instance, rounded to the nearest integer
/// as TSPLIB defines it (`nint(sqrt(dx^2 + dy^2))`), so tour lengths are
/// comparable with published optima.
pub fn euc_2d_distances(tsp: &str) -> Array2<f64> {
    let points: Vec<(f64, f64)> = section(tsp, "NODE_COORD_SECTION")
        .map(|line| {
            let mut fields = line.split_whitespace().skip(1);
            let mut coord = || {
                fields
                    .next()
                    .expect("coordinate")
                    .parse::<f64>()
                    .expect("number")
            };
            (coord(), coord())
        })
        .collect();
    let n = points.len();
    Array2::from_shape_fn((n, n), |(i, j)| {
        let (dx, dy) = (points[i].0 - points[j].0, points[i].1 - points[j].1);
        (dx * dx + dy * dy).sqrt().round()
    })
}

/// The tour of a `.opt.tour` file, as 0-based city indices.
pub fn tour(opt_tour: &str) -> Vec<usize> {
    section(opt_tour, "TOUR_SECTION")
        .map(|line| line.parse::<i64>().expect("city index"))
        .take_while(|&city| city != -1)
        .map(|city| (city - 1) as usize)
        .collect()
}

fn section<'a>(file: &'a str, header: &'a str) -> impl Iterator<Item = &'a str> {
    file.lines()
        .map(str::trim)
        .skip_while(move |line| *line != header)
        .skip(1)
        .take_while(|line| !line.is_empty() && *line != "EOF")
}
