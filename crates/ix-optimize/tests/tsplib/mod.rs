//! Minimal TSPLIB reader for the `EUC_2D` instances under `tests/fixtures`.
//!
//! The `.tsp` and `.opt.tour` files are the TSPLIB95 files from
//! comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/, with line endings
//! normalised to LF. Shared by `tests/aco_tsplib.rs` and
//! `examples/aco_tsplib.rs`.

use ndarray::Array2;

/// A TSPLIB instance with its published optimal tour.
pub struct Instance {
    pub name: &'static str,
    pub tsp: &'static str,
    pub opt_tour: &'static str,
    /// Proven optimal tour length (TSPLIB).
    pub optimum: f64,
}

pub const BERLIN52: Instance = Instance {
    name: "berlin52",
    tsp: include_str!("../fixtures/berlin52.tsp"),
    opt_tour: include_str!("../fixtures/berlin52.opt.tour"),
    optimum: 7542.0,
};

pub const KROA100: Instance = Instance {
    name: "kroA100",
    tsp: include_str!("../fixtures/kroA100.tsp"),
    opt_tour: include_str!("../fixtures/kroA100.opt.tour"),
    optimum: 21282.0,
};

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
