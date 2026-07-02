//! `ix_code_health` — a code-health map of the IX workspace, computed by IX over IX.
//!
//! Pure dogfood: `read_text('crates/**/*.rs')` streams every Rust source file into the
//! bench, the `ix_code_*` UDFs score each one (cyclomatic + cognitive complexity, SLOC,
//! lexical smells), and a GROUP BY rolls them up per crate. No external data — the input
//! is the repository itself, so it runs on any checkout.
//!
//! Three stages:
//!   1. **Health table** — per-crate complexity + smell density.
//!   2. **Map** — `ix_pca_project` the standardized per-crate feature vectors to 2-D and
//!      report the multivariate outliers (crates unlike the rest).
//!   3. **Validation** — *does code-health predict change-proneness?* Correlate each
//!      health metric against per-crate git churn, with a permutation null so the answer
//!      is a measured p-value, not a just-so story. (An external-validity check, matched
//!      to a *predictive* claim — the analog of the mesh's null model.)
//!
//! Run: `cargo run -p ix-duck --example ix_code_health --features duck`

use std::process::Command;

use ix_duck::open_bench;
use ix_math::inference::pearson;

const NULLS: usize = 2000;
const SEED: u64 = 0xC0DE_11EA_17C0_DE11;

fn main() -> ix_duck::Result<()> {
    let conn = open_bench()?;

    // ── 1. per-file scores → per-crate roll-up ───────────────────────────────────
    let sql = "
        WITH files AS (
            SELECT
                regexp_extract(replace(filename, chr(92), '/'), 'crates/([^/]+)/', 1) AS crate,
                ix_code_complexity(content, filename) AS cc,
                CAST(json_extract(ix_code_metrics(content, filename), '$.file_scope.cognitive') AS DOUBLE) AS cog,
                CAST(json_extract(ix_code_metrics(content, filename), '$.file_scope.sloc') AS DOUBLE) AS sloc,
                json_array_length(ix_code_smells(content, filename)) AS smells
            FROM read_text('crates/**/*.rs')
        )
        SELECT crate, count(*) AS files, sum(sloc) AS sloc, avg(cc) AS mean_cc,
               max(cc) AS max_cc, avg(cog) AS mean_cog, sum(smells) AS smells,
               1000.0 * sum(smells) / nullif(sum(sloc),0) AS smell_density
        FROM files WHERE crate <> '' GROUP BY crate
        ORDER BY smell_density DESC NULLS LAST";
    let mut stmt = conn.prepare(sql)?;
    let crates: Vec<CrateHealth> = stmt
        .query_map([], |r| {
            Ok(CrateHealth {
                name: r.get(0)?,
                files: r.get(1)?,
                sloc: r.get::<_, f64>(2)?,
                mean_cc: r.get(3)?,
                max_cc: r.get(4)?,
                mean_cog: r.get(5)?,
                smells: r.get(6)?,
                smell_density: r.get::<_, Option<f64>>(7)?.unwrap_or(0.0),
            })
        })?
        .collect::<Result<_, _>>()?;

    println!(
        "ix over ix: {} crates, {} files, {:.0} SLOC, {} lexical smells\n",
        crates.len(),
        crates.iter().map(|c| c.files).sum::<i64>(),
        crates.iter().map(|c| c.sloc).sum::<f64>(),
        crates.iter().map(|c| c.smells).sum::<i64>()
    );
    println!("highest smell-density crates (lexical smells per KLOC):");
    println!("   {:<20} {:>6} {:>7} {:>7} {:>7} {:>11}", "crate", "files", "SLOC", "mean_cc", "mean_cog", "smells/KLOC");
    for c in crates.iter().take(10) {
        println!("   {:<20} {:>6} {:>7.0} {:>7.1} {:>7.1} {:>11.1}", c.name, c.files, c.sloc, c.mean_cc, c.mean_cog, c.smell_density);
    }

    // ── 2. map: ix_pca_project the standardized feature vectors → 2-D ─────────────
    // Features: complexity (mean/max), cognitive load, smell density, log size.
    let feats: Vec<Vec<f64>> = crates
        .iter()
        .map(|c| vec![c.mean_cc, c.max_cc, c.mean_cog, c.smell_density, (1.0 + c.sloc).ln()])
        .collect();
    let z = standardize(&feats);
    let json = matrix_json(&z);
    let mut pca = conn.prepare("SELECT row, coords[1], coords[2] FROM ix_pca_project(?, 2) ORDER BY row")?;
    let coords: Vec<(usize, f64, f64)> = pca
        .query_map([json], |r| Ok((r.get::<_, i64>(0)? as usize, r.get::<_, f64>(1)?, r.get::<_, f64>(2)?)))?
        .collect::<Result<_, _>>()?;
    let mut outliers: Vec<(usize, f64)> = coords.iter().map(|&(i, x, y)| (i, x * x + y * y)).collect();
    outliers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nix_pca_project health-space map — most multivariate-outlier crates (PC1²+PC2²):");
    for &(i, d2) in outliers.iter().take(8) {
        println!("   {:<20} dist {:.2}  (mean_cc {:.1}, smells/KLOC {:.1})", crates[i].name, d2.sqrt(), crates[i].mean_cc, crates[i].smell_density);
    }

    // ── 3. validation: does code-health predict churn? ───────────────────────────
    // churn = commits touching crates/<name>/; normalize by size (per KLOC) so the test
    // isn't just "bigger crates change more".
    let churn_per_kloc: Vec<f64> = crates
        .iter()
        .map(|c| churn(&c.name) as f64 / (c.sloc / 1000.0).max(0.1))
        .collect();
    // Without a usable git checkout (no `git`, or a shallow clone), `churn()` returns 0
    // for every crate → a constant vector, on which Pearson is undefined and the
    // permutation test is meaningless. Skip the validation with a clear note (Codex P2).
    if churn_per_kloc.iter().all(|&c| c == churn_per_kloc[0]) {
        println!(
            "\nvalidation skipped — churn is constant across crates (no usable git history; \
             run from a full clone with `git` on PATH)."
        );
        return Ok(());
    }
    println!("\nvalidation — does code-health predict git churn (commits/KLOC)? [{NULLS} permutation nulls]");
    for (label, pred) in [
        ("mean cyclomatic", crates.iter().map(|c| c.mean_cc).collect::<Vec<_>>()),
        ("smell density", crates.iter().map(|c| c.smell_density).collect::<Vec<_>>()),
    ] {
        let r = pearson(&pred, &churn_per_kloc).unwrap_or(0.0);
        let p = perm_p(&pred, &churn_per_kloc, r, SEED ^ label.len() as u64);
        let verdict = if p < 0.05 { "predictive" } else { "NOT predictive (within null)" };
        println!("   {label:<16}: r = {r:+.3}  p = {p:.4}  → {verdict}");
    }

    Ok(())
}

struct CrateHealth {
    name: String,
    files: i64,
    sloc: f64,
    mean_cc: f64,
    max_cc: f64,
    mean_cog: f64,
    smells: i64,
    smell_density: f64,
}

/// Commits in history touching `crates/<crate>/`.
fn churn(crate_name: &str) -> i64 {
    Command::new("git")
        .args(["rev-list", "--count", "HEAD", "--", &format!("crates/{crate_name}/")])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// Column-wise z-score standardization (mean 0, unit variance; constant column → 0).
fn standardize(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, d) = (m.len(), m[0].len());
    let mut out = vec![vec![0.0; d]; n];
    for j in 0..d {
        let col: Vec<f64> = m.iter().map(|r| r[j]).collect();
        let mean = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = var.sqrt();
        for i in 0..n {
            out[i][j] = if sd > 0.0 { (m[i][j] - mean) / sd } else { 0.0 };
        }
    }
    out
}

fn matrix_json(m: &[Vec<f64>]) -> String {
    let rows: Vec<String> = m
        .iter()
        .map(|r| format!("[{}]", r.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(",")))
        .collect();
    format!("[{}]", rows.join(","))
}

/// Two-sided permutation p-value: fraction of label-shuffled correlations with
/// |r_null| ≥ |r_real|. Deterministic (seeded xorshift), so the p is reproducible.
fn perm_p(x: &[f64], y: &[f64], r_real: f64, seed: u64) -> f64 {
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut yp = y.to_vec();
    let mut ge = 0usize;
    for _ in 0..NULLS {
        for i in (1..yp.len()).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            yp.swap(i, j);
        }
        if pearson(x, &yp).map(|r| r.abs() >= r_real.abs()).unwrap_or(false) {
            ge += 1;
        }
    }
    (1 + ge) as f64 / (NULLS + 1) as f64
}
