//! `ix_maintain_snapshot` — the **scheduled maintain producer** (Phase A tracer-bullet).
//! Evaluates the maintain gate over the LIVE sibling `../ga` telemetry, then writes a
//! **current-verdict snapshot** in the `ga/state/quality` scorecard shape (latest hexavalent
//! verdict + a `maintain_trend` rollup over the append-only ledger). End-to-end on the IX
//! side: GA data → fused verdict → snapshot file.
//!
//!   cargo run -p ix-duck --features duck --example ix_maintain_snapshot
//!   cargo run -p ix-duck --features duck --example ix_maintain_snapshot -- <out.json>
//!   GA_ROOT=ga-sibling cargo run … --example ix_maintain_snapshot       # CI sibling checkout
//!
//! Discipline (per issue #145 / ADR-0002):
//!   * **formats-not-coupling** — writes ONLY the IX tree; Phase B federates into ga/state/quality.
//!   * **absent-ga → skip** — no sibling corpus ⇒ exit 0 (nothing to produce, not a failure).
//!   * **absent-hits → skip** — no IX-local metric ledger ⇒ exit 0. Same rule, other
//!     load-bearing input: publishing "cannot decide" as a snapshot makes a structural
//!     gap look like a metric regression on the dashboard.
//!   * **fail-closed** — ga present but unreadable ⇒ nonzero exit, no snapshot written.
//!   * **atomic** — temp + rename, so a reader never sees a half-written scorecard.
//!   * **advisory until Phase 3b** — the verdict is not binding.

use std::path::PathBuf;

use ix_duck::maintain::{self, MaintainConfig, MaintainInputs};

/// `GA_ROOT` (CI sibling checkout) else `../ga` relative to the IX repo root.
fn ga_root() -> PathBuf {
    if let Ok(root) = std::env::var("GA_ROOT") {
        return PathBuf::from(root);
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let ix_root = cwd.canonicalize().unwrap_or(cwd);
    ix_root
        .parent()
        .map(|p| p.join("ga"))
        .unwrap_or_else(|| PathBuf::from("../ga"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("state/quality/maintain-gate/last.json"));

    let ga = ga_root();
    let ga_quality = ga.join("state/quality");
    let corpus = ga_quality.join("chatbot-qa");

    // absent-ga → skip (exit 0): there is nothing to produce, which is not a failure.
    // The guardrail corpus is the load-bearing input; without it the gate can't decide.
    if !corpus.exists() {
        eprintln!(
            "ix_maintain_snapshot: no GA sibling corpus at {} — skipping (exit 0)",
            corpus.display()
        );
        return Ok(());
    }

    // Metric is IX-local + harness-written (the externally-derived yield ledger).
    let hits = PathBuf::from("state/thinking-machine/hits.jsonl");

    // absent-hits → skip (exit 0), mirroring absent-ga above. hits.jsonl is gitignored
    // by design (.gitignore: "per-environment runtime data, not a committed artifact"),
    // so a CI checkout never has it. Without it `metric_delta` returns None, `metric_up`
    // is None, and `fuse` correctly yields U/escalate "metric evidence missing — cannot
    // decide". That verdict is honest, but writing it as a SNAPSHOT publishes
    // metric_value 0.0 + oracle_status=warn to the dashboard, where a structural gap
    // ("this gate cannot run in this environment") is indistinguishable from a live
    // regression ("the gate ran and the metric fell"). The scorecard read DEGRADED on
    // that basis while the nightly went green 8 runs running.
    //
    // The metric lens is load-bearing exactly as the guardrail corpus is: absent ⇒
    // nothing to produce, which is not a failure. Skip instead of publishing a verdict
    // this environment cannot earn.
    if !hits.exists() {
        eprintln!(
            "ix_maintain_snapshot: no metric evidence at {} — skipping (exit 0). \
             hits.jsonl is per-environment runtime data; the maintain gate is only \
             decidable where the harness has written it.",
            hits.display()
        );
        return Ok(());
    }
    let loops_dir = ga_quality.join("loops");
    let emb_dir = ga_quality.join("query-embeddings");
    let run_at = chrono::Utc::now().to_rfc3339();

    let inputs = MaintainInputs {
        hits_path: &hits,
        corpus_dir: &corpus,
        loops_dir: Some(&loops_dir),
        query_embeddings_dir: Some(&emb_dir),
        iteration: None, // whole-history advisory (Phase 1) — scheduled, not per-iteration
        run_at: &run_at,
    };

    let conn = ix_duck::open_bench()?;
    // fail-closed: ga present but a lens errors (unreadable corpus, malformed JSON) → the `?`
    // propagates and main returns Err ⇒ nonzero exit, and no snapshot is written.
    let verdict = maintain::evaluate(&conn, &MaintainConfig::default(), &inputs)?;

    // Reuse maintain_trend over the IX-local append-only ledger for the convergence rollup.
    let ledger = PathBuf::from("state/thinking-machine/maintain-gate.jsonl");
    let trend = maintain::maintain_trend(&conn, &ledger)?;

    let snapshot = maintain::build_snapshot(&verdict, &trend);
    maintain::write_snapshot_atomic(&snapshot, &out)?;

    eprintln!(
        "ix_maintain_snapshot: {} ({}) → {}",
        snapshot.status,
        snapshot.oracle_status,
        out.display()
    );
    Ok(())
}
