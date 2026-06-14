//! Render the Campus index (`docs/streeling/README.md`) from the catalog.
//! Generated sections are deterministic; a hand-written intro between
//! `<!-- streeling:intro -->` markers is preserved across regenerations.

use crate::model::LearningRecord;
use std::collections::BTreeMap;
use std::fmt::Write as _;

pub const INTRO_START: &str = "<!-- streeling:intro -->";
pub const INTRO_END: &str = "<!-- /streeling:intro -->";

const DEFAULT_INTRO: &str = "\
Streeling University is the front door to everything the ecosystem has learned —\n\
the \"expose\" half of the `/learnings` loop (see `docs/LEARNING.md`). It indexes,\n\
it does not own: every entry lives in its source repo's `docs/solutions` (or\n\
knowledge/plans/brainstorms). Query the corpus via the Registrar (`docs/streeling/queries.sql`).";

/// Pull the intro text out of an existing README (between the markers).
pub fn extract_intro(existing: &str) -> Option<String> {
    let start = existing.find(INTRO_START)? + INTRO_START.len();
    let end = existing[start..].find(INTRO_END)? + start;
    Some(existing[start..end].trim().to_string())
}

/// Render the full campus index. `existing` (if any) supplies the preserved intro.
pub fn render(records: &[LearningRecord], existing: Option<&str>) -> String {
    let intro = existing
        .and_then(extract_intro)
        .unwrap_or_else(|| DEFAULT_INTRO.to_string());

    let mut out = String::new();
    let _ = writeln!(out, "# 📚 Streeling University\n");
    let _ = writeln!(out, "{INTRO_START}");
    let _ = writeln!(out, "{intro}");
    let _ = writeln!(out, "{INTRO_END}\n");
    let _ = writeln!(
        out,
        "> Generated from `state/streeling/catalog.jsonl` by `streeling campus`. Do not edit below the intro by hand — edits are overwritten; the intro block is preserved.\n"
    );

    // Summary.
    let mut by_repo: BTreeMap<&str, usize> = BTreeMap::new();
    for r in records {
        *by_repo.entry(r.repo.as_str()).or_default() += 1;
    }
    let _ = writeln!(out, "## Enrollment\n");
    let _ = writeln!(out, "- **{}** learnings total", records.len());
    for (repo, n) in &by_repo {
        let _ = writeln!(out, "  - `{repo}`: {n}");
    }
    let _ = writeln!(out);

    // Faculties (by category), titles linked back to source (ix links resolve; others shown as refs).
    let mut by_cat: BTreeMap<&str, Vec<&LearningRecord>> = BTreeMap::new();
    for r in records {
        by_cat.entry(r.category.as_str()).or_default().push(r);
    }
    let _ = writeln!(out, "## Faculties\n");
    for (cat, recs) in &by_cat {
        let _ = writeln!(out, "### {cat} ({})\n", recs.len());
        for r in recs {
            let label = format!("{} · `{}`", r.title, r.repo);
            if r.repo == "ix" {
                // README is at docs/streeling/; repo root is ../../.
                let _ = writeln!(out, "- [{label}](../../{})", r.path);
            } else {
                let _ = writeln!(out, "- {label} — `{}`", r.path);
            }
        }
        let _ = writeln!(out);
    }

    let _ = writeln!(
        out,
        "## Registrar\n\nQuery the whole corpus with DuckDB over `state/streeling/catalog.jsonl` — see `docs/streeling/queries.sql` and `docs/DUCKDB.md`.\n"
    );
    out
}
