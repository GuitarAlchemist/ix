//! Sentinel report generator — writes a human-readable markdown
//! report for each Sentinel cycle.

use ix_fuzzy::observations::{HexObservation, MergedState};
use ix_types::Hexavalent;

/// Generate a markdown report for one Sentinel cycle.
pub fn generate(
    round: u32,
    merged: &MergedState,
    fixes_applied: &[String],
    escalated: bool,
    escalation_reason: &str,
) -> String {
    let mut out = String::new();

    out.push_str(&format!("# Sentinel Report — Round {round}\n\n"));

    if escalated {
        out.push_str(&format!(
            "## ESCALATED\n\n**Reason:** {escalation_reason}\n\n\
             The Sentinel stopped before applying any fixes. Human review required.\n\n"
        ));
    }

    // Observation summary
    let f = merged
        .observations
        .iter()
        .filter(|o| o.variant == Hexavalent::False)
        .count();
    let d = merged
        .observations
        .iter()
        .filter(|o| o.variant == Hexavalent::Doubtful)
        .count();
    let t = merged
        .observations
        .iter()
        .filter(|o| o.variant == Hexavalent::True)
        .count();
    let p = merged
        .observations
        .iter()
        .filter(|o| o.variant == Hexavalent::Probable)
        .count();
    let u = merged
        .observations
        .iter()
        .filter(|o| o.variant == Hexavalent::Unknown)
        .count();
    let c = merged.contradictions.len();

    out.push_str("## Observation Summary\n\n");
    out.push_str(&format!(
        "| Variant | Count |\n|---|---|\n\
         | T (verified) | {t} |\n\
         | P (probable) | {p} |\n\
         | U (unknown) | {u} |\n\
         | D (doubtful) | {d} |\n\
         | F (refuted) | {f} |\n\
         | C (contradictions) | {c} |\n\n"
    ));

    // Distribution
    out.push_str("## Merged Distribution\n\n");
    for (label, v) in [
        ("T", Hexavalent::True),
        ("P", Hexavalent::Probable),
        ("U", Hexavalent::Unknown),
        ("D", Hexavalent::Doubtful),
        ("F", Hexavalent::False),
        ("C", Hexavalent::Contradictory),
    ] {
        let mass = merged.distribution.get(&v);
        out.push_str(&format!("- **{label}**: {mass:.3}\n"));
    }
    out.push('\n');

    // Contradictions detail
    if !merged.contradictions.is_empty() {
        out.push_str("## Contradictions\n\n");
        for c_obs in &merged.contradictions {
            out.push_str(&format!(
                "- `{}` — weight {:.2} — {}\n",
                c_obs.claim_key,
                c_obs.weight,
                c_obs.evidence.as_deref().unwrap_or("(no evidence)")
            ));
        }
        out.push('\n');
    }

    // Fixes applied
    if !fixes_applied.is_empty() {
        out.push_str("## Remediations Applied\n\n");
        for fix in fixes_applied {
            out.push_str(&format!("- Catalog pattern: `{fix}`\n"));
        }
        out.push('\n');
    }

    // Top F/D observations
    let findings: Vec<&HexObservation> = merged
        .observations
        .iter()
        .filter(|o| matches!(o.variant, Hexavalent::False | Hexavalent::Doubtful))
        .take(20)
        .collect();
    if !findings.is_empty() {
        out.push_str("## Top Findings (F and D observations)\n\n");
        out.push_str("| Claim | Source | Variant | Weight | Evidence |\n");
        out.push_str("|---|---|---|---|---|\n");
        for o in findings {
            out.push_str(&format!(
                "| `{}` | {} | {:?} | {:.2} | {} |\n",
                o.claim_key,
                o.source,
                o.variant,
                o.weight,
                o.evidence.as_deref().unwrap_or("-"),
            ));
        }
        out.push('\n');
    }

    out
}
