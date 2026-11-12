//! `ix check <noun>` — validation + governance + environment diagnostics.

use crate::exit;
use crate::output::{self, Format};
use serde_json::json;

/// `ix check doctor` — retained alias for [`ix doctor`](crate::doctor).
///
/// The environment self-diagnosis that used to live here asserted
/// `skill_count >= 34`, one more hand-typed inventory number of exactly the
/// kind ix#185 set out to remove. The real checks now live in
/// [`crate::doctor`], which compares the live registry against a committed
/// snapshot instead. This wrapper keeps the old invocation working.
pub fn doctor(format: Format) -> Result<i32, String> {
    crate::doctor::main(format, crate::doctor::Options::default())
}

/// Check a proposed action against the Demerzel constitution. Returns a
/// hexavalent exit code: D (3) when a rule fires, P (1) for relevant articles
/// only, U (2) when nothing matched.
pub fn action(action_text: &str, _context: Option<&str>, format: Format) -> Result<i32, String> {
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let const_path = format!("{gov_dir}/constitutions/default.constitution.md");

    let constitution = ix_governance::Constitution::load(std::path::Path::new(&const_path))
        .map_err(|e| format!("loading {const_path}: {e}"))?;

    // Simple substring/keyword semantic scan over article names, plus the
    // constitution's own rules (the same ones `ix_governance_check` uses).
    let action_lower = action_text.to_lowercase();
    let mut relevant: Vec<(u8, String)> = Vec::new();
    for art in &constitution.articles {
        // Relevance: any keyword from the article name appears in the action.
        for word in art.name.to_lowercase().split_whitespace() {
            if word.len() > 3 && action_lower.contains(word) {
                relevant.push((art.number, art.name.clone()));
                break;
            }
        }
    }
    let rules = constitution.check_action(action_text);
    for art in &rules.relevant_articles {
        if !relevant.iter().any(|(n, _)| *n == art.number) {
            relevant.push((art.number, art.name.clone()));
        }
    }
    relevant.sort_by_key(|(n, _)| *n);

    // Heuristic verdict: a dangerous keyword or a constitution rule firing
    // → D (doubtful). Relevant articles only → P (probable, review).
    // Nothing matched → U: keyword matching can't tell a benign action from
    // one the rules don't cover, so no match is not approval.
    let danger_words = [
        "delete",
        "drop table",
        "rm -rf",
        "force push",
        "--force",
        "truncate",
    ];
    let dangerous = danger_words.iter().any(|w| action_lower.contains(w));

    let verdict = if dangerous || !rules.compliant {
        "D"
    } else if !relevant.is_empty() {
        "P"
    } else {
        "U"
    };
    let exit_code = match verdict {
        "P" => exit::PROBABLE,
        "D" => exit::DOUBTFUL,
        _ => exit::UNKNOWN,
    };

    let mut payload = json!({
        "verdict": verdict,
        "exit_code": exit_code,
        "basis": "keyword-heuristic",
        "action": action_text,
        "relevant_articles": relevant
            .iter()
            .map(|(n, name)| json!({ "number": n, "name": name }))
            .collect::<Vec<_>>(),
        "dangerous_keywords_matched": dangerous,
        "warnings": rules.warnings,
    });
    if verdict == "U" {
        payload["note"] = json!(
            "No rule matched. This check only recognizes English keywords, so no match is not \
             evidence of compliance: read the constitution for actions it does not cover."
        );
    }
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(exit_code)
}
