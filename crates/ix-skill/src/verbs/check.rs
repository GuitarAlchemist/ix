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
/// hexavalent-friendly exit code based on compliance.
pub fn action(action_text: &str, _context: Option<&str>, format: Format) -> Result<i32, String> {
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let const_path = format!("{gov_dir}/constitutions/default.constitution.md");

    let constitution = ix_governance::Constitution::load(std::path::Path::new(&const_path))
        .map_err(|e| format!("loading {const_path}: {e}"))?;

    // Simple substring/keyword semantic scan over article texts.
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

    // Heuristic verdict: no relevant articles hit → T (no constraint fired).
    // Relevant hits → P (probable compliance, review). Keywords like
    // "delete", "drop", "rm -rf", "force-push" → D (doubtful).
    let danger_words = [
        "delete",
        "drop table",
        "rm -rf",
        "force push",
        "--force",
        "truncate",
    ];
    let dangerous = danger_words.iter().any(|w| action_lower.contains(w));

    let verdict = if dangerous {
        "D"
    } else if !relevant.is_empty() {
        "P"
    } else {
        "T"
    };
    let exit_code = match verdict {
        "T" => exit::OK_TRUE,
        "P" => exit::PROBABLE,
        "D" => exit::DOUBTFUL,
        _ => exit::UNKNOWN,
    };

    let payload = json!({
        "verdict": verdict,
        "exit_code": exit_code,
        "action": action_text,
        "relevant_articles": relevant
            .iter()
            .map(|(n, name)| json!({ "number": n, "name": name }))
            .collect::<Vec<_>>(),
        "dangerous_keywords_matched": dangerous,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(exit_code)
}
