//! `ix check <noun>` — validation + governance + environment diagnostics.

use crate::exit;
use crate::output::{self, Format};
use serde_json::{json, Value};

/// Environment self-diagnosis: rust toolchain, governance submodule, registry
/// health, key directories. Returns OK_TRUE on full green, PROBABLE on
/// non-fatal warnings, DOUBTFUL on missing optional pieces, FALSE on broken.
pub fn doctor(format: Format) -> Result<i32, String> {
    let mut checks: Vec<Value> = Vec::new();
    let mut any_fail = false;
    let mut any_warn = false;

    // Registry skill count
    let skill_count = ix_registry::count();
    let registry_ok = skill_count >= 34; // batch1(6) + batch2(28)
    if !registry_ok {
        any_fail = true;
    }
    checks.push(json!({
        "check": "capability-registry",
        "status": if registry_ok { "ok" } else { "fail" },
        "skills": skill_count,
        "expected_minimum": 34,
    }));

    // Governance submodule presence
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let gov_ok = std::path::Path::new(&gov_dir).is_dir();
    if !gov_ok {
        any_warn = true;
    }
    checks.push(json!({
        "check": "demerzel-governance",
        "status": if gov_ok { "ok" } else { "warn" },
        "path": gov_dir,
    }));

    // Constitution
    let constitution_path = format!("{gov_dir}/constitutions/default.constitution.md");
    let const_ok = std::path::Path::new(&constitution_path).is_file();
    if gov_ok && !const_ok {
        any_warn = true;
    }
    checks.push(json!({
        "check": "default-constitution",
        "status": if const_ok { "ok" } else if gov_ok { "warn" } else { "skip" },
        "path": constitution_path,
    }));

    // State directory (optional)
    let state_ok = std::path::Path::new("state").is_dir();
    checks.push(json!({
        "check": "state-directory",
        "status": if state_ok { "ok" } else { "absent" },
        "path": "state/",
    }));

    let verdict = if any_fail {
        "F"
    } else if any_warn {
        "P"
    } else {
        "T"
    };
    let exit_code = match verdict {
        "T" => exit::OK_TRUE,
        "P" => exit::PROBABLE,
        "F" => exit::FALSE,
        _ => exit::UNKNOWN,
    };

    let payload = json!({
        "verdict": verdict,
        "exit_code": exit_code,
        "checks": checks,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(exit_code)
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
