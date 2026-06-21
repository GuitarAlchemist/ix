//! Evidence provenance — hash the *inputs*, not just the verdict line, so a later audit
//! can detect a swapped input (input provenance > verdict provenance). The metric source
//! is content-hashed via [`super::hash`]; the guardrail carries its baseline ref; and an
//! iteration scope contributes its verified `git:<sha>` as provenance.

use super::hash;
use super::{Evidence, IterationScope, MaintainInputs};
use crate::chatbot::GateReport;

/// Build the evidence vec for one verdict: metric (content-hashed `hits.jsonl`),
/// guardrail-baseline (the gate's baseline ref), and — when scoped — the iteration commit.
pub(crate) fn build_evidence(inputs: &MaintainInputs, report: &GateReport) -> Vec<Evidence> {
    let mut evidence = vec![
        Evidence {
            kind: "metric".into(),
            source: inputs.hits_path.to_string_lossy().into_owned(),
            hash: hash::fnv1a64_file(inputs.hits_path).unwrap_or_else(|| "absent".into()),
        },
        Evidence {
            kind: "guardrail-baseline".into(),
            source: inputs.corpus_dir.to_string_lossy().into_owned(),
            hash: report.baseline_ref.clone(),
        },
    ];
    // The correlation key IS provenance — it ties this verdict to a verified commit.
    if let Some(IterationScope { loop_id, commit_sha, repo_dir }) = &inputs.iteration {
        evidence.push(Evidence {
            kind: format!("iteration-commit:{loop_id}"),
            source: repo_dir.to_string_lossy().into_owned(),
            hash: format!("git:{commit_sha}"),
        });
    }
    evidence
}
