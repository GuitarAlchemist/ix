//! Regenerate `examples/canonical-showcase/<demo>/pipeline.json` from the
//! hand-chained `DemoScenario` definitions.
//!
//! This is the R1 migration helper: it walks the existing `ChaosDetective`,
//! `GovernanceGauntlet`, and `SprintOracle` scenarios, serialises each
//! `StepInput::Static` payload into the pipeline.json schema consumed by
//! `ix_pipeline_run`, and writes the result under
//! `examples/canonical-showcase/`. Because the input data is produced by
//! the scenario's own generator functions, the resulting JSON is
//! guaranteed to be bit-identical to what the hand-chained runner produces
//! for the default seed (42).
//!
//! The test is `#[ignore]`-gated so it only runs when explicitly requested:
//!
//! ```bash
//! cargo test -p ix-agent --test dump_showcase_pipelines -- --ignored --nocapture
//! ```

use ix_agent::demo::{find_scenario, DemoScenario, StepInput};
use serde_json::{json, Value};
use std::path::PathBuf;

const SEED: u64 = 42;
const VERBOSITY: u8 = 1;

struct DumpSpec {
    scenario_id: &'static str,
    folder: &'static str,
    name: &'static str,
    description: &'static str,
    asset_prefix: &'static str,
}

const DUMPS: &[DumpSpec] = &[
    DumpSpec {
        scenario_id: "chaos-detective",
        folder: "02-chaos-detective",
        name: "chaos-detective",
        description: "A logistic-map signal that looks like noise turns out to be deterministic chaos. \
                      stats → FFT → Lyapunov → persistent homology. R1 migration of the chaos_detective \
                      DemoScenario, seed=42.",
        asset_prefix: "chaos_detective",
    },
    DumpSpec {
        scenario_id: "governance-gauntlet",
        folder: "03-governance-gauntlet",
        name: "governance-gauntlet",
        description: "Multi-agent action audit gauntlet: Demerzel persona → policy → belief → \
                      ix_governance_check verdicts. R1 migration of the governance_gauntlet \
                      DemoScenario.",
        asset_prefix: "governance_gauntlet",
    },
    DumpSpec {
        scenario_id: "sprint-oracle",
        folder: "04-sprint-oracle",
        name: "sprint-oracle",
        description: "Sprint velocity history → time-series baseline → ML forecast → Bayesian \
                      confidence band. R1 migration of the sprint_oracle DemoScenario, seed=42.",
        asset_prefix: "sprint_oracle",
    },
];

fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p
}

fn slug(label: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('_');
            last_dash = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

fn dump_one(spec: &DumpSpec) {
    let scenario: Box<dyn DemoScenario> = find_scenario(spec.scenario_id)
        .unwrap_or_else(|| panic!("scenario '{}' not registered", spec.scenario_id));
    let steps = scenario.steps(SEED, VERBOSITY);
    assert!(
        !steps.is_empty(),
        "scenario '{}' has no steps",
        spec.scenario_id
    );

    let mut json_steps: Vec<Value> = Vec::with_capacity(steps.len());
    let mut prev_id: Option<String> = None;
    let mut tools_used: Vec<String> = Vec::with_capacity(steps.len());

    for (idx, step) in steps.iter().enumerate() {
        let arguments = match &step.input {
            StepInput::Static(v) => v.clone(),
            StepInput::Glue(_) => panic!(
                "scenario '{}' step {idx} uses StepInput::Glue — not yet supported by the dumper",
                spec.scenario_id
            ),
        };
        let id = format!("s{:02}_{}", idx + 1, slug(&step.label));
        let asset_name = format!("{}.{}", spec.asset_prefix, slug(&step.label));
        let mut step_json = json!({
            "id": id,
            "tool": step.tool,
            "asset_name": asset_name,
            "arguments": arguments,
        });
        if let Some(prev) = &prev_id {
            step_json["depends_on"] = json!([prev]);
        }
        tools_used.push(step.tool.clone());
        prev_id = Some(id);
        json_steps.push(step_json);
    }

    let spec_json = json!({
        "$schema": "https://ix.guitaralchemist.com/schemas/pipeline-v1.json",
        "name": spec.name,
        "description": spec.description,
        "version": "1.0",
        "seed": SEED,
        "tools_used": tools_used,
        "steps": json_steps,
    });

    let mut out_dir = workspace_root();
    out_dir.push("examples");
    out_dir.push("canonical-showcase");
    out_dir.push(spec.folder);
    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("mkdir {}: {e}", out_dir.display()));
    let mut out_path = out_dir.clone();
    out_path.push("pipeline.json");

    let pretty = serde_json::to_string_pretty(&spec_json).expect("serialise pipeline spec");
    std::fs::write(&out_path, pretty + "\n")
        .unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));

    eprintln!(
        "[dump_showcase_pipelines] wrote {} ({} steps)",
        out_path.display(),
        steps.len()
    );
}

#[test]
#[ignore = "writes files — run with --ignored to regenerate canonical pipeline specs"]
fn dump_showcase_pipelines() {
    for spec in DUMPS {
        dump_one(spec);
    }
}
