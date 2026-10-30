//! L1 over tracer A (S6 of the polyglot meta-pipelines plan): it rewrites the node that fails,
//! the chatbot request, and is judged on questions it never saw.
//!
//!   cargo run -p ix-stream --features world --example l1     (needs GaApi :5232, the chatbot
//!                                                             :5252 and the duckdb CLI)
//!   cargo run -p ix-stream --features world --example l1 -- table   (a finished run's table
//!                                                             from its receipts; no call)
//!
//! - **What it may change.** Only the chatbot's message, one of the fixed templates of plan.rs;
//!   no model proposes them. The extract, judge, report and GaApi nodes of
//!   examples/stream/music-theory-oracle/ix.yaml are hashed at the start; the spec is read again
//!   before each run, and a run whose frozen nodes hash otherwise is refused.
//! - **Train.** Each template, twice, on the 10 questions of the spec, the runs interleaved
//!   (every template's first run, then every second run). Score: T minus F over both runs. A
//!   template with fewer claims checked per question, or more U, than the baseline cannot win;
//!   a template wins only by scoring above the baseline. Beside the score: the precision of the
//!   claims (verified over checked), which T minus F per question does not see when a template
//!   makes answers more checkable; it is reported and does not choose.
//! - **Held out.** examples/stream/music-theory-oracle/held-out.json, read only once the choice
//!   is made: the baseline and the chosen template, twice each. Its questions continue the
//!   spec's pick: the labels transpose, interval, diatonicchords, commontones, chordinfo taken
//!   round-robin, each time the first unused fresh-test prompt by numeric id whose answer
//!   GaApi's triads determine and the frozen judge has a kind for (all 7 chords of a key, a
//!   transposition, the notes of a triad, the notes two triads share, an interval between two
//!   named notes). Only 7 qualify: f-37, f-40, f-116, f-93, f-96, f-48, f-107.
//! - **Budget.** 150 chatbot calls, one per question per run, counted by ix-loop's `Budget`;
//!   one ix-loop `Ledger` line per run. The loop stops when a run would overspend.
//!
//! L1 proposes; it changes no spec. The receipts go to state/stream-tracer/2026-10-18-l1/.

mod plan;

use ix_loop::{Budget, Grant, Ledger};
use ix_pipeline::spec::PipelineSpec;
use ix_stream::skill::stream_run;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const CALLS: i64 = 150;
const RUNS: u32 = 2;

struct L1 {
    spec: PathBuf,
    out: PathBuf,
    frozen: String,
    budget: Budget,
    ledger: Ledger,
}

impl L1 {
    fn graph(&self) -> Result<Value, String> {
        let spec = PipelineSpec::from_file(&self.spec).map_err(|e| e.to_string())?;
        Ok(spec.stages["tracer"].args["graph"].clone())
    }

    /// One run of `template` over `values`, its receipts under `<split>/<template>-r<run>/`.
    fn run(
        &self,
        split: &str,
        name: &str,
        run: u32,
        values: &[Value],
    ) -> Result<(plan::Run, Value), String> {
        let n = values.len() as i64;
        if self.budget.calls() < n {
            return Err(format!(
                "budget: {} chatbot calls left, a run needs {n}",
                self.budget.calls()
            ));
        }
        let template = plan::TEMPLATES
            .iter()
            .find(|t| t.0 == name)
            .ok_or("no such template")?
            .1;
        let graph = plan::candidate(&self.graph()?, &self.frozen, template, values)?;
        for _ in 0..n {
            self.budget.take_call()?;
        }
        let dir = self.out.join(split).join(format!("{name}-r{run}"));
        let out = stream_run(json!({
            "graph": graph, "max_events": n, "budget": 5000, "out_dir": dir.display().to_string()
        }))?;
        for scratch in ["extract", "judge", "report"] {
            let _ = std::fs::remove_file(dir.join(format!("{scratch}.batch.jsonl")));
        }
        let (r, report) = tally(name, run, &dir)?;
        let verdicts: Vec<String> = rows(&dir.join("judge.jsonl"))?
            .iter()
            .filter(|r| r["row"] == "question")
            .map(|r| {
                format!(
                    "{}:{}{}/{}",
                    r["id"].as_str().unwrap_or("?"),
                    r["verdict"].as_str().unwrap_or("?"),
                    r["verified"],
                    r["claims"]
                )
            })
            .collect();
        let entry = json!({
            "split": split, "template": name, "run": run, "T": r.t, "F": r.f, "U": r.u,
            "claims_checked": r.claims, "claims_verified": r.verified, "questions": r.questions,
            "verdicts": verdicts,
            "chatbot_failures": report["chatbot_failures"], "chatbot_ms_median": report["chatbot_ms_median"],
            "faults": out["receipts"]["faults"], "calls_used": self.budget.calls_used(),
        });
        self.ledger.record(entry.clone());
        println!("{}  calls {}", line(split, &r), self.budget.calls_used());
        Ok((r, entry))
    }
}

/// The counts of the report a run left in `dir`.
fn tally(name: &str, run: u32, dir: &Path) -> Result<(plan::Run, Value), String> {
    let report = rows(&dir.join("report.jsonl"))?
        .into_iter()
        .next()
        .ok_or("no report")?;
    let count = |k: &str| report[k].as_i64().unwrap_or(0);
    let r = plan::Run {
        template: name.into(),
        run,
        questions: count("questions"),
        t: count("T"),
        f: count("F"),
        u: count("U"),
        claims: count("claims_checked"),
        verified: count("claims_verified"),
    };
    Ok((r, report))
}

fn line(split: &str, r: &plan::Run) -> String {
    format!(
        "{split:8} {:16} r{}  T{} F{} U{}  T-F {:2}  claims {:2} ({:.1}/q)  verified {:2}  precision {:.3}",
        r.template,
        r.run,
        r.t,
        r.f,
        r.u,
        r.t - r.f,
        r.claims,
        r.claims as f64 / r.questions.max(1) as f64,
        r.verified,
        plan::precision(r.verified, r.claims)
    )
}

fn scores(scores: &[plan::Score]) {
    for s in scores {
        println!(
            "score {:16} {:3}  claims/q {:.2}  U {}  precision {:.3}  eligible {}",
            s.template, s.score, s.claims_per_question, s.u, s.precision, s.eligible
        );
    }
}

/// `l1 -- table`: the table of a finished run, read again from the reports it stored, in the
/// order of its l1.json; no call is made.
fn table(out: &Path) -> Result<(), String> {
    let path = out.join("l1.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let summary: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut train = Vec::new();
    for key in ["train", "held_out"] {
        for e in summary[key].as_array().ok_or("l1.json has no runs")? {
            let (split, name) = (e["split"].as_str().unwrap_or(""), e["template"].as_str());
            let run = e["run"].as_u64().ok_or("a run has no number")? as u32;
            let name = name.ok_or("a run has no template")?;
            let (r, _) = tally(name, run, &out.join(split).join(format!("{name}-r{run}")))?;
            println!("{}", line(split, &r));
            if key == "train" {
                train.push(r);
            }
        }
    }
    let (s, chosen) = plan::choose(&train);
    scores(&s);
    println!("chosen: {chosen} (l1.json: {})", summary["chosen"]);
    Ok(())
}

fn rows(path: &Path) -> Result<Vec<Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .map(|l| serde_json::from_str(l).map_err(|e| e.to_string()))
        .collect()
}

fn main() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = root.join("state/stream-tracer/2026-10-18-l1");
    if std::env::args().nth(1).as_deref() == Some("table") {
        return table(&out);
    }
    if out.exists() {
        return Err(format!(
            "{} exists: a run's receipts are not mixed with another's",
            out.display()
        ));
    }
    let spec = root.join("examples/stream/music-theory-oracle/ix.yaml");
    let mut l1 = L1 {
        spec,
        out: out.clone(),
        frozen: String::new(),
        budget: Budget::root(&Grant {
            calls: CALLS,
            fuel: i64::MAX,
            seconds: f64::INFINITY,
        }),
        ledger: Ledger::at(out.join("ledger.jsonl")),
    };
    let graph = l1.graph()?;
    l1.frozen = plan::frozen(&graph)?;
    let train = plan::node(&graph, "questions")
        .and_then(|q| q["values"].as_array())
        .ok_or("no questions")?
        .clone();
    println!("frozen nodes {:?}: {}", plan::FROZEN, l1.frozen);

    let mut runs = Vec::new();
    let mut table = Vec::new();
    let mut stopped = None;
    'train: for run in 1..=RUNS {
        for (name, _) in plan::TEMPLATES {
            match l1.run("train", name, run, &train) {
                Ok((r, entry)) => {
                    runs.push(r);
                    table.push(entry);
                }
                Err(e) => {
                    stopped = Some(e);
                    break 'train;
                }
            }
        }
    }
    let (s, chosen) = plan::choose(&runs);
    scores(&s);
    println!("chosen: {chosen}");

    let mut held = Vec::new();
    if stopped.is_none() {
        let path = root.join("examples/stream/music-theory-oracle/held-out.json");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let values: Vec<Value> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let base = plan::TEMPLATES[0].0;
        let names = if chosen == base {
            vec![base]
        } else {
            vec![base, chosen.as_str()]
        };
        'held: for run in 1..=RUNS {
            for name in &names {
                match l1.run("held-out", name, run, &values) {
                    Ok((_, entry)) => held.push(entry),
                    Err(e) => {
                        stopped = Some(e);
                        break 'held;
                    }
                }
            }
        }
    }
    let summary = json!({
        "frozen": { "nodes": plan::FROZEN, "blake3": l1.frozen },
        "templates": plan::TEMPLATES.iter().map(|(n, t)| json!({ "name": n, "message": t })).collect::<Vec<_>>(),
        "calls": { "cap": CALLS, "used": l1.budget.calls_used() },
        "runs_per_template": RUNS,
        "train": table,
        "scores": s.iter().map(|s| json!({
            "template": s.template, "runs": s.runs, "score_T_minus_F": s.score,
            "claims_per_question": s.claims_per_question, "U": s.u, "precision": s.precision,
            "eligible": s.eligible
        })).collect::<Vec<_>>(),
        "chosen": chosen,
        "held_out": held,
        "stopped": stopped,
        "proposal": "the chosen template for the chatbot node's message; a person accepts it or not",
    });
    std::fs::write(
        out.join("l1.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("calls used: {} of {CALLS}", l1.budget.calls_used());
    match stopped {
        Some(e) => Err(format!("stopped: {e}")),
        None => Ok(()),
    }
}
