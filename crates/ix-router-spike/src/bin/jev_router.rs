//! Jev arm for the Spike A held-out routing comparison — Stage 0 (offline).
//!
//! Asks TypeSafe Jev one Choice question per held-out prompt ("which skill
//! should handle this message?", 16 intents + `__none__`) and scores the
//! answers against the SAME 126-prompt TEST set on which the production router
//! and the learned head were already measured (see state/router-spike/RESULTS.md).
//!
//! This binary never talks to the network and never reads a credential:
//!   plan  — build one request per prompt, write requests.jsonl + plan.json
//!           (request SHA-256, UTF-8 byte totals, input-cost PROXY).
//!   mock  — score a deterministic synthetic response set (labelled MOCK) to
//!           exercise validation + scoring end to end.
//!   score — score a receipt file of real responses produced elsewhere, after
//!           checking every request digest against the current plan.
//!
//! Routing policy lives in code: an invalid or missing response is a scored
//! failure (never a silent decline, never zero cost); `__none__` is the only
//! pre-registered decline. Confidence thresholds are reported as exploratory
//! only — tuning one on TEST would be teaching-to-test.
//!
//! Build: cargo run --manifest-path crates/ix-router-spike/Cargo.toml --bin jev-router -- plan

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const MODEL: &str = "jev-1.13.0";
const NONE: &str = "__none__";
const QUESTION_ID: &str = "intent";
/// Reviewed rate card (docs.typesafe.ai/models, 2026-09-22): USD per 1M input tokens.
const INPUT_PRICE_PER_MILLION_USD: f64 = 0.042;
/// Exploratory confidence cut-offs — reported, never used for the verdict.
const EXPLORATORY_TAUS: [f64; 5] = [0.0, 0.3, 0.5, 0.7, 0.9];
/// Probabilities may be rounded by the provider; 17 values rounded to 4 dp can
/// drift ~1e-4 from 1. A format quirk must not be scored as a routing failure.
const SUM_TOL: f64 = 1e-3;

/// One pre-registered arm. `corpus_sha256` is the SHA-256 of the canonical JSON
/// of [(id, prompt, expectedIntentId)] (line-ending independent); the verdict
/// bands were registered against exactly that corpus and those counts.
struct Arm {
    name: &'static str,
    corpus: &'static str,
    reversed_options: bool,
    corpus_sha256: &'static str,
    inscope: usize,
    oos: usize,
    /// Robustness bands (arms other than `base`): (in-scope, OOS) for ROBUST,
    /// in-scope floor for DEGRADED; below it FAIL.
    robust: (usize, usize),
    degraded_floor: usize,
}

const BASE_SHA: &str = "692b1c9d1460e72074dafaad0223d81c2aa5a50fb58b1ea5a91dbd4d4ef2b588";

/// Stage 2 arms (RESULTS.md, "Jev arm — Stage 2 robustness pre-registration").
const ARMS: [Arm; 5] = [
    Arm {
        name: "base",
        corpus: "state/router-spike/heldout-test.json",
        reversed_options: false,
        corpus_sha256: BASE_SHA,
        inscope: 110,
        oos: 16,
        robust: (0, 0),
        degraded_floor: 0,
    },
    Arm {
        name: "reversed",
        corpus: "state/router-spike/heldout-test.json",
        reversed_options: true,
        corpus_sha256: BASE_SHA,
        inscope: 110,
        oos: 16,
        robust: (102, 14),
        degraded_floor: 90,
    },
    Arm {
        name: "fr",
        corpus: "state/router-spike/jev/corpora/heldout-fr.json",
        reversed_options: false,
        corpus_sha256: "d7e9651d97c37ce624ab1cf2fc531a9659e6b90ab501434fb4573a3eab1798e8",
        inscope: 110,
        oos: 16,
        robust: (99, 13),
        degraded_floor: 90,
    },
    Arm {
        name: "es",
        corpus: "state/router-spike/jev/corpora/heldout-es.json",
        reversed_options: false,
        corpus_sha256: "c509430babf0887e62e1ec2ba6ba16db03d1f1d6f1dc36888b7ea6af0056f564",
        inscope: 110,
        oos: 16,
        robust: (99, 13),
        degraded_floor: 90,
    },
    Arm {
        name: "fresh",
        corpus: "state/router-spike/jev/corpora/fresh-test.json",
        reversed_options: false,
        corpus_sha256: "e9e8c0e1b51e67abb2543ba77c6992b72bf4c98d18d71d3af5ac7ab777d07385",
        inscope: 112,
        oos: 16,
        robust: (101, 13),
        degraded_floor: 92,
    },
];

fn arm(name: &str) -> &'static Arm {
    ARMS.iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("unknown arm {name}"))
}

/// Stage 2 verdict for a non-base arm.
fn robustness_verdict(a: &Arm, m: &Metrics, wrong_model: bool) -> &'static str {
    if wrong_model || m.invalid > 2 {
        "KILL"
    } else if m.inscope_correct >= a.robust.0 && m.oos_declined >= a.robust.1 {
        "ROBUST"
    } else if m.inscope_correct >= a.degraded_floor {
        "DEGRADED"
    } else {
        "FAIL"
    }
}

#[derive(Deserialize)]
struct HeldOut {
    version: String,
    prompts: Vec<Prompt>,
}

#[derive(Deserialize, Clone)]
struct Prompt {
    id: String,
    prompt: String,
    #[serde(rename = "expectedIntentId")]
    expected: String,
}

#[derive(Deserialize)]
struct Options {
    version: String,
    instructions: String,
    criteria: BTreeMap<String, String>,
}

/// One validated Choice answer.
#[derive(Debug, Clone, PartialEq)]
struct Answer {
    choice: String,
    confidence: f64,
    probabilities: BTreeMap<String, f64>,
    input_tokens: u64,
    output_tokens: u64,
}

/// Why a response was rejected. Every variant is scored as a failure.
#[derive(Debug, PartialEq)]
enum Invalid {
    Missing,
    DigestMismatch,
    WrongModel(String),
    Contract(String),
}

/// Request bytes. Canonical (sorted) unless `reversed`: then the criteria
/// object is re-emitted in reverse key order — the only byte change — so that
/// `__none__` is presented last instead of first.
fn body(prompt: &Prompt, opts: &Options, reversed: bool) -> Vec<u8> {
    let canonical = encode(&request(prompt, opts));
    if !reversed {
        return canonical;
    }
    let sorted = String::from_utf8(encode(&json!(opts.criteria))).unwrap();
    let pairs: Vec<String> = opts
        .criteria
        .iter()
        .rev()
        .map(|(k, v)| format!("{}:{}", json!(k), json!(v)))
        .collect();
    let text = String::from_utf8(canonical).unwrap();
    assert_eq!(
        text.matches(&sorted).count(),
        1,
        "criteria must appear once"
    );
    text.replace(&sorted, &format!("{{{}}}", pairs.join(",")))
        .into_bytes()
}

fn request(prompt: &Prompt, opts: &Options) -> Value {
    json!({
        "model": MODEL,
        "state": { "user_message": prompt.prompt },
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": opts.instructions,
                "criteria": opts.criteria,
            }
        }
    })
}

/// Canonical bytes: compact JSON, keys sorted (serde_json's default map is a
/// BTreeMap), UTF-8 kept raw — same encoding as learn's typesafe_lab.py.
fn encode(v: &Value) -> Vec<u8> {
    serde_json::to_vec(v).expect("request is serializable")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn num(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite())
}

/// Fail-closed validation of one response against the options it was asked.
fn validate(resp: &Value, options: &BTreeSet<String>) -> Result<Answer, Invalid> {
    let c = |m: &str| Invalid::Contract(m.to_string());
    let model = resp
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| c("model missing"))?;
    if model != MODEL {
        return Err(Invalid::WrongModel(model.to_string()));
    }
    let answers = resp
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| c("answers missing"))?;
    if answers.len() != 1 || !answers.contains_key(QUESTION_ID) {
        return Err(c("answers must contain exactly the intent question"));
    }
    let a = &answers[QUESTION_ID];
    if a.get("type").and_then(Value::as_str) != Some("choice") {
        return Err(c("intent is not a choice answer"));
    }
    let choice = a
        .get("choice")
        .and_then(Value::as_str)
        .ok_or_else(|| c("choice missing"))?;
    if !options.contains(choice) {
        return Err(c("choice is not one of the requested options"));
    }
    let probs = a
        .get("probabilities")
        .and_then(Value::as_object)
        .ok_or_else(|| c("probabilities missing"))?;
    let keys: BTreeSet<String> = probs.keys().cloned().collect();
    if &keys != options {
        return Err(c(
            "probability keys must exactly match the requested options",
        ));
    }
    let mut probabilities = BTreeMap::new();
    for (k, v) in probs {
        let p = num(v)
            .filter(|p| (0.0..=1.0).contains(p))
            .ok_or_else(|| c("probability outside [0,1]"))?;
        probabilities.insert(k.clone(), p);
    }
    if (probabilities.values().sum::<f64>() - 1.0).abs() > SUM_TOL {
        return Err(c("probabilities must sum to 1"));
    }
    let max = probabilities.values().cloned().fold(f64::MIN, f64::max);
    if probabilities[choice] != max {
        return Err(c("choice must carry the maximum probability"));
    }
    let confidence = a
        .get("confidence")
        .and_then(num)
        .filter(|x| (0.0..=1.0).contains(x))
        .ok_or_else(|| c("confidence missing or outside [0,1]"))?;
    // Missing usage is a contract failure, never an implicit zero cost.
    let usage = resp
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| c("usage missing"))?;
    let tok = |k: &str| {
        usage
            .get(k)
            .and_then(|v| {
                v.as_u64().or_else(|| {
                    v.as_f64()
                        .filter(|x| *x >= 0.0 && x.fract() == 0.0)
                        .map(|x| x as u64)
                })
            })
            .ok_or_else(|| c("usage token count missing"))
    };
    Ok(Answer {
        choice: choice.to_string(),
        confidence,
        probabilities,
        input_tokens: tok("input_tokens")?,
        output_tokens: tok("output_tokens")?,
    })
}

/// Pre-registered routing: the answer's choice, with `__none__` as the only
/// decline. `tau` > 0 is exploratory: below it, decline.
fn route(a: &Answer, tau: f64) -> String {
    if a.confidence < tau {
        NONE.to_string()
    } else {
        a.choice.clone()
    }
}

#[derive(Debug, Default, PartialEq)]
struct Metrics {
    inscope_total: usize,
    inscope_correct: usize,
    oos_total: usize,
    oos_declined: usize,
    invalid: usize,
    macro_f1: f64,
    min_f1: (String, f64),
    per_intent_f1: BTreeMap<String, f64>,
}

impl Metrics {
    fn inscope_acc(&self) -> f64 {
        ratio(self.inscope_correct, self.inscope_total)
    }
    fn oos_decline(&self) -> f64 {
        ratio(self.oos_declined, self.oos_total)
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

/// Scores routed predictions. `None` prediction = invalid response: wrong for
/// an in-scope prompt, NOT a correct decline for an OOS prompt.
fn score(gold: &[&str], pred: &[Option<String>], intents: &[String]) -> Metrics {
    let mut m = Metrics::default();
    let mut tp: BTreeMap<&str, usize> = BTreeMap::new();
    let mut fp: BTreeMap<&str, usize> = BTreeMap::new();
    let mut fne: BTreeMap<&str, usize> = BTreeMap::new();
    for (g, p) in gold.iter().zip(pred) {
        let p = p.as_deref();
        if p.is_none() {
            m.invalid += 1;
        }
        if *g == NONE {
            m.oos_total += 1;
            if p == Some(NONE) {
                m.oos_declined += 1;
            }
        } else {
            m.inscope_total += 1;
            if p == Some(*g) {
                m.inscope_correct += 1;
                *tp.entry(g).or_default() += 1;
            } else {
                *fne.entry(g).or_default() += 1;
            }
        }
        if let Some(p) = p {
            if p != NONE && p != *g {
                *fp.entry(p).or_default() += 1;
            }
        }
    }
    let mut min = (String::new(), f64::MAX);
    for i in intents {
        let (t, f, n) = (tp.get(i.as_str()), fp.get(i.as_str()), fne.get(i.as_str()));
        let (t, f, n) = (
            *t.unwrap_or(&0) as f64,
            *f.unwrap_or(&0) as f64,
            *n.unwrap_or(&0) as f64,
        );
        let f1 = if t == 0.0 {
            0.0
        } else {
            2.0 * t / (2.0 * t + f + n)
        };
        if f1 < min.1 {
            min = (i.clone(), f1);
        }
        m.per_intent_f1.insert(i.clone(), f1);
    }
    m.macro_f1 = m.per_intent_f1.values().sum::<f64>() / intents.len().max(1) as f64;
    m.min_f1 = min;
    m
}

/// Multiclass Brier score over valid answers (lower is better).
fn brier(gold: &[&str], answers: &[Option<Answer>]) -> Option<f64> {
    let scored: Vec<f64> = gold
        .iter()
        .zip(answers)
        .filter_map(|(g, a)| a.as_ref().map(|a| (g, a)))
        .map(|(g, a)| {
            a.probabilities
                .iter()
                .map(|(k, p)| (p - if k == g { 1.0 } else { 0.0 }).powi(2))
                .sum()
        })
        .collect();
    (!scored.is_empty()).then(|| scored.iter().sum::<f64>() / scored.len() as f64)
}

fn read_json<T: for<'de> Deserialize<'de>>(p: &Path) -> T {
    let s = fs::read_to_string(p).unwrap_or_else(|e| panic!("read {p:?}: {e}"));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("parse {p:?}: {e}"))
}

struct Setup {
    arm: &'static Arm,
    heldout: HeldOut,
    options: Options,
    option_set: BTreeSet<String>,
    intents: Vec<String>,
    digests: Vec<String>,
    bodies: Vec<Vec<u8>>,
}

fn setup(root: &Path, arm: &'static Arm) -> Setup {
    let heldout: HeldOut = read_json(&root.join(arm.corpus));
    let options: Options = read_json(&root.join("state/router-spike/jev/options.json"));
    let option_set: BTreeSet<String> = options.criteria.keys().cloned().collect();
    let gold_labels: BTreeSet<String> =
        heldout.prompts.iter().map(|p| p.expected.clone()).collect();
    assert_eq!(
        gold_labels, option_set,
        "options must cover exactly the TEST label set"
    );
    assert!(
        option_set.len() <= 255,
        "Choice supports at most 255 options"
    );
    let corpus: Vec<Value> = heldout
        .prompts
        .iter()
        .map(|p| json!([p.id, p.prompt, p.expected]))
        .collect();
    let corpus_sha = sha256_hex(&encode(&Value::Array(corpus)));
    let oos = heldout
        .prompts
        .iter()
        .filter(|p| p.expected == NONE)
        .count();
    assert_eq!(
        (heldout.prompts.len() - oos, oos, corpus_sha.as_str()),
        (arm.inscope, arm.oos, arm.corpus_sha256),
        "held-out corpus differs from the one the verdict bands were registered against"
    );
    let intents = option_set.iter().filter(|o| *o != NONE).cloned().collect();
    let bodies: Vec<Vec<u8>> = heldout
        .prompts
        .iter()
        .map(|p| body(p, &options, arm.reversed_options))
        .collect();
    let digests = bodies.iter().map(|b| sha256_hex(b)).collect();
    Setup {
        arm,
        heldout,
        options,
        option_set,
        intents,
        digests,
        bodies,
    }
}

/// Output directory: `jev/` for the Stage 1 base arm, `jev/<arm>/` otherwise.
fn arm_dir(root: &Path, a: &Arm) -> PathBuf {
    let base = root.join("state/router-spike/jev");
    let dir = if a.name == "base" {
        base
    } else {
        base.join(a.name)
    };
    fs::create_dir_all(&dir).expect("create arm dir");
    dir
}

fn plan(root: &Path, s: &Setup) {
    let out = arm_dir(root, s.arm);
    let mut lines = String::new();
    for (p, b) in s.heldout.prompts.iter().zip(&s.bodies) {
        lines.push_str(&format!(
            "{{\"id\":{},\"request\":{}}}\n",
            json!(p.id),
            String::from_utf8_lossy(b)
        ));
    }
    fs::write(out.join("requests.jsonl"), lines).expect("write requests.jsonl");
    let total: usize = s.bodies.iter().map(Vec::len).sum();
    let max = s.bodies.iter().map(Vec::len).max().unwrap_or(0);
    let plan = json!({
        "heldout_version": s.heldout.version,
        "arm": s.arm.name,
        "reversed_options": s.arm.reversed_options,
        "corpus_sha256": s.arm.corpus_sha256,
        "options_version": s.options.version,
        "model": MODEL,
        "calls": s.bodies.len(),
        "retries": 0,
        "request_body_utf8_bytes_total": total,
        "request_body_utf8_bytes_max": max,
        // Bytes are NOT provider-billed tokens; this is a size proxy only.
        "input_cost_proxy_usd": total as f64 / 1e6 * INPUT_PRICE_PER_MILLION_USD,
        "rate_card": format!("{INPUT_PRICE_PER_MILLION_USD} USD per 1M input tokens (docs.typesafe.ai/models, reviewed 2026-09-22; recheck before any call)"),
        "request_sha256": s.heldout.prompts.iter().zip(&s.digests).map(|(p, d)| (p.id.clone(), d.clone())).collect::<BTreeMap<_, _>>(),
    });
    fs::write(
        out.join("plan.json"),
        serde_json::to_string_pretty(&plan).unwrap() + "\n",
    )
    .expect("write plan.json");
    println!(
        "plan: {} requests, {} bytes total (max {}), cost proxy ${:.6}",
        s.bodies.len(),
        total,
        max,
        plan["input_cost_proxy_usd"].as_f64().unwrap()
    );
}

/// Receipt line: {"id", "request_sha256", "response"} — one per prompt. The
/// runner hashes the bytes it actually sent. A repeated id means a retry,
/// which the pre-registration forbids, so it aborts instead of last-wins.
fn load_receipt(path: &Path) -> BTreeMap<String, (String, Value)> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut out = BTreeMap::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(l).unwrap_or_else(|e| panic!("receipt line: {e}"));
        let id = v["id"].as_str().expect("receipt id").to_string();
        let digest = v["request_sha256"].as_str().unwrap_or("").to_string();
        let prev = out.insert(id.clone(), (digest, v["response"].clone()));
        assert!(
            prev.is_none(),
            "duplicate receipt id {id}: retries are not allowed"
        );
    }
    out
}

/// Exact two-sided McNemar p-value from the discordant counts.
fn mcnemar_exact(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    // P(X <= k), X ~ Binomial(n, 1/2), accumulated in log space.
    let ln_choose = |n: usize, k: usize| -> f64 {
        (1..=k)
            .map(|i| ((n - k + i) as f64).ln() - (i as f64).ln())
            .sum()
    };
    let tail: f64 = (0..=k)
        .map(|i| (ln_choose(n, i) - n as f64 * std::f64::consts::LN_2).exp())
        .sum();
    (2.0 * tail).min(1.0)
}

/// The pre-registered verdict (RESULTS.md, "Jev arm — pre-registration"),
/// stated in counts over 110 in-scope / 16 OOS prompts.
fn verdict(m: &Metrics, wrong_model: bool) -> &'static str {
    if wrong_model || m.invalid > 2 || m.inscope_correct < 83 || m.oos_declined < 6 {
        "KILL"
    } else if m.inscope_correct >= 87 && m.oos_declined >= 11 && m.macro_f1 >= 0.787 {
        "COMPETITIVE_WITH_HEAD"
    } else if m.inscope_correct >= 87 && m.macro_f1 >= 0.745 {
        "BEATS_PRODUCTION"
    } else {
        "INCONCLUSIVE"
    }
}

fn validate_all(
    s: &Setup,
    receipt: &BTreeMap<String, (String, Value)>,
) -> Vec<Result<Answer, Invalid>> {
    // An unplanned id is an extra call; the pre-registration fixes the call set.
    let planned: BTreeSet<&str> = s.heldout.prompts.iter().map(|p| p.id.as_str()).collect();
    if let Some(extra) = receipt.keys().find(|k| !planned.contains(k.as_str())) {
        panic!("receipt id {extra} is not in the plan: extra calls are not allowed");
    }
    s.heldout
        .prompts
        .iter()
        .zip(&s.digests)
        .map(|(p, d)| match receipt.get(&p.id) {
            None => Err(Invalid::Missing),
            Some((got, _)) if got != d => Err(Invalid::DigestMismatch),
            Some((_, resp)) => validate(resp, &s.option_set),
        })
        .collect()
}

/// Reported usage over EVERY response in the receipt, valid or not: a
/// rejected answer was still billed.
fn usage_totals(receipt: &BTreeMap<String, (String, Value)>) -> (u64, u64) {
    receipt.values().fold((0, 0), |(i, o), (_, r)| {
        let tok = |k: &str| r["usage"][k].as_f64().filter(|x| *x >= 0.0).unwrap_or(0.0) as u64;
        (i + tok("input_tokens"), o + tok("output_tokens"))
    })
}

fn report(
    root: &Path,
    s: &Setup,
    receipt: &BTreeMap<String, (String, Value)>,
    label: &str,
) -> Value {
    let results = validate_all(s, receipt);
    let results = results.as_slice();
    let gold: Vec<&str> = s
        .heldout
        .prompts
        .iter()
        .map(|p| p.expected.as_str())
        .collect();
    let answers: Vec<Option<Answer>> = results.iter().map(|r| r.as_ref().ok().cloned()).collect();
    let routed = |tau: f64| -> Vec<Option<String>> {
        answers
            .iter()
            .map(|a| a.as_ref().map(|a| route(a, tau)))
            .collect()
    };
    let primary = score(&gold, &routed(0.0), &s.intents);
    let argmax_inscope = {
        let pairs = gold.iter().zip(&answers).filter(|(g, _)| **g != NONE);
        let (n, ok) = pairs.fold((0, 0), |(n, ok), (g, a)| {
            let best = a.as_ref().and_then(|a| {
                a.probabilities
                    .iter()
                    .filter(|(k, _)| *k != NONE)
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(k, _)| k.clone())
            });
            (n + 1, ok + usize::from(best.as_deref() == Some(*g)))
        });
        ratio(ok, n)
    };
    let invalid: BTreeMap<String, String> = s
        .heldout
        .prompts
        .iter()
        .zip(results)
        .filter_map(|(p, r)| r.as_ref().err().map(|e| (p.id.clone(), format!("{e:?}"))))
        .collect();
    let (input_tokens, output_tokens) = usage_totals(receipt);
    let exploratory: Vec<Value> = EXPLORATORY_TAUS
        .iter()
        .map(|&t| {
            let m = score(&gold, &routed(t), &s.intents);
            json!({ "tau": t, "inscope_acc": m.inscope_acc(), "oos_decline": m.oos_decline(), "macro_f1": m.macro_f1 })
        })
        .collect();
    let prod: Value = read_json(&root.join("state/router-spike/production-baseline-heldout.json"));
    let head: Value = read_json(&root.join("state/router-spike/head-eval.json"));
    // Paired comparison with production ("correct" = right intent, or a decline
    // for an OOS prompt) — only where production ran on this exact corpus.
    let mcnemar = (s.arm.corpus_sha256 == BASE_SHA).then(|| {
        let prod_correct: BTreeMap<&str, bool> = prod["prompts"]
            .as_array()
            .expect("production per-prompt records")
            .iter()
            .map(|r| (r["Id"].as_str().unwrap(), r["Correct"].as_bool().unwrap()))
            .collect();
        let (mut jev_only, mut prod_only) = (0, 0);
        for ((p, g), r) in s.heldout.prompts.iter().zip(&gold).zip(&routed(0.0)) {
            let j = r.as_deref() == Some(*g);
            match (j, prod_correct[p.id.as_str()]) {
                (true, false) => jev_only += 1,
                (false, true) => prod_only += 1,
                _ => {}
            }
        }
        json!({
            "jev_right_prod_wrong": jev_only,
            "prod_right_jev_wrong": prod_only,
            "p_two_sided": mcnemar_exact(jev_only, prod_only),
            "note": "Reported alongside the verdict; the verdict does not claim significance.",
        })
    });
    let wrong_model = results
        .iter()
        .any(|r| matches!(r, Err(Invalid::WrongModel(_))));
    let predictions: BTreeMap<&str, Option<String>> = s
        .heldout
        .prompts
        .iter()
        .map(|p| p.id.as_str())
        .zip(routed(0.0))
        .collect();
    json!({
        "label": label,
        "arm": s.arm.name,
        "verdict": if s.arm.name == "base" {
            verdict(&primary, wrong_model)
        } else {
            robustness_verdict(s.arm, &primary, wrong_model)
        },
        "counts": {
            "inscope_correct": primary.inscope_correct,
            "inscope_total": primary.inscope_total,
            "oos_declined": primary.oos_declined,
            "oos_total": primary.oos_total,
        },
        "mcnemar_vs_production": mcnemar,
        "predictions": predictions,
        "model": MODEL,
        "heldout_version": s.heldout.version,
        "options_version": s.options.version,
        "prompts": gold.len(),
        "invalid_responses": invalid,
        "jev": {
            "inscope_acc": primary.inscope_acc(),
            "inscope_argmax_excluding_none": argmax_inscope,
            "oos_decline": primary.oos_decline(),
            "macro_f1": primary.macro_f1,
            "min_f1": { "intent": primary.min_f1.0, "f1": primary.min_f1.1 },
            "brier": brier(&gold, &answers),
            "per_intent_f1": primary.per_intent_f1,
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "rate_card_usd": input_tokens as f64 / 1e6 * INPUT_PRICE_PER_MILLION_USD,
            "note": "Rate-card arithmetic on reported input tokens (output tokens are free on the reviewed card), not an account charge.",
        },
        "exploratory_confidence_taus": exploratory,
        "baselines": {
            "production": { "inscope_acc": prod["overall"]["InScopeAccuracy"], "oos_decline": prod["overall"]["OosDeclineRate"] },
            "learned_head_iter1_clean": { "inscope_acc": 0.818, "oos_decline": 0.688, "macro_f1": 0.817, "source": "RESULTS.md iteration 1 table" },
            "learned_head_current": { "inscope_acc": head["test_inscope_acc_with_tau"], "oos_decline": head["test_oos_decline_rate"], "macro_f1": head["test_macro_f1"] },
        },
    })
}

/// Deterministic synthetic responses for exercising the pipeline. Gold with a
/// fixed error pattern, plus one malformed and one missing entry. MOCK only.
fn mock_receipt(s: &Setup) -> BTreeMap<String, (String, Value)> {
    let opts: Vec<&String> = s.option_set.iter().collect();
    s.heldout
        .prompts
        .iter()
        .zip(&s.digests)
        .enumerate()
        .filter(|(i, _)| i % 50 != 7) // one prompt per 50 left unanswered
        .map(|(i, (p, d))| {
            let pick = if i % 5 == 0 { opts[(i / 5) % opts.len()].clone() } else { p.expected.clone() };
            let rest = (1.0 - 0.6) / (opts.len() - 1) as f64;
            let probs: BTreeMap<&String, f64> = opts.iter().map(|o| (*o, if **o == pick { 0.6 } else { rest })).collect();
            let mut resp = json!({
                "model": MODEL,
                "answers": { QUESTION_ID: { "type": "choice", "choice": pick, "confidence": 0.6, "probabilities": probs } },
                "usage": { "input_tokens": 300, "output_tokens": 0 },
            });
            if i == 3 {
                resp["answers"][QUESTION_ID]["probabilities"][NONE] = json!(0.9); // no longer sums to 1
            }
            (p.id.clone(), (d.clone(), resp))
        })
        .collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let arm_name = match args.iter().position(|a| a == "--arm") {
        Some(i) => {
            let name = args.get(i + 1).expect("--arm needs a name").clone();
            args.drain(i..=i + 1);
            name
        }
        None => "base".to_string(),
    };
    let root = PathBuf::from(std::env::var("IX_ROOT").unwrap_or_else(|_| ".".into()));
    let s = setup(&root, arm(&arm_name));
    let out = arm_dir(&root, s.arm);
    match args.get(1).map(String::as_str) {
        Some("plan") => plan(&root, &s),
        Some("mock") => {
            let r = report(&root, &s, &mock_receipt(&s), "MOCK — synthetic responses, not a Jev measurement");
            fs::write(out.join("mock-report.json"), serde_json::to_string_pretty(&r).unwrap() + "\n").expect("write mock report");
            println!("{}", serde_json::to_string_pretty(&r["jev"]).unwrap());
        }
        Some("score") => {
            let path = args.get(2).expect("usage: score <receipt.jsonl>");
            let r = report(&root, &s, &load_receipt(Path::new(path)), "LIVE receipt");
            fs::write(out.join("jev-eval.json"), serde_json::to_string_pretty(&r).unwrap() + "\n").expect("write jev-eval.json");
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
        }
        _ => eprintln!("usage: jev-router plan | mock | score <receipt.jsonl> [--arm base|reversed|fr|es|fresh]   (run from the ix root or set IX_ROOT)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> BTreeSet<String> {
        ["a", "b", NONE].iter().map(|s| s.to_string()).collect()
    }

    fn resp(choice: &str, pa: f64, pb: f64, pn: f64) -> Value {
        json!({
            "model": MODEL,
            "answers": { "intent": { "type": "choice", "choice": choice, "confidence": 0.8,
                "probabilities": { "a": pa, "b": pb, NONE: pn } } },
            "usage": { "input_tokens": 10, "output_tokens": 0 },
        })
    }

    #[test]
    fn test_validate_accepts_well_formed_choice() {
        let a = validate(&resp("a", 0.7, 0.2, 0.1), &opts()).unwrap();
        assert_eq!(a.choice, "a");
        assert_eq!(a.input_tokens, 10);
    }

    #[test]
    fn test_validate_rejects_contract_breaks() {
        assert!(matches!(
            validate(&resp("a", 0.7, 0.2, 0.2), &opts()),
            Err(Invalid::Contract(_))
        )); // sum != 1
        assert!(matches!(
            validate(&resp("b", 0.7, 0.2, 0.1), &opts()),
            Err(Invalid::Contract(_))
        )); // not argmax
        assert!(matches!(
            validate(&resp("z", 0.7, 0.2, 0.1), &opts()),
            Err(Invalid::Contract(_))
        )); // unknown option
        let mut r = resp("a", 0.7, 0.2, 0.1);
        r["answers"]["intent"]["probabilities"]
            .as_object_mut()
            .unwrap()
            .remove(NONE);
        assert!(matches!(validate(&r, &opts()), Err(Invalid::Contract(_)))); // missing option key
        let mut r = resp("a", 0.7, 0.2, 0.1);
        r["model"] = json!("jev-latest");
        assert_eq!(
            validate(&r, &opts()),
            Err(Invalid::WrongModel("jev-latest".into()))
        );
    }

    #[test]
    fn test_missing_usage_is_invalid_not_zero_cost() {
        let mut r = resp("a", 0.7, 0.2, 0.1);
        r.as_object_mut().unwrap().remove("usage");
        assert!(matches!(validate(&r, &opts()), Err(Invalid::Contract(_))));
    }

    #[test]
    fn test_invalid_response_is_never_a_correct_decline() {
        let intents = vec!["a".to_string(), "b".to_string()];
        let m = score(&[NONE, "a"], &[None, None], &intents);
        assert_eq!((m.oos_declined, m.inscope_correct, m.invalid), (0, 0, 2));
    }

    #[test]
    fn test_score_counts_accuracy_decline_and_f1() {
        let intents = vec!["a".to_string(), "b".to_string()];
        let pred = [Some("a"), Some("b"), Some(NONE), Some("a")].map(|p| p.map(String::from));
        let m = score(&["a", "a", NONE, NONE], &pred, &intents);
        assert_eq!(
            (
                m.inscope_correct,
                m.inscope_total,
                m.oos_declined,
                m.oos_total
            ),
            (1, 2, 1, 2)
        );
        // a: tp1 fp1 fn1 -> 0.5 ; b: tp0 -> 0.0
        assert_eq!(m.per_intent_f1["a"], 0.5);
        assert_eq!(m.macro_f1, 0.25);
        assert_eq!(m.min_f1, ("b".to_string(), 0.0));
    }

    #[test]
    fn test_exploratory_tau_declines_low_confidence() {
        let a = validate(&resp("a", 0.7, 0.2, 0.1), &opts()).unwrap();
        assert_eq!(route(&a, 0.0), "a");
        assert_eq!(route(&a, 0.9), NONE);
    }

    #[test]
    fn test_usage_counts_rejected_responses() {
        let mut bad = resp("a", 0.7, 0.2, 0.2); // invalid: sums to 1.1
        bad["usage"] = json!({ "input_tokens": 40, "output_tokens": 5 });
        let receipt = BTreeMap::from([
            ("x".to_string(), (String::new(), resp("a", 0.7, 0.2, 0.1))),
            ("y".to_string(), (String::new(), bad)),
        ]);
        assert_eq!(usage_totals(&receipt), (50, 5));
    }

    #[test]
    fn test_reversed_body_changes_only_option_order() {
        let opts = Options {
            version: "t".into(),
            instructions: "pick".into(),
            criteria: BTreeMap::from([
                (NONE.to_string(), "none".to_string()),
                ("skill.a".to_string(), "A".to_string()),
            ]),
        };
        let p = Prompt {
            id: "x".into(),
            prompt: "hi".into(),
            expected: NONE.into(),
        };
        let fwd = String::from_utf8(body(&p, &opts, false)).unwrap();
        let rev = String::from_utf8(body(&p, &opts, true)).unwrap();
        assert!(fwd.contains(r#"{"__none__":"none","skill.a":"A"}"#));
        assert!(rev.contains(r#"{"skill.a":"A","__none__":"none"}"#));
        assert_eq!(fwd.len(), rev.len());
        // Same JSON value: only the key order differs.
        assert_eq!(
            serde_json::from_str::<Value>(&fwd).unwrap(),
            serde_json::from_str::<Value>(&rev).unwrap()
        );
    }

    #[test]
    fn test_robustness_bands() {
        let a = arm("fresh");
        let m = |ic, od, inv| Metrics {
            inscope_correct: ic,
            oos_declined: od,
            invalid: inv,
            ..Default::default()
        };
        assert_eq!(robustness_verdict(a, &m(101, 13, 0), false), "ROBUST");
        assert_eq!(robustness_verdict(a, &m(101, 12, 0), false), "DEGRADED");
        assert_eq!(robustness_verdict(a, &m(92, 16, 0), false), "DEGRADED");
        assert_eq!(robustness_verdict(a, &m(91, 16, 0), false), "FAIL");
        assert_eq!(robustness_verdict(a, &m(112, 16, 3), false), "KILL");
    }

    #[test]
    fn test_mcnemar_exact_matches_binomial() {
        assert_eq!(mcnemar_exact(0, 0), 1.0);
        // b=0, c=6: 2 * 0.5^6 = 0.03125
        assert!((mcnemar_exact(0, 6) - 0.03125).abs() < 1e-12);
        assert!((mcnemar_exact(3, 3) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_verdict_bands_follow_preregistration() {
        let m = |ic, od, inv, f1| Metrics {
            inscope_correct: ic,
            oos_declined: od,
            invalid: inv,
            macro_f1: f1,
            ..Default::default()
        };
        assert_eq!(verdict(&m(83, 6, 0, 0.75), false), "INCONCLUSIVE"); // tie with production
        assert_eq!(verdict(&m(82, 6, 0, 0.75), false), "KILL");
        assert_eq!(verdict(&m(87, 6, 0, 0.75), false), "BEATS_PRODUCTION");
        assert_eq!(verdict(&m(87, 11, 0, 0.79), false), "COMPETITIVE_WITH_HEAD");
        assert_eq!(verdict(&m(95, 12, 3, 0.9), false), "KILL");
        assert_eq!(verdict(&m(95, 12, 0, 0.9), true), "KILL");
    }

    #[test]
    fn test_rounded_probabilities_and_float_tokens_are_accepted() {
        let mut r = resp("a", 0.7004, 0.2, 0.1);
        r["usage"]["input_tokens"] = json!(10.0);
        assert!(validate(&r, &opts()).is_ok());
    }

    #[test]
    fn test_encoding_is_canonical_and_digest_stable() {
        let v = json!({ "b": 1, "a": "é" });
        assert_eq!(encode(&v), "{\"a\":\"é\",\"b\":1}".as_bytes());
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
