//! The tracer of ix-stream: guitar questions routed to experts under budgets, with
//! back-pressure, a gate that learns, algedonic signals and a live dashboard.
//!
//! ```text
//! cargo run -p ix-stream --example route -- [--policy learned|local|jev] [--epochs 3]
//!     [--jev-share 0.4] [--jev-ms 400] [--jev-workers 4] [--wait] [--claude N]
//!     [--serve 127.0.0.1:8791] [--linger S] [--seed 7] [--out DIR] [--data DIR]
//! ```
//!
//! The stream is the 221 labelled questions of `state/router-spike/jev/corpora/full-sol.json`
//! (32 skills and `__none__`), `--epochs` times, in an order DuckDB shuffles by a seeded hash.
//! Each question goes through:
//!
//! 1. `local`: the answer the local embedding router gave it, recorded in
//!    `full/local-c2-merge.json`, with its confidence;
//! 2. `kw`: a Zig stage built for WASI and run in ix-wasm's sandbox, which scores the
//!    question's words against the skills' descriptions (`IX_ZIG`, else `zig`);
//! 3. `gate`: an expert for the context (how sure `local` was, whether `kw` agrees): accept
//!    `local`, ask Jev, or ask Claude. A lane that is full is passed over for the next choice,
//!    as a mixture of experts drops what overflows an expert's capacity; with `--wait` the
//!    gate waits for its first choice instead;
//! 4. `jev`: Jev's recorded answer (`full/jev-eval.json`) after a wait like a live call's,
//!    charged what the recorded call reported. Nothing is sent to Jev;
//! 5. `claude`: Claude Haiku, live, on the person's plan (`claude -p`), at most `--claude`
//!    calls; none by default;
//! 6. `judge`: compares with the label and says so on the bus, where the gate learns it;
//! 7. `acc`: a DuckDB window over every 100 judged answers (accuracy per expert), shown live;
//!    `parquet`: every judged answer, kept in `results.parquet`.
//!
//! The Jev budget is `--jev-share` of the stream at the recorded price per call. `--policy`
//! swaps the gate for a fixed rule (always `local`; Jev until its budget is spent) on the same
//! stream, for comparison. Results go to `--out` as JSON lines, with every bus event.

use ix_loop::{Ask, ClaudeCode, SemanticNode};
use ix_stream::gate::{Arm, Budget, Gate, Ledger};
use ix_stream::petri::{check, Abstraction};
use ix_stream::sql::{ParquetSink, Window};
use ix_stream::sse::Feed;
use ix_stream::{source, Bus, Graph, Refused, Signal};
use ix_wasm::build::Toolchain;
use ix_wasm::{Limits, Program};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PAGE: &str = include_str!("../assets/dash.html");
const CONTEXTS: [&str; 6] = [
    "local refuse · kw differs",
    "local refuse · kw agrees",
    "local < 0.7 · kw differs",
    "local < 0.7 · kw agrees",
    "local >= 0.7 · kw differs",
    "local >= 0.7 · kw agrees",
];
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "what", "how", "from", "that", "this", "are", "you", "your",
    "any", "not", "all", "one", "its", "into", "out", "can", "doe", "which", "when", "use",
    "using", "return", "answer", "question", "user", "message", "about", "give", "like", "than",
    "them", "then", "there", "these", "they", "have", "ha", "each", "per", "only", "pure",
    "lookup", "llm", "call", "ask", "list", "given", "also", "such", "via", "other", "more",
    "most", "some", "skill", "none",
];

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    Learned,
    Local,
    Jev,
}

struct Args {
    policy: Policy,
    epochs: u32,
    jev_share: f64,
    jev_ms: u64,
    jev_workers: usize,
    wait: bool,
    claude: u32,
    serve: Option<String>,
    linger: u64,
    seed: u64,
    out: PathBuf,
    data: PathBuf,
}

fn args() -> Args {
    let mut a = Args {
        policy: Policy::Learned,
        epochs: 3,
        jev_share: 0.4,
        jev_ms: 400,
        jev_workers: 4,
        wait: false,
        claude: 0,
        serve: None,
        linger: 0,
        seed: 7,
        out: PathBuf::new(),
        data: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../state/router-spike/jev"),
    };
    let mut out = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--policy" => {
                a.policy = match value().as_str() {
                    "learned" => Policy::Learned,
                    "local" => Policy::Local,
                    "jev" => Policy::Jev,
                    p => panic!("no policy {p}: learned, local or jev"),
                }
            }
            "--epochs" => a.epochs = value().parse().expect("--epochs"),
            "--jev-share" => a.jev_share = value().parse().expect("--jev-share"),
            "--jev-ms" => a.jev_ms = value().parse().expect("--jev-ms"),
            "--jev-workers" => a.jev_workers = value().parse().expect("--jev-workers"),
            "--wait" => a.wait = true,
            "--claude" => a.claude = value().parse().expect("--claude"),
            "--serve" => a.serve = Some(value()),
            "--linger" => a.linger = value().parse().expect("--linger"),
            "--seed" => a.seed = value().parse().expect("--seed"),
            "--out" => out = Some(PathBuf::from(value())),
            "--data" => a.data = PathBuf::from(value()),
            f => panic!("unknown flag {f}"),
        }
    }
    let name = match a.policy {
        Policy::Learned => "learned",
        Policy::Local => "local",
        Policy::Jev => "jev",
    };
    a.out = out.unwrap_or_else(|| {
        std::env::temp_dir().join("ix-stream").join(format!(
            "route-{name}{}-seed{}",
            if a.wait { "-wait" } else { "" },
            a.seed
        ))
    });
    a
}

fn read(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The words of `text` as the Zig stage cuts them: ASCII letters and digits, lowercased, at
/// least three long, a final `s` dropped from words longer than three.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| {
            let w = w.to_ascii_lowercase();
            if w.len() > 3 && w.ends_with('s') {
                w[..w.len() - 1].to_string()
            } else {
                w
            }
        })
        .collect()
}

/// The keyword expert's source: each skill's description words, weighted by how few skills
/// share them.
fn keyword_stage(criteria: &BTreeMap<String, String>) -> String {
    let labels: Vec<&String> = criteria.keys().filter(|k| *k != "__none__").collect();
    let sets: Vec<BTreeSet<String>> = labels
        .iter()
        .map(|l| {
            let mut s: BTreeSet<String> = words(&criteria[*l]).into_iter().collect();
            s.extend(words(l.trim_start_matches("skill.")));
            s.retain(|w| !STOPWORDS.contains(&w.as_str()));
            s
        })
        .collect();
    let mut df: HashMap<&str, usize> = HashMap::new();
    for s in &sets {
        for w in s {
            *df.entry(w.as_str()).or_default() += 1;
        }
    }
    let n = labels.len() as f64;
    let mut table = String::new();
    for (i, s) in sets.iter().enumerate() {
        for w in s {
            let d = df[w.as_str()];
            if d <= 6 {
                let weight = (n / d as f64).ln();
                table.push_str(&format!(
                    "    .{{ .w = \"{w}\", .label = {i}, .weight = {weight:.3} }},\n"
                ));
            }
        }
    }
    let names: String = labels.iter().map(|l| format!("    \"{l}\",\n")).collect();
    format!(
        r#"const std = @import("std");
const wasi = std.os.wasi;

const Word = struct {{ w: []const u8, label: u8, weight: f32 }};
const labels = [_][]const u8{{
{names}}};
const table = [_]Word{{
{table}}};

fn readAll(gpa: std.mem.Allocator) ![]u8 {{
    var list: std.ArrayList(u8) = .empty;
    var chunk: [4096]u8 = undefined;
    while (true) {{
        const iov = [_]wasi.iovec_t{{.{{ .base = &chunk, .len = chunk.len }}}};
        var n: usize = 0;
        if (wasi.fd_read(0, &iov, 1, &n) != .SUCCESS) return error.ReadFailed;
        if (n == 0) return list.items;
        try list.appendSlice(gpa, chunk[0..n]);
    }}
}}

fn writeAll(fd: wasi.fd_t, bytes: []const u8) void {{
    var rest = bytes;
    while (rest.len > 0) {{
        const iov = [_]wasi.ciovec_t{{.{{ .base = rest.ptr, .len = rest.len }}}};
        var n: usize = 0;
        if (wasi.fd_write(fd, &iov, 1, &n) != .SUCCESS or n == 0) return;
        rest = rest[n..];
    }}
}}

const In = struct {{ text: []const u8 }};

pub fn main() u8 {{
    const gpa = std.heap.wasm_allocator;
    const raw = readAll(gpa) catch return 2;
    const x = std.json.parseFromSliceLeaky(In, gpa, raw, .{{ .ignore_unknown_fields = true }}) catch return 2;
    var score: [labels.len]f32 = @splat(0);
    var tok: [64]u8 = undefined;
    const t = x.text;
    var i: usize = 0;
    while (i < t.len) {{
        while (i < t.len and !std.ascii.isAlphanumeric(t[i])) : (i += 1) {{}}
        var n: usize = 0;
        while (i < t.len and std.ascii.isAlphanumeric(t[i])) : (i += 1) {{
            if (n < tok.len) {{
                tok[n] = std.ascii.toLower(t[i]);
                n += 1;
            }}
        }}
        if (n < 3) continue;
        var w: []const u8 = tok[0..n];
        if (w.len > 3 and w[w.len - 1] == 's') w = w[0 .. w.len - 1];
        for (table) |e| {{
            if (std.mem.eql(u8, e.w, w)) score[e.label] += e.weight;
        }}
    }}
    var best: usize = 0;
    var b1: f32 = 0;
    var b2: f32 = 0;
    for (score, 0..) |s, k| {{
        if (s > b1) {{
            b2 = b1;
            b1 = s;
            best = k;
        }} else if (s > b2) {{
            b2 = s;
        }}
    }}
    var out: [256]u8 = undefined;
    const line = blk: {{
        if (b1 <= 0) break :blk std.fmt.bufPrint(&out, "{{{{\"chosen\":\"__none__\",\"score\":0,\"margin\":0}}}}\n", .{{}}) catch return 1;
        break :blk std.fmt.bufPrint(&out, "{{{{\"chosen\":\"{{s}}\",\"score\":{{d:.3}},\"margin\":{{d:.3}}}}}}\n", .{{ labels[best], b1, b1 - b2 }}) catch return 1;
    }};
    writeAll(1, line);
    return 0;
}}
"#
    )
}

fn context(v: &Value) -> usize {
    let conf = v["local"]["confidence"].as_f64().unwrap_or(0.0);
    let bucket = if conf <= 0.0 {
        0
    } else if conf < 0.7 {
        1
    } else {
        2
    };
    bucket * 2 + usize::from(v["kw"]["chosen"] == v["local"]["chosen"])
}

fn key(v: &Value) -> String {
    format!("{}:{}", v["epoch"], v["id"].as_str().unwrap_or(""))
}

fn main() {
    let a = args();
    std::fs::create_dir_all(&a.out).expect("out folder");
    let corpus = a.data.join("corpora/full-sol.json");
    let jev_eval = read(&a.data.join("full/jev-eval.json"));
    let local: HashMap<String, (String, f64)> = read(&a.data.join("full/local-c2-merge.json"))
        .as_object()
        .expect("local predictions")
        .iter()
        .map(|(id, p)| {
            (
                id.clone(),
                (
                    p["chosen"].as_str().unwrap_or("__none__").to_string(),
                    p["confidence"].as_f64().unwrap_or(0.0),
                ),
            )
        })
        .collect();
    let jev: HashMap<String, String> = jev_eval["predictions"]
        .as_object()
        .expect("jev predictions")
        .iter()
        .map(|(id, p)| (id.clone(), p.as_str().unwrap_or("__none__").to_string()))
        .collect();
    let criteria: BTreeMap<String, String> = read(&a.data.join("options-full.json"))["criteria"]
        .as_object()
        .expect("criteria")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
        .collect();
    // What one recorded call cost: the rate card over the reported input tokens, per prompt.
    let per_call = jev_eval["usage"]["rate_card_usd"]
        .as_f64()
        .expect("rate card")
        / jev_eval["prompts"].as_f64().expect("prompts");
    let items = 221 * a.epochs as u64;
    let jev_cap = per_call * a.jev_share * items as f64;

    // The keyword expert, built once into the shared stage cache.
    let zig = std::env::var_os("IX_ZIG").unwrap_or_else(|| "zig".into());
    let cache = std::env::var_os("IX_WASM_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(|l| PathBuf::from(l).join("ix-wasm")))
        .unwrap_or_else(|| a.out.join("wasm"));
    let built = Toolchain::zig(zig, &cache)
        .build(
            &keyword_stage(&criteria),
            Instant::now() + Duration::from_secs(300),
        )
        .unwrap_or_else(|e| panic!("the keyword stage does not build:\n{e}"));
    let kw = Arc::new(Program::new(&std::fs::read(&built.path).unwrap()).unwrap());
    println!(
        "kw stage {} ({}, {} ms)",
        built.id,
        if built.cached { "cached" } else { "built" },
        built.ms
    );

    let mut budgets = vec![Budget::new("jev", "usd", jev_cap)];
    let mut arms = vec![Arm::free("local"), Arm::paid("jev", 0.02, "jev", per_call)];
    if a.claude > 0 {
        budgets.push(Budget::new("claude", "calls", a.claude as f64));
        arms.push(Arm::paid("claude", 0.1, "claude", 1.0));
    }
    let ledger = Ledger::new(budgets);
    let arm_names: Vec<String> = arms.iter().map(|x| x.name.clone()).collect();
    let mut gate = Gate::new(CONTEXTS.len(), arms, ledger.clone(), items);

    let bus = Bus::new();
    let record = bus.subscribe(&[], 1 << 20);
    let signals = bus.subscribe(&["signal"], 1 << 16);
    let mut g = Graph::new(&bus);

    let sql = format!(
        "SELECT e.epoch, p.id, p.prompt, p.expectedIntentId AS label \
         FROM range({}) e(epoch), (SELECT unnest(prompts) AS p FROM read_json('{}')) \
         ORDER BY e.epoch, hash(p.id || '/' || e.epoch || '/' || {})",
        a.epochs,
        corpus.display().to_string().replace('\\', "/"),
        a.seed
    );
    let src = g.source("duckdb", move |out| match source::duckdb(&sql, out) {
        Ok(sent) => out.bus().publish(
            "source",
            json!({ "rows": sent.rows, "cancelled": sent.cancelled }),
        ),
        Err(e) => eprintln!("{e}"),
    });

    let local_n = g.stage("local", 32, 1, move |mut v, out| {
        let (chosen, conf) = local
            .get(v["id"].as_str().unwrap_or(""))
            .cloned()
            .unwrap_or(("__none__".into(), 0.0));
        v["local"] = json!({ "chosen": chosen, "confidence": conf });
        let _ = out.emit(0, v);
    });

    let kw_n = g.stage("kw", 32, 4, move |mut v, out| {
        let stdin = serde_json::to_vec(&json!({ "text": v["prompt"] })).unwrap();
        let limits = Limits {
            fuel: 200_000_000,
            ..Limits::default()
        };
        v["kw"] = match kw.run(&stdin, &limits) {
            Ok(o) if o.exit == 0 => {
                let mut k: Value = serde_json::from_slice(&o.stdout).unwrap_or(Value::Null);
                k["fuel"] = json!(o.fuel_used);
                k
            }
            Ok(o) => json!({ "error": format!("exit {}", o.exit) }),
            Err(e) => json!({ "error": e.to_string() }),
        };
        let _ = out.emit(0, v);
    });

    let policy = a.policy;
    let wait = a.wait;
    let names = arm_names.clone();
    let gate_n = g.stage_mut("gate", 16, move |mut v, out| {
        for e in signals.try_iter() {
            if let Some(s) = Signal::from_json(&e.data) {
                if let (true, Some(item)) = (s.node == "judge", &s.item) {
                    gate.learn(item, if s.valence > 0 { 1.0 } else { 0.0 });
                }
            }
        }
        let ctx = context(&v);
        let item = key(&v);
        let choice = gate.choose(ctx);
        let order = match policy {
            Policy::Learned => choice.ranked.clone(),
            Policy::Local => vec![0],
            Policy::Jev if choice.prices[1].is_finite() => vec![1, 0],
            Policy::Jev => vec![0],
        };
        let wanted = order[0];
        let mut held = Some(v.take());
        let mut arm = 0;
        for &k in &order {
            if k == 0 {
                break;
            }
            let mut w = held.take().unwrap();
            w["route"] = json!({ "ctx": ctx, "arm": names[k] });
            if wait && k == wanted {
                if out.emit(k, w).is_ok() {
                    arm = k;
                }
                break;
            }
            match out.try_emit(k, w) {
                Ok(()) => {
                    arm = k;
                    break;
                }
                Err(Refused::Full(w)) | Err(Refused::Closed(w)) => {
                    out.signal(
                        &Signal::pain("gate", 0.2, &format!("{} lane full", names[k])).about(&item),
                    );
                    held = Some(w);
                }
            }
        }
        if let Some(mut w) = held {
            w["route"] = json!({ "ctx": ctx, "arm": "local" });
            w["answer"] = w["local"]["chosen"].clone();
            let _ = out.emit(0, w);
        }
        gate.commit(&item, ctx, arm);
        let (q, n) = gate.beliefs(ctx);
        out.bus().publish(
            "route",
            json!({
                "item": item, "ctx": ctx, "arm": names[arm], "wanted": names[wanted],
                "prices": choice.prices.iter().map(|p| if p.is_finite() { json!((p * 1e4).round() / 1e4) } else { json!(null) }).collect::<Vec<_>>(),
                "pace": choice.pace, "budgets": gate.ledger().json(),
                "beliefs": { "q": q, "n": n },
            }),
        );
    });

    let lane_ledger = ledger.clone();
    let jev_ms = a.jev_ms;
    let jev_n = g.stage("jev", 4, a.jev_workers, move |mut v, out| {
        let item = key(&v);
        if lane_ledger.spend("jev", per_call) {
            std::thread::sleep(Duration::from_millis(jev_ms));
            v["answer"] = json!(jev
                .get(v["id"].as_str().unwrap_or(""))
                .cloned()
                .unwrap_or("__none__".into()));
            v["cost"] = json!({ "usd": per_call });
            let share = per_call / jev_cap.max(1e-12);
            out.signal(&Signal::pain("jev", share, "spent a share of the jev budget").about(&item));
        } else {
            v["answer"] = v["local"]["chosen"].clone();
            v["fallback"] = json!("jev budget");
            out.signal(&Signal::pain("jev", 1.0, "budget refused the call").about(&item));
        }
        let _ = out.emit(0, v);
    });

    let claude_n = (a.claude > 0).then(|| {
        let node = ClaudeCode::new(a.out.join("claude-work"));
        let lane_ledger = ledger.clone();
        let ids: Vec<&String> = criteria.keys().collect();
        let menu: String = criteria
            .iter()
            .map(|(k, d)| format!("- {k}: {d}\n"))
            .collect();
        let schema = json!({
            "type": "object",
            "properties": { "skill": { "type": "string", "enum": ids } },
            "required": ["skill"],
            "additionalProperties": false,
        });
        let cap = a.claude as f64;
        g.stage("claude", 2, 2, move |mut v, out| {
            let item = key(&v);
            let answer = if lane_ledger.spend("claude", 1.0) {
                let ask = Ask {
                    prompt: format!("Skills:\n{menu}\nMessage: {}\n", v["prompt"].as_str().unwrap_or("")),
                    model: "haiku".into(),
                    schema: Some(schema.clone()),
                    system: Some("You route a message to one skill of a guitar assistant. Choose __none__ when no skill fits.".into()),
                };
                match node.ask(&ask, Instant::now() + Duration::from_secs(120)) {
                    Ok(ans) => {
                        out.signal(&Signal::pain("claude", 1.0 / cap, "spent one call of the plan").about(&item));
                        v["cost"] = json!({ "calls": 1, "notional_usd": ans.notional_usd });
                        ans.value["skill"].as_str().map(str::to_string)
                    }
                    Err(e) => {
                        out.signal(&Signal::pain("claude", 1.0, &format!("call failed: {e}")).about(&item));
                        None
                    }
                }
            } else {
                out.signal(&Signal::pain("claude", 1.0, "budget refused the call").about(&item));
                None
            };
            match answer {
                Some(s) => v["answer"] = json!(s),
                None => {
                    v["answer"] = v["local"]["chosen"].clone();
                    v["fallback"] = json!("claude");
                }
            }
            let _ = out.emit(0, v);
        })
    });

    let results = Arc::new(Mutex::new(Vec::new()));
    let kept = results.clone();
    let (mut n, mut good) = (0u64, 0u64);
    let judge_n = g.stage_mut("judge", 64, move |v, out| {
        let right = v["answer"] == v["label"];
        let item = key(&v);
        let s = if right {
            Signal::pleasure("judge", 1.0, "right")
        } else {
            Signal::pain("judge", 1.0, "wrong")
        };
        out.signal(&s.about(&item));
        n += 1;
        good += u64::from(right);
        let arm = v["route"]["arm"].clone();
        out.bus().publish(
            "judged",
            json!({ "n": n, "good": good, "acc": good as f64 / n as f64, "arm": arm, "ctx": v["route"]["ctx"], "right": right }),
        );
        let record = json!({
            "epoch": v["epoch"], "id": v["id"], "ctx": v["route"]["ctx"], "arm": arm,
            "answer": v["answer"], "label": v["label"], "correct": right,
            "fallback": v["fallback"], "local": v["local"], "kw": v["kw"], "cost": v["cost"],
        });
        for port in 0..out.ports() {
            let _ = out.emit(port, record.clone());
        }
        kept.lock().unwrap().push(record);
    });

    // SQL in the stream: accuracy per expert over each window of 100 judged answers.
    let acc_n = g.batch(
        "acc",
        64,
        Window::new(
            "acc",
            "SELECT arm, count(*) AS n, round(avg(correct::INT), 3) AS acc FROM batch GROUP BY arm ORDER BY arm",
            100,
            &a.out,
        ),
    );
    let windows_n = g.stage_mut("windows", 64, |row, out| out.bus().publish("window", row));
    let parquet_n = g.batch(
        "parquet",
        64,
        ParquetSink::new("parquet", &a.out.join("results.parquet")).expect("parquet sink"),
    );

    g.link(src, local_n);
    g.link(local_n, kw_n);
    g.link(kw_n, gate_n);
    g.link(gate_n, judge_n);
    g.link(gate_n, jev_n);
    if let Some(c) = claude_n {
        g.link(gate_n, c);
        g.link(c, judge_n);
    }
    g.link(jev_n, judge_n);
    g.link(judge_n, acc_n);
    g.link(acc_n, windows_n);
    g.link(judge_n, parquet_n);

    let shape = g.shape();
    let mut petri = check(&shape, Abstraction::exact(), 50_000).expect("petri net");
    if petri.deadlock_free.is_none() {
        petri = check(&shape, Abstraction::clamp(1, 1), 2_000_000).expect("petri net");
    }
    println!(
        "petri: deadlock_free={:?} bounded={:?} states={} exact={}",
        petri.deadlock_free,
        petri.bounded,
        petri.states,
        petri.abstraction.is_exact()
    );

    let feed = a.serve.as_deref().map(|addr| {
        let f = Feed::serve(addr, PAGE, &bus, &[], 100_000).expect("feed");
        println!("dashboard: http://{}/", f.addr());
        f
    });
    bus.publish(
        "meta",
        json!({
            "title": "route", "policy": match a.policy { Policy::Learned => "learned", Policy::Local => "local", Policy::Jev => "jev" },
            "items": items, "contexts": CONTEXTS, "arms": arm_names,
            "budgets": ledger.json(), "petri": petri.json(), "jev_ms": a.jev_ms,
            "jev_workers": a.jev_workers, "wait": a.wait,
        }),
    );
    if feed.is_some() && a.linger > 0 {
        // A moment for a browser to connect before the stream starts.
        std::thread::sleep(Duration::from_secs(3));
    }

    let report = g.run(Some(Duration::from_millis(250)));
    bus.publish("done", json!({ "elapsed_s": report.elapsed.as_secs_f64() }));

    let results = results.lock().unwrap();
    let path = a.out.join("results.jsonl");
    let mut f = std::fs::File::create(&path).unwrap();
    for r in results.iter() {
        writeln!(f, "{r}").unwrap();
    }
    let mut ev = std::fs::File::create(a.out.join("events.jsonl")).unwrap();
    for e in record.try_iter() {
        writeln!(ev, "{}", e.json()).unwrap();
    }
    std::fs::write(a.out.join("petri.json"), petri.json().to_string()).unwrap();

    let total = results.len() as f64;
    let right = results.iter().filter(|r| r["correct"] == true).count() as f64;
    println!(
        "\n{} items in {:.1} s: accuracy {:.3}; jev ${:.5} of ${:.5}; bus lost {}",
        results.len(),
        report.elapsed.as_secs_f64(),
        right / total.max(1.0),
        ledger.get("jev").unwrap().spent,
        jev_cap,
        report.bus_lost
    );
    if let Some(c) = ledger.get("claude") {
        println!("claude calls {} of {}", c.spent, c.cap);
    }
    for s in &report.nodes {
        println!(
            "  {:<8} in {:>5} out {:>5} max queued {:>3}/{:<3} busy {:>9.1} ms blocked {:>9.1} ms refused {}",
            s.name, s.received, s.emitted, s.max_queued, s.cap, s.busy_ms, s.blocked_ms, s.refused
        );
    }
    let sql = format!(
        "SELECT ctx, arm, count(*) AS n, round(avg(correct::INT), 3) AS acc \
         FROM read_json('{0}') GROUP BY ALL ORDER BY ctx, arm; \
         SELECT arm, count(*) AS n, round(avg(correct::INT), 3) AS acc FROM read_json('{0}') GROUP BY ALL ORDER BY n DESC;",
        path.display().to_string().replace('\\', "/")
    );
    match std::process::Command::new(source::duckdb_exe())
        .args(["-c", &sql])
        .output()
    {
        Ok(o) => print!("{}", String::from_utf8_lossy(&o.stdout)),
        Err(e) => println!("no duckdb report: {e}"),
    }
    println!("results in {}", a.out.display());
    if let Some(f) = feed {
        if a.linger > 0 {
            println!("the dashboard stays up for {} s", a.linger);
            std::thread::sleep(Duration::from_secs(a.linger));
        }
        f.close();
    }
}
