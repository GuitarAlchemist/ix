//! `ga-chatbot` CLI — stub MCP server, single-shot test mode, and QA runner.

use clap::{Parser, Subcommand};
use ga_chatbot::aggregate::{JudgeVerdict, QaResult};
use ga_chatbot::mcp_bridge::{McpBridge, McpBridgeConfig};
use ga_chatbot::qa::{load_corpus_ids, run_deterministic_checks};
use ga_chatbot::{ask_stub, load_fixtures, ChatbotRequest, Instrument};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(name = "ga-chatbot", about = "Domain-specific voicing chatbot")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a minimal JSON-RPC stdio MCP server (stub mode).
    Serve {
        /// Use stub fixtures for responses.
        #[arg(long)]
        stub: bool,
        /// Serve HTTP on this port instead of stdio JSON-RPC.
        #[arg(long)]
        http: Option<u16>,
        /// Path to stub fixtures JSONL file.
        #[arg(
            long,
            default_value = "tests/adversarial/fixtures/stub-responses.jsonl"
        )]
        fixtures: PathBuf,
    },
    /// Single-shot ask mode for testing.
    Ask {
        /// The question to ask.
        #[arg(long)]
        question: String,
        /// Target instrument.
        #[arg(long, default_value = "guitar")]
        instrument: String,
        /// Path to stub fixtures JSONL file.
        #[arg(
            long,
            default_value = "tests/adversarial/fixtures/stub-responses.jsonl"
        )]
        fixtures: PathBuf,
    },
    /// Start the live HTTP server backed by real GA + IX MCP tools.
    ServeLive {
        /// HTTP port to listen on.
        #[arg(long, default_value = "7184")]
        port: u16,
        /// Executable for the GA MCP server.
        #[arg(long, default_value = "dotnet")]
        ga_command: String,
        /// Arguments for the GA MCP server (comma-separated, e.g. "run,--project,GaMcpServer").
        #[arg(long, value_delimiter = ',')]
        ga_args: Vec<String>,
        /// Executable for the IX MCP server.
        #[arg(long, default_value = "cargo")]
        ix_command: String,
        /// Arguments for the IX MCP server (comma-separated, e.g. "run,-p,ix-agent").
        #[arg(long, value_delimiter = ',')]
        ix_args: Vec<String>,
    },
    /// Run the deterministic QA pipeline on the adversarial corpus.
    Qa {
        /// Directory containing adversarial prompt corpus (*.jsonl files).
        #[arg(long)]
        corpus: PathBuf,
        /// Path to stub response fixtures (JSONL).
        #[arg(long)]
        fixtures: PathBuf,
        /// Directory containing voicing corpus JSON files.
        #[arg(long)]
        corpus_dir: PathBuf,
        /// Output path for findings (JSONL).
        #[arg(long, default_value = "findings.jsonl")]
        output: PathBuf,
        /// Compute Shapley prompt attribution after QA and append to output.
        #[arg(long)]
        shapley: bool,
        /// Benchmark mode: run prompts against live LLMs instead of stub fixtures.
        #[arg(long)]
        benchmark: bool,
        /// Comma-separated model list for benchmark (e.g. "gpt-4o-mini,llama3.1:latest").
        /// First tries OpenAI model names, then Ollama. Default: gpt-4o-mini.
        #[arg(long, default_value = "gpt-4o-mini")]
        models: String,
    },
}

/// A prompt entry from the adversarial corpus.
#[derive(serde::Deserialize)]
struct CorpusEntry {
    id: String,
    category: String,
    prompt: String,
    expected_check: String,
    expected_verdict: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve {
            stub: _,
            http,
            fixtures,
        } => {
            let fixture_map = load_fixtures(&fixtures);
            if let Some(port) = http {
                serve_http(port, &fixture_map);
            } else {
                serve_jsonrpc(&fixture_map);
            }
        }
        Commands::ServeLive {
            port,
            ga_command,
            ga_args,
            ix_command,
            ix_args,
        } => {
            let config = McpBridgeConfig {
                ga_command,
                ga_args,
                ix_command,
                ix_args,
            };
            serve_http_live(port, &config);
        }
        Commands::Ask {
            question,
            instrument,
            fixtures,
        } => {
            let fixture_map = load_fixtures(&fixtures);
            let inst = match instrument.to_lowercase().as_str() {
                "guitar" => Some(Instrument::Guitar),
                "bass" => Some(Instrument::Bass),
                "ukulele" => Some(Instrument::Ukulele),
                _ => None,
            };
            let req = ChatbotRequest {
                question,
                instrument: inst,
            };
            let resp = ask_stub(&req, &fixture_map);
            println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        }
        Commands::Qa {
            corpus,
            fixtures,
            corpus_dir,
            output,
            shapley,
            benchmark,
            models,
        } => {
            if benchmark {
                let model_list: Vec<&str> = models.split(',').map(|s| s.trim()).collect();
                std::process::exit(run_benchmark(&corpus, &corpus_dir, &output, &model_list));
            } else {
                std::process::exit(run_qa(&corpus, &fixtures, &corpus_dir, &output, shapley));
            }
        }
    }
}

/// Run the deterministic QA pipeline over all adversarial prompts.
///
/// Returns 0 if no F/D verdicts, 1 otherwise.
fn run_qa(
    corpus_path: &std::path::Path,
    fixtures_path: &std::path::Path,
    corpus_dir: &std::path::Path,
    output_path: &std::path::Path,
    shapley: bool,
) -> i32 {
    // Load fixtures for stub responses
    let fixture_map = load_fixtures(fixtures_path);

    // Load voicing corpus IDs from all *-corpus.json files in corpus_dir
    let mut all_corpus_ids = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(corpus_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json")
                && path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("-corpus"))
            {
                let ids = load_corpus_ids(&path);
                all_corpus_ids.extend(ids);
            }
        }
    }
    eprintln!("Loaded {} voicing IDs from corpus", all_corpus_ids.len());

    // Load all adversarial prompts from corpus directory
    let prompts = load_adversarial_prompts(corpus_path);
    eprintln!("Loaded {} adversarial prompts", prompts.len());

    if prompts.is_empty() {
        eprintln!("No prompts found in {:?}", corpus_path);
        return 1;
    }

    // Run pipeline
    let mut results: Vec<QaResult> = Vec::new();
    let mut fail_count = 0;
    let mut pass_count = 0;

    // Regression-gauge tallies: compare actual deterministic verdict to the
    // expected_verdict declared in each corpus entry. Prompts tagged
    // expected_check="llm" are deferred (not graded here).
    let mut match_count = 0;
    let mut mismatch_count = 0;
    let mut deferred_count = 0;
    let mut mismatches: Vec<(String, String, char, String)> = Vec::new();

    for entry in &prompts {
        let req = ChatbotRequest {
            question: entry.prompt.clone(),
            instrument: Some(Instrument::Guitar),
        };
        let response = ask_stub(&req, &fixture_map);
        let findings =
            run_deterministic_checks(&entry.id, &entry.prompt, &response, &all_corpus_ids);

        // Determine worst verdict from deterministic checks
        let det_verdict = worst_verdict(&findings);

        // Gauge: actual vs expected. Deferred when the corpus labels this
        // prompt as requiring an LLM judge.
        let expected = entry.expected_verdict.chars().next().unwrap_or('?');
        if entry.expected_check == "llm" {
            deferred_count += 1;
        } else if det_verdict == expected {
            match_count += 1;
        } else {
            mismatch_count += 1;
            mismatches.push((
                entry.id.clone(),
                entry.category.clone(),
                det_verdict,
                entry.expected_verdict.clone(),
            ));
        }

        // Create a single "deterministic" judge verdict for aggregation
        let judge_verdict = JudgeVerdict {
            judge: "deterministic".to_string(),
            verdict: det_verdict,
            grounded: !findings.iter().any(|f| f.layer == 1 && f.verdict == 'F'),
            accurate: true, // deterministic layer doesn't check accuracy
            safe: !findings.iter().any(|f| f.layer == 0 && f.verdict == 'F'),
            reasoning: findings
                .iter()
                .map(|f| f.reason.clone())
                .collect::<Vec<_>>()
                .join("; "),
            flags: findings
                .iter()
                .filter(|f| f.verdict == 'F')
                .map(|f| format!("layer{}:{}", f.layer, f.reason))
                .collect(),
        };

        let aggregate =
            ga_chatbot::aggregate::aggregate_verdicts(std::slice::from_ref(&judge_verdict));

        let result = QaResult {
            prompt_id: entry.id.clone(),
            deterministic_verdict: Some(det_verdict),
            judge_verdicts: vec![judge_verdict],
            aggregate,
        };

        match det_verdict {
            'F' | 'D' => fail_count += 1,
            _ => pass_count += 1,
        }

        results.push(result);
    }

    // Write findings to output JSONL
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out_file = match std::fs::File::create(output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create output file {:?}: {}", output_path, e);
            return 1;
        }
    };
    for result in &results {
        if let Ok(json) = serde_json::to_string(result) {
            writeln!(out_file, "{}", json).ok();
        }
    }

    // Print summary
    let total = results.len();
    println!();
    println!("=== Adversarial QA Summary ===");
    println!("Total prompts: {}", total);
    println!("Pass (T/P):    {}", pass_count);
    println!("Fail (F/D):    {}", fail_count);
    println!("Output:        {:?}", output_path);
    println!();

    // Regression gauge: actual deterministic verdict vs corpus-declared
    // expected_verdict. "Deferred" = expected_check="llm" (not graded here).
    let graded = match_count + mismatch_count;
    let match_pct = if graded > 0 {
        100.0 * match_count as f64 / graded as f64
    } else {
        0.0
    };
    println!("=== Regression Gauge (deterministic-graded) ===");
    println!(
        "Graded:        {} ({} match, {} mismatch)",
        graded, match_count, mismatch_count
    );
    println!("Match rate:    {:.1}%", match_pct);
    println!("Deferred(LLM): {}", deferred_count);
    if !mismatches.is_empty() {
        println!();
        println!("Regression candidates (actual != expected):");
        for (id, category, actual, expected) in &mismatches {
            println!(
                "  {:<20} [{}] actual={} expected={}",
                id, category, actual, expected
            );
        }
    }
    println!();

    // Emit a machine-readable gauge summary alongside findings.jsonl so CI
    // and dashboards can consume the structured metrics without re-parsing.
    let summary_path = output_path.with_file_name("summary.json");
    let mut verdict_counts: HashMap<char, usize> = HashMap::new();
    for r in &results {
        if let Some(v) = r.deterministic_verdict {
            *verdict_counts.entry(v).or_insert(0) += 1;
        }
    }
    let summary = serde_json::json!({
        "total": total,
        "graded": graded,
        "match": match_count,
        "mismatch": mismatch_count,
        "deferred_llm": deferred_count,
        "match_rate": if graded > 0 { match_count as f64 / graded as f64 } else { 0.0 },
        "verdict_counts": {
            "T": verdict_counts.get(&'T').copied().unwrap_or(0),
            "P": verdict_counts.get(&'P').copied().unwrap_or(0),
            "U": verdict_counts.get(&'U').copied().unwrap_or(0),
            "C": verdict_counts.get(&'C').copied().unwrap_or(0),
            "D": verdict_counts.get(&'D').copied().unwrap_or(0),
            "F": verdict_counts.get(&'F').copied().unwrap_or(0),
        },
        "mismatches": mismatches.iter().map(|(id, category, actual, expected)| serde_json::json!({
            "id": id,
            "category": category,
            "actual": actual.to_string(),
            "expected": expected,
        })).collect::<Vec<_>>(),
        "pipeline_version": "phase2-deterministic-only",
    });
    if let Ok(f) = std::fs::File::create(&summary_path) {
        serde_json::to_writer_pretty(f, &summary).ok();
    }

    // Print worst-scoring prompts
    let failures: Vec<_> = results
        .iter()
        .filter(|r| matches!(r.deterministic_verdict, Some('F') | Some('D')))
        .collect();
    if !failures.is_empty() {
        println!("Failing prompts:");
        for r in &failures {
            let reasons: Vec<_> = r.judge_verdicts.iter().flat_map(|j| &j.flags).collect();
            println!(
                "  {} [{}] {}",
                r.prompt_id,
                r.deterministic_verdict.unwrap_or('?'),
                reasons
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        println!();
    }

    // Shapley attribution (post-QA, not in critical path)
    if shapley && !results.is_empty() {
        let sample: Vec<QaResult> = if results.len() > 20 {
            // Sample 20 stratified by category (take first 20 for simplicity)
            eprintln!(
                "Shapley: sampling 20 of {} results (exact Shapley capped at 20)",
                results.len()
            );
            results.iter().take(20).cloned().collect()
        } else {
            results.clone()
        };

        let scores = ga_chatbot::shapley::compute_prompt_shapley(&sample);

        // Append shapley_summary to output file
        let shapley_summary = serde_json::json!({
            "shapley_summary": {
                "sample_size": sample.len(),
                "total_prompts": results.len(),
                "scores": scores,
            }
        });
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(output_path) {
            writeln!(f, "{}", shapley_summary).ok();
        }

        // Print top-5 and bottom-5
        println!("=== Shapley Prompt Attribution ===");
        println!("Sample size: {} / {}", sample.len(), results.len());
        println!();

        let top_n = scores.len().min(5);
        println!("Top-{} most diagnostic prompts:", top_n);
        for s in scores.iter().take(top_n) {
            println!(
                "  {:<25} Shapley={:.4}  category={}  fail_rate={:.1}",
                s.prompt_id, s.shapley_value, s.category, s.failure_rate
            );
        }

        let bottom_n = scores.len().min(5);
        println!();
        println!("Bottom-{} least diagnostic (pruning candidates):", bottom_n);
        for s in scores
            .iter()
            .rev()
            .take(bottom_n)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            println!(
                "  {:<25} Shapley={:.4}  category={}  fail_rate={:.1}",
                s.prompt_id, s.shapley_value, s.category, s.failure_rate
            );
        }
        println!();
    }

    // Exit code reflects the regression gauge: a mismatch (actual verdict
    // != expected_verdict on a graded prompt) is a real regression. Raw F/D
    // counts are noisy because many adversarial prompts are expected to
    // fail (injection, hallucination) and LLM-deferred prompts have no
    // deterministic ground truth.
    if mismatch_count > 0 {
        1
    } else {
        0
    }
}

/// Benchmark the chatbot across multiple LLM models.
///
/// For each model, runs all adversarial prompts through the live LLM,
/// scores responses with deterministic checks, and outputs a comparison table.
fn run_benchmark(
    corpus_path: &std::path::Path,
    corpus_dir: &std::path::Path,
    output_path: &std::path::Path,
    models: &[&str],
) -> i32 {
    let mut all_corpus_ids = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(corpus_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json")
                && path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("-corpus"))
            {
                let ids = load_corpus_ids(&path);
                all_corpus_ids.extend(ids);
            }
        }
    }

    let prompts = load_adversarial_prompts(corpus_path);
    if prompts.is_empty() {
        eprintln!("No prompts found in {:?}", corpus_path);
        return 1;
    }

    #[derive(serde::Serialize)]
    struct ModelScore {
        model: String,
        total: usize,
        pass: usize,
        fail: usize,
        pass_rate: f64,
        avg_response_ms: u64,
    }

    let mut scores: Vec<ModelScore> = Vec::new();
    let mut all_results: Vec<serde_json::Value> = Vec::new();

    for model_name in models {
        eprintln!("\n=== Benchmarking: {} ===", model_name);
        std::env::set_var("GA_CHATBOT_MODEL", model_name);

        let mut pass = 0usize;
        let mut fail = 0usize;
        let mut total_ms = 0u128;

        for entry in &prompts {
            let start = std::time::Instant::now();
            let history = vec![serde_json::json!({"role": "user", "content": entry.prompt})];
            let answer = call_llm(&entry.prompt, &history);
            let elapsed = start.elapsed().as_millis();
            total_ms += elapsed;

            let response = ga_chatbot::ChatbotResponse {
                answer: answer.clone(),
                voicing_ids: vec![],
                confidence: 0.5,
                sources: vec![],
            };

            let findings =
                run_deterministic_checks(&entry.id, &entry.prompt, &response, &all_corpus_ids);
            let verdict = worst_verdict(&findings);

            let passed = !matches!(verdict, 'F' | 'D');
            if passed {
                pass += 1;
            } else {
                fail += 1;
            }

            all_results.push(serde_json::json!({
                "model": model_name,
                "prompt_id": entry.id,
                "verdict": verdict.to_string(),
                "response_ms": elapsed,
                "answer_preview": if answer.len() > 100 { &answer[..100] } else { &answer },
            }));

            eprint!("  {} [{}] {}ms ", entry.id, verdict, elapsed);
            if passed {
                eprintln!("PASS");
            } else {
                eprintln!("FAIL");
            }
        }

        let total = pass + fail;
        scores.push(ModelScore {
            model: model_name.to_string(),
            total,
            pass,
            fail,
            pass_rate: if total > 0 {
                pass as f64 / total as f64
            } else {
                0.0
            },
            avg_response_ms: if total > 0 {
                (total_ms / total as u128) as u64
            } else {
                0
            },
        });
    }

    // Write detailed results
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    if let Ok(mut f) = std::fs::File::create(output_path) {
        for r in &all_results {
            writeln!(f, "{}", r).ok();
        }
        writeln!(f, "{}", serde_json::json!({"benchmark_summary": &scores})).ok();
    }

    // Print comparison table
    println!();
    println!("╔══════════════════════════╦═══════╦══════╦══════╦══════════╦═════════╗");
    println!("║ Model                    ║ Total ║ Pass ║ Fail ║ Pass Rate║ Avg ms  ║");
    println!("╠══════════════════════════╬═══════╬══════╬══════╬══════════╬═════════╣");
    for s in &scores {
        println!(
            "║ {:<24} ║ {:>5} ║ {:>4} ║ {:>4} ║ {:>7.1}% ║ {:>6}ms ║",
            s.model,
            s.total,
            s.pass,
            s.fail,
            s.pass_rate * 100.0,
            s.avg_response_ms
        );
    }
    println!("╚══════════════════════════╩═══════╩══════╩══════╩══════════╩═════════╝");
    println!();
    println!("Results written to {:?}", output_path);

    0
}

/// Find the worst verdict in a set of findings (F > D > U > C > P > T).
fn worst_verdict(findings: &[ga_chatbot::qa::Finding]) -> char {
    let priority = |c: char| -> u8 {
        match c {
            'F' => 5,
            'D' => 4,
            'C' => 3,
            'U' => 2,
            'P' => 1,
            'T' => 0,
            _ => 0,
        }
    };
    findings
        .iter()
        .map(|f| f.verdict)
        .max_by_key(|&c| priority(c))
        .unwrap_or('U')
}

/// Load all adversarial prompts from JSONL files in a directory.
fn load_adversarial_prompts(dir: &std::path::Path) -> Vec<CorpusEntry> {
    let mut prompts = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read corpus directory {:?}: {}", dir, e);
            return prompts;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "jsonl") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<CorpusEntry>(line) {
                        Ok(entry) => prompts.push(entry),
                        Err(e) => {
                            eprintln!("Failed to parse line in {:?}: {}", path, e);
                        }
                    }
                }
            }
        }
    }
    prompts
}

const SYSTEM_PROMPT: &str = r#"You are the Guitar Alchemist chatbot — a music theory assistant specialized in chord voicings across guitar, bass, and ukulele.

You have access to a voicing corpus of 1,500 analyzed voicings (500 per instrument):

GUITAR (standard tuning EADGBE, 24 frets): 5 voicing families (silhouette=0.199), connected topology (betti_0=1, betti_1=561), transition costs 3.0-7.0, I-IV-V grammar (64 parses).
BASS (standard tuning EADG, 21 frets): 5 families (silhouette=0.206), connected (betti_0=1, betti_1=587).
UKULELE (standard tuning GCEA, 15 frets): 5 families (silhouette=0.204), connected (betti_0=1, betti_1=586), richer progressions (125 parses).

Rules: Be specific about fret positions. Use tab notation (e.g. x-3-2-0-0-0 low to high). Don't hallucinate positions. Keep answers concise."#;

const TOOLS: &str = r#"[
  {
    "type": "function",
    "function": {
      "name": "search_voicings",
      "description": "Search the voicing corpus for chord voicings on a specific instrument. Returns real voicing data with fret positions and MIDI notes.",
      "parameters": {
        "type": "object",
        "properties": {
          "instrument": { "type": "string", "enum": ["guitar", "bass", "ukulele"] },
          "query": { "type": "string", "description": "Search term: chord name, quality, or fret range (e.g. 'maj7', 'open position', 'barre')" }
        },
        "required": ["instrument"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_instrument_info",
      "description": "Get instrument specifications: tuning, string count, fret count, range.",
      "parameters": {
        "type": "object",
        "properties": {
          "instrument": { "type": "string", "enum": ["guitar", "bass", "ukulele"] }
        },
        "required": ["instrument"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "verify_voicing",
      "description": "Verify a voicing is physically possible on an instrument. Checks string count, fret span, and range.",
      "parameters": {
        "type": "object",
        "properties": {
          "instrument": { "type": "string", "enum": ["guitar", "bass", "ukulele"] },
          "frets": { "type": "string", "description": "Fret notation like x-3-2-0-1-0 (low to high)" }
        },
        "required": ["instrument", "frets"]
      }
    }
  }
]"#;

fn parse_chord_pitch_classes(query: &str) -> Option<Vec<u8>> {
    let q = query.trim().to_lowercase();
    // Extract root note
    let (root_pc, rest) = if let Some(r) = q.strip_prefix("c#").or_else(|| q.strip_prefix("db")) {
        (1, r)
    } else if let Some(r) = q.strip_prefix("d#").or_else(|| q.strip_prefix("eb")) {
        (3, r)
    } else if let Some(r) = q.strip_prefix("f#").or_else(|| q.strip_prefix("gb")) {
        (6, r)
    } else if let Some(r) = q.strip_prefix("g#").or_else(|| q.strip_prefix("ab")) {
        (8, r)
    } else if let Some(r) = q.strip_prefix("a#").or_else(|| q.strip_prefix("bb")) {
        (10, r)
    } else if let Some(r) = q.strip_prefix('c') {
        (0, r)
    } else if let Some(r) = q.strip_prefix('d') {
        (2, r)
    } else if let Some(r) = q.strip_prefix('e') {
        (4, r)
    } else if let Some(r) = q.strip_prefix('f') {
        (5, r)
    } else if let Some(r) = q.strip_prefix('g') {
        (7, r)
    } else if let Some(r) = q.strip_prefix('a') {
        (9, r)
    } else if let Some(r) = q.strip_prefix('b') {
        (11, r)
    } else {
        return None;
    };

    // Parse quality → interval set (semitones from root)
    let intervals: Vec<u8> =
        if rest.contains("maj7") || rest.contains("major7") || rest.contains("Δ") {
            vec![0, 4, 7, 11] // maj7
        } else if rest.contains("m7b5") || rest.contains("min7b5") || rest.contains("ø") {
            vec![0, 3, 6, 10] // half-dim
        } else if rest.contains("dim7") || rest.contains("°7") {
            vec![0, 3, 6, 9] // dim7
        } else if rest.contains("m7") || rest.contains("min7") || rest.contains("-7") {
            vec![0, 3, 7, 10] // min7
        } else if rest.contains("7") {
            vec![0, 4, 7, 10] // dom7
        } else if rest.contains("m") || rest.contains("min") || rest.contains("-") {
            vec![0, 3, 7] // minor
        } else if rest.contains("aug") || rest.contains("+") {
            vec![0, 4, 8] // aug
        } else if rest.contains("dim") || rest.contains("°") {
            vec![0, 3, 6] // dim
        } else if rest.contains("sus4") {
            vec![0, 5, 7]
        } else if rest.contains("sus2") {
            vec![0, 2, 7]
        } else {
            vec![0, 4, 7] // major
        };

    Some(intervals.iter().map(|i| (root_pc + i) % 12).collect())
}

fn execute_tool(name: &str, args: &serde_json::Value) -> String {
    match name {
        "search_voicings" => {
            let instrument = args
                .get("instrument")
                .and_then(|i| i.as_str())
                .unwrap_or("guitar");
            let query = args.get("query").and_then(|q| q.as_str()).unwrap_or("");
            let corpus_path = format!("state/voicings/{}-corpus.json", instrument);
            let content = match std::fs::read_to_string(&corpus_path) {
                Ok(c) => c,
                Err(_) => return format!("Corpus not found at {}", corpus_path),
            };
            let voicings: Vec<serde_json::Value> =
                serde_json::from_str(&content).unwrap_or_default();
            let query_lower = query.to_lowercase();

            // Try pitch-class matching first
            let target_pcs = parse_chord_pitch_classes(&query_lower);

            let matches: Vec<_> = voicings
                .iter()
                .filter(|v| {
                    let midi_notes = v.get("midiNotes").and_then(|m| m.as_array());
                    let diagram = v.get("diagram").and_then(|d| d.as_str()).unwrap_or("");
                    let fret_span = v.get("fretSpan").and_then(|f| f.as_i64()).unwrap_or(0);
                    let min_fret = v.get("minFret").and_then(|f| f.as_i64()).unwrap_or(0);

                    // Pitch-class match: voicing must contain ALL target pitch classes
                    if let (Some(ref target), Some(midi)) = (&target_pcs, &midi_notes) {
                        let voicing_pcs: std::collections::HashSet<u8> = midi
                            .iter()
                            .filter_map(|n| n.as_i64())
                            .map(|n| (n % 12) as u8)
                            .collect();
                        let has_all = target.iter().all(|pc| voicing_pcs.contains(pc));
                        let no_extras = voicing_pcs.len() <= target.len() + 1; // allow one doubled note
                        return has_all && no_extras;
                    }

                    // Fallback: keyword matching
                    if query_lower.is_empty() {
                        return true;
                    }
                    if query_lower.contains("open") && min_fret <= 1 && fret_span <= 3 {
                        return true;
                    }
                    if query_lower.contains("barre") && min_fret >= 2 {
                        return true;
                    }
                    if query_lower.contains("drop") && (2..=4).contains(&fret_span) {
                        return true;
                    }
                    diagram.to_lowercase().contains(&query_lower)
                })
                .take(15)
                .cloned()
                .collect();

            let result = serde_json::json!({
                "instrument": instrument,
                "query": query,
                "pitch_class_search": target_pcs.is_some(),
                "target_pitch_classes": target_pcs.as_ref().map(|pcs| pcs.iter().map(|p| {
                    ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][*p as usize]
                }).collect::<Vec<_>>()),
                "results_count": matches.len(),
                "total_in_corpus": voicings.len(),
                "voicings": matches
            });
            serde_json::to_string_pretty(&result).unwrap_or_default()
        }
        "get_instrument_info" => {
            let instrument = args
                .get("instrument")
                .and_then(|i| i.as_str())
                .unwrap_or("guitar");
            let info = match instrument {
                "guitar" => serde_json::json!({
                    "instrument": "guitar", "strings": 6, "tuning": "EADGBE",
                    "tuning_midi": [40, 45, 50, 55, 59, 64], "frets": 24,
                    "range_low": "E2 (MIDI 40)", "range_high": "E6 (MIDI 88 at 24th fret)"
                }),
                "bass" => serde_json::json!({
                    "instrument": "bass", "strings": 4, "tuning": "EADG",
                    "tuning_midi": [28, 33, 38, 43], "frets": 21,
                    "range_low": "E1 (MIDI 28)", "range_high": "Eb4 (MIDI 63 at 21st fret)",
                    "note": "Bass has 4 strings, NOT 6. Voicings must use 4 or fewer notes."
                }),
                "ukulele" => serde_json::json!({
                    "instrument": "ukulele", "strings": 4, "tuning": "GCEA",
                    "tuning_midi": [55, 48, 52, 57], "frets": 15,
                    "range_low": "C4 (MIDI 48)", "range_high": "A5 (MIDI 72 at 15th fret)",
                    "note": "Ukulele has 4 strings with re-entrant tuning (G string is higher than C)."
                }),
                _ => serde_json::json!({"error": "Unknown instrument"}),
            };
            serde_json::to_string_pretty(&info).unwrap_or_default()
        }
        "verify_voicing" => {
            let instrument = args
                .get("instrument")
                .and_then(|i| i.as_str())
                .unwrap_or("guitar");
            let frets_str = args.get("frets").and_then(|f| f.as_str()).unwrap_or("");
            let frets: Vec<&str> = frets_str.split('-').collect();
            let expected_strings = match instrument {
                "guitar" => 6,
                "bass" => 4,
                "ukulele" => 4,
                _ => 0,
            };
            let mut issues = Vec::new();
            if frets.len() != expected_strings {
                issues.push(format!(
                    "{} has {} strings but voicing has {} positions",
                    instrument,
                    expected_strings,
                    frets.len()
                ));
            }
            let played_frets: Vec<i32> = frets
                .iter()
                .filter(|f| **f != "x" && **f != "X")
                .filter_map(|f| f.parse().ok())
                .collect();
            if let (Some(&min), Some(&max)) = (played_frets.iter().min(), played_frets.iter().max())
            {
                let span = max - min;
                if span > 5 {
                    issues.push(format!(
                        "Fret span {} is likely unplayable (max comfortable span is ~4-5 frets)",
                        span
                    ));
                }
            }
            if played_frets.iter().any(|&f| f > 24) {
                issues.push("Fret number exceeds 24 (most instruments don't go that high)".into());
            }
            let result = if issues.is_empty() {
                serde_json::json!({"valid": true, "instrument": instrument, "frets": frets_str, "played_notes": played_frets.len()})
            } else {
                serde_json::json!({"valid": false, "instrument": instrument, "frets": frets_str, "issues": issues})
            };
            serde_json::to_string_pretty(&result).unwrap_or_default()
        }
        _ => format!("Unknown tool: {}", name),
    }
}

fn call_llm(question: &str, history: &[serde_json::Value]) -> String {
    let mut messages = vec![serde_json::json!({"role": "system", "content": SYSTEM_PROMPT})];
    for msg in history {
        if let (Some(role), Some(content)) = (
            msg.get("role").and_then(|r| r.as_str()),
            msg.get("content").and_then(|c| c.as_str()),
        ) {
            if role == "user" || role == "assistant" {
                messages.push(serde_json::json!({"role": role, "content": content}));
            }
        }
    }
    if messages.len() <= 1
        || messages
            .last()
            .and_then(|m| m.get("role").and_then(|r| r.as_str()))
            != Some("user")
    {
        messages.push(serde_json::json!({"role": "user", "content": question}));
    }

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap();

        let tools: serde_json::Value = serde_json::from_str(TOOLS).unwrap();

        // Tool-use loop: up to 5 rounds
        for _ in 0..5 {
            let model =
                std::env::var("GA_CHATBOT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
            let body = serde_json::json!({
                "model": model,
                "messages": messages,
                "tools": tools,
                "max_tokens": 2048,
            });

            let resp = client
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", openai_key))
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .send()
                .await;

            let json: serde_json::Value = match resp {
                Ok(r) => r.json().await.unwrap_or_default(),
                Err(e) => return format!("API error: {}", e),
            };

            if json.get("error").is_some() {
                eprintln!("[ga-chatbot] API error: {}", json);
                return json["error"]["message"]
                    .as_str()
                    .unwrap_or("API error")
                    .to_string();
            }

            let choice = &json["choices"][0]["message"];
            let finish_reason = json["choices"][0]["finish_reason"].as_str().unwrap_or("");

            if finish_reason == "tool_calls" {
                if let Some(tool_calls) = choice.get("tool_calls").and_then(|t| t.as_array()) {
                    // Add assistant message with tool calls
                    messages.push(choice.clone());

                    for tc in tool_calls {
                        let tool_name = tc["function"]["name"].as_str().unwrap_or("");
                        let tool_args: serde_json::Value = tc["function"]["arguments"]
                            .as_str()
                            .and_then(|s| serde_json::from_str(s).ok())
                            .unwrap_or_default();
                        let tool_id = tc["id"].as_str().unwrap_or("");

                        eprintln!("[ga-chatbot] Tool call: {}({})", tool_name, tool_args);
                        let result = execute_tool(tool_name, &tool_args);
                        eprintln!(
                            "[ga-chatbot] Tool result: {}...",
                            &result[..result.len().min(200)]
                        );

                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }));
                    }
                    continue; // Next loop iteration with tool results
                }
            }

            // Final text response
            return choice
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("No response.")
                .to_string();
        }

        "Tool use loop exceeded 5 rounds.".to_string()
    })
}

// ---------------------------------------------------------------------------
// Live mode — real MCP bridge tools
// ---------------------------------------------------------------------------

const LIVE_SYSTEM_PROMPT: &str = r#"You are a music theory assistant with access to real computation tools. Use ga__ tools for chord parsing, voicing search, and music theory. Use ix__ tools for structural analysis, clustering, topology, and voice leading. NEVER invent voicings — always call a tool to get real data."#;

/// Convert MCP tool descriptors into the OpenAI function-calling tools array.
fn mcp_tools_to_openai(tools: &[ga_chatbot::mcp_bridge::ToolDescriptor]) -> serde_json::Value {
    let funcs: Vec<serde_json::Value> = tools
        .iter()
        .filter(|t| {
            // Skip tools with invalid schemas (arrays missing "items" sub-schema)
            let schema_str = t.input_schema.to_string();
            !schema_str.contains(r#""type":"array"}"#)
        })
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": if t.input_schema.is_null() {
                        serde_json::json!({"type": "object", "properties": {}})
                    } else {
                        t.input_schema.clone()
                    },
                }
            })
        })
        .collect();
    serde_json::Value::Array(funcs)
}

/// Call the LLM with tool-use loop, dispatching tool calls through the MCP bridge.
///
/// Returns the final assistant text response.
fn call_llm_live(
    question: &str,
    history: &[serde_json::Value],
    bridge: &Arc<Mutex<McpBridge>>,
    openai_tools: &serde_json::Value,
) -> String {
    let mut messages = vec![serde_json::json!({"role": "system", "content": LIVE_SYSTEM_PROMPT})];
    for msg in history {
        if let (Some(role), Some(content)) = (
            msg.get("role").and_then(|r| r.as_str()),
            msg.get("content").and_then(|c| c.as_str()),
        ) {
            if role == "user" || role == "assistant" {
                messages.push(serde_json::json!({"role": role, "content": content}));
            }
        }
    }
    if messages.len() <= 1
        || messages
            .last()
            .and_then(|m| m.get("role").and_then(|r| r.as_str()))
            != Some("user")
    {
        messages.push(serde_json::json!({"role": "user", "content": question}));
    }

    let openai_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap();

        for round in 0..5 {
            let model =
                std::env::var("GA_CHATBOT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
            let body = serde_json::json!({
                "model": model,
                "messages": messages,
                "tools": openai_tools,
                "max_tokens": 2048,
            });

            let resp = client
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", openai_key))
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .send()
                .await;

            let json: serde_json::Value = match resp {
                Ok(r) => r.json().await.unwrap_or_default(),
                Err(e) => return format!("API error: {}", e),
            };

            if json.get("error").is_some() {
                eprintln!("[ga-chatbot-live] API error: {}", json);
                return json["error"]["message"]
                    .as_str()
                    .unwrap_or("API error")
                    .to_string();
            }

            let choice = &json["choices"][0]["message"];
            let finish_reason = json["choices"][0]["finish_reason"].as_str().unwrap_or("");

            if finish_reason == "tool_calls" {
                if let Some(tool_calls) = choice.get("tool_calls").and_then(|t| t.as_array()) {
                    messages.push(choice.clone());

                    for tc in tool_calls {
                        let tool_name = tc["function"]["name"].as_str().unwrap_or("");
                        let tool_args: serde_json::Value = tc["function"]["arguments"]
                            .as_str()
                            .and_then(|s| serde_json::from_str(s).ok())
                            .unwrap_or_default();
                        let tool_id = tc["id"].as_str().unwrap_or("");

                        eprintln!(
                            "[ga-chatbot-live] Round {} tool call: {}({})",
                            round, tool_name, tool_args
                        );

                        let result = {
                            let mut b = bridge.lock().unwrap();
                            match b.execute_tool(tool_name, tool_args) {
                                Ok(val) => {
                                    // MCP tools/call returns {content: [{type:"text",text:"..."}]}
                                    // Extract the text if possible, otherwise stringify
                                    val.get("content")
                                        .and_then(|c| c.as_array())
                                        .and_then(|arr| arr.first())
                                        .and_then(|item| item.get("text").and_then(|t| t.as_str()))
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| val.to_string())
                                }
                                Err(e) => format!("Tool error: {}", e),
                            }
                        };

                        eprintln!(
                            "[ga-chatbot-live] Tool result: {}...",
                            &result[..result.len().min(200)]
                        );

                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": result,
                        }));
                    }
                    continue;
                }
            }

            // Final text response
            return choice
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("No response.")
                .to_string();
        }

        "Tool use loop exceeded 5 rounds.".to_string()
    })
}

/// Live HTTP server backed by real GA + IX MCP tools.
///
/// Spawns the MCP bridge once at startup and shares it across all requests.
/// Endpoints mirror `serve_http` but route tool calls through the bridge.
fn serve_http_live(port: u16, config: &McpBridgeConfig) {
    use std::net::TcpListener;

    eprintln!("[ga-chatbot-live] Spawning MCP bridge...");
    eprintln!(
        "[ga-chatbot-live]   GA: {} {:?}",
        config.ga_command, config.ga_args
    );
    eprintln!(
        "[ga-chatbot-live]   IX: {} {:?}",
        config.ix_command, config.ix_args
    );

    let bridge = match McpBridge::new(config) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[ga-chatbot-live] Failed to spawn MCP bridge: {}", e);
            std::process::exit(1);
        }
    };

    let mut all_tools = bridge.merged_tools();
    // OpenAI function calling limit is 128 tools
    if all_tools.len() > 128 {
        all_tools.truncate(128);
    }
    let tool_count = all_tools.len();
    let openai_tools = mcp_tools_to_openai(&all_tools);
    eprintln!(
        "[ga-chatbot-live] Bridge ready — {} tools available",
        tool_count
    );

    let bridge = Arc::new(Mutex::new(bridge));

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind to {}: {}", addr, e);
        std::process::exit(1);
    });
    eprintln!("[ga-chatbot-live] HTTP server listening on http://{}", addr);

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Read the full request (headers + body). For large bodies we may need
        // multiple reads, but 16 KiB is sufficient for chat messages.
        let mut buf = [0u8; 16384];
        let n = match std::io::Read::read(&mut stream, &mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let request_str = String::from_utf8_lossy(&buf[..n]);

        let first_line = request_str.lines().next().unwrap_or("");
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        let (method, path) = if parts.len() >= 2 {
            (parts[0], parts[1])
        } else {
            ("GET", "/")
        };

        let (status, content_type, body) = match (method, path) {
            ("GET", "/api/chatbot/status") => {
                let tool_count = {
                    let b = bridge.lock().unwrap();
                    b.merged_tools().len()
                };
                let resp = serde_json::json!({
                    "isAvailable": true,
                    "message": format!("ix ga-chatbot (live mode, {} tools)", tool_count),
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "mode": "live",
                    "tool_count": tool_count,
                });
                ("200 OK", "application/json", resp.to_string())
            }
            ("GET", "/api/chatbot/tools") => {
                let tools = {
                    let b = bridge.lock().unwrap();
                    b.merged_tools()
                };
                let summary: Vec<serde_json::Value> = tools
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "name": t.name,
                            "description": t.description,
                        })
                    })
                    .collect();
                (
                    "200 OK",
                    "application/json",
                    serde_json::to_string_pretty(&summary).unwrap_or_default(),
                )
            }
            ("GET", "/api/chatbot/examples") => {
                let examples = serde_json::json!([
                    "Show me drop-2 voicings for Cmaj7 on guitar",
                    "What voicings work for a ii-V-I in C?",
                    "Compare Am7 voicings across guitar and ukulele",
                    "Find voicings with minimal finger movement from Dm7 to G7",
                    "What are the most common voicing families on bass?",
                    "Cluster these voicings by topology",
                    "What is the Betti number of the voicing graph?"
                ]);
                ("200 OK", "application/json", examples.to_string())
            }
            ("POST", p) if p.starts_with("/api/chatbot/chat") => {
                let body_start = request_str
                    .find("\r\n\r\n")
                    .map(|i| i + 4)
                    .or_else(|| request_str.find("\n\n").map(|i| i + 2))
                    .unwrap_or(n);
                let json_body = &request_str[body_start..];
                let parsed = serde_json::from_str::<serde_json::Value>(json_body).ok();

                let user_messages = parsed
                    .as_ref()
                    .and_then(|v| v.get("messages"))
                    .and_then(|m| m.as_array())
                    .cloned()
                    .unwrap_or_default();

                let question = parsed
                    .as_ref()
                    .and_then(|v| {
                        v.get("message")
                            .or(v.get("question"))
                            .and_then(|m| m.as_str().map(String::from))
                    })
                    .or_else(|| {
                        user_messages.last().and_then(|m| {
                            m.get("content").and_then(|c| c.as_str().map(String::from))
                        })
                    })
                    .unwrap_or_else(|| "Hello".to_string());

                let answer = call_llm_live(&question, &user_messages, &bridge, &openai_tools);

                let json_resp = serde_json::json!({
                    "response": answer,
                    "content": answer,
                    "mode": "live",
                });
                ("200 OK", "application/json", json_resp.to_string())
            }
            ("OPTIONS", _) => ("200 OK", "text/plain", String::new()),
            _ => ("404 Not Found", "text/plain", "Not found".to_string()),
        };

        let response = format!(
            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\n\r\n{}",
            status,
            content_type,
            body.len(),
            body
        );
        std::io::Write::write_all(&mut stream, response.as_bytes()).ok();
    }
}

/// Simple HTTP server for the ga-chatbot frontend integration.
///
/// Serves three endpoints matching what the React chatService.ts expects:
/// - GET /api/chatbot/status
/// - GET /api/chatbot/examples
/// - POST /api/chatbot/chat (non-streaming, returns full JSON)
/// - POST /api/chatbot/chat/stream (SSE, single data event)
fn serve_http(port: u16, _fixtures: &HashMap<String, ga_chatbot::ChatbotResponse>) {
    use std::net::TcpListener;

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind to {}: {}", addr, e);
        std::process::exit(1);
    });
    eprintln!("[ga-chatbot] HTTP server listening on http://{}", addr);

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };

        let mut buf = [0u8; 8192];
        let n = match std::io::Read::read(&mut stream, &mut buf) {
            Ok(n) => n,
            Err(_) => continue,
        };
        let request_str = String::from_utf8_lossy(&buf[..n]);

        let first_line = request_str.lines().next().unwrap_or("");
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        let (method, path) = if parts.len() >= 2 {
            (parts[0], parts[1])
        } else {
            ("GET", "/")
        };

        let (status, content_type, body) = match (method, path) {
            ("GET", "/api/chatbot/status") => {
                let resp = serde_json::json!({
                    "isAvailable": true,
                    "message": "ix ga-chatbot (stub mode)",
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                });
                ("200 OK", "application/json", resp.to_string())
            }
            ("GET", "/api/chatbot/examples") => {
                let examples = serde_json::json!([
                    "Show me drop-2 voicings for Cmaj7 on guitar",
                    "What voicings work for a ii-V-I in C?",
                    "Compare Am7 voicings across guitar and ukulele",
                    "Find voicings with minimal finger movement from Dm7 to G7",
                    "What are the most common voicing families on bass?"
                ]);
                ("200 OK", "application/json", examples.to_string())
            }
            ("POST", p) if p.starts_with("/api/chatbot/chat") => {
                let body_start = request_str
                    .find("\r\n\r\n")
                    .map(|i| i + 4)
                    .or_else(|| request_str.find("\n\n").map(|i| i + 2))
                    .unwrap_or(n);
                let json_body = &request_str[body_start..];
                let parsed = serde_json::from_str::<serde_json::Value>(json_body).ok();

                let user_messages = parsed
                    .as_ref()
                    .and_then(|v| v.get("messages"))
                    .and_then(|m| m.as_array())
                    .cloned()
                    .unwrap_or_default();

                let question = parsed
                    .as_ref()
                    .and_then(|v| {
                        v.get("message")
                            .or(v.get("question"))
                            .and_then(|m| m.as_str().map(String::from))
                    })
                    .or_else(|| {
                        user_messages.last().and_then(|m| {
                            m.get("content").and_then(|c| c.as_str().map(String::from))
                        })
                    })
                    .unwrap_or_else(|| "Hello".to_string());

                let answer = call_llm(&question, &user_messages);

                let json_resp = serde_json::json!({
                    "response": answer,
                    "content": answer,
                });
                ("200 OK", "application/json", json_resp.to_string())
            }
            ("OPTIONS", _) => ("200 OK", "text/plain", String::new()),
            _ => ("404 Not Found", "text/plain", "Not found".to_string()),
        };

        let response = format!(
            "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\n\r\n{}",
            status,
            content_type,
            body.len(),
            body
        );
        std::io::Write::write_all(&mut stream, response.as_bytes()).ok();
    }
}

/// Minimal JSON-RPC stdio loop implementing the `ga_chatbot_ask` tool.
///
/// Reads one JSON-RPC request per line from stdin, dispatches to `ask_stub`,
/// writes one JSON-RPC response per line to stdout. No async, no tokio.
fn serve_jsonrpc(fixtures: &HashMap<String, ga_chatbot::ChatbotResponse>) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {}", e)
                    }
                });
                writeln!(stdout, "{}", err_resp).ok();
                stdout.flush().ok();
                continue;
            }
        };

        let id = request
            .get("id")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");

        let response = match method {
            "ga_chatbot_ask" => {
                let params = request.get("params").cloned().unwrap_or_default();
                let question = params
                    .get("question")
                    .and_then(|q| q.as_str())
                    .unwrap_or("")
                    .to_string();
                let instrument = params
                    .get("instrument")
                    .and_then(|i| i.as_str())
                    .and_then(|i| match i {
                        "guitar" => Some(Instrument::Guitar),
                        "bass" => Some(Instrument::Bass),
                        "ukulele" => Some(Instrument::Ukulele),
                        _ => None,
                    });

                let req = ChatbotRequest {
                    question,
                    instrument,
                };
                let resp = ask_stub(&req, fixtures);
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": resp
                })
            }
            _ => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("Method not found: {}", method)
                    }
                })
            }
        };

        writeln!(stdout, "{}", response).ok();
        stdout.flush().ok();
    }
}
