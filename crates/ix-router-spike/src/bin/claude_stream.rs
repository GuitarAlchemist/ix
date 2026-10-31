//! Offline classifier for Claude Code `--output-format stream-json` stdout (#484).
//!
//! Throwaway, like `jev-router`: it answers #484's smallest discriminating check on
//! hand-written fixtures built from the shapes documented for Claude Code 2.1.314
//! (docs/research/agent-composition/2026-10-30-484-claude-code-programmatic-evidence.md).
//! The fixtures are synthetic until a separately admitted live run records a real stream.
//! No process is spawned and no model is called.
//!
//! What a caller can prove from one process's stdout: which of the turns it sent ended on a
//! `result` line of its own session, in order. Request acceptance and end of input have no
//! documented signal, so nothing here claims them.
//!
//! Run: cargo run --manifest-path crates/ix-router-spike/Cargo.toml --bin claude-stream --
//!      <stdout.jsonl> --session <id> --turns <n> [--exit <code>]

use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

/// The outcome of one turn the caller sent. Only `Success` is a success.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum Turn {
    /// A `result` line of this session with `subtype: "success"` and `is_error: false`.
    /// Denied permissions are reported, not judged: a CLI success is not business completion.
    Success {
        result: String,
        permission_denials: usize,
    },
    /// A `result` line of this session that is not a success (any other subtype, or
    /// `is_error` not exactly false).
    ResultError { subtype: String },
    /// A line of the turn did not parse.
    Malformed,
    /// The stream ended inside a line (no final newline).
    Truncated,
    /// A line or the stream exceeded `Limits`.
    LimitExceeded,
    /// The process exited non-zero before the turn's result (143 = SIGTERM).
    Terminated { exit: i32 },
    /// The stream ended cleanly, or is still open, without the turn's result.
    NoResult,
}

struct Expect<'a> {
    /// The session of this process incarnation: lines of any other session are stale.
    session_id: &'a str,
    /// How many turns the caller sent; results are matched to them in order.
    turns: usize,
}

struct Limits {
    max_line_bytes: usize,
    max_lines: usize,
    max_diagnostics: usize,
    max_diagnostic_bytes: usize,
}

const LIMITS: Limits = Limits {
    max_line_bytes: 1 << 20,
    max_lines: 100_000,
    max_diagnostics: 8,
    max_diagnostic_bytes: 200,
};

#[derive(Debug, Default, Serialize)]
struct Report {
    turns: Vec<Turn>,
    events: BTreeMap<&'static str, usize>,
    diagnostics: Vec<String>,
}

/// Classify one process's stdout. `exit` is the process exit code, `None` while it runs.
fn classify(stdout: &str, exit: Option<i32>, expect: &Expect, limits: &Limits) -> Report {
    let mut r = Report::default();
    // The first problem seen in the open turn; it outranks that turn's result.
    let mut open_fault: Option<Turn> = None;
    for (i, raw) in stdout.split_inclusive('\n').enumerate() {
        if i >= limits.max_lines {
            r.note(limits, "line limit reached; rest of the stream unread");
            open_fault.get_or_insert(Turn::LimitExceeded);
            break;
        }
        let line = raw.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            continue;
        }
        // A JSONL writer ends every record with a newline: a last line without one was cut,
        // even when what arrived happens to parse.
        if !raw.ends_with('\n') {
            r.bump("truncated");
            r.note(limits, line);
            open_fault = Some(Turn::Truncated);
            break;
        }
        if line.len() > limits.max_line_bytes {
            r.bump("over_limit");
            r.note(limits, "line over the byte limit");
            open_fault.get_or_insert(Turn::LimitExceeded);
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            r.bump("malformed");
            r.note(limits, line);
            open_fault.get_or_insert(Turn::Malformed);
            continue;
        };
        if v["session_id"]
            .as_str()
            .is_some_and(|s| s != expect.session_id)
        {
            r.bump("stale");
            continue;
        }
        let class = match (v["type"].as_str(), v["subtype"].as_str()) {
            (Some("result"), _) => {
                r.bump("result");
                if r.turns.len() >= expect.turns {
                    r.bump("unexpected_result");
                } else {
                    let outcome = open_fault.take().unwrap_or_else(|| result_outcome(&v));
                    r.turns.push(outcome);
                }
                continue;
            }
            (Some("system"), Some("init")) => "init",
            (Some("system"), Some("api_retry")) => "retry",
            (Some("system"), Some("permission_denied")) => "permission_denied",
            (Some("stream_event"), _) if v["event"]["delta"]["type"] == "text_delta" => {
                "output_delta"
            }
            (Some("stream_event"), _) => "stream_event",
            (Some("assistant" | "user"), _) if has_tool_block(&v) => "tool_activity",
            (Some("assistant" | "user"), _) => "message",
            _ => "unknown",
        };
        r.bump(class);
    }
    let unanswered = || match exit {
        Some(code) if code != 0 => Turn::Terminated { exit: code },
        _ => Turn::NoResult,
    };
    if r.turns.len() < expect.turns {
        let first = open_fault.unwrap_or_else(unanswered);
        r.turns.push(first);
    }
    while r.turns.len() < expect.turns {
        r.turns.push(unanswered());
    }
    r
}

fn result_outcome(v: &Value) -> Turn {
    let subtype = v["subtype"].as_str().unwrap_or("<missing>");
    if subtype == "success" && v["is_error"] == Value::Bool(false) {
        Turn::Success {
            result: v["result"].as_str().unwrap_or_default().to_string(),
            permission_denials: v["permission_denials"].as_array().map_or(0, Vec::len),
        }
    } else {
        Turn::ResultError {
            subtype: subtype.to_string(),
        }
    }
}

fn has_tool_block(v: &Value) -> bool {
    v["message"]["content"].as_array().is_some_and(|blocks| {
        blocks
            .iter()
            .any(|b| matches!(b["type"].as_str(), Some("tool_use" | "tool_result")))
    })
}

impl Report {
    fn bump(&mut self, class: &'static str) {
        *self.events.entry(class).or_default() += 1;
    }

    /// Keep a bounded sample of what went wrong: at most `max_diagnostics` notes, each cut to
    /// `max_diagnostic_bytes` on a char boundary; the rest is only counted.
    fn note(&mut self, limits: &Limits, text: &str) {
        if self.diagnostics.len() >= limits.max_diagnostics {
            self.bump("diagnostics_dropped");
            return;
        }
        let mut end = text.len().min(limits.max_diagnostic_bytes);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.diagnostics.push(text[..end].to_string());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let (Some(path), Some(session), Some(turns)) =
        (args.first(), flag("--session"), flag("--turns"))
    else {
        eprintln!("usage: claude-stream <stdout.jsonl> --session <id> --turns <n> [--exit <code>]");
        std::process::exit(2);
    };
    let stdout = std::fs::read_to_string(path).expect("read stdout file");
    let expect = Expect {
        session_id: &session,
        turns: turns.parse().expect("--turns is a number"),
    };
    let exit = flag("--exit").map(|c| c.parse().expect("--exit is a number"));
    let report = classify(&stdout, exit, &expect, &LIMITS);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize")
    );
}

#[cfg(test)]
mod tests {
    //! Fixtures are synthetic, written from the documented shapes; none is a recorded stream.
    use super::*;
    use serde_json::json;

    const S: &str = "sess-b";

    fn jsonl(lines: &[Value]) -> String {
        lines.iter().map(|v| format!("{v}\n")).collect()
    }

    fn init(session: &str) -> Value {
        json!({"type": "system", "subtype": "init", "session_id": session, "capabilities": []})
    }

    fn ok(session: &str, text: &str) -> Value {
        json!({"type": "result", "subtype": "success", "is_error": false, "result": text,
               "session_id": session, "permission_denials": []})
    }

    fn one(session: &str) -> Expect<'_> {
        Expect {
            session_id: session,
            turns: 1,
        }
    }

    fn success(text: &str) -> Turn {
        Turn::Success {
            result: text.into(),
            permission_denials: 0,
        }
    }

    #[test]
    fn test_valid_turn_is_success_and_its_events_are_classified() {
        let out = jsonl(&[
            init(S),
            json!({"type": "stream_event", "session_id": S,
                   "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}}),
            json!({"type": "assistant", "session_id": S, "parent_tool_use_id": null,
                   "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}}),
            json!({"type": "user", "session_id": S, "parent_tool_use_id": null,
                   "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}}),
            json!({"type": "system", "subtype": "api_retry", "session_id": S,
                   "attempt": 1, "max_retries": 3, "error": "overloaded"}),
            ok(S, "done"),
        ]);
        let r = classify(&out, Some(0), &one(S), &LIMITS);
        assert_eq!(r.turns, vec![success("done")]);
        for (class, n) in [
            ("init", 1),
            ("output_delta", 1),
            ("tool_activity", 2),
            ("retry", 1),
            ("result", 1),
        ] {
            assert_eq!(r.events.get(class), Some(&n), "{class}");
        }
    }

    #[test]
    fn test_two_sequential_turns_close_in_order_on_their_results() {
        let out = jsonl(&[init(S), ok(S, "first"), ok(S, "second")]);
        let expect = Expect {
            session_id: S,
            turns: 2,
        };
        let r = classify(&out, Some(0), &expect, &LIMITS);
        assert_eq!(r.turns, vec![success("first"), success("second")]);
    }

    #[test]
    fn test_result_beyond_the_turns_sent_is_unexpected_not_a_success() {
        let out = jsonl(&[init(S), ok(S, "first"), ok(S, "extra")]);
        let r = classify(&out, Some(0), &one(S), &LIMITS);
        assert_eq!(r.turns, vec![success("first")]);
        assert_eq!(r.events.get("unexpected_result"), Some(&1));
    }

    #[test]
    fn test_late_result_from_an_old_incarnation_is_stale_not_success() {
        let out = jsonl(&[init("sess-a"), ok("sess-a", "late"), init(S)]);
        let r = classify(&out, Some(0), &one(S), &LIMITS);
        assert_eq!(r.turns, vec![Turn::NoResult]);
        assert_eq!(r.events.get("stale"), Some(&2));
    }

    #[test]
    fn test_malformed_line_voids_its_turn_even_before_a_success_result() {
        let out = format!(
            "{}{{\"type\": \"assistant\", \n{}",
            jsonl(&[init(S)]),
            jsonl(&[ok(S, "x")])
        );
        let r = classify(&out, Some(0), &one(S), &LIMITS);
        assert_eq!(r.turns, vec![Turn::Malformed]);
        assert_eq!(r.events.get("malformed"), Some(&1));
    }

    #[test]
    fn test_last_line_without_newline_is_truncated_even_when_it_parses() {
        let cut = format!(
            "{}{{\"type\":\"result\",\"subtype\":\"succ",
            jsonl(&[init(S)])
        );
        assert_eq!(
            classify(&cut, Some(0), &one(S), &LIMITS).turns,
            vec![Turn::Truncated]
        );
        let unterminated = format!("{}{}", jsonl(&[init(S)]), ok(S, "x"));
        assert_eq!(
            classify(&unterminated, Some(0), &one(S), &LIMITS).turns,
            vec![Turn::Truncated]
        );
    }

    #[test]
    fn test_unknown_type_is_counted_and_never_a_success() {
        let out = jsonl(&[init(S), json!({"type": "future_event", "session_id": S})]);
        let r = classify(&out, Some(0), &one(S), &LIMITS);
        assert_eq!(r.turns, vec![Turn::NoResult]);
        assert_eq!(r.events.get("unknown"), Some(&1));
    }

    #[test]
    fn test_result_that_is_not_a_clean_success_is_a_result_error() {
        let cases = [
            (
                json!({"type": "result", "subtype": "error_max_turns", "is_error": true, "session_id": S}),
                "error_max_turns",
            ),
            (
                json!({"type": "result", "subtype": "success", "is_error": true, "session_id": S}),
                "success",
            ),
            (
                json!({"type": "result", "subtype": "success", "session_id": S}),
                "success",
            ),
        ];
        for (result, subtype) in cases {
            let r = classify(
                &jsonl(&[init(S), result.clone()]),
                Some(0),
                &one(S),
                &LIMITS,
            );
            assert_eq!(
                r.turns,
                vec![Turn::ResultError {
                    subtype: subtype.into()
                }],
                "{result}"
            );
        }
    }

    #[test]
    fn test_permission_denial_is_reported_with_the_result() {
        let out = jsonl(&[
            init(S),
            json!({"type": "system", "subtype": "permission_denied", "session_id": S, "tool_name": "Bash"}),
            json!({"type": "result", "subtype": "success", "is_error": false, "result": "partial",
                   "session_id": S, "permission_denials": [{"tool_name": "Bash"}]}),
        ]);
        let r = classify(&out, Some(0), &one(S), &LIMITS);
        assert_eq!(
            r.turns,
            vec![Turn::Success {
                result: "partial".into(),
                permission_denials: 1
            }]
        );
        assert_eq!(r.events.get("permission_denied"), Some(&1));
    }

    #[test]
    fn test_exit_without_result_is_terminated_or_no_result() {
        let out = jsonl(&[init(S)]);
        assert_eq!(
            classify(&out, Some(143), &one(S), &LIMITS).turns,
            vec![Turn::Terminated { exit: 143 }]
        );
        assert_eq!(
            classify(&out, Some(0), &one(S), &LIMITS).turns,
            vec![Turn::NoResult]
        );
        assert_eq!(
            classify(&out, None, &one(S), &LIMITS).turns,
            vec![Turn::NoResult]
        );
    }

    #[test]
    fn test_answered_turn_survives_a_later_kill() {
        let out = jsonl(&[init(S), ok(S, "first")]);
        let expect = Expect {
            session_id: S,
            turns: 2,
        };
        let r = classify(&out, Some(143), &expect, &LIMITS);
        assert_eq!(
            r.turns,
            vec![success("first"), Turn::Terminated { exit: 143 }]
        );
    }

    #[test]
    fn test_limits_void_the_turn_and_bound_the_diagnostics() {
        let small = Limits {
            max_line_bytes: 64,
            max_lines: 1000,
            max_diagnostics: 2,
            max_diagnostic_bytes: 20,
        };
        let long = json!({"type": "assistant", "session_id": S, "pad": "x".repeat(100)});
        let r = classify(
            &jsonl(&[init(S), long, ok(S, "x")]),
            Some(0),
            &one(S),
            &small,
        );
        assert_eq!(r.turns, vec![Turn::LimitExceeded]);

        let few = Limits {
            max_lines: 2,
            ..small
        };
        let r = classify(
            &jsonl(&[init(S), init(S), ok(S, "x")]),
            Some(0),
            &one(S),
            &few,
        );
        assert_eq!(r.turns, vec![Turn::LimitExceeded]);

        let noisy = "not json é\n".repeat(10) + &jsonl(&[ok(S, "x")]);
        let wide_lines = Limits {
            max_line_bytes: 1 << 20,
            ..small
        };
        let r = classify(&noisy, Some(0), &one(S), &wide_lines);
        assert_eq!(r.turns, vec![Turn::Malformed]);
        assert_eq!(r.diagnostics.len(), 2);
        assert!(
            r.diagnostics.iter().all(|d| d.len() <= 20),
            "{:?}",
            r.diagnostics
        );
        assert_eq!(r.events.get("diagnostics_dropped"), Some(&8));
    }
}
