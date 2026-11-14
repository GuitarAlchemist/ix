//! `ix beliefs <subverb>` — belief state management.
//!
//! Belief files live under `./state/beliefs/*.belief.json`. Each file holds
//! one `BeliefState` record. Phase 1 supports `show` (list) and `get <key>`;
//! `set` / `snapshot` / `diff` arrive in Week 5.

use crate::output::{self, Format};
use serde_json::{json, Value};
use std::path::Path;

fn beliefs_dir() -> String {
    std::env::var("IX_BELIEFS_DIR").unwrap_or_else(|_| "state/beliefs".to_string())
}

pub fn show(format: Format) -> Result<(), String> {
    let dir = beliefs_dir();
    let path = Path::new(&dir);

    let mut entries: Vec<Value> = Vec::new();
    if path.is_dir() {
        let read = std::fs::read_dir(path).map_err(|e| format!("reading {dir}: {e}"))?;
        for entry in read.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("json") {
                let name = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let text = std::fs::read_to_string(&p).unwrap_or_default();
                let value: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                entries.push(json!({
                    "key": name,
                    "path": p.display().to_string(),
                    "state": value,
                }));
            }
        }
    }

    let payload = json!({
        "count": entries.len(),
        "dir": dir,
        "beliefs": entries,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}

/// Write a belief state to `state/beliefs/<key>.belief.json`.
///
/// `truth_value` is a single-letter hexavalent symbol (`T`/`P`/`U`/`D`/`F`/`C`).
pub fn set(
    key: &str,
    proposition: &str,
    truth_value: &str,
    confidence: f64,
    format: Format,
) -> Result<(), String> {
    let hex = parse_hex(truth_value)?;
    let confidence = confidence.clamp(0.0, 1.0);

    let dir = beliefs_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("creating {dir}: {e}"))?;
    let file = format!("{dir}/{key}.belief.json");

    let belief = json!({
        "proposition": proposition,
        "truth_value": hex.symbol().to_string(),
        "confidence": confidence,
        "supporting": [],
        "contradicting": [],
        "updated_at": current_iso_timestamp(),
    });
    let text = serde_json::to_string_pretty(&belief).unwrap();
    std::fs::write(&file, &text).map_err(|e| format!("writing {file}: {e}"))?;

    output::emit(
        &json!({ "action": "set", "key": key, "path": file, "state": belief }),
        format,
    )
    .map_err(|e| format!("{e}"))?;
    Ok(())
}

/// Capture every belief file into `state/snapshots/{YYYY-MM-DD}-{desc}.snapshot.json`.
pub fn snapshot(description: &str, format: Format) -> Result<(), String> {
    let dir = beliefs_dir();
    let snapshot_dir =
        std::env::var("IX_SNAPSHOTS_DIR").unwrap_or_else(|_| "state/snapshots".to_string());
    std::fs::create_dir_all(&snapshot_dir).map_err(|e| format!("creating {snapshot_dir}: {e}"))?;

    let mut beliefs: Vec<Value> = Vec::new();
    let path = Path::new(&dir);
    if path.is_dir() {
        let read = std::fs::read_dir(path).map_err(|e| format!("reading {dir}: {e}"))?;
        for entry in read.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("json") {
                let text = std::fs::read_to_string(&p).unwrap_or_default();
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    beliefs.push(v);
                }
            }
        }
    }

    let ts = current_iso_timestamp();
    let date = ts.split('T').next().unwrap_or(&ts);
    let kebab: String = description
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let snapshot_file = format!("{snapshot_dir}/{date}-{kebab}.snapshot.json");

    let payload = json!({
        "timestamp": ts,
        "trigger": "manual",
        "description": description,
        "beliefs": beliefs,
        "count": beliefs.len(),
    });
    let text = serde_json::to_string_pretty(&payload).unwrap();
    std::fs::write(&snapshot_file, &text).map_err(|e| format!("writing {snapshot_file}: {e}"))?;

    output::emit(
        &json!({
            "action": "snapshot",
            "path": snapshot_file,
            "captured_beliefs": beliefs.len(),
        }),
        format,
    )
    .map_err(|e| format!("{e}"))?;
    Ok(())
}

fn parse_hex(s: &str) -> Result<ix_types::Hexavalent, String> {
    match s.trim() {
        "T" | "t" | "True" | "true" => Ok(ix_types::Hexavalent::True),
        "P" | "p" | "Probable" | "probable" => Ok(ix_types::Hexavalent::Probable),
        "U" | "u" | "Unknown" | "unknown" => Ok(ix_types::Hexavalent::Unknown),
        "D" | "d" | "Doubtful" | "doubtful" => Ok(ix_types::Hexavalent::Doubtful),
        "F" | "f" | "False" | "false" => Ok(ix_types::Hexavalent::False),
        "C" | "c" | "Contradictory" | "contradictory" => Ok(ix_types::Hexavalent::Contradictory),
        other => Err(format!(
            "invalid hexavalent value '{other}' — expected one of T/P/U/D/F/C"
        )),
    }
}

/// Minimal ISO-8601 timestamp without external deps — uses SystemTime +
/// integer arithmetic. Format: `YYYY-MM-DDTHH:MM:SSZ`.
fn current_iso_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since epoch → date (proleptic Gregorian, epoch = 1970-01-01).
    let days = (secs / 86_400) as i64;
    let (year, month, day) = days_to_ymd(days);
    let h = (secs % 86_400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: i64) -> (i64, u32, u32) {
    // Civil-from-days algorithm (Howard Hinnant).
    days += 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = (days - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

pub fn get(key: &str, format: Format) -> Result<(), String> {
    let dir = beliefs_dir();
    let file = format!("{dir}/{key}.belief.json");
    let p = Path::new(&file);
    if !p.is_file() {
        return Err(format!("belief not found: {file}"));
    }
    let text = std::fs::read_to_string(p).map_err(|e| format!("reading {file}: {e}"))?;
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("parsing {file}: {e}"))?;
    output::emit(&value, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}
