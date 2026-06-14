//! Unified quality gate ledger (v1).
//!
//! One JSONL file per repo at `state/quality/gate-ledger.jsonl`, append-only.
//! Each line is either a **v1 entry** (this module's [`GateLedgerEntry`]) or a
//! **legacy v0 row** (the chatbot-PR shape from `ga/docs/schemas/gate-ledger.schema.json`).
//!
//! Producers should write v1. Consumers MUST be able to read both — see
//! [`read_ledger`], which yields [`LedgerLine::V1`] / [`LedgerLine::LegacyV0`].
//!
//! See `docs/contracts/2026-05-24-quality-gate-ledger.contract.md` for the
//! cross-repo schema, field reference, and rationale.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

/// A v1 quality-gate ledger entry. See the contract for field semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateLedgerEntry {
    pub schema_version: u32,
    pub schema: String,
    pub id: String,
    pub run_at: DateTime<Utc>,
    pub source: String,
    pub domain: String,
    pub decision: GateDecision,
    pub metric: GateMetric,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<GateEvidence>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supersedes: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator_ack: Option<OperatorAck>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// Decision categories. `Skip` covers degraded environments (e.g., backend down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateDecision {
    Pass,
    Fail,
    Warn,
    Skip,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateMetric {
    pub name: String,
    pub value: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trend: Option<MetricTrendDir>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricTrendDir {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Evidence pointer. The JSON field is `ref` (a Rust keyword), so we store
/// it as `ref_` and rename on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateEvidence {
    pub kind: EvidenceKind,
    #[serde(rename = "ref")]
    pub ref_: String,
}

impl GateEvidence {
    pub fn new(kind: EvidenceKind, ref_: impl Into<String>) -> Self {
        Self {
            kind,
            ref_: ref_.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceKind {
    Url,
    File,
    Sha,
    RunId,
    Pr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorAck {
    pub by: String,
    pub at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl GateLedgerEntry {
    /// Construct a new v1 entry with a freshly-minted UUID v7 (sortable) and
    /// the current UTC timestamp. Callers can mutate the result before
    /// appending.
    pub fn new(
        source: impl Into<String>,
        domain: impl Into<String>,
        decision: GateDecision,
        metric: GateMetric,
    ) -> Self {
        Self {
            schema_version: 1,
            schema: "quality-gate-ledger-v1".to_string(),
            id: Uuid::now_v7().to_string(),
            run_at: Utc::now(),
            source: source.into(),
            domain: domain.into(),
            decision,
            metric,
            evidence: None,
            supersedes: Vec::new(),
            operator_ack: None,
            extra: None,
        }
    }
}

/// A line read from the ledger. Old PR-shaped rows are returned as raw JSON
/// in [`LedgerLine::LegacyV0`] so callers can fold them in without losing data.
///
/// `V1` is boxed because [`GateLedgerEntry`] is significantly larger than a
/// `serde_json::Value` and we want a tight enum payload.
#[derive(Debug, Clone)]
pub enum LedgerLine {
    V1(Box<GateLedgerEntry>),
    LegacyV0(Value),
}

#[derive(Debug, Error)]
pub enum LedgerError {
    #[error("i/o error on {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("json parse error on {path} line {line}: {source}")]
    Json {
        path: std::path::PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
}

/// Append a v1 entry as one JSON line. Creates the parent directory and file
/// as needed. Atomic-per-line on POSIX/NTFS append semantics; safe for
/// concurrent producers within a single host.
pub fn append_entry(path: &Path, entry: &GateLedgerEntry) -> Result<(), LedgerError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| LedgerError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| LedgerError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

    let line = serde_json::to_string(entry).map_err(|e| LedgerError::Json {
        path: path.to_path_buf(),
        line: 0,
        source: e,
    })?;
    writeln!(file, "{}", line).map_err(|e| LedgerError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

/// Stream-read the ledger. Lines with `schema_version == 1` parse as v1;
/// everything else is returned as legacy v0 raw JSON. Blank lines are skipped.
///
/// Returns an error only on i/o failure or hard JSON-parse failure (a line
/// that isn't even valid JSON). v1 schema validation errors fall back to
/// [`LedgerLine::LegacyV0`] (so a partially-shaped v1 line doesn't break the
/// reader).
pub fn read_ledger(path: &Path) -> Result<Vec<LedgerLine>, LedgerError> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(LedgerError::Io {
                path: path.to_path_buf(),
                source: e,
            })
        }
    };

    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| LedgerError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value: Value = serde_json::from_str(trimmed).map_err(|e| LedgerError::Json {
            path: path.to_path_buf(),
            line: i + 1,
            source: e,
        })?;

        let is_v1 = value
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .map(|n| n == 1)
            .unwrap_or(false);

        if is_v1 {
            match serde_json::from_value::<GateLedgerEntry>(value.clone()) {
                Ok(entry) => out.push(LedgerLine::V1(Box::new(entry))),
                Err(_) => out.push(LedgerLine::LegacyV0(value)),
            }
        } else {
            out.push(LedgerLine::LegacyV0(value));
        }
    }
    Ok(out)
}

/// Filter for [`query`]. All fields are AND-combined; `None` means "no filter".
#[derive(Debug, Default, Clone)]
pub struct LedgerQuery {
    pub source: Option<String>,
    pub domain: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub decision: Option<GateDecision>,
    pub limit: Option<usize>,
}

/// Read + filter v1 entries (legacy v0 rows are excluded — query that
/// substrate via [`read_ledger`] if you need it). Sorted most-recent-first
/// by `run_at`, then `limit` applied.
pub fn query(path: &Path, q: &LedgerQuery) -> Result<Vec<GateLedgerEntry>, LedgerError> {
    let lines = read_ledger(path)?;
    let mut matches: Vec<GateLedgerEntry> = lines
        .into_iter()
        .filter_map(|l| match l {
            LedgerLine::V1(e) => Some(*e),
            LedgerLine::LegacyV0(_) => None,
        })
        .filter(|e| match q.source.as_deref() {
            Some(s) => e.source == s,
            None => true,
        })
        .filter(|e| match q.domain.as_deref() {
            Some(d) => e.domain == d,
            None => true,
        })
        .filter(|e| match q.since {
            Some(t) => e.run_at >= t,
            None => true,
        })
        .filter(|e| match q.decision {
            Some(d) => e.decision == d,
            None => true,
        })
        .collect();

    matches.sort_by_key(|e| std::cmp::Reverse(e.run_at));
    if let Some(n) = q.limit {
        matches.truncate(n);
    }
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample() -> GateLedgerEntry {
        let mut e = GateLedgerEntry::new(
            "ix-quality-trend",
            "structural",
            GateDecision::Pass,
            GateMetric {
                name: "quality_signal".to_string(),
                value: 3015.0,
                threshold: Some(2500.0),
                trend: Some(MetricTrendDir::Improving),
            },
        );
        e.evidence = Some(GateEvidence::new(
            EvidenceKind::File,
            "state/quality/embeddings/2026-05-24.json",
        ));
        e
    }

    #[test]
    fn round_trip_v1_entry() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gate-ledger.jsonl");

        let entry = sample();
        append_entry(&path, &entry).unwrap();

        let lines = read_ledger(&path).unwrap();
        assert_eq!(lines.len(), 1);
        match &lines[0] {
            LedgerLine::V1(got) => {
                let got = got.as_ref();
                assert_eq!(got.id, entry.id);
                assert_eq!(got.source, "ix-quality-trend");
                assert_eq!(got.metric.value, 3015.0);
                assert_eq!(got.metric.threshold, Some(2500.0));
                assert_eq!(got.decision, GateDecision::Pass);
                let ev = got.evidence.as_ref().unwrap();
                assert_eq!(ev.kind, EvidenceKind::File);
                assert!(ev.ref_.ends_with("2026-05-24.json"));
            }
            _ => panic!("expected v1 line"),
        }
    }

    #[test]
    fn evidence_serializes_with_ref_key() {
        let entry = sample();
        let s = serde_json::to_string(&entry).unwrap();
        assert!(s.contains("\"ref\":\"state/quality/embeddings/2026-05-24.json\""));
        assert!(!s.contains("\"ref_\""));
    }

    #[test]
    fn coexists_with_legacy_v0_rows() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gate-ledger.jsonl");

        // Legacy v0 (chatbot PR row — no schema_version).
        let legacy = r#"{"pr":155,"branch":"chatbot/x","mergedAt":"2026-05-11T00:32:05Z","gates":{},"decision":"merged-clean"}"#;
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, format!("{}\n", legacy)).unwrap();

        // Then a v1 entry.
        let entry = sample();
        append_entry(&path, &entry).unwrap();

        let lines = read_ledger(&path).unwrap();
        assert_eq!(lines.len(), 2);
        assert!(matches!(lines[0], LedgerLine::LegacyV0(_)));
        assert!(matches!(lines[1], LedgerLine::V1(_)));
    }

    #[test]
    fn query_filters_by_source_and_domain() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gate-ledger.jsonl");

        let mut a = GateLedgerEntry::new(
            "sentrux",
            "structural",
            GateDecision::Pass,
            GateMetric {
                name: "complexity".to_string(),
                value: 0.42,
                threshold: None,
                trend: None,
            },
        );
        a.run_at = "2026-05-20T10:00:00Z".parse().unwrap();

        let mut b = GateLedgerEntry::new(
            "chatbot-qa",
            "chatbot",
            GateDecision::Fail,
            GateMetric {
                name: "pass_pct".to_string(),
                value: 70.0,
                threshold: Some(90.0),
                trend: Some(MetricTrendDir::Degrading),
            },
        );
        b.run_at = "2026-05-23T10:00:00Z".parse().unwrap();

        append_entry(&path, &a).unwrap();
        append_entry(&path, &b).unwrap();

        let only_sentrux = query(
            &path,
            &LedgerQuery {
                source: Some("sentrux".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(only_sentrux.len(), 1);
        assert_eq!(only_sentrux[0].source, "sentrux");

        let only_chatbot_domain = query(
            &path,
            &LedgerQuery {
                domain: Some("chatbot".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(only_chatbot_domain.len(), 1);
        assert_eq!(only_chatbot_domain[0].domain, "chatbot");

        let since = query(
            &path,
            &LedgerQuery {
                since: Some("2026-05-22T00:00:00Z".parse().unwrap()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(since.len(), 1);
        assert_eq!(since[0].source, "chatbot-qa");
    }

    #[test]
    fn query_sorts_descending_and_applies_limit() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gate-ledger.jsonl");

        for i in 0..5 {
            let mut e = GateLedgerEntry::new(
                "ix-quality-trend",
                "structural",
                GateDecision::Pass,
                GateMetric {
                    name: "x".to_string(),
                    value: i as f64,
                    threshold: None,
                    trend: None,
                },
            );
            e.run_at = format!("2026-05-2{}T10:00:00Z", i).parse().unwrap();
            append_entry(&path, &e).unwrap();
        }

        let got = query(
            &path,
            &LedgerQuery {
                limit: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(got.len(), 3);
        // Newest first.
        assert_eq!(got[0].metric.value, 4.0);
        assert_eq!(got[1].metric.value, 3.0);
        assert_eq!(got[2].metric.value, 2.0);
    }

    #[test]
    fn read_returns_empty_for_missing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gate-ledger.jsonl");
        let lines = read_ledger(&path).unwrap();
        assert!(lines.is_empty());
    }
}
