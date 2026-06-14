//! The GA↔IX acoustic-tune JSON contract (`urn:ix:acoustic-tune:v0.1-draft`).
//!
//! IX writes a [`TuneRequest`] (a CMA-ES generation of candidate param vectors to
//! render); GA renders + scores each and writes a [`TuneResult`]; IX ingests it.
//! Transport is JSON-on-disk (the canonical GA↔IX pattern). Signed off
//! 2026-06-07 — see `docs/plans/2026-06-07-ix-acoustic-tune.md`. v0.1.x is a
//! DRAFT; freeze only at the named Phase-4 milestone (use `links.supersedes` for
//! any later baseline shift).

use serde::{Deserialize, Serialize};

/// `$schema` value for a request document.
pub const REQUEST_SCHEMA: &str = "urn:ix:acoustic-tune:v0.1-draft/request";
/// `$schema` value for a result document.
pub const RESULT_SCHEMA: &str = "urn:ix:acoustic-tune:v0.1-draft/result";

/// IX → GA: a generation of candidate parameter vectors to render and score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneRequest {
    pub schema: String,
    pub session_id: String,
    pub iteration: usize,
    /// Optimizer id (e.g. `"cmaes"`) — informational for GA.
    pub optimizer: String,
    /// Index→name for the `params` vectors (the locked actuator order).
    pub param_names: Vec<String>,
    pub candidates: Vec<Candidate>,
    /// Categorical guitar model (outer-enumerated; fixed for this session).
    pub guitar_type: i64,
    pub render: RenderSpec,
}

/// One candidate parameter vector to render.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub params: Vec<f64>,
}

/// How GA should render each candidate (the eval conditions IX fixes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderSpec {
    pub note_hz: f64,
    pub velocity: f64,
    pub seconds: f64,
    pub sample_rate: u32,
}

/// GA → IX: the score (and optional diagnostics) for each rendered candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneResult {
    pub schema: String,
    pub session_id: String,
    pub iteration: usize,
    /// Reference recording the candidates were scored against.
    #[serde(default)]
    pub reference: Option<String>,
    pub scores: Vec<CandidateScore>,
}

/// The outcome for one candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateScore {
    pub id: String,
    pub params: Vec<f64>,
    /// Phase A metric: GA's spectral match score, higher = closer (IX minimizes
    /// `1 - spectral_score`).
    pub spectral_score: f64,
    /// Phase B (optional): the descriptor vector IX can recompute its own loss
    /// from. Kept opaque (`Value`) so the schema spans A→B without a break.
    #[serde(default)]
    pub features: Option<serde_json::Value>,
    /// Guardrails — a clipped or silent render is a failed evaluation.
    #[serde(default)]
    pub guardrail: Option<Guardrail>,
}

/// Per-render sanity guardrails (the paired instrument to the metric).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guardrail {
    #[serde(default)]
    pub clipped: bool,
    #[serde(default)]
    pub silent: bool,
    #[serde(default)]
    pub rms: f64,
}

impl Guardrail {
    /// A render that clipped or fell silent is unusable regardless of its score.
    pub fn failed(&self) -> bool {
        self.clipped || self.silent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_through_json() {
        let req = TuneRequest {
            schema: REQUEST_SCHEMA.to_string(),
            session_id: "s".into(),
            iteration: 3,
            optimizer: "cmaes".into(),
            param_names: vec!["decay".into(), "brightness".into()],
            candidates: vec![Candidate {
                id: "c0".into(),
                params: vec![0.99, 0.4],
            }],
            guitar_type: 0,
            render: RenderSpec {
                note_hz: 110.0,
                velocity: 0.9,
                seconds: 4.0,
                sample_rate: 48000,
            },
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: TuneRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.candidates[0].params, vec![0.99, 0.4]);
        assert_eq!(back.iteration, 3);
    }

    #[test]
    fn result_tolerates_missing_optional_fields() {
        // Minimal GA output: no reference, no features, no guardrail.
        let json = r#"{
            "schema": "urn:ix:acoustic-tune:v0.1-draft/result",
            "session_id": "s", "iteration": 0,
            "scores": [ { "id": "c0", "params": [0.99, 0.4], "spectral_score": 0.7 } ]
        }"#;
        let res: TuneResult = serde_json::from_str(json).unwrap();
        assert_eq!(res.scores.len(), 1);
        assert!(res.scores[0].features.is_none());
        assert!(res.scores[0].guardrail.is_none());
        assert!((res.scores[0].spectral_score - 0.7).abs() < 1e-12);
    }
}
