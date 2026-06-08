//! Tuning session orchestrator — drives a CMA-ES over the synth's continuous
//! parameters via the JSON contract: [`next_request`](TuneSession::next_request)
//! emits a generation to render, [`ingest`](TuneSession::ingest) consumes the
//! scored result and advances the optimizer.
//!
//! One session covers one `guitar_type` (the categorical dimension is
//! outer-enumerated: run one session per type and compare final scores). The
//! session lives in memory; the GA exchange is the on-disk request/result files.

use std::path::Path;

use ndarray::Array1;

use crate::cmaes::CmaEs;
use crate::contract::{Candidate, RenderSpec, TuneRequest, TuneResult, REQUEST_SCHEMA};
use crate::AskTell;

/// Loss assigned to a render that tripped a guardrail (clipped/silent) — strictly
/// worse than any real `1 - spectral_score` (which is ≤ 1 for a score in [0, 1]).
const GUARDRAIL_FAIL_LOSS: f64 = 2.0;

/// Configuration for a tuning session over the continuous synth parameters.
pub struct SessionConfig {
    pub session_id: String,
    /// The locked actuator order, e.g. `[decay, brightness, dispersion, attack_decay, reverb_mix]`.
    pub param_names: Vec<String>,
    pub init: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub sigma: f64,
    pub seed: u64,
    pub guitar_type: i64,
    pub render: RenderSpec,
}

/// A live tuning session: holds the CMA-ES optimizer and converts its ask/tell
/// into contract requests/results.
pub struct TuneSession {
    session_id: String,
    param_names: Vec<String>,
    guitar_type: i64,
    render: RenderSpec,
    opt: CmaEs,
    iteration: usize,
}

impl TuneSession {
    /// Build a session from `cfg`, validating the parameter dimensions/bounds.
    pub fn new(cfg: SessionConfig) -> Result<Self, String> {
        let n = cfg.param_names.len();
        if n == 0 {
            return Err("need at least one parameter".into());
        }
        if cfg.init.len() != n || cfg.lower.len() != n || cfg.upper.len() != n {
            return Err("param_names / init / lower / upper length mismatch".into());
        }
        for i in 0..n {
            if cfg.lower[i] >= cfg.upper[i] {
                return Err(format!(
                    "lower[{i}] ({}) must be < upper[{i}] ({})",
                    cfg.lower[i], cfg.upper[i]
                ));
            }
            if cfg.init[i] < cfg.lower[i] || cfg.init[i] > cfg.upper[i] {
                return Err(format!("init[{i}] ({}) is outside its bounds", cfg.init[i]));
            }
        }
        let opt = CmaEs::new(Array1::from_vec(cfg.init), cfg.sigma, cfg.seed)
            .with_bounds(Array1::from_vec(cfg.lower), Array1::from_vec(cfg.upper));
        Ok(Self {
            session_id: cfg.session_id,
            param_names: cfg.param_names,
            guitar_type: cfg.guitar_type,
            render: cfg.render,
            opt,
            iteration: 0,
        })
    }

    /// Number of completed iterations (ingested results).
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Build the next request: a CMA-ES generation of candidates to render.
    pub fn next_request(&mut self) -> TuneRequest {
        let candidates: Vec<Candidate> = self
            .opt
            .ask()
            .into_iter()
            .enumerate()
            .map(|(i, p)| Candidate {
                id: format!("i{}c{}", self.iteration, i),
                params: p.to_vec(),
            })
            .collect();
        TuneRequest {
            schema: REQUEST_SCHEMA.to_string(),
            session_id: self.session_id.clone(),
            iteration: self.iteration,
            optimizer: "cmaes".to_string(),
            param_names: self.param_names.clone(),
            candidates,
            guitar_type: self.guitar_type,
            render: self.render.clone(),
        }
    }

    /// Ingest a scored result and advance the optimizer. Each candidate's loss is
    /// `1 - spectral_score`; a clipped/silent guardrail failure is penalized to
    /// [`GUARDRAIL_FAIL_LOSS`] so the search avoids unusable renders.
    pub fn ingest(&mut self, result: &TuneResult) -> Result<(), String> {
        // Reject a stale / out-of-order / wrong-session result before touching the
        // optimizer: the JSON-on-disk exchange can leave an old `tune-result.json`
        // behind, and applying losses for a different generation/session would
        // silently corrupt the CMA-ES state.
        if result.session_id != self.session_id {
            return Err(format!(
                "result session_id '{}' does not match this session '{}'",
                result.session_id, self.session_id
            ));
        }
        if result.iteration != self.iteration {
            return Err(format!(
                "stale result: iteration {} does not match the pending generation {}",
                result.iteration, self.iteration
            ));
        }
        if result.scores.is_empty() {
            return Err("result has no scores".into());
        }
        let evaluated: Vec<(Array1<f64>, f64)> = result
            .scores
            .iter()
            .map(|s| {
                let loss = match &s.guardrail {
                    Some(g) if g.failed() => GUARDRAIL_FAIL_LOSS,
                    _ => 1.0 - s.spectral_score,
                };
                (Array1::from_vec(s.params.clone()), loss)
            })
            .collect();
        self.opt.tell(&evaluated);
        self.iteration += 1;
        Ok(())
    }

    /// Best `(params, loss)` so far (`loss = 1 - spectral_score`).
    pub fn recommend(&self) -> Option<(Vec<f64>, f64)> {
        self.opt.recommend().map(|(p, l)| (p.to_vec(), l))
    }
}

/// Write a request to `path` as pretty JSON (the IX→GA half of the contract).
pub fn write_request(path: impl AsRef<Path>, req: &TuneRequest) -> Result<(), String> {
    let json = serde_json::to_string_pretty(req).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

/// Read a result from `path` (the GA→IX half of the contract).
pub fn read_result(path: impl AsRef<Path>) -> Result<TuneResult, String> {
    let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{CandidateScore, Guardrail, RESULT_SCHEMA};

    fn cfg(seed: u64) -> SessionConfig {
        SessionConfig {
            session_id: "test".into(),
            param_names: vec!["a".into(), "b".into()],
            init: vec![0.2, 0.8],
            lower: vec![0.0, 0.0],
            upper: vec![1.0, 1.0],
            sigma: 0.15,
            seed,
            guitar_type: 0,
            render: RenderSpec {
                note_hz: 110.0,
                velocity: 0.9,
                seconds: 1.0,
                sample_rate: 48000,
            },
        }
    }

    // The full contract loop, in-process: simulate GA by scoring each candidate
    // with exp(-||p - target||²) (1 at the target). The session must converge the
    // recommendation toward the target — and we round-trip the request through
    // JSON each iteration to prove the on-disk contract carries everything.
    #[test]
    fn session_converges_over_the_simulated_contract_loop() {
        let target = [0.5, 0.5];
        let mut session = TuneSession::new(cfg(2024)).expect("session");

        for _ in 0..60 {
            let req = session.next_request();
            // Round-trip the request as GA would read it.
            let wire = serde_json::to_string(&req).unwrap();
            let req: TuneRequest = serde_json::from_str(&wire).unwrap();

            // "GA" renders + scores: higher score = closer to target.
            let scores = req
                .candidates
                .iter()
                .map(|c| {
                    let d2: f64 = c
                        .params
                        .iter()
                        .zip(target.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum();
                    CandidateScore {
                        id: c.id.clone(),
                        params: c.params.clone(),
                        spectral_score: (-d2).exp(),
                        features: None,
                        guardrail: None,
                    }
                })
                .collect();
            let result = TuneResult {
                schema: RESULT_SCHEMA.into(),
                session_id: req.session_id.clone(),
                iteration: req.iteration,
                reference: None,
                scores,
            };
            session.ingest(&result).expect("ingest");
        }

        let (best, loss) = session.recommend().expect("recommendation");
        assert!(
            (best[0] - target[0]).abs() < 0.05 && (best[1] - target[1]).abs() < 0.05,
            "recommended {best:?} should converge to {target:?}"
        );
        assert!(loss < 0.02, "best loss {loss} should be near 0");
    }

    // A clipped/silent render is penalized above any real score: given a clipped
    // candidate with a HIGH spectral_score and a clean candidate with a lower one,
    // the optimizer's best must be the clean candidate.
    #[test]
    fn guardrail_failure_loses_to_a_clean_render() {
        let mut session = TuneSession::new(cfg(7)).expect("session");
        let req = session.next_request();
        let clean = req.candidates[0].clone();
        let clipped = req.candidates[1].clone();

        let mut scores = vec![
            CandidateScore {
                id: clean.id.clone(),
                params: clean.params.clone(),
                spectral_score: 0.6, // clean, decent
                features: None,
                guardrail: Some(Guardrail {
                    clipped: false,
                    silent: false,
                    rms: 0.2,
                }),
            },
            CandidateScore {
                id: clipped.id.clone(),
                params: clipped.params.clone(),
                spectral_score: 0.99, // looks great, but...
                features: None,
                guardrail: Some(Guardrail {
                    clipped: true, // ...it clipped → unusable
                    silent: false,
                    rms: 0.99,
                }),
            },
        ];
        // remaining candidates: neutral
        for c in req.candidates.iter().skip(2) {
            scores.push(CandidateScore {
                id: c.id.clone(),
                params: c.params.clone(),
                spectral_score: 0.1,
                features: None,
                guardrail: None,
            });
        }

        let result = TuneResult {
            schema: RESULT_SCHEMA.into(),
            session_id: req.session_id.clone(),
            iteration: 0,
            reference: None,
            scores,
        };
        session.ingest(&result).expect("ingest");

        let (best, _) = session.recommend().expect("rec");
        assert_eq!(
            best, clean.params,
            "the clean render must win over the higher-scoring clipped one"
        );
    }

    // Codex P2: a stale / wrong-session result must be rejected before it can
    // corrupt the optimizer state (the on-disk exchange can leave an old file).
    #[test]
    fn ingest_rejects_stale_or_wrong_session_result() {
        let mut session = TuneSession::new(cfg(11)).expect("session");
        let req = session.next_request();
        let score = |it: usize, sid: &str| TuneResult {
            schema: RESULT_SCHEMA.into(),
            session_id: sid.into(),
            iteration: it,
            reference: None,
            scores: vec![CandidateScore {
                id: req.candidates[0].id.clone(),
                params: req.candidates[0].params.clone(),
                spectral_score: 0.5,
                features: None,
                guardrail: None,
            }],
        };
        assert!(
            session.ingest(&score(0, "OTHER")).is_err(),
            "wrong session_id must be rejected"
        );
        assert!(
            session.ingest(&score(99, "test")).is_err(),
            "stale iteration must be rejected"
        );
        assert!(
            session.ingest(&score(0, "test")).is_ok(),
            "the matching current-generation result is accepted"
        );
    }
}
