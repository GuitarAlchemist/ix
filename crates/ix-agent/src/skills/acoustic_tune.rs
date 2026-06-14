//! `#[ix_skill]` wrappers for the acoustic-tune analysis — making the reference
//! descriptor and the decomposed spectral distance pipeline-/NL-/MCP-callable.
//! Delegate to `crate::acoustic_tune` handlers (pure, offline `f64`).

use crate::acoustic_tune;
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

// --- ix_analyze_reference --------------------------------------------------

fn analyze_reference_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "signal": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Mono audio samples of a reference note/recording"
            },
            "sample_rate": { "type": "number", "default": 48000, "description": "Sample rate in Hz" }
        },
        "required": ["signal"]
    })
}

fn analyze_reference_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "f0_hz": { "type": ["number", "null"], "description": "Fundamental frequency (Hz), or null" },
            "centroid_hz": { "type": "number", "description": "Spectral centroid (brightness), Hz" },
            "rolloff_hz": { "type": "number", "description": "85% spectral rolloff, Hz" },
            "rms": { "type": "number", "description": "Root-mean-square level" },
            "attack_seconds": { "type": "number", "description": "Onset→peak attack time, s" },
            "inharmonicity_b": { "type": "number", "description": "Stiff-string inharmonicity coefficient B" },
            "band_decay_slopes": { "type": "array", "items": { "type": "number" }, "description": "Per-band decay slope (log-energy/s; more negative = faster)" },
            "band_edges": { "type": "array", "items": { "type": "number" }, "description": "Band edge frequencies (Hz)" }
        }
    })
}

/// Analyze a reference recording's samples into its perceptual descriptor:
/// f0, spectral centroid/rolloff, RMS, attack time, inharmonicity, and the
/// per-band decay slopes (the naturalness fingerprint).
#[ix_skill(
    domain = "signal",
    name = "analyze_reference",
    governance = "deterministic",
    schema_fn = "crate::skills::acoustic_tune::analyze_reference_schema",
    output_schema_fn = "crate::skills::acoustic_tune::analyze_reference_output_schema"
)]
pub fn analyze_reference(params: Value) -> Result<Value, String> {
    acoustic_tune::analyze_reference(params)
}

// --- ix_spectral_distance --------------------------------------------------

fn spectral_distance_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "target": { "type": "array", "items": { "type": "number" }, "description": "Reference samples" },
            "candidate": { "type": "array", "items": { "type": "number" }, "description": "Candidate render samples to score" },
            "sample_rate": { "type": "number", "default": 48000, "description": "Sample rate in Hz" }
        },
        "required": ["target", "candidate"]
    })
}

fn spectral_distance_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "total": { "type": "number", "description": "Layered perceptual loss (lower = closer/more natural)" },
            "mss": { "type": "number", "description": "Multi-resolution STFT term" },
            "mel": { "type": "number", "description": "Perceptual (log-mel) term" },
            "decay": { "type": "number", "description": "Per-band decay-slope term (the naturalness fingerprint)" },
            "f0": { "type": "number", "description": "Relative f0 term" },
            "inharmonicity": { "type": "number", "description": "Inharmonicity term" },
            "dominant": { "type": "string", "description": "The largest term — the bottleneck to address" },
            "recommendation": { "type": "string", "description": "Retune vs extend-kernel guidance for the dominant term" }
        }
    })
}

/// Compute the layered perceptual distance between a target recording and a
/// candidate render, decomposed into its per-term contributions plus the
/// dominant-term diagnosis (retune vs extend the synth kernel).
#[ix_skill(
    domain = "signal",
    name = "spectral_distance",
    governance = "deterministic",
    schema_fn = "crate::skills::acoustic_tune::spectral_distance_schema",
    output_schema_fn = "crate::skills::acoustic_tune::spectral_distance_output_schema"
)]
pub fn spectral_distance(params: Value) -> Result<Value, String> {
    acoustic_tune::spectral_distance(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn decaying_tone(freq: f64, decay: f64, sr: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|t| {
                let tt = t as f64 / sr;
                (-decay * tt).exp() * (TAU * freq * tt).sin()
            })
            .collect()
    }

    #[test]
    fn analyze_reference_skill_extracts_descriptor() {
        let sig = decaying_tone(220.0, 3.0, 8000.0, 8000);
        let out = analyze_reference(json!({ "signal": sig, "sample_rate": 8000.0 }))
            .expect("analyze_reference runs");
        assert!(
            out["f0_hz"]
                .as_f64()
                .map(|f| (f - 220.0).abs() < 10.0)
                .unwrap_or(false),
            "f0 should be ≈220, got {:?}",
            out["f0_hz"]
        );
        let slopes = out["band_decay_slopes"]
            .as_array()
            .expect("band_decay_slopes");
        assert!(!slopes.is_empty() && slopes.iter().any(|s| s.as_f64().unwrap() < 0.0));
    }

    // The decomposed distance must flag the right bottleneck: a wrong-decay
    // candidate is decay-dominated and recommends the kernel loop filter.
    #[test]
    fn spectral_distance_skill_diagnoses_wrong_decay() {
        let sr = 8000.0;
        let target = decaying_tone(300.0, 3.0, sr, 8000);
        let no_decay = decaying_tone(300.0, 0.0, sr, 8000);
        let out = spectral_distance(
            json!({ "target": target, "candidate": no_decay, "sample_rate": sr }),
        )
        .expect("spectral_distance runs");
        assert!(out["total"].as_f64().unwrap() > 0.0);
        assert_eq!(out["dominant"].as_str().unwrap(), "decay");
        assert!(out["recommendation"]
            .as_str()
            .unwrap()
            .contains("loop filter"));
    }

    #[test]
    fn acoustic_tune_skills_registered_and_pipeline_callable() {
        for name in ["analyze_reference", "spectral_distance"] {
            let d = ix_registry::all()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("{name} not registered"));
            assert_eq!(d.inputs.len(), 1, "{name} must be arity-1");
        }
    }
}
