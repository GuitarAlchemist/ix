//! Handlers for the acoustic-tune analysis skills — pure, one-shot spectral
//! analysis wrapped from `ix-acoustic-tune` and exposed as `#[ix_skill]`s, so the
//! reference descriptor and the decomposed spectral distance are pipeline-, NL-,
//! and MCP-callable. Offline `f64`; renders nothing.

use serde_json::{json, Value};

/// Parse a required 1-D numeric array field.
fn parse_signal(p: &Value, field: &str) -> Result<Vec<f64>, String> {
    p.get(field)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing or invalid array field '{field}'"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| format!("non-numeric value in '{field}'"))
        })
        .collect()
}

/// Sample rate (Hz), defaulting to 48 kHz.
fn sample_rate(p: &Value) -> f64 {
    p.get("sample_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(48000.0)
}

/// `analyze_reference`: extract the perceptual descriptor (f0, centroid, rolloff,
/// rms, attack, inharmonicity-B, per-band decay slopes) from a recording's samples.
pub fn analyze_reference(p: Value) -> Result<Value, String> {
    let signal = parse_signal(&p, "signal")?;
    if signal.is_empty() {
        return Err("signal must not be empty".into());
    }
    let descriptor = ix_acoustic_tune::reference::analyze(&signal, sample_rate(&p));
    serde_json::to_value(descriptor).map_err(|e| e.to_string())
}

/// `spectral_distance`: the layered perceptual loss between a target and a
/// candidate render, **decomposed** into its per-term contributions with the
/// dominant term and the retune-vs-extend-kernel recommendation.
pub fn spectral_distance(p: Value) -> Result<Value, String> {
    use ix_acoustic_tune::spectral_loss::{decompose_residual, LossWeights};
    let target = parse_signal(&p, "target")?;
    let candidate = parse_signal(&p, "candidate")?;
    if target.is_empty() || candidate.is_empty() {
        return Err("target and candidate must both be non-empty".into());
    }
    let r = decompose_residual(
        &target,
        &candidate,
        sample_rate(&p),
        &LossWeights::default(),
    );
    Ok(json!({
        "total": r.total,
        "mss": r.mss,
        "mel": r.mel,
        "decay": r.decay,
        "f0": r.f0,
        "inharmonicity": r.inharmonicity,
        "dominant": r.dominant,
        "recommendation": r.recommendation(),
    }))
}
