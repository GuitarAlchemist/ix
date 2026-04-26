//! Phase A of the harness signature layer — canonical
//! serialization, Ed25519 sign/verify, and a minimal key-registry
//! type.
//!
//! Implementation of the design in
//! `demerzel/docs/governance/harness-signature-layer.md`. This
//! crate provides the primitives; wiring them into the merge
//! function (Phase B) and the first Tier-1 adapter (Phase C) are
//! follow-on commits.
//!
//! # Three public operations
//!
//! - [`canonical_form`] — deterministic byte form of a
//!   [`HexObservation`] suitable for signing
//! - [`sign`] — produce a [`Signature`] over a HexObservation
//!   using an Ed25519 keypair
//! - [`verify`] — check a Signature against a HexObservation and
//!   a public key
//!
//! # Non-goals
//!
//! - Not a replacement for `ix-fuzzy::observations::merge` — that
//!   crate is unchanged by this one. The verification step gets
//!   wired in during Phase B.
//! - Not HSM-backed. Keys live in memory / on disk for now.
//! - Not post-quantum. Ed25519 only. The `SignatureAlgorithm`
//!   enum is an extension point for future algorithms.

use base64::engine::general_purpose::STANDARD as B64_STANDARD;
use base64::Engine as _;
use ed25519_dalek::{Signer, Verifier};
use ix_fuzzy::observations::HexObservation;
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Signature algorithm discriminator. Ed25519 is the only
/// supported algorithm today; the enum shape is the extension
/// point for post-quantum or ECDSA variants later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignatureAlgorithm {
    /// Ed25519 (RFC 8032). 32-byte public keys, 64-byte signatures.
    Ed25519,
}

/// A detached signature over a [`HexObservation`]. Carries the
/// algorithm, the key identifier, and the signature bytes.
///
/// `key_id` is the first 16 bytes of `SHA-256(public_key)`
/// rendered as hex — 32 characters. Used as a compact registry
/// lookup key without shipping the full public key inline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    /// Algorithm used to produce `bytes`.
    pub algorithm: SignatureAlgorithm,
    /// 32-char hex fingerprint of the public key.
    pub key_id: String,
    /// Base64-encoded signature bytes.
    pub bytes: String,
}

/// Errors produced by this crate.
#[derive(Debug, thiserror::Error)]
pub enum SigningError {
    /// The signature bytes could not be base64-decoded.
    #[error("signature base64: {0}")]
    Base64(String),
    /// The signature was the wrong length for the declared algorithm.
    #[error("signature length: expected {expected}, got {actual}")]
    SignatureLength { expected: usize, actual: usize },
    /// The signature did not verify against the given public key.
    #[error("signature verification failed")]
    VerificationFailed,
    /// The canonical-form serializer produced invalid JSON.
    #[error("canonical form: {0}")]
    Canonical(String),
}

/// Produce the canonical byte form of an observation for signing.
///
/// Contract (from the design doc):
/// - Alphabetical field order (not the Rust declaration order)
/// - No optional whitespace
/// - `evidence: None` renders as `"evidence":null` literally
/// - Weights are quantized to 0.01 resolution before rendering
///   so float-printing drift across platforms is impossible
/// - Variant is rendered as a single uppercase letter (T/P/U/D/F/C)
///
/// The return type is `String` rather than `Vec<u8>` because the
/// canonical form is always valid UTF-8 and callers typically
/// want to log or compare it as text.
pub fn canonical_form(obs: &HexObservation) -> Result<String, SigningError> {
    // Quantize weight to 0.01 resolution. Round-to-nearest.
    let quantized = (obs.weight * 100.0).round() / 100.0;
    // Render the weight manually so we control the format.
    let weight_str = format_weight(quantized);

    // Build the canonical JSON by hand in alphabetical field
    // order. Using serde_json::to_string on a BTreeMap would
    // also work but adds a dependency on BTreeMap's key ordering;
    // manual assembly makes the order explicit and audit-friendly.
    let evidence_json = match &obs.evidence {
        Some(s) => serde_json::to_string(s).map_err(|e| SigningError::Canonical(e.to_string()))?,
        None => "null".to_string(),
    };
    let claim_key_json = serde_json::to_string(&obs.claim_key)
        .map_err(|e| SigningError::Canonical(e.to_string()))?;
    let diagnosis_id_json = serde_json::to_string(&obs.diagnosis_id)
        .map_err(|e| SigningError::Canonical(e.to_string()))?;
    let source_json =
        serde_json::to_string(&obs.source).map_err(|e| SigningError::Canonical(e.to_string()))?;

    Ok(format!(
        "{{\"claim_key\":{claim_key_json},\"diagnosis_id\":{diagnosis_id_json},\"evidence\":{evidence_json},\"ordinal\":{ordinal},\"round\":{round},\"source\":{source_json},\"variant\":\"{variant}\",\"weight\":{weight_str}}}",
        ordinal = obs.ordinal,
        round = obs.round,
        variant = variant_letter(obs.variant),
        weight_str = weight_str,
    ))
}

/// Single-letter variant representation used in canonical form.
fn variant_letter(v: Hexavalent) -> char {
    match v {
        Hexavalent::True => 'T',
        Hexavalent::Probable => 'P',
        Hexavalent::Unknown => 'U',
        Hexavalent::Doubtful => 'D',
        Hexavalent::False => 'F',
        Hexavalent::Contradictory => 'C',
    }
}

/// Format a weight at 0.01 resolution without trailing zeros and
/// without scientific notation. Always prints at least one
/// fractional digit so `0.0` and `1.0` render as `0.00` and
/// `1.00` — predictable text.
fn format_weight(w: f64) -> String {
    // Already quantized to 0.01. Use two decimal places.
    format!("{w:.2}")
}

/// Compute the `key_id` for a public key. First 16 bytes of
/// `SHA-256(public_key_bytes)`, hex-encoded.
pub fn compute_key_id(public_key: &ed25519_dalek::VerifyingKey) -> String {
    let pk_bytes = public_key.to_bytes();
    let mut hasher = Sha256::new();
    hasher.update(pk_bytes);
    let hash = hasher.finalize();
    let mut out = String::with_capacity(32);
    for byte in hash.iter().take(16) {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Sign an observation with an Ed25519 signing key. Returns a
/// [`Signature`] that can be attached to the observation (Phase B)
/// or transmitted alongside it (Phase C).
pub fn sign(
    key: &ed25519_dalek::SigningKey,
    obs: &HexObservation,
) -> Result<Signature, SigningError> {
    let canonical = canonical_form(obs)?;
    let sig_bytes = key.sign(canonical.as_bytes());
    let verifying_key = key.verifying_key();
    Ok(Signature {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: compute_key_id(&verifying_key),
        bytes: B64_STANDARD.encode(sig_bytes.to_bytes()),
    })
}

/// Verify a [`Signature`] against an observation and a public
/// key. Returns `Ok(())` on success, [`SigningError::VerificationFailed`]
/// on any failure (bad length, bad base64, bad signature, algorithm
/// mismatch).
pub fn verify(
    public_key: &ed25519_dalek::VerifyingKey,
    obs: &HexObservation,
    signature: &Signature,
) -> Result<(), SigningError> {
    if signature.algorithm != SignatureAlgorithm::Ed25519 {
        return Err(SigningError::VerificationFailed);
    }
    let raw = B64_STANDARD
        .decode(&signature.bytes)
        .map_err(|e| SigningError::Base64(e.to_string()))?;
    if raw.len() != 64 {
        return Err(SigningError::SignatureLength {
            expected: 64,
            actual: raw.len(),
        });
    }
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(&raw);
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);

    let canonical = canonical_form(obs)?;
    public_key
        .verify(canonical.as_bytes(), &sig)
        .map_err(|_| SigningError::VerificationFailed)
}

/// Generate a fresh Ed25519 signing key from 32 random bytes.
/// Convenience for tests and adapter bootstrapping.
///
/// Uses `rand::rngs::OsRng` to fill the seed bytes directly,
/// bypassing `SigningKey::generate` (which depends on an older
/// `rand_core` version than our workspace's `rand 0.9`).
pub fn generate_key() -> ed25519_dalek::SigningKey {
    use rand::rngs::OsRng;
    use rand::TryRngCore;
    let mut seed = [0u8; 32];
    let mut rng = OsRng;
    rng.try_fill_bytes(&mut seed)
        .expect("OsRng should not fail");
    ed25519_dalek::SigningKey::from_bytes(&seed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn obs_fixture() -> HexObservation {
        HexObservation {
            source: "cargo".to_string(),
            diagnosis_id: "abc123def456".to_string(),
            round: 3,
            ordinal: 42,
            claim_key: "test:ix_math::stats::valuable".to_string(),
            variant: Hexavalent::True,
            weight: 0.9,
            evidence: Some("ok".to_string()),
        }
    }

    // ── Canonical form determinism ─────────────────────────────

    #[test]
    fn canonical_form_is_deterministic_for_identical_input() {
        let o = obs_fixture();
        let c1 = canonical_form(&o).unwrap();
        let c2 = canonical_form(&o).unwrap();
        assert_eq!(c1, c2);
    }

    #[test]
    fn canonical_form_has_alphabetical_field_order() {
        let o = obs_fixture();
        let c = canonical_form(&o).unwrap();
        // The fields should appear in this exact order:
        // claim_key, diagnosis_id, evidence, ordinal, round,
        // source, variant, weight
        let positions: Vec<usize> = [
            "\"claim_key\":",
            "\"diagnosis_id\":",
            "\"evidence\":",
            "\"ordinal\":",
            "\"round\":",
            "\"source\":",
            "\"variant\":",
            "\"weight\":",
        ]
        .iter()
        .map(|needle| c.find(needle).expect("field present"))
        .collect();
        for i in 1..positions.len() {
            assert!(
                positions[i - 1] < positions[i],
                "field {i} out of order in {c}"
            );
        }
    }

    #[test]
    fn canonical_form_renders_none_evidence_as_null() {
        let mut o = obs_fixture();
        o.evidence = None;
        let c = canonical_form(&o).unwrap();
        assert!(c.contains("\"evidence\":null"));
    }

    #[test]
    fn canonical_form_quantizes_weight_to_two_decimals() {
        let mut o = obs_fixture();
        o.weight = 0.8947; // Not a 0.01 multiple
        let c = canonical_form(&o).unwrap();
        // Rounded to 0.89, rendered with two decimal places.
        assert!(c.contains("\"weight\":0.89"), "got {c}");
    }

    #[test]
    fn canonical_form_renders_integer_weights_with_two_decimals() {
        let mut o = obs_fixture();
        o.weight = 1.0;
        let c = canonical_form(&o).unwrap();
        assert!(c.contains("\"weight\":1.00"));
    }

    #[test]
    fn canonical_form_escapes_special_characters_in_strings() {
        let mut o = obs_fixture();
        o.evidence = Some("line1\nline2\"quote".to_string());
        let c = canonical_form(&o).unwrap();
        // serde_json's string escape should handle this.
        assert!(c.contains(r#""evidence":"line1\nline2\"quote""#));
    }

    #[test]
    fn canonical_form_renders_variant_as_single_letter() {
        let variants = [
            (Hexavalent::True, 'T'),
            (Hexavalent::Probable, 'P'),
            (Hexavalent::Unknown, 'U'),
            (Hexavalent::Doubtful, 'D'),
            (Hexavalent::False, 'F'),
            (Hexavalent::Contradictory, 'C'),
        ];
        for (v, letter) in variants {
            let mut o = obs_fixture();
            o.variant = v;
            let c = canonical_form(&o).unwrap();
            let expected = format!("\"variant\":\"{letter}\"");
            assert!(c.contains(&expected), "variant {v:?} → {c}");
        }
    }

    // ── Sign / verify round trip ───────────────────────────────

    #[test]
    fn sign_verify_round_trip() {
        let key = generate_key();
        let o = obs_fixture();
        let sig = sign(&key, &o).unwrap();
        assert_eq!(sig.algorithm, SignatureAlgorithm::Ed25519);
        assert_eq!(sig.key_id.len(), 32); // 16 bytes hex = 32 chars

        verify(&key.verifying_key(), &o, &sig).unwrap();
    }

    #[test]
    fn verify_rejects_modified_observation() {
        let key = generate_key();
        let mut o = obs_fixture();
        let sig = sign(&key, &o).unwrap();

        // Mutate the observation after signing.
        o.weight = 0.5;
        let err = verify(&key.verifying_key(), &o, &sig).unwrap_err();
        assert!(matches!(err, SigningError::VerificationFailed));
    }

    #[test]
    fn verify_rejects_wrong_public_key() {
        let key_a = generate_key();
        let key_b = generate_key();
        let o = obs_fixture();
        let sig = sign(&key_a, &o).unwrap();

        let err = verify(&key_b.verifying_key(), &o, &sig).unwrap_err();
        assert!(matches!(err, SigningError::VerificationFailed));
    }

    #[test]
    fn verify_rejects_bad_signature_length() {
        let key = generate_key();
        let o = obs_fixture();
        let sig = Signature {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: compute_key_id(&key.verifying_key()),
            bytes: B64_STANDARD.encode([0u8; 32]), // 32 bytes, not 64
        };
        let err = verify(&key.verifying_key(), &o, &sig).unwrap_err();
        assert!(
            matches!(err, SigningError::SignatureLength { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn verify_rejects_bad_base64() {
        let key = generate_key();
        let o = obs_fixture();
        let sig = Signature {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: compute_key_id(&key.verifying_key()),
            bytes: "!!! not base64 !!!".to_string(),
        };
        let err = verify(&key.verifying_key(), &o, &sig).unwrap_err();
        assert!(matches!(err, SigningError::Base64(_)));
    }

    // ── Key id properties ─────────────────────────────────────

    #[test]
    fn key_id_is_deterministic() {
        let key = generate_key();
        let k1 = compute_key_id(&key.verifying_key());
        let k2 = compute_key_id(&key.verifying_key());
        assert_eq!(k1, k2);
    }

    #[test]
    fn key_id_is_32_hex_chars() {
        let key = generate_key();
        let id = compute_key_id(&key.verifying_key());
        assert_eq!(id.len(), 32);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn different_keys_have_different_ids() {
        let k1 = generate_key();
        let k2 = generate_key();
        assert_ne!(
            compute_key_id(&k1.verifying_key()),
            compute_key_id(&k2.verifying_key())
        );
    }

    // ── Signature serde ───────────────────────────────────────

    #[test]
    fn signature_serde_round_trip() {
        let key = generate_key();
        let o = obs_fixture();
        let sig = sign(&key, &o).unwrap();
        let json = serde_json::to_string(&sig).unwrap();
        let back: Signature = serde_json::from_str(&json).unwrap();
        assert_eq!(back, sig);
    }
}
