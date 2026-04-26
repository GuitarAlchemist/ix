//! Holographic Reduced Representations — Tony Plate, 1991.
//!
//! HRRs represent role-filler bindings as fixed-size vectors using
//! circular convolution. Multiple bindings SUM into one vector;
//! retrieval is by correlation with a probe.
//!
//! # Properties
//!
//! - **Fixed-size output.** A "memory" is always one `[f64; D]`
//!   regardless of how many events were bound into it.
//! - **Associative recall.** Probe with a role vector, recover an
//!   approximate filler vector; compare against candidates to
//!   identify the match.
//! - **Graceful capacity degradation.** Can store ~`D / 4` distinct
//!   bindings with reliable recall before noise dominates.
//! - **Composable.** `bind` and `bundle` are algebraic operations —
//!   you can bind an already-composed vector, or bundle multiple
//!   memories together.
//!
//! # Why this matters for session memory
//!
//! A session with thousands of events can be "summarized" into one
//! D-dimensional vector where each event contributes a
//! `bind(role, filler)` term to the bundle. Retrieval — "what tool
//! was called with target X?" — becomes a correlation lookup:
//! `correlate(memory, role_target_X) ≈ tool_vector`.
//!
//! This is a *fuzzy* memory — retrieval is noisy and capacity is
//! bounded. Pair with an exact log for when precision matters.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Default HRR vector dimension. 2048 gives reliable recall for
/// ~500 distinct bindings, which covers most realistic session
/// sizes with headroom.
pub const DEFAULT_DIM: usize = 2048;

/// A fixed-size memory vector in an HRR algebra.
///
/// Cheap to clone (boxed Vec<f64>). Addition is bundling; see
/// [`HrrVector::bundle`]. Multiplication-like composition is
/// circular convolution; see [`HrrVector::bind`].
#[derive(Debug, Clone, PartialEq)]
pub struct HrrVector {
    /// Underlying dense vector. Length = `dim()`.
    components: Vec<f64>,
}

impl HrrVector {
    /// Construct the zero vector of dimension `dim`.
    pub fn zero(dim: usize) -> Self {
        Self {
            components: vec![0.0; dim],
        }
    }

    /// Construct a random unit-length vector used as a role or
    /// filler primitive. Uses a seeded PRNG so the same seed always
    /// returns the same vector — this is what lets different parts
    /// of the system agree on what "the role `tool_name`" means
    /// without any shared registry beyond the seed.
    pub fn random_seeded(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        // Draw from a normal-ish distribution (sum of 12 uniforms
        // ≈ N(0, 1); scaled down to keep components small) then
        // normalize to unit length.
        let mut components = vec![0.0; dim];
        for c in components.iter_mut() {
            let mut s = 0.0;
            for _ in 0..12 {
                s += rng.random::<f64>();
            }
            *c = (s - 6.0) / (dim as f64).sqrt();
        }
        let norm: f64 = components.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for c in components.iter_mut() {
                *c /= norm;
            }
        }
        Self { components }
    }

    /// Dimension of this vector. Must match when binding, bundling,
    /// or correlating against other vectors.
    pub fn dim(&self) -> usize {
        self.components.len()
    }

    /// Raw component access. Returns a slice of length [`Self::dim`].
    pub fn components(&self) -> &[f64] {
        &self.components
    }

    /// Bundle (vector addition + renormalization). The bundle of
    /// several HRR vectors is itself a vector that is *similar to*
    /// each of them under correlation. This is how we store many
    /// bindings in one memory.
    pub fn bundle(&self, other: &HrrVector) -> HrrVector {
        assert_eq!(self.dim(), other.dim(), "dimension mismatch in bundle");
        let mut out = Vec::with_capacity(self.dim());
        for (a, b) in self.components.iter().zip(other.components.iter()) {
            out.push(a + b);
        }
        // Normalize so the bundle stays on the unit sphere. This
        // keeps correlation scores in (-1, 1).
        let norm: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for c in out.iter_mut() {
                *c /= norm;
            }
        }
        HrrVector { components: out }
    }

    /// Bind two vectors via circular convolution. The result is
    /// *dissimilar* to either input under correlation, but the
    /// original operands can be recovered (approximately) by
    /// correlation with the other operand.
    ///
    /// `bind(role, filler)` ≈ a tag for "this role is bound to this
    /// filler". Multiple such bindings can be bundled into one
    /// memory.
    pub fn bind(&self, other: &HrrVector) -> HrrVector {
        let n = self.dim();
        assert_eq!(n, other.dim(), "dimension mismatch in bind");
        let mut out = vec![0.0; n];
        // Naive O(n^2) circular convolution. For the dimensions we
        // use (up to 4096), this is well under a millisecond per
        // bind. FFT-based convolution is a future optimization.
        for (i, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..n {
                let k = ((i + n) - j) % n;
                acc += self.components[j] * other.components[k];
            }
            *slot = acc;
        }
        HrrVector { components: out }
    }

    /// Correlation: approximate inverse of [`HrrVector::bind`].
    /// `correlate(bind(a, b), a) ≈ b` with HRR noise.
    ///
    /// Concretely, correlation is circular convolution with the
    /// *involution* (reversed + rotated) of the probe.
    pub fn correlate(&self, probe: &HrrVector) -> HrrVector {
        let n = self.dim();
        assert_eq!(n, probe.dim(), "dimension mismatch in correlate");
        let mut out = vec![0.0; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..n {
                let k = (i + j) % n;
                acc += self.components[k] * probe.components[j];
            }
            *slot = acc;
        }
        HrrVector { components: out }
    }

    /// Cosine similarity between two HRR vectors. Returns a value in
    /// `[-1.0, 1.0]`; higher means more similar. Used to compare a
    /// correlation result against candidate fillers.
    pub fn cosine_similarity(&self, other: &HrrVector) -> f64 {
        assert_eq!(self.dim(), other.dim(), "dimension mismatch in cosine");
        let dot: f64 = self
            .components
            .iter()
            .zip(other.components.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f64 = self.components.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = other.components.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }
}

/// A named-symbol → vector registry used to bind events into HRR
/// memory without coordinating vector choice across sources.
///
/// Uses deterministic seeding (hash of the symbol name) so two
/// processes can independently allocate vectors for the symbol
/// `"tool:ix_stats"` and get bit-identical results.
pub struct SymbolRegistry {
    dim: usize,
    cache: HashMap<String, HrrVector>,
}

impl SymbolRegistry {
    /// Construct a new registry with the given HRR dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            cache: HashMap::new(),
        }
    }

    /// Get-or-allocate the vector bound to `name`. Seeded from a
    /// stable hash of the name so the same name always yields the
    /// same vector across processes.
    pub fn get(&mut self, name: &str) -> HrrVector {
        if let Some(v) = self.cache.get(name) {
            return v.clone();
        }
        let seed = stable_hash(name);
        let vec = HrrVector::random_seeded(self.dim, seed);
        self.cache.insert(name.to_string(), vec.clone());
        vec
    }

    /// Dimension shared by all vectors in this registry.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Deterministic string hash → u64 seed for [`HrrVector::random_seeded`].
/// Not cryptographically strong — just stable across runs and
/// platforms. Matches `DefaultHasher` would NOT be stable; this
/// one is (FNV-1a 64-bit).
fn stable_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // Mix the bits so nearby strings don't produce nearby seeds.
    let _ = PI; // silence unused import
    h ^ (h >> 33)
}

/// Serialize an HRR vector to a compact little-endian byte layout.
/// Header: `u32` dimension. Then `dim` × `f64` little-endian.
pub fn encode(vector: &HrrVector) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + vector.dim() * 8);
    out.extend_from_slice(&(vector.dim() as u32).to_le_bytes());
    for c in vector.components() {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out
}

/// Deserialize an HRR vector from a byte slice produced by
/// [`encode`]. Returns `None` if the slice is malformed.
pub fn decode(bytes: &[u8]) -> Option<HrrVector> {
    if bytes.len() < 4 {
        return None;
    }
    let dim = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let expected = 4 + dim * 8;
    if bytes.len() != expected {
        return None;
    }
    let mut components = Vec::with_capacity(dim);
    for i in 0..dim {
        let off = 4 + i * 8;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[off..off + 8]);
        components.push(f64::from_le_bytes(buf));
    }
    Some(HrrVector { components })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A small dimension keeps the O(n^2) convolution fast enough
    /// for repeated test runs while still being large enough to
    /// show the retrieval signal clearly.
    const TEST_DIM: usize = 256;

    #[test]
    fn random_vector_is_approximately_unit_length() {
        let v = HrrVector::random_seeded(TEST_DIM, 42);
        let norm: f64 = v.components().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9, "expected unit norm, got {norm}");
    }

    #[test]
    fn same_seed_yields_same_vector() {
        let a = HrrVector::random_seeded(TEST_DIM, 123);
        let b = HrrVector::random_seeded(TEST_DIM, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_yield_different_vectors() {
        let a = HrrVector::random_seeded(TEST_DIM, 1);
        let b = HrrVector::random_seeded(TEST_DIM, 2);
        assert_ne!(a, b);
        // Unrelated random vectors should have near-zero cosine.
        assert!(
            a.cosine_similarity(&b).abs() < 0.3,
            "unrelated vectors should have low cosine similarity"
        );
    }

    #[test]
    fn bind_and_correlate_round_trip() {
        // Core HRR property: correlate(bind(role, filler), role) ≈ filler
        let role = HrrVector::random_seeded(TEST_DIM, 100);
        let filler = HrrVector::random_seeded(TEST_DIM, 200);
        let bound = role.bind(&filler);

        // Correlation should recover the filler with higher cosine
        // similarity than a random distractor.
        let recovered = bound.correlate(&role);
        let distractor = HrrVector::random_seeded(TEST_DIM, 999);

        let sim_filler = recovered.cosine_similarity(&filler);
        let sim_distractor = recovered.cosine_similarity(&distractor);

        assert!(
            sim_filler > sim_distractor,
            "expected filler similarity ({sim_filler}) > distractor ({sim_distractor})"
        );
        // At this dimension, the filler similarity should be well
        // above random.
        assert!(
            sim_filler > 0.2,
            "filler similarity should be clearly above zero, got {sim_filler}"
        );
    }

    #[test]
    fn bundle_is_similar_to_its_parts() {
        // Bundling (vector sum + renormalize) produces a vector
        // similar to each operand. This is the "multi-event
        // memory" property.
        let a = HrrVector::random_seeded(TEST_DIM, 10);
        let b = HrrVector::random_seeded(TEST_DIM, 20);
        let bundle = a.bundle(&b);
        let sim_a = bundle.cosine_similarity(&a);
        let sim_b = bundle.cosine_similarity(&b);
        assert!(sim_a > 0.5, "bundle should be similar to a, got {sim_a}");
        assert!(sim_b > 0.5, "bundle should be similar to b, got {sim_b}");
    }

    #[test]
    fn registry_returns_same_vector_for_same_name() {
        let mut reg = SymbolRegistry::new(TEST_DIM);
        let v1 = reg.get("tool:ix_stats");
        let v2 = reg.get("tool:ix_stats");
        assert_eq!(v1, v2);
    }

    #[test]
    fn registry_returns_different_vectors_for_different_names() {
        let mut reg = SymbolRegistry::new(TEST_DIM);
        let a = reg.get("tool:ix_stats");
        let b = reg.get("tool:ix_fft");
        assert_ne!(a, b);
        // Unrelated names → low cosine similarity.
        assert!(a.cosine_similarity(&b).abs() < 0.3);
    }

    #[test]
    fn session_memory_stores_and_retrieves_multiple_events() {
        // Build a "session memory" bundling several role-filler
        // bindings, then probe for each role and verify the
        // strongest cosine match is the right filler.
        let mut reg = SymbolRegistry::new(TEST_DIM);

        // Three events — each binds "round_N" to "tool_X".
        let round_1 = reg.get("round:1");
        let round_2 = reg.get("round:2");
        let round_3 = reg.get("round:3");
        let tool_stats = reg.get("tool:ix_stats");
        let tool_fft = reg.get("tool:ix_fft");
        let tool_distance = reg.get("tool:ix_distance");

        let event_1 = round_1.bind(&tool_stats);
        let event_2 = round_2.bind(&tool_fft);
        let event_3 = round_3.bind(&tool_distance);

        let memory = event_1.bundle(&event_2).bundle(&event_3);

        // Probe: "what tool ran in round 2?"
        let probe = memory.correlate(&round_2);

        // Compare against all known tools; pick the best match.
        let candidates = [
            ("ix_stats", &tool_stats),
            ("ix_fft", &tool_fft),
            ("ix_distance", &tool_distance),
        ];
        let best = candidates
            .iter()
            .map(|(name, v)| (*name, probe.cosine_similarity(v)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        assert_eq!(
            best.0, "ix_fft",
            "expected round:2 probe to recover ix_fft, got {} (score {})",
            best.0, best.1
        );
    }

    #[test]
    fn encode_decode_round_trip() {
        let v = HrrVector::random_seeded(TEST_DIM, 7);
        let bytes = encode(&v);
        let decoded = decode(&bytes).expect("decode");
        assert_eq!(v, decoded);
    }

    #[test]
    fn decode_rejects_malformed_input() {
        assert!(decode(&[]).is_none());
        assert!(decode(&[0, 0, 0]).is_none());
        // Claims dim=100 but only provides 4 bytes of data.
        let mut bytes = 100u32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        assert!(decode(&bytes).is_none());
    }
}
