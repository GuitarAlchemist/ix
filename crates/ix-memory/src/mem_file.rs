//! `.mem` binary wrapper format.
//!
//! A minimal container that can carry any of the memory payload
//! types produced by this crate. The container is *dumb* — it
//! doesn't interpret the payload; it just stamps a magic header,
//! a kind tag, the length, the bytes, and an integrity hash.
//!
//! # Layout
//!
//! ```text
//! Offset  Size  Field
//! ──────────────────────────────────────────────────────
//! 0       4     Magic "IXMM" (4 ASCII bytes)
//! 4       1     Version (currently 0x01)
//! 5       1     Kind ('H' | 'D' | 'S')
//! 6       4     Payload length (u32, little-endian)
//! 10      N     Payload bytes
//! 10+N    32    SHA-256 of bytes [0 .. 10+N]
//! ```
//!
//! Total overhead: 10 bytes header + 32 bytes trailer = 42 bytes.
//!
//! # Kind semantics
//!
//! - `'H'` — HRR vector (see [`crate::hrr::encode`])
//! - `'D'` — DNA codon stream (see [`crate::dna::pack`])
//! - `'S'` — Sedenion session signature (128 bytes)
//!
//! # What this does NOT do
//!
//! - No streaming / partial reads. The file must be read in full.
//! - No versioning beyond a single byte — if the format needs to
//!   evolve, future versions must be additive (new kinds, same
//!   layout) or a new magic string.
//! - No encryption. If you need that, wrap the whole file.

use sha2::{Digest, Sha256};

use crate::MEM_FORMAT_VERSION;

/// Magic bytes identifying a `.mem` file.
pub const MAGIC: [u8; 4] = *b"IXMM";

/// Kind byte for HRR payload.
pub const KIND_HRR: u8 = b'H';
/// Kind byte for DNA codon stream payload.
pub const KIND_DNA: u8 = b'D';
/// Kind byte for sedenion signature payload.
pub const KIND_SEDENION: u8 = b'S';

/// Errors produced while reading or writing a `.mem` file.
#[derive(Debug, thiserror::Error)]
pub enum MemFileError {
    /// The input byte slice was shorter than the minimum header + trailer.
    #[error("input too short: got {got} bytes, need at least {need}")]
    TooShort { got: usize, need: usize },
    /// The magic bytes at the start didn't match.
    #[error("bad magic: expected {expected:?}, got {got:?}")]
    BadMagic { expected: [u8; 4], got: [u8; 4] },
    /// The version byte was unsupported.
    #[error("unsupported version: {0}")]
    BadVersion(u8),
    /// The kind byte was unrecognized.
    #[error("unknown kind: {0}")]
    BadKind(u8),
    /// The declared payload length exceeds the input size.
    #[error("payload length {declared} exceeds available {available}")]
    BadPayloadLength { declared: usize, available: usize },
    /// The SHA-256 trailer did not match the recomputed hash.
    #[error("integrity check failed — file tampered or truncated")]
    IntegrityFailure,
}

/// A decoded `.mem` file with its kind and payload.
#[derive(Debug, Clone, PartialEq)]
pub struct MemFile {
    /// Kind byte — one of [`KIND_HRR`], [`KIND_DNA`], [`KIND_SEDENION`].
    pub kind: u8,
    /// Raw payload bytes. Caller interprets according to kind.
    pub payload: Vec<u8>,
}

impl MemFile {
    /// Construct a new mem file from a kind byte and payload.
    /// Validates that the kind is recognized.
    pub fn new(kind: u8, payload: Vec<u8>) -> Result<Self, MemFileError> {
        match kind {
            KIND_HRR | KIND_DNA | KIND_SEDENION => Ok(Self { kind, payload }),
            other => Err(MemFileError::BadKind(other)),
        }
    }

    /// Serialize to the wire format — header + payload + SHA-256
    /// trailer.
    pub fn encode(&self) -> Vec<u8> {
        let payload_len = self.payload.len() as u32;
        let mut out = Vec::with_capacity(10 + self.payload.len() + 32);
        out.extend_from_slice(&MAGIC);
        out.push(MEM_FORMAT_VERSION);
        out.push(self.kind);
        out.extend_from_slice(&payload_len.to_le_bytes());
        out.extend_from_slice(&self.payload);

        // Trailer: SHA-256 of everything written so far.
        let mut hasher = Sha256::new();
        hasher.update(&out);
        let hash = hasher.finalize();
        out.extend_from_slice(&hash);

        out
    }

    /// Parse a `.mem` file from its wire bytes, verifying the
    /// magic, version, kind, length, and integrity hash.
    pub fn decode(bytes: &[u8]) -> Result<Self, MemFileError> {
        const MIN_LEN: usize = 10 + 32; // header + empty payload + trailer

        if bytes.len() < MIN_LEN {
            return Err(MemFileError::TooShort {
                got: bytes.len(),
                need: MIN_LEN,
            });
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(MemFileError::BadMagic {
                expected: MAGIC,
                got: magic,
            });
        }

        let version = bytes[4];
        if version != MEM_FORMAT_VERSION {
            return Err(MemFileError::BadVersion(version));
        }

        let kind = bytes[5];
        match kind {
            KIND_HRR | KIND_DNA | KIND_SEDENION => {}
            other => return Err(MemFileError::BadKind(other)),
        }

        let len_bytes: [u8; 4] = bytes[6..10].try_into().unwrap();
        let payload_len = u32::from_le_bytes(len_bytes) as usize;

        let expected_total = 10 + payload_len + 32;
        if bytes.len() != expected_total {
            return Err(MemFileError::BadPayloadLength {
                declared: payload_len,
                available: bytes.len().saturating_sub(10 + 32),
            });
        }

        let payload = bytes[10..10 + payload_len].to_vec();
        let trailer: &[u8] = &bytes[10 + payload_len..];

        // Recompute the SHA-256 over everything before the trailer.
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..10 + payload_len]);
        let expected_hash = hasher.finalize();
        if trailer != expected_hash.as_slice() {
            return Err(MemFileError::IntegrityFailure);
        }

        Ok(Self { kind, payload })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_hrr() {
        let payload = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mem = MemFile::new(KIND_HRR, payload.clone()).unwrap();
        let bytes = mem.encode();
        let decoded = MemFile::decode(&bytes).unwrap();
        assert_eq!(decoded.kind, KIND_HRR);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn round_trip_dna() {
        let payload = b"ATGCATGCATGC".to_vec();
        let mem = MemFile::new(KIND_DNA, payload.clone()).unwrap();
        let bytes = mem.encode();
        let decoded = MemFile::decode(&bytes).unwrap();
        assert_eq!(decoded.kind, KIND_DNA);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn round_trip_sedenion() {
        let payload = vec![0x42u8; 128];
        let mem = MemFile::new(KIND_SEDENION, payload.clone()).unwrap();
        let bytes = mem.encode();
        let decoded = MemFile::decode(&bytes).unwrap();
        assert_eq!(decoded.kind, KIND_SEDENION);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn empty_payload_round_trip() {
        let mem = MemFile::new(KIND_HRR, vec![]).unwrap();
        let bytes = mem.encode();
        let decoded = MemFile::decode(&bytes).unwrap();
        assert_eq!(decoded.payload.len(), 0);
    }

    #[test]
    fn reject_short_input() {
        let bytes = [0u8; 5];
        assert!(matches!(
            MemFile::decode(&bytes),
            Err(MemFileError::TooShort { .. })
        ));
    }

    #[test]
    fn reject_bad_magic() {
        let mut bytes = MemFile::new(KIND_HRR, vec![1, 2, 3]).unwrap().encode();
        bytes[0] = b'X';
        assert!(matches!(
            MemFile::decode(&bytes),
            Err(MemFileError::BadMagic { .. })
        ));
    }

    #[test]
    fn reject_bad_version() {
        let mut bytes = MemFile::new(KIND_HRR, vec![1, 2, 3]).unwrap().encode();
        bytes[4] = 0xFF;
        assert!(matches!(
            MemFile::decode(&bytes),
            Err(MemFileError::BadVersion(0xFF))
        ));
    }

    #[test]
    fn reject_bad_kind() {
        assert!(matches!(
            MemFile::new(b'X', vec![]),
            Err(MemFileError::BadKind(b'X'))
        ));
    }

    #[test]
    fn reject_tampered_payload() {
        let mut bytes = MemFile::new(KIND_HRR, vec![1, 2, 3, 4, 5])
            .unwrap()
            .encode();
        // Flip a bit in the payload area.
        bytes[11] ^= 0x01;
        assert!(matches!(
            MemFile::decode(&bytes),
            Err(MemFileError::IntegrityFailure)
        ));
    }

    #[test]
    fn reject_tampered_header() {
        let mut bytes = MemFile::new(KIND_HRR, vec![1, 2, 3, 4, 5])
            .unwrap()
            .encode();
        // Flip a bit in the length field.
        bytes[6] ^= 0x01;
        // This will either produce BadPayloadLength (size mismatch)
        // or IntegrityFailure depending on which bit flipped. Both
        // are acceptable rejection modes.
        let err = MemFile::decode(&bytes).unwrap_err();
        assert!(matches!(
            err,
            MemFileError::BadPayloadLength { .. } | MemFileError::IntegrityFailure
        ));
    }
}
