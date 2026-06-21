//! FNV-1a 64-bit hashing for evidence provenance — stable across platforms so a later
//! audit can detect a swapped input. The evidence layer hashes its input bytes (input
//! provenance > verdict provenance), so the verdict records *what it was computed from*.

use std::path::Path;

/// FNV-1a 64-bit hash over a byte stream — the platform-stable primitive both the
/// in-memory and file hashes share.
pub(crate) fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a 64-bit hash of a file's bytes, rendered as `fnv1a64:<hex>`; `None` if absent.
pub(crate) fn fnv1a64_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(format!("fnv1a64:{:016x}", fnv1a64(bytes)))
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_is_stable_and_seeded() {
        // Empty input → the FNV offset basis (unchanged seed).
        assert_eq!(fnv1a64(std::iter::empty()), 0xcbf2_9ce4_8422_2325);
        // Known vector: "a" → 0xaf63dc4c8601ec8c (standard FNV-1a 64).
        assert_eq!(fnv1a64(*b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn fnv1a64_file_renders_prefixed_hex_and_none_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.bin");
        std::fs::write(&p, b"a").unwrap();
        assert_eq!(fnv1a64_file(&p).as_deref(), Some("fnv1a64:af63dc4c8601ec8c"));
        assert_eq!(fnv1a64_file(&dir.path().join("nope")), None);
    }
}
