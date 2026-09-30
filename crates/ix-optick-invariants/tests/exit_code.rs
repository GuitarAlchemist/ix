//! Runs the binary on a tiny synthetic OPTK v4 index that has dead dims, to
//! pin the exit-code contract `ix-autoresearch` relies on: #37/#38 failures
//! exit 0 by default and 1 only with `--fail-on-dead`.

use std::path::PathBuf;
use std::process::Command;

const DIM: usize = 124;

/// Minimal OPTK v4 image: 3 guitar voicings, no metadata records (so the
/// PC-set invariants #25/#32/#36 are vacuous), CONTEXT dims 48..60 always zero.
fn write_index(name: &str) -> PathBuf {
    let count = 3usize;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"OPTK");
    buf.extend_from_slice(&4u32.to_le_bytes()); // version
    buf.extend_from_slice(&616u32.to_le_bytes()); // header_size
    buf.extend_from_slice(&ix_optick::compute_schema_hash().to_le_bytes());
    buf.extend_from_slice(&0xFEFFu16.to_le_bytes()); // endian marker
    buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
    buf.extend_from_slice(&(DIM as u32).to_le_bytes());
    buf.extend_from_slice(&(count as u64).to_le_bytes());
    buf.push(3u8); // instruments
    buf.extend_from_slice(&[0u8; 7]);

    let header_size = 616u64;
    let offsets_table = header_size;
    let vectors_offset = offsets_table + (count as u64) * 8;
    let metadata_offset = vectors_offset + (count * DIM * 4) as u64;

    // Instrument slices: all voicings are guitar.
    buf.extend_from_slice(&vectors_offset.to_le_bytes());
    buf.extend_from_slice(&(count as u64).to_le_bytes());
    for _ in 0..2 {
        buf.extend_from_slice(&metadata_offset.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
    }
    buf.extend_from_slice(&offsets_table.to_le_bytes());
    buf.extend_from_slice(&vectors_offset.to_le_bytes());
    buf.extend_from_slice(&metadata_offset.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_length
    for _ in 0..DIM {
        buf.extend_from_slice(&0.5f32.to_le_bytes()); // header scales
    }
    assert_eq!(buf.len() as u64, header_size);

    buf.extend_from_slice(&vec![0u8; count * 8]); // metadata offsets table
    for row in 0..count {
        for d in 0..DIM {
            let x = if (48..60).contains(&d) {
                0.0f32
            } else {
                row as f32 * 0.1 + 0.01
            };
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }

    let path = std::env::temp_dir().join(format!(
        "ix_optick_invariants_{}_{}.optk",
        name,
        std::process::id()
    ));
    std::fs::write(&path, &buf).unwrap();
    path
}

fn run(index: &PathBuf, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ix-optick-invariants"))
        .arg("--index")
        .arg(index)
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn dead_dims_exit_zero_by_default_and_one_with_fail_on_dead() {
    let index = write_index("exit");

    let default = run(&index, &[]);
    let stderr = String::from_utf8_lossy(&default.stderr);
    assert!(
        stderr.contains("invariant #37: 112/124 compact dims PASS, 12 FAIL"),
        "{stderr}"
    );
    assert!(
        stderr.contains("invariant #38: 5/6 partitions PASS, 1 FAIL"),
        "{stderr}"
    );
    assert_eq!(default.status.code(), Some(0), "{stderr}");

    let strict = run(&index, &["--fail-on-dead"]);
    assert_eq!(
        strict.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&strict.stderr)
    );

    let _ = std::fs::remove_file(&index);
}

#[test]
fn negative_or_non_finite_tolerance_is_rejected() {
    let index = write_index("tolerance");
    for bad in ["--dead-tolerance=-1", "--dead-tolerance=NaN"] {
        let out = run(&index, &[bad]);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{bad}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let _ = std::fs::remove_file(&index);
}
