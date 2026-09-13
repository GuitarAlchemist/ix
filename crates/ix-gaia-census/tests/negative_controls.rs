//! Negative controls for the v0a census tracer.
//!
//! Every test calls only `ix_gaia_census::census`. Every mutation is applied to
//! a copy of the fixture inside a `tempfile::tempdir()`; the checked-in evidence
//! fixture is never mutated by any test.

use std::fs;
use std::path::{Path, PathBuf};

use ix_gaia_census::{census, CensusRefusal, CensusRequest, EntryClass, Hexavalent, NonFileEntry};
use tempfile::TempDir;

const MANIFEST: &str = "gaia-s1-r2-bundle-manifest.md";
const GRAMMAR: &str = "gaia-uncertainty-grammar-v0.1.ebnf";

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gaia-s1-r18")
}

fn request_for(root: &Path) -> CensusRequest {
    CensusRequest {
        evidence_root: root.to_path_buf(),
        manifest_file_name: MANIFEST.to_string(),
    }
}

/// A writable copy of the fourteen fixture files.
fn staged_copy() -> TempDir {
    let stage = tempfile::tempdir().expect("tempdir");
    for entry in fs::read_dir(fixture_root()).expect("fixture root") {
        let entry = entry.expect("fixture entry");
        let staged = stage.path().join(entry.file_name());
        fs::copy(entry.path(), &staged).expect("copy fixture file");
        // The fixture files carry the read-only attribute of the immutable
        // subject they were copied from, and `fs::copy` preserves it. The stage
        // is the mutable copy, so clear it there and only there.
        let mut permissions = fs::metadata(&staged).expect("staged metadata").permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        permissions.set_readonly(false);
        fs::set_permissions(&staged, permissions).expect("stage is writable");
    }
    stage
}

/// Rewrite the staged manifest, replacing `find` with `replace_with` once.
fn rewrite_manifest(stage: &TempDir, find: &str, replace_with: &str) {
    let path = stage.path().join(MANIFEST);
    let text = fs::read_to_string(&path).expect("read staged manifest");
    assert!(text.contains(find), "staged manifest must contain {find:?}");
    fs::write(&path, text.replacen(find, replace_with, 1)).expect("write staged manifest");
}

/// The staged manifest's inventory row for `name`, verbatim.
fn inventory_row(stage: &TempDir, name: &str) -> String {
    let text = fs::read_to_string(stage.path().join(MANIFEST)).expect("read staged manifest");
    text.lines()
        .find(|line| line.starts_with('|') && line.contains(&format!("`{name}`")))
        .unwrap_or_else(|| panic!("no inventory row for {name}"))
        .to_string()
}

/// Manifest §3, the declared fixed point over the thirteen listed files.
const DECLARED_AGGREGATE: &str =
    "970bb50c3f40aac8da50f5822fc8e6ad48f7e27643dd67176591a7788f0b311c";

/// SHA-256 of the empty byte string — a value, not a failure.
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// The 0-based line indices of the manifest's §2 Inventory section.
fn inventory_line_span(text: &str) -> std::ops::Range<usize> {
    let start = text
        .lines()
        .position(|line| line.trim() == "## 2. Inventory")
        .expect("the manifest has an inventory section");
    let end = text
        .lines()
        .enumerate()
        .skip(start + 1)
        .find(|(_, line)| line.starts_with("## "))
        .map(|(index, _)| index)
        .unwrap_or(text.lines().count());
    start..end
}

/// The 1-based line number of the first line containing `needle`.
fn line_number_of(stage: &TempDir, needle: &str) -> usize {
    let text = fs::read_to_string(stage.path().join(MANIFEST)).expect("read staged manifest");
    text.lines()
        .position(|line| line.contains(needle))
        .map(|index| index + 1)
        .unwrap_or_else(|| panic!("no line containing {needle:?}"))
}

#[test]
fn absent_declared_file_refuses_and_emits_no_artifact() {
    let stage = staged_copy();
    fs::remove_file(stage.path().join(GRAMMAR)).expect("remove a declared file");

    match census(&request_for(stage.path())) {
        Err(CensusRefusal::BinderIncomplete { field, .. }) => assert_eq!(field, GRAMMAR),
        Err(other) => panic!("a missing declared file must refuse as binder-incomplete: {other}"),
        Ok(_) => panic!("a missing declared file must emit no artifact"),
    }
}

#[test]
fn malformed_manifest_rows_refuse_with_the_offending_line() {
    let row = {
        let stage = staged_copy();
        inventory_row(&stage, GRAMMAR)
    };
    let cases = [
        // digest one character short
        format!("| `{GRAMMAR}` | 10,322 | `ee9f61b6be06aed76cdc0493c95c1540c6b148c31d75801ef9b67729b739451` |"),
        // two cells instead of three
        format!("| `{GRAMMAR}` | 10,322 |"),
        // byte count that is not a non-negative integer
        format!("| `{GRAMMAR}` | ten thousand | `ee9f61b6be06aed76cdc0493c95c1540c6b148c31d75801ef9b67729b7394517` |"),
    ];

    for malformed in cases {
        let stage = staged_copy();
        rewrite_manifest(&stage, &row, &malformed);
        let expected_line = line_number_of(&stage, &malformed);

        match census(&request_for(stage.path())) {
            Err(CensusRefusal::ManifestUnparseable { line, .. }) => assert_eq!(
                line, expected_line,
                "the refusal must name the offending line of {malformed:?}"
            ),
            Err(other) => panic!("{malformed:?} must refuse as unparseable, got {other}"),
            Ok(_) => panic!("{malformed:?} must emit no artifact"),
        }
    }
}

#[test]
fn a_manifest_declaring_no_rows_refuses() {
    let stage = staged_copy();
    rewrite_manifest(&stage, "## 2. Inventory", "## 2. Inventory withdrawn");

    match census(&request_for(stage.path())) {
        Err(CensusRefusal::ManifestDeclaresNoRows) => {}
        other => panic!("a manifest declaring no row must refuse, got {other:?}"),
    }
}

#[test]
fn an_unreadable_evidence_root_refuses() {
    let stage = staged_copy();
    let absent = stage.path().join("no-such-directory");

    match census(&request_for(&absent)) {
        Err(CensusRefusal::EvidenceRootUnreadable { .. }) => {}
        other => panic!("an unreadable evidence root must refuse, got {other:?}"),
    }
}

#[test]
fn a_single_byte_mutation_marks_exactly_one_row_false() {
    let stage = staged_copy();
    let path = stage.path().join(GRAMMAR);
    let mut bytes = fs::read(&path).expect("read staged file");
    bytes[0] ^= 0x01;
    fs::write(&path, &bytes).expect("write mutated file");

    let artifact = census(&request_for(stage.path())).expect("drift is observed, not refused");

    let false_rows: Vec<&str> = artifact
        .rows
        .iter()
        .filter(|row| row.state == Hexavalent::False)
        .map(|row| row.name.as_str())
        .collect();
    assert_eq!(false_rows, vec![GRAMMAR]);
    assert_eq!(artifact.agreement, Hexavalent::False);
    assert_ne!(
        artifact.aggregate_digest_measured, DECLARED_AGGREGATE,
        "one mutated byte must move the aggregate"
    );
}

#[test]
fn a_mutated_declared_digest_retains_both_values() {
    let stage = staged_copy();
    let declared = "ee9f61b6be06aed76cdc0493c95c1540c6b148c31d75801ef9b67729b7394517";
    let drifted = "ee9f61b6be06aed76cdc0493c95c1540c6b148c31d75801ef9b67729b7394510";
    rewrite_manifest(&stage, declared, drifted);

    let artifact = census(&request_for(stage.path())).expect("drift is observed, not refused");

    let row = artifact
        .rows
        .iter()
        .find(|row| row.name == GRAMMAR)
        .expect("the declared row survives");
    assert_eq!(row.state, Hexavalent::False);
    assert_eq!(row.declared_sha256, drifted, "the declared value is retained");
    assert_eq!(row.measured_sha256, declared, "the measured value is retained");
    assert_eq!(
        artifact.aggregate_digest_measured, DECLARED_AGGREGATE,
        "the bytes on disk did not move, so the measured aggregate must not either"
    );
}

#[test]
fn a_zero_byte_declared_file_is_true_not_a_failure() {
    let stage = staged_copy();
    fs::write(stage.path().join("empty-value.txt"), b"").expect("write an empty file");
    let row = inventory_row(&stage, GRAMMAR);
    rewrite_manifest(
        &stage,
        &row,
        &format!("{row}\n| `empty-value.txt` | 0 | `{EMPTY_SHA256}` |"),
    );

    let artifact = census(&request_for(stage.path())).expect("an empty value is not a refusal");

    let row = artifact
        .rows
        .iter()
        .find(|row| row.name == "empty-value.txt")
        .expect("the empty file is declared and measured");
    assert_eq!(row.state, Hexavalent::True);
    assert_eq!(row.measured_bytes, 0);
    assert_eq!(artifact.totals.unlisted_files, 0);
}

#[test]
fn reordered_manifest_rows_produce_an_identical_census() {
    let stage = staged_copy();
    let ordered = census(&request_for(stage.path())).expect("ordered manifest");

    let path = stage.path().join(MANIFEST);
    let text = fs::read_to_string(&path).expect("read staged manifest");
    let inventory = inventory_line_span(&text);
    let is_data_row = |index: usize, line: &str| {
        inventory.contains(&index) && line.starts_with("| `gaia-")
    };
    let mut data_rows: Vec<String> = text
        .lines()
        .enumerate()
        .filter(|(index, line)| is_data_row(*index, line))
        .map(|(_, line)| line.to_string())
        .collect();
    assert_eq!(data_rows.len(), 13, "the fixture declares thirteen rows");
    data_rows.reverse();
    let mut remaining = data_rows.into_iter();
    let shuffled: Vec<String> = text
        .lines()
        .enumerate()
        .map(|(index, line)| {
            if is_data_row(index, line) {
                remaining.next().expect("one replacement per data row")
            } else {
                line.to_string()
            }
        })
        .collect();
    fs::write(&path, shuffled.join("\n")).expect("write shuffled manifest");

    let shuffled = census(&request_for(stage.path())).expect("shuffled manifest");

    assert_eq!(shuffled.rows, ordered.rows, "rows are in ordinal order");
    assert_eq!(shuffled.totals, ordered.totals);
    assert_eq!(
        shuffled.aggregate_digest_measured,
        ordered.aggregate_digest_measured
    );
}

#[test]
fn an_unlisted_file_is_reported_rather_than_ignored() {
    let stage = staged_copy();
    fs::write(stage.path().join("fifteenth-entry.txt"), b"debris").expect("write unlisted file");

    let artifact = census(&request_for(stage.path())).expect("an unlisted file is not a refusal");

    assert_eq!(artifact.totals.unlisted_files, 1);
    assert_eq!(artifact.totals.unlisted_names, vec!["fifteenth-entry.txt"]);
    assert_eq!(
        artifact.aggregate_digest_measured, DECLARED_AGGREGATE,
        "the declared rows are untouched, so their aggregate must not move"
    );
    assert_ne!(
        artifact.agreement,
        Hexavalent::True,
        "a directory carrying an undeclared file does not agree with its manifest"
    );
}

#[test]
fn a_non_file_entry_is_reported_rather_than_ignored() {
    let stage = staged_copy();
    fs::create_dir(stage.path().join("build")).expect("create an empty subdirectory");
    let nested = stage.path().join("undeclared-evidence");
    fs::create_dir(&nested).expect("create a populated subdirectory");
    fs::write(nested.join("payload.bin"), b"undeclared evidence").expect("write a nested payload");

    let artifact = census(&request_for(stage.path())).expect("a non-file entry is not a refusal");

    assert_ne!(
        artifact.agreement,
        Hexavalent::True,
        "a root carrying entries the census cannot measure does not agree with its manifest"
    );
    assert_eq!(artifact.totals.non_file_entries, 2);
    assert_eq!(
        artifact.totals.non_file_names,
        vec![
            NonFileEntry {
                name: "build".to_string(),
                entry_class: EntryClass::Directory,
            },
            NonFileEntry {
                name: "undeclared-evidence".to_string(),
                entry_class: EntryClass::Directory,
            },
        ],
        "every unsupported entry is named with its class, in ordinal order"
    );
    assert_eq!(
        artifact.totals.unlisted_files, 0,
        "a nested payload is not an entry of the evidence root"
    );
    assert_eq!(
        artifact.aggregate_digest_measured, DECLARED_AGGREGATE,
        "the declared rows are untouched, so their aggregate must not move"
    );
    let json = artifact.to_canonical_json();
    for named in ["build", "undeclared-evidence"] {
        assert!(
            json.contains(named),
            "the artifact must name the {named:?} entry it cannot measure; got {json}"
        );
    }
}

#[test]
fn a_second_inventory_section_refuses_rather_than_collapsing_into_a_pass() {
    let stage = staged_copy();
    let path = stage.path().join(MANIFEST);
    let text = fs::read_to_string(&path).expect("read staged manifest");
    let inventory = inventory_line_span(&text);
    let contradiction: Vec<String> = text
        .lines()
        .enumerate()
        .filter(|(index, _)| inventory.contains(index))
        .map(|(_, line)| contradicting_row(line))
        .collect();
    fs::write(&path, format!("{text}\n{}\n", contradiction.join("\n")))
        .expect("write the twice-declared manifest");

    let headings: Vec<usize> = fs::read_to_string(&path)
        .expect("read the twice-declared manifest")
        .lines()
        .enumerate()
        .filter(|(_, line)| line.trim() == "## 2. Inventory")
        .map(|(index, _)| index + 1)
        .collect();
    assert_eq!(headings.len(), 2, "the staged manifest declares §2 twice");

    match census(&request_for(stage.path())) {
        Err(CensusRefusal::DuplicateInventorySection { first, second }) => {
            assert_eq!(
                first, headings[0],
                "the refusal names the first declaration"
            );
            assert_eq!(
                second, headings[1],
                "and the second, so neither is chosen over the other"
            );
        }
        Err(other) => panic!("a second inventory section must refuse as a duplicate: {other}"),
        Ok(artifact) => panic!(
            "a second inventory section must not collapse into a pass: agreement={:?}, rows={}",
            artifact.agreement,
            artifact.rows.len()
        ),
    }
}

/// An inventory line with any declared digest altered, so that re-declaring the
/// line contradicts the first declaration rather than merely repeating it.
fn contradicting_row(line: &str) -> String {
    line.split('|')
        .map(|cell| {
            let bare = cell.trim().trim_matches('`');
            if bare.len() == 64 && bare.chars().all(|c| c.is_ascii_hexdigit()) {
                let last = if bare.ends_with('0') { '1' } else { '0' };
                format!(" `{}{last}` ", &bare[..63])
            } else {
                cell.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("|")
}

#[test]
fn tables_outside_the_inventory_section_are_still_ignored() {
    let stage = staged_copy();
    let path = stage.path().join(MANIFEST);
    let text = fs::read_to_string(&path).expect("read staged manifest");
    let not_an_inventory = format!(
        "\n## 8. Not an inventory\n\n| Relative path | Bytes | SHA-256 |\n| --- | --- | --- |\n| `{GRAMMAR}` | 1 | `{}` |\n",
        "0".repeat(64)
    );
    fs::write(&path, format!("{text}{not_an_inventory}")).expect("write the extended manifest");

    let artifact = census(&request_for(stage.path())).expect("a table outside §2 declares nothing");

    assert_eq!(
        artifact.totals.listed_files, 13,
        "only the §2 inventory declares rows"
    );
    assert_eq!(artifact.agreement, Hexavalent::True);
}

#[test]
fn manifest_declaring_a_name_outside_the_evidence_root_refuses() {
    for escaping in [
        "../gaia-uncertainty-grammar-v0.1.ebnf",
        "nested/gaia-uncertainty-grammar-v0.1.ebnf",
        "..\\gaia-uncertainty-grammar-v0.1.ebnf",
        "/etc/passwd",
    ] {
        let stage = staged_copy();
        rewrite_manifest(&stage, &format!("`{GRAMMAR}`"), &format!("`{escaping}`"));

        match census(&request_for(stage.path())) {
            Err(CensusRefusal::UnsafeManifestName { name }) => assert_eq!(name, escaping),
            Err(other) => panic!("{escaping:?} must refuse as an unsafe name, got {other}"),
            Ok(_) => panic!("{escaping:?} must refuse, an artifact was emitted"),
        }
    }
}

#[test]
fn manifest_declaring_the_same_name_twice_refuses() {
    let stage = staged_copy();
    let row = inventory_row(&stage, GRAMMAR);
    rewrite_manifest(&stage, &row, &format!("{row}\n{row}"));

    let result = census(&request_for(stage.path()));

    match result {
        Err(CensusRefusal::DuplicateManifestRow { name }) => assert_eq!(name, GRAMMAR),
        other => panic!("a duplicate declaration must refuse, got {other:?}"),
    }
}
