//! Read-only v0a census tracer over copied immutable Gaia evidence.
//!
//! The crate has exactly one public entry point, [`census`]. It reads a copied
//! evidence directory and its declared inventory manifest, measures the raw
//! bytes of every declared file, and returns one typed deterministic advisory
//! artifact. It routes no work, mutates nothing, fits no model, contacts no
//! bus, and carries no score, verdict, freshness, or authority value.

use std::path::PathBuf;

use serde::Serialize;
use thiserror::Error;

mod controls;
mod manifest;
mod measure;

/// The workspace's canonical six-valued truth type, re-exported because it is
/// part of this crate's public surface. There is exactly one hexavalent truth
/// table in the workspace and this crate does not fork it.
pub use ix_types::Hexavalent;

/// Bumped on any breaking change to the serialized artifact shape.
pub const SCHEMA_VERSION: u32 = 1;

/// The declared inputs of one census run.
#[derive(Debug, Clone)]
pub struct CensusRequest {
    /// Directory holding the copied immutable evidence.
    pub evidence_root: PathBuf,
    /// File name, inside `evidence_root`, of the manifest declaring the inventory.
    pub manifest_file_name: String,
}

/// One typed deterministic advisory artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CensusArtifact {
    pub schema_version: u32,
    pub subject: SubjectRef,
    /// Declared rows in ordinal (raw UTF-8 byte) file-name order.
    pub rows: Vec<CensusRow>,
    pub totals: CensusTotals,
    /// SHA-256 over the LF-joined `name|bytes|sha256` records of the declared
    /// rows, in ordinal file-name order, with no trailing newline, built from
    /// the *measured* bytes.
    pub aggregate_digest_measured: String,
    /// Fold of the row states through the canonical hexavalent `and`.
    pub agreement: Hexavalent,
    /// One row per enumerated control in the union of specification §11 and
    /// engineering-doctrine §4. No control is silently omitted: a control with
    /// no subject in a read-only local census is marked `NotApplicable` and
    /// names the subject it is missing.
    pub controls: Vec<ControlCoverage>,
}

/// One control of the M2 test obligation, and how this census meets it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ControlCoverage {
    /// `spec-11-01`..`spec-11-28`, `doctrine-4-01`..`doctrine-4-14`.
    pub control_id: String,
    pub status: ControlStatus,
    /// When `Covered`, the test that discriminates the control. When
    /// `NotApplicable`, the subject this tracer does not have.
    pub detail: String,
}

/// Whether a control has a subject in this tracer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ControlStatus {
    Covered,
    NotApplicable,
}

impl CensusArtifact {
    /// The artifact's canonical serialization: one setting, no variance.
    ///
    /// Every field is a named object member and no map is serialized, so the
    /// byte sequence is a function of the measured evidence alone.
    pub fn to_canonical_json(&self) -> String {
        serde_json::to_string(self).expect("the artifact holds only strings, integers, and enums")
    }
}

/// Identity of the subject the census was taken over.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubjectRef {
    pub manifest_file_name: String,
    /// SHA-256 of the manifest file itself, measured over raw bytes.
    pub manifest_sha256: String,
    /// The same record construction as `aggregate_digest_measured`, but over
    /// every file present in the evidence root, the manifest included.
    pub ordinal_aggregate_measured: String,
}

/// One declared file, as declared and as measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CensusRow {
    pub name: String,
    pub declared_bytes: u64,
    pub measured_bytes: u64,
    pub declared_sha256: String,
    pub measured_sha256: String,
    /// `T` when declared and measured agree on both size and digest, `F` otherwise.
    pub state: Hexavalent,
}

/// Counts over the census, in bytes and files.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CensusTotals {
    pub listed_files: u64,
    pub listed_bytes: u64,
    pub measured_bytes: u64,
    /// Files present in the evidence root that the manifest does not declare.
    /// The manifest itself is not one of them: it never lists itself.
    pub unlisted_files: u64,
    /// The names of those files, in ordinal order. Reported, never ignored.
    pub unlisted_names: Vec<String>,
    /// Entries in the evidence root that are not plain files. A v0a census
    /// measures raw bytes, so it can measure no other entry class; every such
    /// entry is counted here rather than skipped, and a non-zero count demotes
    /// `agreement` exactly as an unlisted file does.
    pub non_file_entries: u64,
    /// Each of those entries, named with its class, in ordinal order.
    pub non_file_names: Vec<NonFileEntry>,
}

/// One entry in the evidence root that this census cannot measure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NonFileEntry {
    pub name: String,
    pub entry_class: EntryClass,
}

/// The class of an entry that is not a plain file. No class is elided: an
/// entry that is none of the named classes is reported as `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EntryClass {
    Directory,
    Symlink,
    Other,
}

/// A refusal. When the tracer refuses, no artifact is produced at all.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CensusRefusal {
    #[error("evidence root unreadable: {detail}")]
    EvidenceRootUnreadable { detail: String },
    #[error("manifest unparseable at line {line}: {detail}")]
    ManifestUnparseable { line: usize, detail: String },
    #[error("manifest declares no rows")]
    ManifestDeclaresNoRows,
    #[error("binder cannot resolve {field}: {detail}")]
    BinderIncomplete { field: String, detail: String },
    #[error("manifest declares {name} more than once")]
    DuplicateManifestRow { name: String },
    #[error("manifest declares an inventory section at line {first} and again at line {second}")]
    DuplicateInventorySection { first: usize, second: usize },
    #[error("manifest declares a name that is not a plain file name: {name}")]
    UnsafeManifestName { name: String },
}

/// Take a census over copied immutable evidence.
///
/// Returns one typed deterministic advisory artifact, or a refusal. A refusal
/// emits no artifact: the tracer refuses before it emits, it never degrades and
/// emits.
pub fn census(request: &CensusRequest) -> Result<CensusArtifact, CensusRefusal> {
    let listing = measure::measure_root(&request.evidence_root)?;
    let present = &listing.files;

    let manifest_measured = present.get(&request.manifest_file_name).ok_or_else(|| {
        CensusRefusal::BinderIncomplete {
            field: request.manifest_file_name.clone(),
            detail: "declared manifest is not present in the evidence root".to_string(),
        }
    })?;

    let manifest_bytes = std::fs::read(request.evidence_root.join(&request.manifest_file_name))
        .map_err(|e| CensusRefusal::EvidenceRootUnreadable {
            detail: format!("{}: {e}", request.manifest_file_name),
        })?;

    let mut declared = manifest::parse(&manifest_bytes)?;
    declared.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));

    let mut rows = Vec::with_capacity(declared.len());
    for row in &declared {
        let measured =
            present
                .get(&row.name)
                .ok_or_else(|| CensusRefusal::BinderIncomplete {
                    field: row.name.clone(),
                    detail: "declared file is not present in the evidence root".to_string(),
                })?;
        let state = if row.bytes == measured.bytes && row.sha256 == measured.sha256 {
            Hexavalent::True
        } else {
            Hexavalent::False
        };
        rows.push(CensusRow {
            name: row.name.clone(),
            declared_bytes: row.bytes,
            measured_bytes: measured.bytes,
            declared_sha256: row.sha256.clone(),
            measured_sha256: measured.sha256.clone(),
            state,
        });
    }

    let aggregate_digest_measured = aggregate(
        rows.iter()
            .map(|row| record(&row.name, row.measured_bytes, &row.measured_sha256)),
    );
    let ordinal_aggregate_measured = aggregate(
        present
            .iter()
            .map(|(name, measured)| record(name, measured.bytes, &measured.sha256)),
    );

    // Present files the manifest declares neither as a row nor as itself.
    // `BTreeMap` iteration is ordinal, so the reported order is too.
    let declared_names: std::collections::BTreeSet<&str> =
        declared.iter().map(|row| row.name.as_str()).collect();
    let unlisted_names: Vec<String> = present
        .keys()
        .filter(|name| {
            name.as_str() != request.manifest_file_name && !declared_names.contains(name.as_str())
        })
        .cloned()
        .collect();

    let totals = CensusTotals {
        listed_files: rows.len() as u64,
        listed_bytes: rows.iter().map(|row| row.declared_bytes).sum(),
        measured_bytes: rows.iter().map(|row| row.measured_bytes).sum(),
        unlisted_files: unlisted_names.len() as u64,
        unlisted_names,
        non_file_entries: listing.non_files.len() as u64,
        non_file_names: listing.non_files,
    };
    let agreement = rows
        .iter()
        .fold(Hexavalent::True, |acc, row| acc.and(row.state));
    // A root the census cannot fully read does not agree with its manifest,
    // whether the undeclared content is a file it can measure or an entry class
    // it cannot. Neither is allowed to leave `agreement` at `True`.
    let agreement = if totals.unlisted_files > 0 || totals.non_file_entries > 0 {
        agreement.and(Hexavalent::False)
    } else {
        agreement
    };

    Ok(CensusArtifact {
        schema_version: SCHEMA_VERSION,
        subject: SubjectRef {
            manifest_file_name: request.manifest_file_name.clone(),
            manifest_sha256: manifest_measured.sha256.clone(),
            ordinal_aggregate_measured,
        },
        rows,
        totals,
        aggregate_digest_measured,
        agreement,
        controls: controls::coverage(),
    })
}

/// One `name|bytes|sha256` record of the declared digest construction.
fn record(name: &str, bytes: u64, sha256: &str) -> String {
    format!("{name}|{bytes}|{sha256}")
}

/// SHA-256 over the LF-joined records, with no trailing newline.
fn aggregate(records: impl Iterator<Item = String>) -> String {
    let joined = records.collect::<Vec<_>>().join("\n");
    measure::sha256_hex(joined.as_bytes())
}
