//! The corpus generator — a **generator, never a labeller**.
//!
//! Every family here models a drift shape the approved evidence documents, and
//! every family's site set is exactly what its citation identifies as an
//! instance of that shape. A token match is not a site, and **a target is not a
//! site**: `F-DEBRIS`'s citation describes one act on one stream, so the family
//! has one site, not one per file it could have been applied to. Looping over
//! the targets is the natural way to write a perturbation generator and it is
//! wrong here; it looks exactly like thoroughness and inflates every count.
//!
//! Generation is **total**. For every enumerated triple the generator produces
//! exactly one of an admitted case or a **typed rejection**. It never aborts and
//! it never assigns a label: labels come from the oracle, over the staged bytes,
//! and are sealed into the corpus at lock time.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ix_gaia_advisory::Hexavalent;

use super::digest::{ix_agg_1, sha256_hex};
use super::{approved_evidence_root, oracle, MANIFEST_FILE_NAME};

/// Bumped whenever a site set changes, so a case from an older enumeration can
/// never share an identity with a case from this one.
pub const CASE_ID_SALT: &str = "gaia-m3-case-v4";

/// The single licensed `F-DEBRIS` target: the program whose stdout the citation
/// names. `…-output.txt` is the *product* of redirecting outside the directory,
/// which the same citation endorses, so it is not the debris act.
pub const DEBRIS_TARGET: &str = "gaia-s1-r2-validator.py";
pub const DEBRIS_STAGED_FILE: &str = "gaia-s1-r2-validator.py.stdout";
/// unit: bytes — the declared generator constant for the debris file's content.
pub const DEBRIS_BYTES: usize = 64;

/// The `F-RESTAMP` target and the seven lines manifest L63 cites.
pub const RESTAMP_TARGET: &str = "gaia-consolidated-mission-room-factory-spec-v0.2.md";
pub const RESTAMP_CITED_LINES: [u32; 7] = [10, 16, 18, 474, 484, 571, 574];

/// The `F-STALE-FIGURE` target and the three census claims manifest L112 cites.
///
/// The manifest cites `ledger:551, :555, :645`; those physical lines are empty
/// and the `210`-bearing claims sit one line later, so the manifest→ledger
/// citation base is **0-based** while the manifest→spec base is 1-based. Both
/// bases are measured and stated rather than assumed. The 1-based physical
/// lines are used throughout.
pub const STALE_TARGET: &str = "gaia-s1-r2-change-ledger.md";
pub const STALE_CITED_LINES: [u32; 3] = [552, 556, 646];
/// The dated figure, and the current one. Both are citation-derived: manifest
/// L110 records the history and L112 records that rewriting exactly these three
/// figures "was considered and refused". The perturbation is precisely the edit
/// the approved evidence describes and declines to make.
pub const STALE_FROM: &[u8; 3] = b"210";
pub const STALE_TO: &[u8; 3] = b"214";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    /// Every bare LF becomes CRLF — identical displayed characters, different
    /// SHA-256. Cited as a census over all fourteen files, so fourteen sites.
    Crlf,
    /// A fifteenth file appears in the root.
    Debris,
    /// In-place edit at a documented re-stamp site, line count preserved.
    Restamp,
    /// A round-scoped census figure rewritten at a documented claim.
    StaleFigure,
    /// The positive control: the unmodified root. Not a drift family.
    Pristine,
}

impl Family {
    pub fn id(self) -> &'static str {
        match self {
            Family::Crlf => "F-CRLF",
            Family::Debris => "F-DEBRIS",
            Family::Restamp => "F-RESTAMP",
            Family::StaleFigure => "F-STALE-FIGURE",
            Family::Pristine => "F-PRISTINE",
        }
    }
}

/// The 1-based line's byte range in `bytes`, LF excluded.
///
/// Byte offsets into a byte array, never character offsets into a decoded
/// string: the two diverge by hundreds of bytes in these files, and a
/// preregistration that specifies edits at byte granularity cannot carry an
/// offset that is not a byte offset.
pub fn line_range(bytes: &[u8], line: u32) -> Option<(usize, usize)> {
    let mut start = 0usize;
    let mut current = 1u32;
    let mut index = 0usize;
    while current < line {
        match bytes[index..].iter().position(|byte| *byte == b'\n') {
            Some(offset) => {
                index += offset + 1;
                start = index;
                current += 1;
            }
            None => return None,
        }
        if start >= bytes.len() {
            return None;
        }
    }
    let end = bytes[start..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map(|offset| start + offset)
        .unwrap_or(bytes.len());
    Some((start, end))
}

/// Every `210` token offset the `F-STALE-FIGURE` family rewrites, ordinal.
///
/// Four tokens across three claims: the claim at ledger line 646 carries the
/// figure twice and both are rewritten, so the staged line stays internally
/// consistent. Rewriting only one would also be labelled `F`, so nothing in the
/// measurement depends on the choice; it is fixed so no implementer has to make
/// it.
pub fn stale_figure_token_offsets() -> Vec<usize> {
    let bytes =
        std::fs::read(approved_evidence_root().join(STALE_TARGET)).expect("ledger readable");
    let mut out = Vec::new();
    for line in STALE_CITED_LINES {
        let (start, end) = line_range(&bytes, line).expect("the cited line exists");
        out.extend(
            token_offsets(&bytes[start..end], STALE_FROM)
                .into_iter()
                .map(|o| start + o),
        );
    }
    out.sort_unstable();
    out
}

fn token_offsets(haystack: &[u8], needle: &[u8; 3]) -> Vec<usize> {
    (0..haystack.len().saturating_sub(needle.len() - 1))
        .filter(|index| &haystack[*index..index + needle.len()] == needle.as_slice())
        .collect()
}

/// One enumerated `(family, target, site)` triple.
#[derive(Debug, Clone)]
pub struct Site {
    pub family: Family,
    pub target: String,
    pub site: String,
}

#[derive(Debug, Clone)]
pub struct AdmittedCase {
    pub case_id: String,
    pub family: Family,
    pub target: String,
    pub site: String,
    /// Computed by the oracle from the staged bytes, sealed at lock time.
    pub label: Hexavalent,
    /// `IX-AGG-1` over the staged root.
    pub staged_aggregate: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectionKind {
    /// The family does not apply to this target. Recorded, not mislabelled.
    NotApplicable(&'static str),
    /// The staged root would equal the pristine root.
    NoOpEdit,
    /// A later case by ordinal `case_id` collides with an earlier one.
    DuplicateStagedAggregate { collides_with: String },
    /// The cited site is not resolvable in the current bytes.
    SiteAbsent,
}

#[derive(Debug, Clone)]
pub struct RejectedCase {
    pub family: Family,
    pub target: String,
    pub site: String,
    pub kind: RejectionKind,
}

/// A locked corpus.
pub struct Corpus {
    pub root: PathBuf,
    /// unit: count — every `(family, target, site)` triple the licence yields.
    pub enumerated: usize,
    pub admitted: Vec<AdmittedCase>,
    pub rejected: Vec<RejectedCase>,
    /// `IX-AGG-1` over the corpus tree, the rejection ledger included.
    pub digest: String,
}

impl Corpus {
    pub fn truth_balance(&self) -> (usize, usize) {
        let t = self
            .admitted
            .iter()
            .filter(|case| case.label == Hexavalent::True)
            .count();
        (t, self.admitted.len() - t)
    }
}

pub struct Options {
    pub families: Vec<Family>,
    /// Stage one extra case that is byte-identical to an admitted one, so the
    /// duplicate rule has something to reject. A one-case family cannot collide
    /// with itself, so the collision must be injected rather than discovered.
    pub inject_duplicate: bool,
}

impl Options {
    pub fn family(family: Family) -> Self {
        Self {
            families: vec![family],
            inject_duplicate: false,
        }
    }
}

/// `case_id = sha256(salt ‖ family ‖ NUL ‖ target ‖ NUL ‖ site ‖ NUL ‖ target_declared_sha256)`
pub fn case_id(family: Family, target: &str, site: &str, target_declared_sha256: &str) -> String {
    let mut input = Vec::new();
    input.extend_from_slice(CASE_ID_SALT.as_bytes());
    input.extend_from_slice(family.id().as_bytes());
    input.push(0);
    input.extend_from_slice(target.as_bytes());
    input.push(0);
    input.extend_from_slice(site.as_bytes());
    input.push(0);
    input.extend_from_slice(target_declared_sha256.as_bytes());
    sha256_hex(&input)
}

/// The pristine bundle, read once.
pub fn pristine_files() -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for entry in std::fs::read_dir(approved_evidence_root()).expect("approved evidence readable") {
        let entry = entry.expect("entry readable");
        out.insert(
            entry
                .file_name()
                .to_str()
                .expect("fixture names are UTF-8")
                .to_string(),
            std::fs::read(entry.path()).expect("fixture readable"),
        );
    }
    out
}

/// The manifest's declared SHA-256 for each bundle file, plus the manifest's
/// own, so every case identity is anchored to a declared value.
pub fn declared_sha256s(pristine: &BTreeMap<String, Vec<u8>>) -> BTreeMap<String, String> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for (name, bytes) in pristine {
        out.insert(name.clone(), sha256_hex(bytes));
    }
    out
}

/// Every `(family, target, site)` triple the citation licences.
pub fn enumerate(options: &Options) -> Vec<Site> {
    let mut sites = Vec::new();
    for family in &options.families {
        match family {
            // Manifest L105 / L107-108 is a census over all fourteen files,
            // declaring which of them carry bare LF. The cardinality comes from
            // the citation here exactly as it does everywhere else; the two
            // families differ because the two citations differ.
            Family::Crlf => {
                for target in pristine_files().keys() {
                    sites.push(Site {
                        family: Family::Crlf,
                        target: target.clone(),
                        site: "all-bare-lf".to_string(),
                    });
                }
            }
            // Manifest L101 documents one act, one stream, one fifteenth file,
            // one recorded instance. One site.
            Family::Debris => sites.push(Site {
                family: Family::Debris,
                target: DEBRIS_TARGET.to_string(),
                site: "validator-stdout-redirect".to_string(),
            }),
            // Manifest L63 cites seven lines of the specification.
            Family::Restamp => {
                for line in RESTAMP_CITED_LINES {
                    sites.push(Site {
                        family: Family::Restamp,
                        target: RESTAMP_TARGET.to_string(),
                        site: format!("spec-v0.2:{line}"),
                    });
                }
            }
            // Manifest L112 cites three census claims, corroborated by the
            // ledger's own statement that the string appears "in three places".
            Family::StaleFigure => {
                for line in STALE_CITED_LINES {
                    sites.push(Site {
                        family: Family::StaleFigure,
                        target: STALE_TARGET.to_string(),
                        site: format!("ledger:{line}"),
                    });
                }
            }
            Family::Pristine => sites.push(Site {
                family: Family::Pristine,
                target: String::new(),
                site: "root".to_string(),
            }),
        }
    }
    if options.inject_duplicate {
        sites.push(Site {
            family: Family::Debris,
            target: DEBRIS_TARGET.to_string(),
            site: "injected-duplicate-for-nc-4".to_string(),
        });
    }
    sites
}

/// Stage one site, or say precisely why it cannot be staged.
fn stage(
    site: &Site,
    pristine: &BTreeMap<String, Vec<u8>>,
) -> Result<BTreeMap<String, Vec<u8>>, RejectionKind> {
    match site.family {
        Family::Pristine => Ok(pristine.clone()),

        Family::Debris => {
            let source = pristine
                .get(&site.target)
                .ok_or(RejectionKind::SiteAbsent)?;
            if source.len() < DEBRIS_BYTES {
                return Err(RejectionKind::SiteAbsent);
            }
            let mut files = pristine.clone();
            files.insert(
                DEBRIS_STAGED_FILE.to_string(),
                source[..DEBRIS_BYTES].to_vec(),
            );
            Ok(files)
        }

        Family::Crlf => {
            let source = pristine
                .get(&site.target)
                .ok_or(RejectionKind::SiteAbsent)?;
            // A file that already carries no bare LF has nothing for this
            // family to do. Recorded, not mislabelled, not aborted on.
            let bare = source
                .iter()
                .enumerate()
                .any(|(index, byte)| *byte == b'\n' && (index == 0 || source[index - 1] != b'\r'));
            if !bare {
                return Err(RejectionKind::NotApplicable("NoBareLf"));
            }
            let mut rewritten = Vec::with_capacity(source.len());
            for (index, byte) in source.iter().enumerate() {
                if *byte == b'\n' && (index == 0 || source[index - 1] != b'\r') {
                    rewritten.push(b'\r');
                }
                rewritten.push(*byte);
            }
            let mut files = pristine.clone();
            files.insert(site.target.clone(), rewritten);
            Ok(files)
        }

        // Site-selection rule, total and deterministic: the first ASCII digit
        // at or after the start of the cited line and strictly before that
        // line's terminating LF. Substitution: `0x30 + ((d - 0x30 + 1) mod 10)`
        // — one byte out, one byte in.
        //
        // The equal-length edit is a **declared generator constant**, not a
        // citation claim: manifest L63 documents a line-preserving re-stamp
        // that was explicitly not byte-length-preserving. This narrowing is
        // disclosed rather than smuggled, and it is count-neutral.
        Family::Restamp => {
            let line: u32 = site
                .site
                .rsplit(':')
                .next()
                .and_then(|text| text.parse().ok())
                .ok_or(RejectionKind::SiteAbsent)?;
            let source = pristine
                .get(&site.target)
                .ok_or(RejectionKind::SiteAbsent)?;
            let (start, end) = line_range(source, line).ok_or(RejectionKind::SiteAbsent)?;
            let offset = (start..end)
                .find(|index| source[*index].is_ascii_digit())
                .ok_or(RejectionKind::SiteAbsent)?;

            let mut rewritten = source.clone();
            rewritten[offset] = b'0' + ((source[offset] - b'0' + 1) % 10);
            let mut files = pristine.clone();
            files.insert(site.target.clone(), rewritten);
            Ok(files)
        }

        // Within the cited claim's line byte range, **every** occurrence of the
        // dated token is replaced by the current one. A claim is one case,
        // however many tokens it carries.
        Family::StaleFigure => {
            let line: u32 = site
                .site
                .rsplit(':')
                .next()
                .and_then(|text| text.parse().ok())
                .ok_or(RejectionKind::SiteAbsent)?;
            let source = pristine
                .get(&site.target)
                .ok_or(RejectionKind::SiteAbsent)?;
            let (start, end) = line_range(source, line).ok_or(RejectionKind::SiteAbsent)?;
            let offsets = token_offsets(&source[start..end], STALE_FROM);
            if offsets.is_empty() {
                return Err(RejectionKind::SiteAbsent);
            }

            let mut rewritten = source.clone();
            for offset in offsets {
                rewritten[start + offset..start + offset + STALE_TO.len()]
                    .copy_from_slice(STALE_TO);
            }
            let mut files = pristine.clone();
            files.insert(site.target.clone(), rewritten);
            Ok(files)
        }
    }
}

/// Generate, stage, seal labels, and lock.
pub fn generate(dest: &Path, options: &Options) -> Corpus {
    if dest.exists() {
        std::fs::remove_dir_all(dest).expect("corpus root is removable");
    }
    std::fs::create_dir_all(dest.join("dev")).expect("corpus root is creatable");

    let pristine = pristine_files();
    let declared = declared_sha256s(&pristine);
    let frozen_manifest = pristine
        .get(MANIFEST_FILE_NAME)
        .expect("the bundle carries its manifest")
        .clone();

    let sites = enumerate(options);
    let enumerated = sites.len();

    let mut admitted: Vec<AdmittedCase> = Vec::new();
    let mut rejected: Vec<RejectedCase> = Vec::new();

    for site in &sites {
        let declared_sha = declared.get(&site.target).cloned().unwrap_or_default();
        let id = case_id(site.family, &site.target, &site.site, &declared_sha);

        let files = match stage(site, &pristine) {
            Ok(files) => files,
            Err(kind) => {
                rejected.push(RejectedCase {
                    family: site.family,
                    target: site.target.clone(),
                    site: site.site.clone(),
                    kind,
                });
                continue;
            }
        };
        // The positive control *is* the pristine root, so it is exempt by
        // construction — it is the reference every other case is a
        // perturbation of, not a perturbation that failed to change anything.
        if files == pristine && site.family != Family::Pristine {
            rejected.push(RejectedCase {
                family: site.family,
                target: site.target.clone(),
                site: site.site.clone(),
                kind: RejectionKind::NoOpEdit,
            });
            continue;
        }

        let staged_dir = dest.join("dev").join(&id);
        std::fs::create_dir_all(&staged_dir).expect("staged root is creatable");
        for (name, bytes) in &files {
            std::fs::write(staged_dir.join(name), bytes).expect("staged file is writable");
        }

        admitted.push(AdmittedCase {
            case_id: id,
            family: site.family,
            target: site.target.clone(),
            site: site.site.clone(),
            label: oracle::label(&staged_dir, MANIFEST_FILE_NAME, &frozen_manifest),
            staged_aggregate: ix_agg_1(&staged_dir),
        });
    }

    // Duplicate handling is a rejection, not an abort: the later case by
    // ordinal `case_id` is rejected and names the earlier, and generation
    // completes. Double-counting is still prevented.
    admitted.sort_by(|a, b| a.case_id.as_bytes().cmp(b.case_id.as_bytes()));
    let mut first_seen: BTreeMap<String, String> = BTreeMap::new();
    let mut kept: Vec<AdmittedCase> = Vec::new();
    for case in admitted {
        // `F-PRISTINE` is exempt from the duplicate check by construction.
        if case.family == Family::Pristine {
            kept.push(case);
            continue;
        }
        match first_seen.get(&case.staged_aggregate) {
            Some(earlier) => {
                std::fs::remove_dir_all(dest.join("dev").join(&case.case_id))
                    .expect("the rejected staged root is removable");
                rejected.push(RejectedCase {
                    family: case.family,
                    target: case.target.clone(),
                    site: case.site.clone(),
                    kind: RejectionKind::DuplicateStagedAggregate {
                        collides_with: earlier.clone(),
                    },
                });
            }
            None => {
                first_seen.insert(case.staged_aggregate.clone(), case.case_id.clone());
                kept.push(case);
            }
        }
    }
    let admitted = kept;
    rejected.sort_by(|a, b| {
        (a.family.id(), a.target.as_str(), a.site.as_str()).cmp(&(
            b.family.id(),
            b.target.as_str(),
            b.site.as_str(),
        ))
    });

    write_case_index(dest, &admitted);
    write_rejection_ledger(dest, &rejected);

    Corpus {
        digest: ix_agg_1(dest),
        root: dest.to_path_buf(),
        enumerated,
        admitted,
        rejected,
    }
}

/// The sealed labels. Hand-written so key order is fixed by this code rather
/// than by a derive, because the file is bound into `corpus_digest_ix_agg_1`.
fn write_case_index(dest: &Path, admitted: &[AdmittedCase]) {
    let mut text = String::new();
    for case in admitted {
        text.push_str(&format!(
            "{{\"case_id\":\"{}\",\"family\":\"{}\",\"target\":\"{}\",\"site\":\"{}\",\"label\":\"{}\"}}\n",
            case.case_id,
            case.family.id(),
            case.target,
            case.site,
            case.label.as_str()
        ));
    }
    std::fs::write(dest.join("cases.jsonl"), text).expect("case index is writable");
}

/// A rejection is evidence, not an error, so the ledger is part of the corpus
/// tree and is bound into its digest.
fn write_rejection_ledger(dest: &Path, rejected: &[RejectedCase]) {
    let mut text = String::new();
    for case in rejected {
        let (kind, detail) = match &case.kind {
            RejectionKind::NotApplicable(reason) => ("NotApplicable", (*reason).to_string()),
            RejectionKind::NoOpEdit => ("NoOpEdit", String::new()),
            RejectionKind::DuplicateStagedAggregate { collides_with } => {
                ("DuplicateStagedAggregate", collides_with.clone())
            }
            RejectionKind::SiteAbsent => ("SiteAbsent", String::new()),
        };
        text.push_str(&format!(
            "{{\"family\":\"{}\",\"target\":\"{}\",\"site\":\"{}\",\"rejection\":\"{}\",\"detail\":\"{}\"}}\n",
            case.family.id(),
            case.target,
            case.site,
            kind,
            detail
        ));
    }
    std::fs::write(dest.join("rejections.jsonl"), text).expect("rejection ledger is writable");
}
