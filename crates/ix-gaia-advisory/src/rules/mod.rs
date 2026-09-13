//! Rule dispatch, and the harness-computed cost account.
//!
//! Every rule runs through **this** dispatch, and its cost and its `Unknown`s
//! are computed **here**, from the reads the rule declares — never
//! self-reported by the rule. A rule that accounted for itself could look cheap
//! or confident by accounting differently from the others, the comparison would
//! stop being like-for-like, and no test would catch it because both sides
//! would be "correct" under their own accounting. Making that structurally
//! impossible is the whole reason the rule set is a closed enum behind one
//! entry point.

pub(crate) mod window_probe;

use std::path::Path;

use ix_types::Hexavalent;

use crate::manifest::DeclaredRow;
use crate::root::RootListing;
use crate::AdvisoryRefusal;
use crate::RuleId;

use window_probe::WindowReferenceTable;

/// What a rule concluded about one declared row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowVerdict {
    pub(crate) head: Hexavalent,
    pub(crate) tail: Hexavalent,
    pub(crate) state: Hexavalent,
}

impl RowVerdict {
    /// A row nothing bound. `U` never satisfies a pass.
    pub(crate) fn unbound() -> Self {
        Self {
            head: Hexavalent::Unknown,
            tail: Hexavalent::Unknown,
            state: Hexavalent::Unknown,
        }
    }
}

/// What the dispatch measured, for the artifact to report.
pub(crate) struct RuleOutcome {
    /// Parallel to the declared rows, ordinal.
    pub(crate) row_verdicts: Vec<RowVerdict>,
    /// The rule's finding about the root as a whole.
    pub(crate) root_state: Hexavalent,
    /// unit: bytes — declared before the run, from the manifest's own figures.
    pub(crate) budget_bytes: u64,
    /// unit: bytes — content bytes the rule actually read.
    pub(crate) bytes_read: u64,
    /// unit: count
    pub(crate) files_opened: u64,
}

/// Everything a rule may look at. It may look at nothing else.
// Every field is consumed once slices S3, S4 and S5 land their rule arms.
#[allow(dead_code)]
pub(crate) struct RuleCtx<'a> {
    pub(crate) rule: RuleId,
    pub(crate) window_bytes: u32,
    pub(crate) evidence_root: &'a Path,
    pub(crate) manifest_file_name: &'a str,
    pub(crate) declared: &'a [DeclaredRow],
    pub(crate) listing: &'a RootListing,
    pub(crate) wrt: &'a WindowReferenceTable,
}

/// Run one rule over one root.
pub(crate) fn run(ctx: &RuleCtx<'_>) -> Result<RuleOutcome, AdvisoryRefusal> {
    match ctx.rule {
        RuleId::AlwaysAgree => Ok(always_agree(ctx)),
        RuleId::NameSetOnly => Ok(structural(ctx, true, false)),
        RuleId::LengthOnly => Ok(structural(ctx, false, true)),
        RuleId::StructuralOnly => Ok(structural(ctx, true, true)),
        RuleId::WindowProbe => window(ctx),
        RuleId::FullDigest => full_digest(ctx),
    }
}

/// The windowed rule: the content-free ablation, plus head and tail window
/// digests against the pristine Window Reference Table.
///
/// **At width zero it is definitionally the ablation.** It reads no content, so
/// it binds neither window, and its row states are the ablation's — NC-9
/// asserts the two produce identical outcomes and treats any divergence as a
/// harness defect rather than a finding. That is why the width-zero path skips
/// the comparison outright instead of comparing two empty digests, which would
/// report `T` for a window nothing looked at.
fn window(ctx: &RuleCtx<'_>) -> Result<RuleOutcome, AdvisoryRefusal> {
    let mut ablation = structural(ctx, true, true);
    if ctx.window_bytes == 0 {
        return Ok(ablation);
    }

    let mut bytes_read = 0u64;
    let mut budget_bytes = 0u64;
    let mut files_opened = 0u64;

    for (declared, verdict) in ctx.declared.iter().zip(ablation.row_verdicts.iter_mut()) {
        let Some(listed) = ctx.listing.files.get(&declared.name) else {
            continue; // not measurable; the ablation already left it unbound
        };
        budget_bytes += window_probe::read_cost(listed.bytes, ctx.window_bytes);

        let Some(reference) = ctx.wrt.rows.get(&declared.name) else {
            // No pristine reference for this name. The windows are not
            // comparable, so neither binds — and an unbound window never
            // upgrades the row.
            continue;
        };

        let content = std::fs::read(ctx.evidence_root.join(&declared.name)).map_err(|e| {
            AdvisoryRefusal::EvidenceRootUnreadable {
                detail: format!("{}: {e}", declared.name),
            }
        })?;
        bytes_read += window_probe::read_cost(content.len() as u64, ctx.window_bytes);
        files_opened += 1;

        let (head, tail) = window_probe::windows(&content, ctx.window_bytes);
        verdict.head = truth(crate::digest::sha256_hex(head) == reference.head_sha256);
        verdict.tail = truth(crate::digest::sha256_hex(tail) == reference.tail_sha256);
        if verdict.head == Hexavalent::False || verdict.tail == Hexavalent::False {
            verdict.state = Hexavalent::False;
        }
    }

    ablation.budget_bytes = budget_bytes;
    ablation.bytes_read = bytes_read;
    ablation.files_opened = files_opened;
    Ok(ablation)
}

fn truth(matched: bool) -> Hexavalent {
    if matched {
        Hexavalent::True
    } else {
        Hexavalent::False
    }
}

/// The trivial floor. It reads nothing and agrees with everything, including
/// with rows it could not have measured — which is the entire point of a floor
/// and the reason its `false_agreement` is exactly `1.0` wherever the truth-`F`
/// cases bind. It is reported so no other rule can be praised for beating
/// nothing.
fn always_agree(ctx: &RuleCtx<'_>) -> RuleOutcome {
    RuleOutcome {
        row_verdicts: vec![
            RowVerdict {
                head: Hexavalent::Unknown,
                tail: Hexavalent::Unknown,
                state: Hexavalent::True,
            };
            ctx.declared.len()
        ],
        root_state: Hexavalent::True,
        budget_bytes: 0,
        bytes_read: 0,
        files_opened: 0,
    }
}

/// The content-free rules, and the ablation that is their disjunction.
///
/// `NameSetOnly` makes a claim about the **root** and none about any row;
/// `LengthOnly` makes a claim about each **row** and none about the root. Their
/// disjunction is therefore built by taking each rule's own claim, not by
/// re-deriving one from the other — and because `F` absorbs under the canonical
/// `and`, a disagreement found by either limb reaches `reconciles`.
///
/// None of them reads a content byte: every length here comes from directory
/// metadata. That is why all three **miss** a length-preserving edit, and that
/// miss is a measured, predicted blind spot rather than a defect.
fn structural(ctx: &RuleCtx<'_>, name_set: bool, lengths: bool) -> RuleOutcome {
    let row_verdicts = ctx
        .declared
        .iter()
        .map(|declared| match ctx.listing.files.get(&declared.name) {
            // Not measurable: absent, or an entry class raw-byte measurement
            // cannot reach. Nothing binds it, and `U` never satisfies a pass.
            None => RowVerdict::unbound(),
            Some(listed) => RowVerdict {
                head: Hexavalent::Unknown,
                tail: Hexavalent::Unknown,
                state: if !lengths || listed.bytes == declared.bytes {
                    Hexavalent::True
                } else {
                    Hexavalent::False
                },
            },
        })
        .collect();

    let root_state = if name_set && ctx.listing.name_set() != declared_name_set(ctx) {
        Hexavalent::False
    } else {
        Hexavalent::True
    };

    RuleOutcome {
        row_verdicts,
        root_state,
        budget_bytes: 0,
        bytes_read: 0,
        files_opened: 0,
    }
}

/// Every name the root is declared to hold: the manifest's rows, plus the
/// manifest itself, which never lists itself.
fn declared_name_set(ctx: &RuleCtx<'_>) -> std::collections::BTreeSet<String> {
    ctx.declared
        .iter()
        .map(|row| row.name.clone())
        .chain(std::iter::once(ctx.manifest_file_name.to_string()))
        .collect()
}

/// unit: bytes — every content byte the exact reference reads over this root,
/// known from directory metadata before a single one of them is read.
fn full_read_budget(ctx: &RuleCtx<'_>) -> u64 {
    ctx.listing.files.values().map(|file| file.bytes).sum()
}

/// The exact reference, `ix_gaia_census::census` — the M2 crate, unmodified.
///
/// It is a **ceiling, not a rival**: its correctness rests on the M2 reviews,
/// not on anything here, and nothing in this crate claims it is right. It is
/// also a genuinely separate code path, so a defect in a rule cannot silently
/// agree with it.
///
/// When the reference cannot bind a declared row it **refuses**, and a refusal
/// is not agreement: this crate reports `Unknown` for every row rather than
/// promoting a refusal into a pass. That is a rule outcome, not a binder
/// failure — this crate's own binder resolved every declared field, which is
/// what C7 governs.
fn full_digest(ctx: &RuleCtx<'_>) -> Result<RuleOutcome, AdvisoryRefusal> {
    let budget_bytes = full_read_budget(ctx);
    let files_opened = ctx.listing.files.len() as u64;

    let census = ix_gaia_census::census(&ix_gaia_census::CensusRequest {
        evidence_root: ctx.evidence_root.to_path_buf(),
        manifest_file_name: ctx.manifest_file_name.to_string(),
    });

    let artifact = match census {
        Ok(artifact) => artifact,
        // The reference read the root and then declined to bind it. It has
        // already paid for the read, so the read is charged.
        Err(ix_gaia_census::CensusRefusal::BinderIncomplete { .. }) => {
            return Ok(RuleOutcome {
                row_verdicts: vec![RowVerdict::unbound(); ctx.declared.len()],
                root_state: Hexavalent::Unknown,
                budget_bytes,
                bytes_read: budget_bytes,
                files_opened,
            })
        }
        Err(other) => return Err(from_census(other)),
    };

    let states: std::collections::BTreeMap<&str, Hexavalent> = artifact
        .rows
        .iter()
        .map(|row| (row.name.as_str(), row.state))
        .collect();

    let row_verdicts = ctx
        .declared
        .iter()
        .map(|declared| match states.get(declared.name.as_str()) {
            Some(state) => RowVerdict {
                head: Hexavalent::Unknown,
                tail: Hexavalent::Unknown,
                state: *state,
            },
            None => RowVerdict::unbound(),
        })
        .collect();

    // A root the reference cannot fully account for does not agree with its
    // manifest, whether the surplus is a file it can measure or an entry class
    // it cannot.
    let root_state = if artifact.totals.unlisted_files > 0 || artifact.totals.non_file_entries > 0 {
        Hexavalent::False
    } else {
        Hexavalent::True
    };

    Ok(RuleOutcome {
        row_verdicts,
        root_state,
        budget_bytes,
        bytes_read: budget_bytes,
        files_opened,
    })
}

/// Map the reference's refusals onto this crate's. Nothing is invented and
/// nothing is swallowed.
fn from_census(refusal: ix_gaia_census::CensusRefusal) -> AdvisoryRefusal {
    use ix_gaia_census::CensusRefusal as C;
    match refusal {
        C::EvidenceRootUnreadable { detail } => AdvisoryRefusal::EvidenceRootUnreadable { detail },
        C::ManifestUnparseable { line, detail } => {
            AdvisoryRefusal::ManifestUnparseable { line, detail }
        }
        C::ManifestDeclaresNoRows => AdvisoryRefusal::ManifestDeclaresNoRows,
        C::BinderIncomplete { field, detail } => {
            AdvisoryRefusal::BinderIncomplete { field, detail }
        }
        C::DuplicateManifestRow { name } => AdvisoryRefusal::DuplicateManifestRow { name },
        C::DuplicateInventorySection { first, second } => {
            AdvisoryRefusal::DuplicateInventorySection { first, second }
        }
        C::UnsafeManifestName { name } => AdvisoryRefusal::UnsafeManifestName { name },
    }
}
