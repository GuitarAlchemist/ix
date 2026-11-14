//! Offline spike of an explicit, immutable artifact list for #485 (decision D2 of the #483
//! synthesis): a task receives a manifest of pinned items, and resolving it gives one explicit
//! outcome per item, never an empty table.
//!
//! Throwaway, like `jev-router` and `claude-stream`. Digests use `ix-recover`'s format
//! (`sha256:<hex>`, `ix_recover::sha256_hex`), so a list item and a published run agree on what
//! a content hash is. Fixtures are synthetic; no network, no GitHub fetch, no credential.
//!
//! Run: cargo run --manifest-path crates/ix-router-spike/Cargo.toml --bin context-list --
//!      <manifest.json> --root <dir> --accept <schema>[,<schema>...]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

/// Manifest format this spike reads. Anything else is refused whole.
const MANIFEST_VERSION: u32 = 0;
/// Items larger than this are refused: a list carries bounded data, not a dump of anything.
const MAX_ITEM_BYTES: u64 = 16 << 20;
/// Classifications a list may carry. Secrets and anything unknown are refused.
const ALLOWED_CLASSIFICATIONS: [&str; 2] = ["public", "internal"];

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Manifest {
    version: u32,
    /// How many items the writer put in; fewer means the dump is incomplete.
    count: usize,
    items: Vec<Item>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Item {
    id: String,
    source: Source,
    sha256: String,
    schema: String,
    classification: String,
    bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Source {
    /// A path relative to the list's root.
    Local { path: String },
    /// A file pinned by repository, path and full commit sha; a branch or tag moves.
    Github {
        repo: String,
        path: String,
        commit: String,
    },
}

/// What a reader found at a source.
#[derive(Debug, Clone, PartialEq)]
enum Read {
    Bytes(Vec<u8>),
    NotFound,
    /// The source exists but this reader may not read it (revoked or never granted).
    Denied(String),
}

/// Why a whole manifest is refused before any item is read.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "refused", rename_all = "snake_case")]
enum Refusal {
    IncompatibleManifest { version: u32 },
    IncompleteDump { declared: usize, present: usize },
    DuplicateId { id: String },
    MovingReference { id: String, commit: String },
    Classification { id: String, classification: String },
    Oversize { id: String, bytes: u64 },
}

/// One item's outcome. Only `Ready` hands bytes to the task.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum Outcome {
    Ready {
        #[serde(skip)]
        bytes: Vec<u8>,
    },
    Missing,
    ReadDenied {
        reason: String,
    },
    ChangedHash {
        expected: String,
        actual: String,
    },
    SizeMismatch {
        expected: u64,
        actual: u64,
    },
    IncompatibleSchema {
        schema: String,
    },
}

/// Check the manifest whole, then resolve each item through `read`. A refusal reads nothing.
fn resolve(
    manifest: &Manifest,
    accepted_schemas: &[&str],
    read: &mut dyn FnMut(&Source) -> Read,
) -> Result<Vec<(String, Outcome)>, Refusal> {
    if manifest.version != MANIFEST_VERSION {
        return Err(Refusal::IncompatibleManifest {
            version: manifest.version,
        });
    }
    if manifest.count != manifest.items.len() {
        return Err(Refusal::IncompleteDump {
            declared: manifest.count,
            present: manifest.items.len(),
        });
    }
    let mut ids = BTreeSet::new();
    for item in &manifest.items {
        let id = item.id.clone();
        if !ids.insert(id.clone()) {
            return Err(Refusal::DuplicateId { id });
        }
        if let Source::Github { commit, .. } = &item.source {
            let full_sha = commit.len() == 40
                && commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
            if !full_sha {
                return Err(Refusal::MovingReference {
                    id,
                    commit: commit.clone(),
                });
            }
        }
        if !ALLOWED_CLASSIFICATIONS.contains(&item.classification.as_str()) {
            return Err(Refusal::Classification {
                id,
                classification: item.classification.clone(),
            });
        }
        if item.bytes > MAX_ITEM_BYTES {
            return Err(Refusal::Oversize {
                id,
                bytes: item.bytes,
            });
        }
    }
    Ok(manifest
        .items
        .iter()
        .map(|item| (item.id.clone(), resolve_item(item, accepted_schemas, read)))
        .collect())
}

fn resolve_item(
    item: &Item,
    accepted_schemas: &[&str],
    read: &mut dyn FnMut(&Source) -> Read,
) -> Outcome {
    if !accepted_schemas.contains(&item.schema.as_str()) {
        return Outcome::IncompatibleSchema {
            schema: item.schema.clone(),
        };
    }
    let bytes = match read(&item.source) {
        Read::Bytes(bytes) => bytes,
        Read::NotFound => return Outcome::Missing,
        Read::Denied(reason) => return Outcome::ReadDenied { reason },
    };
    if bytes.len() as u64 != item.bytes {
        return Outcome::SizeMismatch {
            expected: item.bytes,
            actual: bytes.len() as u64,
        };
    }
    let actual = ix_recover::sha256_hex(&bytes);
    if actual != item.sha256 {
        return Outcome::ChangedHash {
            expected: item.sha256.clone(),
            actual,
        };
    }
    Outcome::Ready { bytes }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let (Some(path), Some(root), Some(accept)) = (args.first(), flag("--root"), flag("--accept"))
    else {
        eprintln!("usage: context-list <manifest.json> --root <dir> --accept <schema>[,...]");
        std::process::exit(2);
    };
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(path).expect("read manifest"))
        .expect("parse manifest");
    let accepted: Vec<&str> = accept.split(',').collect();
    let root = Path::new(&root);
    let mut read = |source: &Source| match source {
        Source::Local { path } => match std::fs::read(root.join(path)) {
            Ok(bytes) => Read::Bytes(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Read::NotFound,
            Err(e) => Read::Denied(e.kind().to_string()),
        },
        Source::Github { .. } => Read::Denied("github sources are not fetched offline".into()),
    };
    let report = match resolve(&manifest, &accepted, &mut read) {
        Ok(items) => serde_json::json!({
            "usable": items.iter().all(|(_, o)| matches!(o, Outcome::Ready { .. })),
            "items": items.iter().map(|(id, o)| serde_json::json!({"id": id, "result": o})).collect::<Vec<_>>(),
        }),
        Err(refusal) => serde_json::json!({"usable": false, "refusal": refusal}),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize")
    );
}

#[cfg(test)]
mod tests {
    //! Synthetic fixtures: an in-memory reader stands in for the disk and for GitHub.
    use super::*;
    use ix_recover::sha256_hex;
    use std::collections::BTreeMap;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    fn local(id: &str, content: &[u8]) -> Item {
        Item {
            id: id.into(),
            source: Source::Local {
                path: format!("{id}.json"),
            },
            sha256: sha256_hex(content),
            schema: "notes/v1".into(),
            classification: "internal".into(),
            bytes: content.len() as u64,
        }
    }

    fn manifest(items: Vec<Item>) -> Manifest {
        Manifest {
            version: 0,
            count: items.len(),
            items,
        }
    }

    /// A reader over a fixed map from source path to what reading it gives.
    fn reader(entries: &[(&str, Read)]) -> impl FnMut(&Source) -> Read {
        let map: BTreeMap<String, Read> = entries
            .iter()
            .map(|(p, r)| (p.to_string(), r.clone()))
            .collect();
        move |source| {
            let key = match source {
                Source::Local { path } => path.clone(),
                Source::Github { repo, path, commit } => format!("{repo}@{commit}:{path}"),
            };
            map.get(&key).cloned().unwrap_or(Read::NotFound)
        }
    }

    fn outcomes(m: &Manifest, read: &mut dyn FnMut(&Source) -> Read) -> Vec<Outcome> {
        resolve(m, &["notes/v1"], read)
            .expect("not refused")
            .into_iter()
            .map(|(_, o)| o)
            .collect()
    }

    #[test]
    fn test_pinned_items_resolve_when_bytes_match() {
        let m = manifest(vec![local("a", b"alpha"), local("b", b"beta")]);
        let mut read = reader(&[
            ("a.json", Read::Bytes(b"alpha".to_vec())),
            ("b.json", Read::Bytes(b"beta".to_vec())),
        ]);
        let got = resolve(&m, &["notes/v1"], &mut read).expect("not refused");
        assert_eq!(
            got,
            vec![
                (
                    "a".to_string(),
                    Outcome::Ready {
                        bytes: b"alpha".to_vec()
                    }
                ),
                (
                    "b".to_string(),
                    Outcome::Ready {
                        bytes: b"beta".to_vec()
                    }
                ),
            ]
        );
    }

    #[test]
    fn test_missing_item_is_missing_never_an_empty_read() {
        let m = manifest(vec![local("a", b"alpha")]);
        assert_eq!(outcomes(&m, &mut reader(&[])), vec![Outcome::Missing]);
    }

    #[test]
    fn test_one_changed_byte_is_a_changed_hash() {
        let m = manifest(vec![local("a", b"alpha")]);
        let mut read = reader(&[("a.json", Read::Bytes(b"alphb".to_vec()))]);
        assert_eq!(
            outcomes(&m, &mut read),
            vec![Outcome::ChangedHash {
                expected: sha256_hex(b"alpha"),
                actual: sha256_hex(b"alphb"),
            }]
        );
    }

    #[test]
    fn test_size_is_checked_before_the_hash() {
        let m = manifest(vec![local("a", b"alpha")]);
        let mut read = reader(&[("a.json", Read::Bytes(b"alpha!".to_vec()))]);
        assert_eq!(
            outcomes(&m, &mut read),
            vec![Outcome::SizeMismatch {
                expected: 5,
                actual: 6
            }]
        );
    }

    #[test]
    fn test_revoked_read_is_distinct_from_missing() {
        let m = manifest(vec![local("a", b"alpha")]);
        let mut read = reader(&[("a.json", Read::Denied("revoked".into()))]);
        assert_eq!(
            outcomes(&m, &mut read),
            vec![Outcome::ReadDenied {
                reason: "revoked".into()
            }]
        );
    }

    #[test]
    fn test_unaccepted_schema_is_incompatible_and_not_read() {
        let mut item = local("a", b"alpha");
        item.schema = "notes/v2".into();
        let m = manifest(vec![item]);
        let mut reads = 0;
        let mut read = |_: &Source| {
            reads += 1;
            Read::Bytes(b"alpha".to_vec())
        };
        assert_eq!(
            outcomes(&m, &mut read),
            vec![Outcome::IncompatibleSchema {
                schema: "notes/v2".into()
            }]
        );
        assert_eq!(reads, 0, "an item the task cannot use is not read");
    }

    #[test]
    fn test_one_bad_item_leaves_the_others_named() {
        let m = manifest(vec![local("a", b"alpha"), local("b", b"beta")]);
        let mut read = reader(&[("a.json", Read::Bytes(b"alpha".to_vec()))]);
        let got = resolve(&m, &["notes/v1"], &mut read).expect("not refused");
        assert_eq!(got[0].0, "a");
        assert!(matches!(got[0].1, Outcome::Ready { .. }));
        assert_eq!(got[1], ("b".to_string(), Outcome::Missing));
    }

    #[test]
    fn test_manifest_level_refusals_read_nothing() {
        let good = || manifest(vec![local("a", b"alpha")]);
        let cases: Vec<(Manifest, Refusal)> = vec![
            (
                Manifest {
                    version: 1,
                    ..good()
                },
                Refusal::IncompatibleManifest { version: 1 },
            ),
            (
                Manifest { count: 2, ..good() },
                Refusal::IncompleteDump {
                    declared: 2,
                    present: 1,
                },
            ),
            (
                manifest(vec![local("a", b"alpha"), local("a", b"again")]),
                Refusal::DuplicateId { id: "a".into() },
            ),
            (
                manifest(vec![Item {
                    source: Source::Github {
                        repo: "GuitarAlchemist/ix".into(),
                        path: "README.md".into(),
                        commit: "main".into(),
                    },
                    ..local("g", b"x")
                }]),
                Refusal::MovingReference {
                    id: "g".into(),
                    commit: "main".into(),
                },
            ),
            (
                manifest(vec![Item {
                    classification: "secret".into(),
                    ..local("s", b"x")
                }]),
                Refusal::Classification {
                    id: "s".into(),
                    classification: "secret".into(),
                },
            ),
            (
                manifest(vec![Item {
                    bytes: MAX_ITEM_BYTES + 1,
                    ..local("big", b"x")
                }]),
                Refusal::Oversize {
                    id: "big".into(),
                    bytes: MAX_ITEM_BYTES + 1,
                },
            ),
        ];
        for (m, refusal) in cases {
            let mut reads = 0;
            let mut read = |_: &Source| {
                reads += 1;
                Read::NotFound
            };
            assert_eq!(resolve(&m, &["notes/v1"], &mut read), Err(refusal.clone()));
            assert_eq!(reads, 0, "{refusal:?} read an item");
        }
    }

    #[test]
    fn test_github_item_pinned_to_a_full_commit_resolves() {
        let m = manifest(vec![Item {
            source: Source::Github {
                repo: "GuitarAlchemist/ix".into(),
                path: "README.md".into(),
                commit: SHA.into(),
            },
            ..local("g", b"readme")
        }]);
        let key = format!("GuitarAlchemist/ix@{SHA}:README.md");
        let mut read = reader(&[(key.as_str(), Read::Bytes(b"readme".to_vec()))]);
        assert_eq!(
            outcomes(&m, &mut read),
            vec![Outcome::Ready {
                bytes: b"readme".to_vec()
            }]
        );
    }
}
