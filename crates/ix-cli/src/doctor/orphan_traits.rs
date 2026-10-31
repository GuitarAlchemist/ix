//! Orphan-trait check — declared public surface that nothing satisfies.
//!
//! A `pub trait` with **zero implementors and zero uses as a generic bound**
//! is either dead code or an unsatisfied contract. Either way a human should
//! have to say which, so the check fails until the trait is removed, given an
//! implementor, or recorded in the allowlist with a written reason.
//!
//! The motivating case is `ix-io`'s `DataSource` / `DataSink` (ix#299): the
//! module doc claims "Every I/O backend implements DataSource and/or DataSink"
//! while eight backends implement neither.
//!
//! # Why name-matching is enough here
//!
//! The scanner matches by type **name**, with no module resolution. That is
//! deliberately unreliable for *high* counts — a homonym inflates them — but
//! reliable for *zeros*: a name absent from the tree is genuinely absent. This
//! check only ever acts on zeros, so it builds on the reliable half.
//!
//! Every ambiguity is resolved in the **conservative** direction — toward
//! counting a use, never inventing one. A trait mentioned in a comment, in a
//! doc example, or by an unrelated homonym is treated as used, so the check
//! under-reports rather than crying wolf.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Path of the allowlist, relative to the repo root.
pub const ALLOWLIST_PATH: &str = "state/registry/orphan-traits.allow.json";

/// A `pub trait` declaration found in a crate's `src/` tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraitDecl {
    /// Trait identifier, e.g. `DataSource`.
    pub name: String,
    /// Repo-relative path of the declaring file.
    pub file: String,
    /// 1-based line number of the declaration.
    pub line: usize,
    /// Number of `impl .. for ..` headers naming this trait.
    pub impls: usize,
    /// Number of uses as a bound or trait object (`: T`, `+ T`, `dyn T`,
    /// `impl T`, or a `#[derive(T)]` mention).
    pub bounds: usize,
}

impl TraitDecl {
    /// A trait nothing implements and nothing constrains on.
    pub fn is_orphan(&self) -> bool {
        self.impls == 0 && self.bounds == 0
    }
}

/// One human-signed-off exemption.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AllowEntry {
    /// Trait identifier being exempted.
    #[serde(rename = "trait")]
    pub trait_name: String,
    /// Why this trait may have no implementors. Must be non-empty: the whole
    /// point of the allowlist is that a human said which.
    #[serde(default)]
    pub reason: String,
    /// Optional tracking issue, e.g. `ix#299`.
    #[serde(default)]
    pub issue: Option<String>,
}

/// Parsed allowlist file.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Allowlist {
    /// Free-text header explaining the file to whoever opens it next.
    #[serde(default)]
    pub note: String,
    /// The exemptions.
    #[serde(default)]
    pub allow: Vec<AllowEntry>,
}

impl Allowlist {
    /// Load the allowlist, treating an absent file as empty.
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(ALLOWLIST_PATH);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let raw =
            std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        serde_json::from_str(&raw).map_err(|e| format!("parsing {}: {e}", path.display()))
    }
}

/// Outcome of the scan, before it is turned into a check result.
#[derive(Debug, Clone, Serialize)]
pub struct Census {
    /// Every `pub trait` found under the scanned `src/` trees.
    pub traits: Vec<TraitDecl>,
    /// Orphans not covered by the allowlist — these fail the check.
    pub unlisted_orphans: Vec<TraitDecl>,
    /// Orphans covered by the allowlist — recorded, not failing.
    pub allowed_orphans: Vec<String>,
    /// Allowlist entries whose trait is no longer an orphan, or no longer
    /// exists. Stale exemptions warn, so the allowlist cannot quietly rot.
    pub stale_allowlist: Vec<String>,
    /// Allowlist entries with an empty `reason`. These fail: an exemption
    /// without a stated reason is not a human decision, just a silencer.
    pub reasonless_allowlist: Vec<String>,
}

/// Scan `root` for orphan public traits and reconcile against the allowlist.
pub fn scan(root: &Path) -> Result<Census, String> {
    let allowlist = Allowlist::load(root)?;
    scan_with_allowlist(root, &allowlist)
}

/// Scan with an explicit allowlist — the seam the tests drive.
pub fn scan_with_allowlist(root: &Path, allowlist: &Allowlist) -> Result<Census, String> {
    // Declarations come only from `crates/*/src` — a `pub trait` in a test or
    // an example is scaffolding, not shipped surface.
    let mut decl_files = Vec::new();
    collect_rs(&root.join("crates"), &mut decl_files, &|p| {
        p.components().any(|c| c.as_os_str() == "src")
    });

    // Uses are counted across everything: src, tests, benches, examples. A
    // trait implemented only by a test is still a satisfied contract.
    let mut use_files = Vec::new();
    for dir in ["crates", "examples", "tests"] {
        collect_rs(&root.join(dir), &mut use_files, &|_| true);
    }

    let mut decls: Vec<TraitDecl> = Vec::new();
    for file in &decl_files {
        let Ok(src) = std::fs::read_to_string(file) else {
            continue;
        };
        for (idx, line) in src.lines().enumerate() {
            if let Some(name) = parse_pub_trait(line) {
                decls.push(TraitDecl {
                    name,
                    file: rel(root, file),
                    line: idx + 1,
                    impls: 0,
                    bounds: 0,
                });
            }
        }
    }

    let mut names: Vec<String> = decls.iter().map(|d| d.name.clone()).collect();
    names.sort();
    names.dedup();

    let mut impls: BTreeMap<&str, usize> = names.iter().map(|n| (n.as_str(), 0)).collect();
    let mut bounds: BTreeMap<&str, usize> = names.iter().map(|n| (n.as_str(), 0)).collect();

    for file in &use_files {
        let Ok(src) = std::fs::read_to_string(file) else {
            continue;
        };
        // Collapse all whitespace so a multi-line `impl<T> Foo\n  for Bar`
        // header or a wrapped `where` clause reads as one flat string.
        let flat = src.split_whitespace().collect::<Vec<_>>().join(" ");
        for name in &names {
            *impls.get_mut(name.as_str()).expect("seeded above") += count_impls(&flat, name);
            *bounds.get_mut(name.as_str()).expect("seeded above") +=
                count_bounds(&flat, name) + count_derives(&flat, name);
        }
    }

    for d in &mut decls {
        d.impls = impls[d.name.as_str()];
        d.bounds = bounds[d.name.as_str()];
    }
    decls.sort_by(|a, b| a.name.cmp(&b.name).then(a.file.cmp(&b.file)));

    let allowed: Vec<&str> = allowlist
        .allow
        .iter()
        .map(|e| e.trait_name.as_str())
        .collect();

    let mut unlisted_orphans = Vec::new();
    let mut allowed_orphans = Vec::new();
    for d in &decls {
        if !d.is_orphan() {
            continue;
        }
        if allowed.contains(&d.name.as_str()) {
            allowed_orphans.push(d.name.clone());
        } else {
            unlisted_orphans.push(d.clone());
        }
    }

    let orphan_names: Vec<&str> = decls
        .iter()
        .filter(|d| d.is_orphan())
        .map(|d| d.name.as_str())
        .collect();
    let stale_allowlist = allowlist
        .allow
        .iter()
        .filter(|e| !orphan_names.contains(&e.trait_name.as_str()))
        .map(|e| e.trait_name.clone())
        .collect();
    let reasonless_allowlist = allowlist
        .allow
        .iter()
        .filter(|e| e.reason.trim().is_empty())
        .map(|e| e.trait_name.clone())
        .collect();

    Ok(Census {
        traits: decls,
        unlisted_orphans,
        allowed_orphans,
        stale_allowlist,
        reasonless_allowlist,
    })
}

/// Return the trait name if `line` declares a public trait.
///
/// Doc comments (`/// pub trait Foo`) are skipped because the trimmed line
/// starts with `/`, not `pub`.
fn parse_pub_trait(line: &str) -> Option<String> {
    let mut t = line.trim_start().strip_prefix("pub")?;
    // `pub(crate)`, `pub(super)`, `pub(in path)` — drop the restriction.
    if let Some(r) = t.strip_prefix('(') {
        t = r.split_once(')')?.1;
    }
    t = t.trim_start();
    for kw in ["unsafe ", "auto "] {
        if let Some(r) = t.strip_prefix(kw) {
            t = r.trim_start();
        }
    }
    let name: String = t
        .strip_prefix("trait ")?
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Count `impl .. <name> .. for ..` headers in a whitespace-collapsed source.
///
/// The leading generic parameter list is skipped so `impl<T: Foo> Bar for Baz`
/// is not read as an impl of `Foo`. Anything still ambiguous counts as an
/// impl, which suppresses a finding rather than inventing one.
fn count_impls(flat: &str, name: &str) -> usize {
    let mut n = 0;
    for start in ident_positions(flat, "impl") {
        let after = skip_balanced_generics(&flat[start + "impl".len()..]);
        // The impl header ends at the block; bound the scan so a missing brace
        // cannot run away across the whole file.
        let header_end = after.find('{').unwrap_or(after.len()).min(400);
        let header = &after[..header_end];
        let Some(for_at) = find_ident(header, "for") else {
            continue;
        };
        if find_ident(&header[..for_at], name).is_some() {
            n += 1;
        }
    }
    n
}

/// Count uses of `name` as a generic bound or trait object: `: N`, `+ N`,
/// `dyn N`, `impl N`. `::N` is excluded — that is a path segment, not a bound.
fn count_bounds(flat: &str, name: &str) -> usize {
    let mut n = 0;
    for pos in ident_positions(flat, name) {
        let before = flat[..pos].trim_end();
        let is_bound = match before.as_bytes().last() {
            Some(b':') => !before.ends_with("::"),
            Some(b'+') => true,
            _ => before.ends_with("dyn") || before.ends_with("impl"),
        };
        if !is_bound {
            continue;
        }
        // `impl Foo for Bar` is an impl, not a bound; a bare `impl Foo`
        // (return-position impl trait) is a bound. Tell them apart by what
        // follows the name.
        let after = skip_balanced_generics(&flat[pos + name.len()..]);
        if before.ends_with("impl") && find_ident(after.trim_start(), "for") == Some(0) {
            continue;
        }
        n += 1;
    }
    n
}

/// Count `#[derive(.., Name, ..)]` mentions. A derive macro implements the
/// trait without any textual `impl` header, so without this a derive-backed
/// trait would look like an orphan.
fn count_derives(flat: &str, name: &str) -> usize {
    let mut n = 0;
    let mut from = 0;
    while let Some(rel_at) = flat[from..].find("derive(") {
        let open = from + rel_at + "derive(".len();
        let Some(close_rel) = flat[open..].find(')') else {
            break;
        };
        let list = &flat[open..open + close_rel];
        n += list
            .split(',')
            .filter(|item| {
                item.trim()
                    .rsplit("::")
                    .next()
                    .is_some_and(|seg| seg.trim() == name)
            })
            .count();
        from = open + close_rel;
    }
    n
}

/// Byte offsets where `needle` appears as a standalone identifier.
fn ident_positions(hay: &str, needle: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel_at) = hay[from..].find(needle) {
        let at = from + rel_at;
        let end = at + needle.len();
        let before_ok = !hay[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after_ok = !hay[end..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if before_ok && after_ok {
            out.push(at);
        }
        from = end;
    }
    out
}

/// First standalone-identifier occurrence of `needle`, if any.
fn find_ident(hay: &str, needle: &str) -> Option<usize> {
    ident_positions(hay, needle).into_iter().next()
}

/// If `s` starts with a `<..>` generic list, return the remainder after it.
fn skip_balanced_generics(s: &str) -> &str {
    let t = s.trim_start();
    if !t.starts_with('<') {
        return s;
    }
    let mut depth = 0usize;
    for (i, c) in t.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return &t[i + 1..];
                }
            }
            _ => {}
        }
    }
    s
}

/// Recursively collect `.rs` files under `dir` matching `keep`.
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>, keep: &dyn Fn(&Path) -> bool) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == "target" || name == ".git" || name == "node_modules" {
                continue;
            }
            collect_rs(&path, out, keep);
        } else if path.extension().is_some_and(|e| e == "rs") && keep(&path) {
            out.push(path);
        }
    }
}

/// Repo-relative, forward-slashed path for stable output across platforms.
fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
