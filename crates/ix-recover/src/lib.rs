//! Publish a run's result **once** under a caller-chosen run id, and
//! reconcile from durable evidence after a process kill.
//!
//! Provisional (branch `exp/ix-recover`): the layout, the record schemas and
//! the error set are spike choices, not a contract. They carry the semantics
//! of the Indigo recovery canary (2026-10-17) over into a library that
//! `ix pipeline run --run-id` calls.
//!
//! ## Layout
//!
//! ```text
//! <root>/<run_id>/
//!   owner.lock              held for the whole call (`File::try_lock`)
//!   intent.json             the bindings, written before any effect
//!   attempts/<attempt_id>/  the files + manifest.json, each created new and synced
//!   committed.json          hard link to the committed attempt's manifest.json
//!   receipt.json            written after the commit
//! ```
//!
//! The commit is `hard_link(attempts/<a>/manifest.json, committed.json)`, which
//! fails when the target exists on every platform, so a run id has at most one
//! committed attempt.
//!
//! ## Reconcile order ([`Store::begin`])
//!
//! 1. the owner lock is held elsewhere → [`Error::Unknown`];
//! 2. no intent → write one (a commit or receipt without an intent is
//!    [`Error::Corrupt`]); an intent → its bindings must be the caller's, else
//!    [`Error::Conflict`];
//! 3. a receipt → verify it, the commit and every file → [`Begin::Done`];
//! 4. a commit without a receipt → verify, write the receipt →
//!    [`Begin::Repaired`];
//! 5. otherwise → [`Begin::Fresh`], a new attempt.
//!
//! ## Scope
//!
//! Process-kill recovery. On Unix each directory is fsynced after the link
//! that changes it; on Windows no directory flush is made, so there is no
//! power-loss claim there. Hard links need a filesystem that has them (NTFS,
//! ext4, APFS; not FAT). The residue of a killed attempt stays under
//! `attempts/` and is never cleaned. The lock is released by the OS when its
//! owner dies, which is the only proof of a dead owner this crate uses: work
//! done for an attempt must happen inside the owning process.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const LOCK: &str = "owner.lock";
const INTENT: &str = "intent.json";
const ATTEMPTS: &str = "attempts";
const MANIFEST: &str = "manifest.json";
const COMMITTED: &str = "committed.json";
const RECEIPT: &str = "receipt.json";

const INTENT_SCHEMA: &str = "ix-recover-intent/0";
const MANIFEST_SCHEMA: &str = "ix-recover-manifest/0";
const RECEIPT_SCHEMA: &str = "ix-recover-receipt/0";

/// Why a call stopped. A call that fails never leaves a second commit; the
/// next [`Store::begin`] reconciles whatever it did leave.
#[derive(Debug)]
pub enum Error {
    /// An unusable argument: an unsafe run id or file name, bindings that are
    /// not a JSON object, duplicate file names.
    Usage(String),
    /// Another owner holds the run's lock, so its state cannot be decided now.
    Unknown(String),
    /// A filesystem operation failed.
    Io(String),
    /// Durable evidence is malformed or does not agree with itself.
    Corrupt(String),
    /// The run id was first used with different bindings.
    Conflict(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, msg) = match self {
            Error::Usage(m) => ("usage", m),
            Error::Unknown(m) => ("unknown", m),
            Error::Io(m) => ("io", m),
            Error::Corrupt(m) => ("corrupt", m),
            Error::Conflict(m) => ("conflict", m),
        };
        write!(f, "{kind}: {msg}")
    }
}

impl std::error::Error for Error {}

fn io_err(what: &str, path: &Path, e: io::Error) -> Error {
    Error::Io(format!("{what} {}: {e}", path.display()))
}

/// `sha256:<hex>` of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    let mut s = String::from("sha256:");
    for b in Sha256::digest(bytes) {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Compact JSON with every object's keys sorted, so equal values give equal
/// text however they were built.
pub fn canonical_json(v: &Value) -> String {
    let mut out = String::new();
    write_canonical(v, &mut out);
    out
}

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push(':');
                write_canonical(&map[k.as_str()], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// `sha256:<hex>` of [`canonical_json`]: the bindings digest.
pub fn canonical_sha256(v: &Value) -> String {
    sha256_hex(canonical_json(v).as_bytes())
}

/// A name usable as one path segment on every platform: 1 to 128 of
/// `A-Z a-z 0-9 . _ -`, no trailing dot (so not `.` or `..`), not a Windows
/// device name.
fn check_segment(s: &str) -> Result<(), String> {
    if s.is_empty() || s.len() > 128 {
        return Err("must be 1 to 128 characters".into());
    }
    if !s
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return Err("may hold only A-Z a-z 0-9 . _ -".into());
    }
    if s.ends_with('.') {
        return Err("must not end with a dot".into());
    }
    let stem = s.split('.').next().unwrap_or(s).to_ascii_uppercase();
    let numbered = (stem.starts_with("COM") || stem.starts_with("LPT"))
        && stem.len() == 4
        && stem.as_bytes()[3].is_ascii_digit();
    if numbered || matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL") {
        return Err("is a Windows device name".into());
    }
    Ok(())
}

/// Published file names: each a safe segment, not `manifest.json`, and
/// distinct even on a case-insensitive filesystem.
fn check_names(names: &[&str]) -> Result<(), String> {
    let mut folded = BTreeSet::new();
    for name in names {
        check_segment(name).map_err(|why| format!("file name {name:?} {why}"))?;
        if name.eq_ignore_ascii_case(MANIFEST) {
            return Err(format!("file name {name:?} is reserved"));
        }
        if !folded.insert(name.to_ascii_lowercase()) {
            return Err(format!("file name {name:?} is given twice"));
        }
    }
    Ok(())
}

fn token() -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{}-{seq}", std::process::id())
}

fn read_opt(path: &Path) -> Result<Option<Vec<u8>>, Error> {
    match fs::read(path) {
        Ok(b) => Ok(Some(b)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_err("reading", path, e)),
    }
}

fn present(path: &Path) -> Result<bool, Error> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(io_err("inspecting", path, e)),
    }
}

/// Records are parsed into structs that refuse unknown and duplicated keys.
fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8], path: &Path) -> Result<T, Error> {
    serde_json::from_slice(bytes).map_err(|e| Error::Corrupt(format!("{}: {e}", path.display())))
}

fn to_bytes<T: Serialize>(record: &T) -> Vec<u8> {
    let mut b =
        serde_json::to_vec_pretty(record).expect("records hold only strings and JSON values");
    b.push(b'\n');
    b
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| io_err("creating", path, e))?;
    f.write_all(bytes).map_err(|e| io_err("writing", path, e))?;
    f.sync_all().map_err(|e| io_err("syncing", path, e))
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> Result<(), Error> {
    File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| io_err("syncing", dir, e))
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> Result<(), Error> {
    Ok(())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    schema: String,
    run_id: String,
    bindings: Value,
    bindings_sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    run_id: String,
    attempt_id: String,
    bindings_sha256: String,
    files: Vec<FileEntry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileEntry {
    name: String,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema: String,
    run_id: String,
    attempt_id: String,
    bindings_sha256: String,
    manifest_sha256: String,
}

/// A directory of recoverable runs, one subdirectory per run id.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

/// What [`Store::begin`] found.
#[derive(Debug)]
pub enum Begin {
    /// Nothing is committed: do the work, then [`Attempt::publish`].
    Fresh(Attempt),
    /// The run was published and has its receipt.
    Done(Record),
    /// The run was published but its receipt was missing; it is written now.
    Repaired(Record),
}

/// The verified files of a run's one committed attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    attempt_id: String,
    files: BTreeMap<String, Vec<u8>>,
}

impl Record {
    pub fn attempt_id(&self) -> &str {
        &self.attempt_id
    }

    pub fn file(&self, name: &str) -> Option<&[u8]> {
        self.files.get(name).map(Vec::as_slice)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }
}

/// Where [`Attempt::publish_observed`] stands. A test hook.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Point {
    /// Every file and the manifest are written and synced; nothing is committed.
    Staged,
    /// The commit exists; the receipt is not written yet.
    Committed,
}

impl Store {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, Error> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|e| io_err("creating", &root, e))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Take ownership of `run_id` and reconcile it against `bindings` (a JSON
    /// object naming everything the result depends on) before any work.
    pub fn begin(&self, run_id: &str, bindings: &Value) -> Result<Begin, Error> {
        check_segment(run_id).map_err(|why| Error::Usage(format!("run id {run_id:?} {why}")))?;
        if !bindings.is_object() {
            return Err(Error::Usage("bindings must be a JSON object".into()));
        }
        if has_float(bindings) {
            return Err(Error::Usage(
                "bindings must not hold floating-point numbers: JSON float parsing \
                 is not exact, so the recorded intent could read back different"
                    .into(),
            ));
        }
        let dir = self.root.join(run_id);
        fs::create_dir_all(&dir).map_err(|e| io_err("creating", &dir, e))?;
        let lock_path = dir.join(LOCK);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| io_err("opening", &lock_path, e))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                return Err(Error::Unknown(format!(
                    "run {run_id:?} is held by another owner"
                )))
            }
            Err(fs::TryLockError::Error(e)) => return Err(io_err("locking", &lock_path, e)),
        }
        let run = Run {
            dir,
            run_id: run_id.to_string(),
            bindings_sha256: canonical_sha256(bindings),
            _lock: lock,
        };
        run.reconcile(bindings)
    }
}

/// One run directory while this process holds its lock.
#[derive(Debug)]
struct Run {
    dir: PathBuf,
    run_id: String,
    bindings_sha256: String,
    _lock: File,
}

impl Run {
    fn corrupt(&self, what: &str, msg: impl fmt::Display) -> Error {
        Error::Corrupt(format!("{}: {msg}", self.dir.join(what).display()))
    }

    fn reconcile(self, bindings: &Value) -> Result<Begin, Error> {
        let intent_path = self.dir.join(INTENT);
        match read_opt(&intent_path)? {
            Some(bytes) => self.check_intent(&parse(&bytes, &intent_path)?, bindings)?,
            None => {
                for name in [COMMITTED, RECEIPT] {
                    if present(&self.dir.join(name))? {
                        return Err(self.corrupt(name, format!("present without {INTENT}")));
                    }
                }
                let intent = Intent {
                    schema: INTENT_SCHEMA.into(),
                    run_id: self.run_id.clone(),
                    bindings: bindings.clone(),
                    bindings_sha256: self.bindings_sha256.clone(),
                };
                self.link_new(INTENT, &to_bytes(&intent))?;
            }
        }
        let receipt_path = self.dir.join(RECEIPT);
        if let Some(bytes) = read_opt(&receipt_path)? {
            let receipt: Receipt = parse(&bytes, &receipt_path)?;
            return self.verify_receipt(&receipt).map(Begin::Done);
        }
        if let Some(bytes) = read_opt(&self.dir.join(COMMITTED))? {
            let record = self.verify_commit(&bytes)?;
            self.write_receipt(&record.attempt_id, &bytes)?;
            return Ok(Begin::Repaired(record));
        }
        Ok(Begin::Fresh(Attempt {
            attempt_id: format!("att-{}", token()),
            run: self,
        }))
    }

    fn check_intent(&self, intent: &Intent, bindings: &Value) -> Result<(), Error> {
        if intent.schema != INTENT_SCHEMA {
            return Err(self.corrupt(INTENT, format!("schema {:?}", intent.schema)));
        }
        if intent.run_id != self.run_id {
            if intent.run_id.eq_ignore_ascii_case(&self.run_id) {
                return Err(Error::Conflict(format!(
                    "run id {:?} shares its directory with {:?} on this case-insensitive filesystem",
                    self.run_id, intent.run_id
                )));
            }
            return Err(self.corrupt(INTENT, format!("names run {:?}", intent.run_id)));
        }
        if canonical_sha256(&intent.bindings) != intent.bindings_sha256 {
            return Err(self.corrupt(INTENT, "bindings_sha256 does not match its bindings"));
        }
        if intent.bindings_sha256 != self.bindings_sha256 {
            return Err(Error::Conflict(format!(
                "run {:?} was begun with different bindings: {}",
                self.run_id,
                differing(&intent.bindings, bindings).join(", ")
            )));
        }
        Ok(())
    }

    fn verify_receipt(&self, r: &Receipt) -> Result<Record, Error> {
        if r.schema != RECEIPT_SCHEMA {
            return Err(self.corrupt(RECEIPT, format!("schema {:?}", r.schema)));
        }
        if r.run_id != self.run_id {
            return Err(self.corrupt(RECEIPT, format!("names run {:?}", r.run_id)));
        }
        if r.bindings_sha256 != self.bindings_sha256 {
            return Err(self.corrupt(RECEIPT, "binds other bindings than the intent"));
        }
        let committed = read_opt(&self.dir.join(COMMITTED))?
            .ok_or_else(|| self.corrupt(RECEIPT, format!("present without {COMMITTED}")))?;
        if sha256_hex(&committed) != r.manifest_sha256 {
            return Err(self.corrupt(
                RECEIPT,
                format!("manifest_sha256 does not match {COMMITTED}"),
            ));
        }
        let record = self.verify_commit(&committed)?;
        if record.attempt_id != r.attempt_id {
            return Err(self.corrupt(
                RECEIPT,
                format!(
                    "names attempt {:?}, the commit is {:?}",
                    r.attempt_id, record.attempt_id
                ),
            ));
        }
        Ok(record)
    }

    /// Verify the committed manifest against its attempt directory: the same
    /// manifest bytes, exactly the listed files, each matching its digest.
    fn verify_commit(&self, bytes: &[u8]) -> Result<Record, Error> {
        let m: Manifest = parse(bytes, &self.dir.join(COMMITTED))?;
        if m.schema != MANIFEST_SCHEMA {
            return Err(self.corrupt(COMMITTED, format!("schema {:?}", m.schema)));
        }
        if m.run_id != self.run_id {
            return Err(self.corrupt(COMMITTED, format!("names run {:?}", m.run_id)));
        }
        if m.bindings_sha256 != self.bindings_sha256 {
            return Err(self.corrupt(COMMITTED, "binds other bindings than the intent"));
        }
        if !m.attempt_id.starts_with("att-") || check_segment(&m.attempt_id).is_err() {
            return Err(self.corrupt(COMMITTED, format!("attempt id {:?}", m.attempt_id)));
        }
        let names: Vec<&str> = m.files.iter().map(|f| f.name.as_str()).collect();
        check_names(&names).map_err(|why| self.corrupt(COMMITTED, why))?;
        if !names.windows(2).all(|w| w[0] < w[1]) {
            return Err(self.corrupt(COMMITTED, "files are not in ascending order"));
        }
        let adir = self.dir.join(ATTEMPTS).join(&m.attempt_id);
        if read_opt(&adir.join(MANIFEST))?.as_deref() != Some(bytes) {
            return Err(self.corrupt(
                COMMITTED,
                format!("differs from {}/{MANIFEST}", m.attempt_id),
            ));
        }
        let mut on_disk = BTreeSet::new();
        for entry in fs::read_dir(&adir).map_err(|e| io_err("listing", &adir, e))? {
            let entry = entry.map_err(|e| io_err("listing", &adir, e))?;
            on_disk.insert(entry.file_name().to_string_lossy().into_owned());
        }
        let mut listed: BTreeSet<String> = names.iter().map(|n| n.to_string()).collect();
        listed.insert(MANIFEST.to_string());
        if on_disk != listed {
            return Err(self.corrupt(
                COMMITTED,
                format!(
                    "attempt {} holds {on_disk:?}, its manifest lists {listed:?}",
                    m.attempt_id
                ),
            ));
        }
        let mut files = BTreeMap::new();
        for f in m.files {
            let path = adir.join(&f.name);
            let b = fs::read(&path).map_err(|e| io_err("reading", &path, e))?;
            if sha256_hex(&b) != f.sha256 {
                return Err(self.corrupt(
                    COMMITTED,
                    format!("{}/{} does not match its digest", m.attempt_id, f.name),
                ));
            }
            files.insert(f.name, b);
        }
        Ok(Record {
            attempt_id: m.attempt_id,
            files,
        })
    }

    fn write_receipt(&self, attempt_id: &str, committed: &[u8]) -> Result<(), Error> {
        let receipt = Receipt {
            schema: RECEIPT_SCHEMA.into(),
            run_id: self.run_id.clone(),
            attempt_id: attempt_id.to_string(),
            bindings_sha256: self.bindings_sha256.clone(),
            manifest_sha256: sha256_hex(committed),
        };
        self.link_new(RECEIPT, &to_bytes(&receipt))
    }

    /// Make `name` appear whole or not at all: write a synced temp file, hard
    /// link it to `name` (which fails if `name` exists), drop the temp.
    fn link_new(&self, name: &str, bytes: &[u8]) -> Result<(), Error> {
        let tmp = self.dir.join(format!(".{name}.{}.tmp", token()));
        write_new_synced(&tmp, bytes)?;
        let target = self.dir.join(name);
        let linked = fs::hard_link(&tmp, &target);
        let _ = fs::remove_file(&tmp);
        match linked {
            Ok(()) => sync_dir(&self.dir),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                Err(self.corrupt(name, "appeared while this owner held the run"))
            }
            Err(e) => Err(io_err("linking", &target, e)),
        }
    }
}

fn has_float(v: &Value) -> bool {
    match v {
        Value::Number(n) => n.is_f64(),
        Value::Array(items) => items.iter().any(has_float),
        Value::Object(map) => map.values().any(has_float),
        _ => false,
    }
}

/// The binding names whose values differ (bindings are JSON objects).
fn differing(recorded: &Value, given: &Value) -> Vec<String> {
    let (Some(a), Some(b)) = (recorded.as_object(), given.as_object()) else {
        return vec!["<bindings>".into()];
    };
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    keys.into_iter()
        .filter(|k| a.get(k.as_str()) != b.get(k.as_str()))
        .cloned()
        .collect()
}

/// A new attempt. The run's owner lock is held until the attempt is
/// published or dropped.
#[derive(Debug)]
pub struct Attempt {
    run: Run,
    attempt_id: String,
}

impl Attempt {
    pub fn id(&self) -> &str {
        &self.attempt_id
    }

    /// Stage `files`, commit them, then write the receipt.
    pub fn publish(self, files: &[(&str, &[u8])]) -> Result<Record, Error> {
        self.publish_observed(files, &mut |_| {})
    }

    /// [`Attempt::publish`], calling `observe` at each [`Point`]. A test hook:
    /// a test kills the process from inside `observe`.
    pub fn publish_observed(
        self,
        files: &[(&str, &[u8])],
        observe: &mut dyn FnMut(Point),
    ) -> Result<Record, Error> {
        let mut sorted = files.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        let names: Vec<&str> = sorted.iter().map(|f| f.0).collect();
        check_names(&names).map_err(Error::Usage)?;

        let run = &self.run;
        let attempts = run.dir.join(ATTEMPTS);
        fs::create_dir_all(&attempts).map_err(|e| io_err("creating", &attempts, e))?;
        let adir = attempts.join(&self.attempt_id);
        fs::create_dir(&adir).map_err(|e| io_err("creating", &adir, e))?;
        let mut entries = Vec::with_capacity(sorted.len());
        for (name, bytes) in &sorted {
            write_new_synced(&adir.join(name), bytes)?;
            entries.push(FileEntry {
                name: name.to_string(),
                sha256: sha256_hex(bytes),
            });
        }
        let manifest = to_bytes(&Manifest {
            schema: MANIFEST_SCHEMA.into(),
            run_id: run.run_id.clone(),
            attempt_id: self.attempt_id.clone(),
            bindings_sha256: run.bindings_sha256.clone(),
            files: entries,
        });
        write_new_synced(&adir.join(MANIFEST), &manifest)?;
        sync_dir(&adir)?;
        sync_dir(&attempts)?;
        observe(Point::Staged);

        // @ai:invariant a run id has at most one committed attempt: committed.json is only created by hard_link, which refuses an existing target [T:test conf:0.9 src:recover::test_foreign_commit_during_attempt_is_refused]
        let committed = run.dir.join(COMMITTED);
        match fs::hard_link(adir.join(MANIFEST), &committed) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                return Err(run.corrupt(COMMITTED, "appeared while this owner held the run"))
            }
            Err(e) => return Err(io_err("linking", &committed, e)),
        }
        sync_dir(&run.dir)?;
        observe(Point::Committed);

        run.write_receipt(&self.attempt_id, &manifest)?;
        Ok(Record {
            attempt_id: self.attempt_id.clone(),
            files: sorted
                .into_iter()
                .map(|(n, b)| (n.to_string(), b.to_vec()))
                .collect(),
        })
    }
}
