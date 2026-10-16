//! Go source built into a stage (`GOOS=wasip1 GOARCH=wasm`), in a cache keyed by content.
//!
//! The key is a BLAKE3 hash of the source and the toolchain's version, so the same program on
//! the same Go is built once. The host writes `go.mod` itself; the caller gives `main.go` only.
//! The build may not fetch anything: `GOPROXY=off`, `GOTOOLCHAIN=local`, `GOWORK=off`, cgo off,
//! and the caller's `GOFLAGS` is replaced. So a stage uses the standard library alone.
//!
//! Building does not run the program: Go has no build-time code execution once cgo and
//! `go generate` are out. The program runs later, in the sandbox, as wasm.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// The largest `main.go` a build takes.
pub const MAX_SOURCE_BYTES: usize = 256 << 10;
/// The most compiler output a failed build reports.
pub const MAX_REPORT_CHARS: usize = 4000;

/// A Go toolchain and the folder its stages are cached in.
#[derive(Debug, Clone)]
pub struct GoBuild {
    go: PathBuf,
    cache: PathBuf,
    version: OnceLock<String>,
}

/// A stage in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Built {
    /// `go-` and 32 hex digits; the module is `<cache>/<id>.wasm`.
    pub id: String,
    pub path: PathBuf,
    /// True when an earlier build was reused.
    pub cached: bool,
    pub ms: u64,
}

/// Is `id` the name of a cached stage, and nothing that could leave the cache folder?
pub fn is_id(id: &str) -> bool {
    match id.split_once('-') {
        Some((kind, hex)) => {
            matches!(kind, "go" | "wat" | "wasm")
                && hex.len() == 32
                && hex
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        None => false,
    }
}

/// The id of `bytes` under `kind`, keyed also by `salt`.
pub fn id_of(kind: &str, salt: &str, bytes: &[u8]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(b"ix-wasm v1\0");
    h.update(kind.as_bytes());
    h.update(b"\0");
    h.update(salt.as_bytes());
    h.update(b"\0");
    h.update(bytes);
    format!("{kind}-{}", &h.finalize().to_hex()[..32])
}

impl GoBuild {
    /// The toolchain `go` (a path, or a name on `PATH`), caching stages in `cache`.
    pub fn new(go: impl Into<PathBuf>, cache: impl Into<PathBuf>) -> Self {
        Self {
            go: go.into(),
            cache: cache.into(),
            version: OnceLock::new(),
        }
    }

    pub fn cache(&self) -> &Path {
        &self.cache
    }

    /// The toolchain's version, as `go env GOVERSION` prints it; asked once.
    pub fn version(&self) -> Result<String, String> {
        if let Some(v) = self.version.get() {
            return Ok(v.clone());
        }
        let out = self
            .command()
            .args(["env", "GOVERSION"])
            .output()
            .map_err(|e| format!("go: cannot start {}: {e}", self.go.display()))?;
        if !out.status.success() {
            return Err(format!(
                "go env: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
        Ok(self.version.get_or_init(|| v).clone())
    }

    /// Builds `source` (a `package main`), or reuses the stage an earlier build left. Gives up at
    /// `deadline`. A failed build answers the compiler's report.
    pub fn build(&self, source: &str, deadline: Instant) -> Result<Built, String> {
        let started = Instant::now();
        if source.len() > MAX_SOURCE_BYTES {
            return Err(format!(
                "go: {} bytes of source; a stage is at most {MAX_SOURCE_BYTES}",
                source.len()
            ));
        }
        let version = self.version()?;
        let id = id_of("go", &version, source.as_bytes());
        let path = self.cache.join(format!("{id}.wasm"));
        if path.is_file() {
            return Ok(Built {
                id,
                path,
                cached: true,
                ms: started.elapsed().as_millis() as u64,
            });
        }
        // Two builds of one source may run at once: each gets its own folder, and the last
        // rename wins with the same bytes.
        static BUILDS: AtomicU64 = AtomicU64::new(0);
        let n = BUILDS.fetch_add(1, Ordering::Relaxed);
        let dir = self
            .cache
            .join(format!("build-{id}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("go: {}: {e}", dir.display()))?;
        let result = self.compile(&dir, source, &path, deadline);
        let _ = std::fs::remove_dir_all(&dir);
        result?;
        Ok(Built {
            id,
            path,
            cached: false,
            ms: started.elapsed().as_millis() as u64,
        })
    }

    fn command(&self) -> Command {
        let mut c = Command::new(&self.go);
        c.env("GOOS", "wasip1")
            .env("GOARCH", "wasm")
            .env("CGO_ENABLED", "0")
            .env("GOTOOLCHAIN", "local")
            .env("GOPROXY", "off")
            .env("GOWORK", "off")
            .env("GOFLAGS", "-mod=mod")
            .stdin(Stdio::null());
        c
    }

    fn compile(
        &self,
        dir: &Path,
        source: &str,
        path: &Path,
        deadline: Instant,
    ) -> Result<(), String> {
        std::fs::write(dir.join("go.mod"), "module stage\n\ngo 1.21\n")
            .and_then(|()| std::fs::write(dir.join("main.go"), source))
            .map_err(|e| format!("go: {}: {e}", dir.display()))?;
        let partial = dir.join("stage.wasm");
        let mut child = self
            .command()
            .current_dir(dir)
            .args(["build", "-trimpath", "-o"])
            .arg(&partial)
            .arg(".")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("go: cannot start {}: {e}", self.go.display()))?;
        let mut out = child.stdout.take().expect("piped stdout");
        let mut err = child.stderr.take().expect("piped stderr");
        let out_t = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = out.read_to_string(&mut s);
            s
        });
        let err_t = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = err.read_to_string(&mut s);
            s
        });
        let deadline = deadline.min(Instant::now() + Duration::from_secs(300));
        let status = loop {
            if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
                break s;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err("go: the build ran out of time".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let report = out_t.join().unwrap_or_default() + &err_t.join().unwrap_or_default();
        if !status.success() {
            let report = report.replace(&dir.display().to_string(), ".");
            return Err(clip(report.trim(), MAX_REPORT_CHARS));
        }
        std::fs::rename(&partial, path).map_err(|e| format!("go: {}: {e}", path.display()))
    }
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}
