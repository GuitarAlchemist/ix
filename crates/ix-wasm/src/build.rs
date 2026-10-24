//! Source built into a stage, in a cache keyed by content: Go (`GOOS=wasip1`) or Zig
//! (`-target wasm32-wasi`).
//!
//! The key is a BLAKE3 hash of the language, the toolchain's version and the source, so the
//! same program on the same toolchain is built once. The caller gives one source file; the host
//! writes everything else (Go's `go.mod`).
//!
//! A build may not fetch anything. Go runs with `GOPROXY=off`, `GOTOOLCHAIN=local`,
//! `GOWORK=off`, cgo off and its own `GOFLAGS`, so a stage uses the standard library alone.
//! Zig runs `build-exe` on the file, never a `build.zig`, so no build script runs.
//!
//! Building does not run the program: Go has no build-time code execution once cgo and
//! `go generate` are out, and Zig's `comptime` cannot reach the system. The program runs later,
//! in the sandbox, as wasm.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// The largest source file a build takes.
pub const MAX_SOURCE_BYTES: usize = 256 << 10;
/// The most compiler output a failed build reports.
pub const MAX_REPORT_CHARS: usize = 4000;

/// A language a stage can be written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Go,
    Zig,
}

impl Lang {
    /// The prefix of its stage ids.
    pub fn kind(self) -> &'static str {
        match self {
            Lang::Go => "go",
            Lang::Zig => "zig",
        }
    }

    fn file(self) -> &'static str {
        match self {
            Lang::Go => "main.go",
            Lang::Zig => "main.zig",
        }
    }
}

/// A toolchain and the folder its stages are cached in.
#[derive(Debug, Clone)]
pub struct Toolchain {
    lang: Lang,
    exe: PathBuf,
    cache: PathBuf,
    version: OnceLock<String>,
}

/// A stage in the cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Built {
    /// The language's prefix and 32 hex digits; the module is `<cache>/<id>.wasm`.
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
            matches!(kind, "go" | "zig" | "wat" | "wasm")
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

impl Toolchain {
    /// Go: `exe` is a path, or a name on `PATH`; stages are cached in `cache`.
    pub fn go(exe: impl Into<PathBuf>, cache: impl Into<PathBuf>) -> Self {
        Self::new(Lang::Go, exe, cache)
    }

    /// Zig: `exe` is a path, or a name on `PATH`; stages are cached in `cache`.
    pub fn zig(exe: impl Into<PathBuf>, cache: impl Into<PathBuf>) -> Self {
        Self::new(Lang::Zig, exe, cache)
    }

    fn new(lang: Lang, exe: impl Into<PathBuf>, cache: impl Into<PathBuf>) -> Self {
        Self {
            lang,
            exe: exe.into(),
            cache: cache.into(),
            version: OnceLock::new(),
        }
    }

    pub fn lang(&self) -> Lang {
        self.lang
    }

    pub fn cache(&self) -> &Path {
        &self.cache
    }

    /// The toolchain's version; asked once.
    pub fn version(&self) -> Result<String, String> {
        if let Some(v) = self.version.get() {
            return Ok(v.clone());
        }
        let mut c = self.command();
        match self.lang {
            Lang::Go => c.args(["env", "GOVERSION"]),
            Lang::Zig => c.arg("version"),
        };
        let out = c.output().map_err(|e| {
            format!(
                "{}: cannot start {}: {e}",
                self.lang.kind(),
                self.exe.display()
            )
        })?;
        if !out.status.success() {
            return Err(format!(
                "{} version: {}",
                self.lang.kind(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
        Ok(self.version.get_or_init(|| v).clone())
    }

    /// Builds `source`, or reuses the stage an earlier build left. Gives up at `deadline`. A
    /// failed build answers the compiler's report.
    pub fn build(&self, source: &str, deadline: Instant) -> Result<Built, String> {
        let started = Instant::now();
        let kind = self.lang.kind();
        if source.len() > MAX_SOURCE_BYTES {
            return Err(format!(
                "{kind}: {} bytes of source; a stage is at most {MAX_SOURCE_BYTES}",
                source.len()
            ));
        }
        let version = self.version()?;
        let id = id_of(kind, &version, source.as_bytes());
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
        std::fs::create_dir_all(&dir).map_err(|e| format!("{kind}: {}: {e}", dir.display()))?;
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
        let mut c = Command::new(&self.exe);
        c.stdin(Stdio::null());
        if self.lang == Lang::Go {
            c.env("GOOS", "wasip1")
                .env("GOARCH", "wasm")
                .env("CGO_ENABLED", "0")
                .env("GOTOOLCHAIN", "local")
                .env("GOPROXY", "off")
                .env("GOWORK", "off")
                .env("GOFLAGS", "-mod=mod");
        }
        c
    }

    fn compile(
        &self,
        dir: &Path,
        source: &str,
        path: &Path,
        deadline: Instant,
    ) -> Result<(), String> {
        let kind = self.lang.kind();
        let io = |e: std::io::Error| format!("{kind}: {}: {e}", dir.display());
        std::fs::write(dir.join(self.lang.file()), source).map_err(io)?;
        let partial = dir.join("stage.wasm");
        let mut c = self.command();
        c.current_dir(dir);
        match self.lang {
            Lang::Go => {
                std::fs::write(dir.join("go.mod"), "module stage\n\ngo 1.21\n").map_err(io)?;
                c.args(["build", "-trimpath", "-o"]).arg(&partial).arg(".");
            }
            Lang::Zig => {
                // A 64 KiB stack: Zig's default 16 MiB makes every run start by zeroing it.
                c.args([
                    "build-exe",
                    "main.zig",
                    "-target",
                    "wasm32-wasi",
                    "-O",
                    "ReleaseSmall",
                    "-fstrip",
                    "--stack",
                    "65536",
                ])
                .arg(format!("-femit-bin={}", partial.display()))
                .env("ZIG_GLOBAL_CACHE_DIR", self.cache.join("zig-global"))
                .env("ZIG_LOCAL_CACHE_DIR", dir.join("zig-local"));
            }
        }
        let (status, report) = spawn_until(c, deadline, kind)?;
        if !status.success() {
            let report = report.replace(&dir.display().to_string(), ".");
            return Err(clip(report.trim(), MAX_REPORT_CHARS));
        }
        std::fs::rename(&partial, path).map_err(|e| format!("{kind}: {}: {e}", path.display()))
    }
}

/// Runs `c` to its end or to `deadline` (at most 300 s), and gives its status and output.
fn spawn_until(
    mut c: Command,
    deadline: Instant,
    kind: &str,
) -> Result<(ExitStatus, String), String> {
    let mut child = c
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{kind}: cannot start: {e}"))?;
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
            return Err(format!("{kind}: the build ran out of time"));
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let report = out_t.join().unwrap_or_default() + &err_t.join().unwrap_or_default();
    Ok((status, report))
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}
