//! `serve-viz` — local HTTP server for the t-SNE voicing viewer.
//!
//! Serves `crates/ix-voicings/web/index.html` (embedded at compile time)
//! at `/` and the contents of `state/viz/` (or `--data <dir>`) at
//! `/data/<file>`. Stdlib only, blocking, one thread per connection — fine
//! for a localhost demo.
//!
//! ```sh
//! cargo run -p ix-voicings --bin serve-viz
//! cargo run -p ix-voicings --bin serve-viz -- --port 8765 --data state/viz
//! ```

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Instant;

use clap::Parser;
use serde::{Deserialize, Serialize};

const INDEX_HTML: &str = include_str!("../../web/index.html");
const VOICING_LAYOUT: &str = "voicing-layout.json";
const CLUSTER_ASSIGNMENTS: &str = "cluster-assignments.json";

#[derive(Parser, Debug)]
#[command(about = "Serve the ix-voicings t-SNE viewer on localhost")]
struct Cli {
    /// Bind port (0 = pick a free one).
    #[arg(long, default_value_t = 8765)]
    port: u16,

    /// Directory containing the viz JSON files (served at /data/<file>).
    #[arg(long, default_value = "state/viz")]
    data: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let data_dir = cli
        .data
        .canonicalize()
        .unwrap_or_else(|_| cli.data.clone());

    if let Err(e) = ensure_cluster_assignments(&data_dir) {
        eprintln!("warning: could not derive {CLUSTER_ASSIGNMENTS}: {e}");
        eprintln!("         (the viewer will fall back to instrument-only colouring)");
    }

    let listener = TcpListener::bind(("127.0.0.1", cli.port))
        .unwrap_or_else(|e| panic!("bind 127.0.0.1:{}: {e}", cli.port));
    let addr = listener.local_addr().expect("local_addr");
    println!("serving viewer at  http://{addr}/");
    println!("                   data dir: {}", data_dir.display());

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let dir = data_dir.clone();
                thread::spawn(move || {
                    if let Err(e) = handle(s, &dir) {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
}

fn handle(mut stream: TcpStream, data_dir: &Path) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();

    // Drain headers — we don't need them, but the client expects us to read past them.
    let mut header = String::new();
    loop {
        header.clear();
        let n = reader.read_line(&mut header)?;
        if n == 0 || header == "\r\n" || header == "\n" {
            break;
        }
    }

    if path == "/" || path == "/index.html" {
        return write_response(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes());
    }

    if let Some(rest) = path.strip_prefix("/data/") {
        // Reject anything that escapes the data dir.
        if rest.contains("..") || rest.contains('\\') || rest.starts_with('/') {
            return write_response(&mut stream, 400, "text/plain", b"bad path");
        }
        let target = data_dir.join(rest);
        match std::fs::File::open(&target) {
            Ok(mut f) => {
                let ct = content_type(&target);
                let len = f.metadata().map(|m| m.len()).unwrap_or(0);
                write!(
                    stream,
                    "HTTP/1.0 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {len}\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
                )?;
                let mut buf = vec![0u8; 64 * 1024];
                loop {
                    let n = f.read(&mut buf)?;
                    if n == 0 { break; }
                    stream.write_all(&buf[..n])?;
                }
                Ok(())
            }
            Err(_) => write_response(&mut stream, 404, "text/plain", b"not found"),
        }
    } else {
        write_response(&mut stream, 404, "text/plain", b"not found")
    }
}

fn write_response(stream: &mut TcpStream, status: u16, ct: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Status",
    };
    write!(
        stream,
        "HTTP/1.0 {status} {reason}\r\nContent-Type: {ct}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

/// Compact derivative of `voicing-layout.json` keyed for fast lookup in the
/// browser. The layout file is ~160 MB; this output is ~3 MB. Stores two
/// per-voicing categorical attributes the browser can colour by: cluster
/// (interned to ints) and chord family (already int).
#[derive(Serialize)]
struct ClusterAssignments {
    /// Cluster id strings (`"guitar-C0"`, etc.) — index into this array is
    /// the integer used in `cluster_by_inst` below.
    clusters: Vec<String>,
    /// `cluster_by_inst[instrument][i]` = cluster index for that
    /// instrument's voicing whose per-instrument id is `i`. The id matches
    /// the `id` field on each t-SNE point. Missing voicings are -1.
    cluster_by_inst: BTreeMap<String, Vec<i32>>,
    /// `family_by_inst[instrument][i]` = chord_family_id, parallel layout.
    family_by_inst: BTreeMap<String, Vec<i32>>,
}

#[derive(Deserialize)]
struct LayoutRow {
    global_id: String,
    cluster_id: String,
    chord_family_id: i32,
    instrument: String,
}

fn ensure_cluster_assignments(data_dir: &Path) -> std::io::Result<()> {
    let out_path = data_dir.join(CLUSTER_ASSIGNMENTS);
    if out_path.exists() {
        return Ok(());
    }
    let layout_path = data_dir.join(VOICING_LAYOUT);
    if !layout_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{} missing — run `cargo run -p ix-voicings -- viz-precompute`", layout_path.display()),
        ));
    }

    println!("deriving {CLUSTER_ASSIGNMENTS} from {VOICING_LAYOUT} (one-time)...");
    let started = Instant::now();
    let bytes = std::fs::read(&layout_path)?;
    let rows: Vec<LayoutRow> = serde_json::from_slice(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let mut clusters: Vec<String> = Vec::new();
    let mut cluster_idx: BTreeMap<String, i32> = BTreeMap::new();
    let mut cluster_by_inst: BTreeMap<String, Vec<i32>> = BTreeMap::new();
    let mut family_by_inst: BTreeMap<String, Vec<i32>> = BTreeMap::new();

    for row in rows {
        let cidx = if let Some(&i) = cluster_idx.get(&row.cluster_id) {
            i
        } else {
            let i = clusters.len() as i32;
            clusters.push(row.cluster_id.clone());
            cluster_idx.insert(row.cluster_id, i);
            i
        };

        let Some(rest) = row.global_id.strip_prefix(&format!("{}_v", row.instrument)) else {
            continue;
        };
        let Ok(id): Result<usize, _> = rest.parse() else { continue; };

        let cluster_bucket = cluster_by_inst.entry(row.instrument.clone()).or_default();
        if cluster_bucket.len() <= id {
            cluster_bucket.resize(id + 1, -1);
        }
        cluster_bucket[id] = cidx;

        let family_bucket = family_by_inst.entry(row.instrument).or_default();
        if family_bucket.len() <= id {
            family_bucket.resize(id + 1, -1);
        }
        family_bucket[id] = row.chord_family_id;
    }

    let assignments = ClusterAssignments { clusters, cluster_by_inst, family_by_inst };
    let serialized = serde_json::to_vec(&assignments)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&out_path, &serialized)?;

    println!(
        "wrote {} ({} bytes, {} clusters, {:.1}s)",
        out_path.display(),
        serialized.len(),
        assignments.clusters.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => "application/json",
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        _ => "application/octet-stream",
    }
}
