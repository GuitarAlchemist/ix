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

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::thread;

use clap::Parser;

const INDEX_HTML: &str = include_str!("../../web/index.html");

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

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => "application/json",
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        _ => "application/octet-stream",
    }
}
