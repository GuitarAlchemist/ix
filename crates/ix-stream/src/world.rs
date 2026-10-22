//! The world nodes of `stream.run`, behind the `world` feature. They reach outside the process,
//! so a graph that has one is no longer deterministic: its counts are reported, not predicted.
//!
//! - [`Http`], a stage that sends one request per value, or one per element of a list field, to
//!   an allowed loopback service ([`ALLOWED`]) and adds what came back to the value. Any other
//!   origin is refused before a connection is made, a redirect is not followed, and a failed or
//!   timed-out call is a result with an `error`, never a crash.
//! - [`Sql`], a batching node that runs DuckDB over every value it took, as the table `batch`,
//!   and sends each row. The query reaches no file and no network: DuckDB's external access is
//!   turned off once `batch` is loaded.
//! - [`File`], a sink that writes what it takes to a JSON-lines file under the run's `out_dir`.

use crate::skill::send_all;
use crate::sql::query_sealed;
use crate::{Batch, Out};
use reqwest::blocking::Client;
use reqwest::redirect::Policy;
use reqwest::Url;
use serde_json::{json, Map, Value};
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The only origins an `http` node may call: GaApi and the GA chatbot, on loopback.
pub const ALLOWED: &[(&str, u16)] = &[
    ("127.0.0.1", 5232),
    ("127.0.0.1", 5252),
    ("localhost", 5232),
    ("localhost", 5252),
];
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;

/// What the world nodes of a run wrote, and what went wrong, for the receipts of the answer.
#[derive(Debug, Default)]
pub struct Receipts {
    pub files: Vec<Value>,
    pub faults: Vec<String>,
}

pub type Shared = Arc<Mutex<Receipts>>;

#[derive(Debug, Clone, Copy)]
enum Method {
    Get,
    Post,
}

/// An `http` node: `{ url, method, body, each, pick, into, timeout_ms }`.
///
/// `url` may hold `{field}` placeholders, filled from the value and percent-encoded (a `/`
/// stays). With `each`, the node sends one request per element of that list field, and
/// `{each-field}` is the element. `body`, for a POST, is JSON whose strings `"{field}"` are
/// replaced by the value's field, and whose other strings have their `{field}`s filled as text. Each call answers `{ url, status, ms, error, body }`, with
/// `body` cut down to `pick` (`{ name: JSON pointer }`) when given, under the field `into`.
pub struct Http {
    name: String,
    url: String,
    method: Method,
    body: Option<Value>,
    each: Option<String>,
    pick: Option<Map<String, Value>>,
    into: String,
    timeout: Duration,
    allowed: Vec<(String, u16)>,
    client: Client,
}

impl Http {
    pub fn from_json(name: &str, cfg: &Value) -> Result<Self, String> {
        Self::with_allowed(name, cfg, ALLOWED)
    }

    fn with_allowed(name: &str, cfg: &Value, allowed: &[(&str, u16)]) -> Result<Self, String> {
        let text = |key: &str| -> Result<Option<String>, String> {
            match cfg.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                Some(v) => Err(format!("{name}: http `{key}` is a string, got {v}")),
            }
        };
        let url = text("url")?.ok_or_else(|| format!("{name}: an http node needs a `url`"))?;
        let method = match text("method")?.as_deref().unwrap_or("GET") {
            "GET" => Method::Get,
            "POST" => Method::Post,
            m => return Err(format!("{name}: http `method` is GET or POST, got {m}")),
        };
        let ms = match cfg.get("timeout_ms") {
            None | Some(Value::Null) => DEFAULT_TIMEOUT_MS,
            Some(v) => v
                .as_u64()
                .filter(|ms| (1..=MAX_TIMEOUT_MS).contains(ms))
                .ok_or_else(|| format!("{name}: timeout_ms is 1 to {MAX_TIMEOUT_MS}, got {v}"))?,
        };
        let pick = match cfg.get("pick") {
            None | Some(Value::Null) => None,
            Some(Value::Object(m)) if m.values().all(Value::is_string) => Some(m.clone()),
            Some(v) => {
                return Err(format!(
                    "{name}: http `pick` maps names to JSON pointers, got {v}"
                ))
            }
        };
        let timeout = Duration::from_millis(ms);
        let client = Client::builder()
            .redirect(Policy::none())
            .no_proxy()
            .timeout(timeout)
            .build()
            .map_err(|e| format!("{name}: {e}"))?;
        let http = Self {
            name: name.to_string(),
            each: text("each")?,
            into: text("into")?.unwrap_or_else(|| name.to_string()),
            body: cfg.get("body").filter(|b| !b.is_null()).cloned(),
            url,
            method,
            pick,
            timeout,
            allowed: allowed.iter().map(|(h, p)| (h.to_string(), *p)).collect(),
            client,
        };
        // Before the run: what fills a placeholder is encoded and cannot move the origin, and
        // the full URL is checked again before each call.
        http.check(&fill(&http.url, &Value::Null, None, true)?)?;
        Ok(http)
    }

    /// The URL, when it parses and its origin is allowed.
    fn check(&self, url: &str) -> Result<Url, String> {
        let allowed = |u: &Url| {
            let host = u.host_str().unwrap_or_default();
            u.scheme() == "http"
                && u.username().is_empty()
                && u.password().is_none()
                && self
                    .allowed
                    .iter()
                    .any(|(h, p)| h == host && u.port() == Some(*p))
        };
        match Url::parse(url) {
            Ok(u) if allowed(&u) => Ok(u),
            _ => {
                let list: Vec<String> = self
                    .allowed
                    .iter()
                    .map(|(h, p)| format!("http://{h}:{p}"))
                    .collect();
                Err(format!(
                    "{}: {url} is not an allowed origin; an http node calls only {}",
                    self.name,
                    list.join(", ")
                ))
            }
        }
    }

    /// The value with what came back under `into`: one result, or a list of them with `each`.
    pub fn apply(&self, mut v: Value) -> Value {
        let got = match &self.each {
            None => self.call(&v, None),
            Some(f) => match v.get(f).and_then(Value::as_array) {
                Some(items) => items.iter().map(|e| self.call(&v, Some((f, e)))).collect(),
                None => failed(Value::Null, None, 0, format!("no list in `{f}`")),
            },
        };
        if let Value::Object(m) = &mut v {
            m.insert(self.into.clone(), got);
        }
        v
    }

    fn call(&self, v: &Value, each: Option<(&str, &Value)>) -> Value {
        let t = Instant::now();
        let url = match fill(&self.url, v, each, false).and_then(|u| self.check(&u)) {
            Ok(u) => u,
            Err(e) => return failed(Value::Null, None, 0, e),
        };
        let shown = json!(url.as_str());
        let req = match self.method {
            Method::Get => self.client.get(url),
            Method::Post => {
                let body = match &self.body {
                    Some(b) => match fill_json(b, v, each) {
                        Ok(b) => b,
                        Err(e) => return failed(shown, None, 0, e),
                    },
                    None => Value::Null,
                };
                self.client
                    .post(url)
                    .header("content-type", "application/json")
                    .body(body.to_string())
            }
        };
        let ms = |t: Instant| t.elapsed().as_millis() as u64;
        let res = match req.send() {
            Ok(r) => r,
            Err(e) => return failed(shown, None, ms(t), self.why(&e)),
        };
        let status = res.status();
        let bytes = match res.bytes() {
            Ok(b) => b,
            Err(e) => return failed(shown, Some(status.as_u16()), ms(t), self.why(&e)),
        };
        let took = ms(t);
        let body: Option<Value> = serde_json::from_slice(&bytes).ok();
        let error = if status.is_redirection() {
            Some(format!("HTTP {status}: a redirect, not followed"))
        } else if !status.is_success() {
            Some(format!("HTTP {status}"))
        } else if body.is_none() {
            Some("the answer is not JSON".to_string())
        } else {
            None
        };
        let body = match (&self.pick, body) {
            (Some(pick), Some(b)) => Value::Object(
                pick.iter()
                    .map(|(k, p)| {
                        let at = b.pointer(p.as_str().unwrap_or_default());
                        (k.clone(), at.cloned().unwrap_or(Value::Null))
                    })
                    .collect(),
            ),
            (_, b) => b.unwrap_or(Value::Null),
        };
        json!({ "url": shown, "status": status.as_u16(), "ms": took, "error": error, "body": body })
    }

    fn why(&self, e: &reqwest::Error) -> String {
        if e.is_timeout() {
            format!("timed out after {} ms", self.timeout.as_millis())
        } else if e.is_connect() {
            format!("cannot connect: {e}")
        } else {
            e.to_string()
        }
    }
}

fn failed(url: Value, status: Option<u16>, ms: u64, error: String) -> Value {
    json!({ "url": url, "status": status, "ms": ms, "error": error, "body": null })
}

/// `template` with each `{field}` replaced by the value's field, percent-encoded. With
/// `any`, a placeholder is left empty whatever the value holds: the origin check before the
/// run, which so refuses a placeholder in the origin.
fn fill(
    template: &str,
    v: &Value,
    each: Option<(&str, &Value)>,
    any: bool,
) -> Result<String, String> {
    fill_as(template, v, each, any, encode)
}

/// [`fill`], each field's text passed through `enc`.
fn fill_as(
    template: &str,
    v: &Value,
    each: Option<(&str, &Value)>,
    any: bool,
    enc: fn(&str) -> String,
) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let close = rest[open..]
            .find('}')
            .ok_or_else(|| format!("{template}: a `{{` without `}}`"))?;
        let field = &rest[open + 1..open + close];
        if !any {
            let got = match each {
                Some((f, e)) if f == field => e,
                _ => v
                    .get(field)
                    .ok_or_else(|| format!("{template}: the value has no `{field}`"))?,
            };
            let s = match got {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            out.push_str(&enc(&s));
        }
        rest = &rest[open + close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Percent-encodes all but unreserved characters and `/`.
fn encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `body` with each string `"{field}"` replaced by the value's field, and the `{field}`s of
/// any other string filled as text: `"{q} Name the key."`.
fn fill_json(body: &Value, v: &Value, each: Option<(&str, &Value)>) -> Result<Value, String> {
    let whole = |s: &str| {
        s.len() > 1
            && s.starts_with('{')
            && s.ends_with('}')
            && !s[1..s.len() - 1].contains(['{', '}'])
    };
    Ok(match body {
        Value::String(s) if !whole(s) && s.contains('{') => {
            Value::String(fill_as(s, v, each, false, str::to_string)?)
        }
        Value::String(s) if whole(s) => {
            let field = &s[1..s.len() - 1];
            match each {
                Some((f, e)) if f == field => e.clone(),
                _ => v
                    .get(field)
                    .cloned()
                    .ok_or_else(|| format!("body: the value has no `{field}`"))?,
            }
        }
        Value::Array(a) => Value::Array(
            a.iter()
                .map(|b| fill_json(b, v, each))
                .collect::<Result<_, _>>()?,
        ),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, b)| Ok((k.clone(), fill_json(b, v, each)?)))
                .collect::<Result<_, String>>()?,
        ),
        other => other.clone(),
    })
}

/// A `sql` node: holds every value it takes and, when its input ends, runs the query over them
/// as `batch` and sends each row.
pub struct Sql {
    name: String,
    sql: String,
    scratch: PathBuf,
    held: Vec<Value>,
    receipts: Shared,
}

impl Sql {
    pub fn new(name: &str, sql: &str, dir: &Path, receipts: &Shared) -> Self {
        Self {
            name: name.to_string(),
            sql: sql.to_string(),
            scratch: dir.join(format!("{name}.batch.jsonl")),
            held: Vec::new(),
            receipts: receipts.clone(),
        }
    }
}

impl Batch for Sql {
    fn take(&mut self, v: Value, _: &Out) {
        self.held.push(v);
    }

    fn end(&mut self, out: &Out) {
        if self.held.is_empty() {
            return;
        }
        match query_sealed(&self.held, &self.sql, &self.scratch) {
            Ok(rows) => {
                for r in rows {
                    if send_all(out, &r).is_err() {
                        break;
                    }
                }
            }
            Err(e) => fault(&self.receipts, format!("{}: {e}", self.name)),
        }
    }
}

/// A `file` sink: writes each value it takes as a line of JSON.
pub struct File {
    name: String,
    path: PathBuf,
    w: Option<BufWriter<std::fs::File>>,
    rows: u64,
    receipts: Shared,
}

impl File {
    /// A sink writing to `file` under `dir`. `file` is a relative path that stays under it.
    pub fn new(name: &str, file: &str, dir: &Path, receipts: &Shared) -> Result<Self, String> {
        let rel = Path::new(file);
        if rel.as_os_str().is_empty()
            || !rel.components().all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(format!(
                "{name}: `file` is a relative path that stays under out_dir, got {file}"
            ));
        }
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let f = std::fs::File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            name: name.to_string(),
            path,
            w: Some(BufWriter::new(f)),
            rows: 0,
            receipts: receipts.clone(),
        })
    }
}

impl Batch for File {
    fn take(&mut self, v: Value, _: &Out) {
        if let Some(w) = &mut self.w {
            match writeln!(w, "{v}") {
                Ok(()) => self.rows += 1,
                Err(e) => fault(&self.receipts, format!("{}: {e}", self.name)),
            }
        }
    }

    fn end(&mut self, _: &Out) {
        if let Some(mut w) = self.w.take() {
            if let Err(e) = w.flush() {
                fault(&self.receipts, format!("{}: {e}", self.name));
            }
        }
        if let Ok(mut r) = self.receipts.lock() {
            r.files.push(json!({
                "node": self.name,
                "path": self.path.display().to_string(),
                "rows": self.rows,
            }));
        }
    }
}

fn fault(receipts: &Shared, e: String) {
    if let Ok(mut r) = receipts.lock() {
        r.faults.push(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    /// A listener on a free loopback port, and an http node allowed to call only it.
    fn local(cfg: Value) -> (TcpListener, Http) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let mut cfg = cfg;
        cfg["url"] = json!(format!("http://127.0.0.1:{port}/x"));
        let http = Http::with_allowed("t", &cfg, &[("127.0.0.1", port)]).unwrap();
        (l, http)
    }

    #[test]
    fn an_origin_off_the_list_is_refused_before_any_connection() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.set_nonblocking(true).unwrap();
        let port = l.local_addr().unwrap().port();
        for url in [
            format!("http://127.0.0.1:{port}/api"),
            "http://example.com/api".to_string(),
            "https://127.0.0.1:5232/api".to_string(),
            "http://127.0.0.1:5232@127.0.0.1:{port}/api".replace("{port}", &port.to_string()),
        ] {
            let e = Http::from_json("t", &json!({ "url": url })).err().unwrap();
            assert!(e.contains("not an allowed origin"), "{e}");
        }
        // A value cannot move the origin: what fills a placeholder is encoded.
        let http = Http::from_json("t", &json!({ "url": "http://127.0.0.1:5232{p}" })).unwrap();
        let got = http.apply(json!({ "p": format!("@127.0.0.1:{port}/api") }));
        assert!(
            got["t"]["error"]
                .as_str()
                .unwrap()
                .contains("not an allowed origin"),
            "{got}"
        );
        assert!(
            matches!(l.accept(), Err(e) if e.kind() == std::io::ErrorKind::WouldBlock),
            "nothing connected"
        );
    }

    #[test]
    fn a_redirect_is_not_followed() {
        let (l, http) = local(json!({}));
        let t = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            s.write_all(
                b"HTTP/1.1 302 Found\r\nLocation: http://example.com/\r\nContent-Length: 0\r\n\r\n",
            )
            .unwrap();
        });
        let got = http.apply(json!({}));
        t.join().unwrap();
        assert_eq!(got["t"]["status"], 302);
        assert!(got["t"]["error"].as_str().unwrap().contains("not followed"));
    }

    #[test]
    fn a_call_that_times_out_is_a_row_with_an_error() {
        let (l, http) = local(json!({ "timeout_ms": 200 }));
        let t = std::thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            std::thread::sleep(Duration::from_millis(800));
            drop(s);
        });
        let got = http.apply(json!({ "id": 1 }));
        t.join().unwrap();
        assert_eq!(got["id"], 1);
        assert!(got["t"]["status"].is_null());
        assert_eq!(got["t"]["error"], "timed out after 200 ms");
    }

    #[test]
    fn a_body_string_with_text_around_a_field_is_filled_as_text() {
        let v = json!({ "q": "notes of C?", "n": 3 });
        let body = json!({ "message": "{q}\n\nName the key.", "n": "{n}" });
        assert_eq!(
            fill_json(&body, &v, None).unwrap(),
            json!({ "message": "notes of C?\n\nName the key.", "n": 3 })
        );
        let missing = fill_json(&json!({ "m": "{q} {nope}" }), &v, None);
        assert!(missing.unwrap_err().contains("no `nope`"));
    }

    #[test]
    fn a_post_sends_the_body_and_picks_from_the_answer() {
        let (l, http) = local(json!({
            "method": "POST", "body": { "message": "{q}" }, "pick": { "answer": "/a/b" }, "into": "bot"
        }));
        let t = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            while !String::from_utf8_lossy(&req).contains("}") {
                let n = s.read(&mut buf).unwrap();
                req.extend_from_slice(&buf[..n]);
            }
            let body = r#"{"a":{"b":"C E G"},"other":1}"#;
            write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            String::from_utf8_lossy(&req).to_string()
        });
        let got = http.apply(json!({ "q": "notes of C?" }));
        let req = t.join().unwrap();
        assert!(req.starts_with("POST /x "), "{req}");
        assert!(req.contains(r#"{"message":"notes of C?"}"#), "{req}");
        assert_eq!(got["bot"]["status"], 200);
        assert!(got["bot"]["error"].is_null());
        assert_eq!(got["bot"]["body"], json!({ "answer": "C E G" }));
    }

    #[test]
    fn each_sends_one_request_per_element_and_encodes_it() {
        let http = Http::from_json(
            "ga",
            &json!({ "url": "http://127.0.0.1:5232/api/{paths}", "each": "paths" }),
        )
        .unwrap();
        assert_eq!(
            fill(
                &http.url,
                &json!({}),
                Some(("paths", &json!("keys/F# major"))),
                false
            )
            .unwrap(),
            "http://127.0.0.1:5232/api/keys/F%23%20major"
        );
        let got = http.apply(json!({ "paths": 3 }));
        assert_eq!(got["ga"]["error"], "no list in `paths`");
    }
}
