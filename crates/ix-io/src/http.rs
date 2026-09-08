//! HTTP client for fetching data from REST APIs and static endpoints.
//!
//! # This module is the crate's network-egress boundary
//!
//! Everything here dials an address a caller chose, so every entry point runs
//! through [`checked_url`] (scheme refusal) and [`send_bounded`] (declared-size
//! rejection, streamed byte cap, request timeout). There is deliberately no
//! unbounded path left in the module: an 8 MiB ceiling that one convenience
//! function bypasses is decoration, not a bound.
//!
//! `reqwest` supplies none of this. `Response::text()` and `Response::bytes()`
//! buffer the **entire** body with no limit, so a hostile or misconfigured
//! endpoint returns an out-of-memory abort rather than an error, and
//! `reqwest::get` applies no timeout at all — a server that accepts and never
//! answers hangs the caller forever. Following ix#286: reject loudly, at a
//! ceiling the caller can read, rather than inherit a silent one.
//!
//! # What is *not* bounded here
//!
//! **There is no SSRF guard.** A URL naming `127.0.0.1`, `169.254.169.254`, or
//! any RFC1918 address is dialled like any other. That is acceptable while the
//! only callers are in-process Rust, and is **not** acceptable if these
//! functions are ever reached from an MCP tool or a DuckDB UDF, where the URL
//! becomes attacker-influenced. Resolve it before adding that exposure, not
//! after. Redirects are followed with `reqwest`'s default policy, which is the
//! same hole by another route.
//!
//! [`DataSource`]: crate::protocol::DataSource

use std::time::Duration;

use crate::error::IoError;
use crate::protocol::{BatchSource, DataBatch};

/// Ceiling on the response bytes this crate will buffer from one request.
///
/// 8 MiB is ~1M CSV-encoded `f64` fields — far more than a training batch
/// needs, and small enough that a hostile endpoint cannot exhaust memory
/// before the count trips. The comparison that matters is not with a smaller
/// number but with the alternative: `reqwest` imposes **no** body limit
/// whatsoever, so without this constant the effective cap is available RAM.
pub const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// Default whole-request timeout, covering connect, headers, and body.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default connect-phase timeout, so an unroutable host fails fast rather than
/// consuming the whole of [`DEFAULT_TIMEOUT`].
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Bounds applied to one HTTP fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchLimits {
    /// Hard ceiling on buffered response bytes. See [`MAX_RESPONSE_BYTES`].
    pub max_bytes: usize,
    /// Whole-request timeout.
    pub timeout: Duration,
    /// Connect-phase timeout.
    pub connect_timeout: Duration,
}

impl Default for FetchLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_RESPONSE_BYTES,
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

impl FetchLimits {
    /// Override the byte ceiling.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Override the whole-request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// How to interpret a fetched body as records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadFormat {
    /// A JSON array of objects, an array of arrays, or a single object.
    Json,
    /// Newline-delimited JSON objects.
    Ndjson,
    /// CSV, with or without a header row.
    Csv {
        /// Whether the first row names the columns.
        has_header: bool,
    },
}

/// Parse a URL and refuse any scheme this crate is not willing to dial.
///
/// `file:`, `data:`, `ftp:` and friends are rejected here rather than deeper
/// down, so the refusal is a stated policy with a readable error instead of
/// whatever the transport happens to say.
///
/// There is deliberately **no** host check on top. For the two schemes this
/// function admits, WHATWG treats an empty host as a parse failure, so
/// `http://` is already an "empty host" parse error above — and `http:///nohost`
/// normalises to host `nohost` under the same spec's slash tolerance, so a
/// hand-written emptiness test would be unreachable code that looks like a
/// guard. Both behaviours are pinned by
/// [`empty_host_is_rejected_by_the_parser_itself`](tests::empty_host_is_rejected_by_the_parser_itself).
pub fn checked_url(url: &str) -> Result<reqwest::Url, IoError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| IoError::Parse(format!("not a valid URL: {url:?} ({e})")))?;

    match parsed.scheme() {
        "http" | "https" => Ok(parsed),
        other => Err(IoError::Limit(format!(
            "refusing to fetch {other:?} URL {url:?}: ix-io dials http and https only"
        ))),
    }
}

/// Build a client carrying the timeout half of `limits`.
fn client(limits: &FetchLimits) -> Result<reqwest::Client, IoError> {
    reqwest::Client::builder()
        .timeout(limits.timeout)
        .connect_timeout(limits.connect_timeout)
        .build()
        .map_err(IoError::Http)
}

/// Send a request and read at most `max_bytes` of the response body.
///
/// Two independent guards, because either alone is insufficient:
///
/// 1. **Declared length.** A `Content-Length` above the ceiling is refused
///    before a single body byte is read. Cheap, but a server may omit or
///    understate the header.
/// 2. **Streamed count.** Chunks are accumulated and the running total is
///    checked, so an undeclared or chunked body is bounded by what actually
///    arrives rather than by what was promised.
async fn send_bounded(
    request: reqwest::RequestBuilder,
    max_bytes: usize,
) -> Result<Vec<u8>, IoError> {
    let mut response = request.send().await?.error_for_status()?;

    if let Some(declared) = response.content_length() {
        if declared > max_bytes as u64 {
            return Err(IoError::Limit(format!(
                "response declares Content-Length {declared} bytes, above the {max_bytes}-byte \
                 ceiling; refusing to read it"
            )));
        }
    }

    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > max_bytes {
            return Err(IoError::Limit(format!(
                "response body exceeds the {max_bytes}-byte ceiling (read {} bytes before \
                 stopping); the endpoint declared {}",
                body.len(),
                match response.content_length() {
                    Some(n) => format!("{n} bytes"),
                    None => "no length".to_string(),
                }
            )));
        }
        body.extend_from_slice(&chunk);
    }

    Ok(body)
}

/// Fetch a body as bytes under explicit limits.
pub async fn fetch_bytes_limited(url: &str, limits: &FetchLimits) -> Result<Vec<u8>, IoError> {
    let url = checked_url(url)?;
    send_bounded(client(limits)?.get(url), limits.max_bytes).await
}

/// Fetch a body as UTF-8 text under explicit limits.
pub async fn fetch_text_limited(url: &str, limits: &FetchLimits) -> Result<String, IoError> {
    let bytes = fetch_bytes_limited(url, limits).await?;
    String::from_utf8(bytes).map_err(|e| IoError::Parse(format!("response is not UTF-8: {e}")))
}

/// Fetch raw bytes from a URL under [`FetchLimits::default`].
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, IoError> {
    fetch_bytes_limited(url, &FetchLimits::default()).await
}

/// Fetch JSON data from a URL and parse into a [`DataBatch`].
pub async fn fetch_json(url: &str) -> Result<DataBatch, IoError> {
    let text = fetch_text_limited(url, &FetchLimits::default()).await?;
    crate::json_io::read_json_string(&text)
}

/// Fetch CSV data from a URL.
pub async fn fetch_csv(url: &str, has_header: bool) -> Result<DataBatch, IoError> {
    let text = fetch_text_limited(url, &FetchLimits::default()).await?;
    crate::csv_io::read_csv_string(&text, has_header)
}

/// Fetch NDJSON (newline-delimited JSON) from a URL.
pub async fn fetch_ndjson(url: &str) -> Result<DataBatch, IoError> {
    let text = fetch_text_limited(url, &FetchLimits::default()).await?;
    crate::json_io::read_ndjson_string(&text)
}

/// POST a batch as JSON and parse the response batch.
///
/// The request body is the caller's own data and is not capped; the *response*
/// is, under [`FetchLimits::default`].
pub async fn post_json(url: &str, batch: &DataBatch) -> Result<DataBatch, IoError> {
    let limits = FetchLimits::default();
    let url = checked_url(url)?;
    let json_str = crate::json_io::batch_to_json(batch)?;

    let request = client(&limits)?
        .post(url)
        .header("Content-Type", "application/json")
        .body(json_str);

    let body = send_bounded(request, limits.max_bytes).await?;
    let text = String::from_utf8(body)
        .map_err(|e| IoError::Parse(format!("response is not UTF-8: {e}")))?;
    crate::json_io::read_json_string(&text)
}

/// A configured HTTP endpoint: URL, method, headers, payload format, bounds.
///
/// # Reaching the synchronous protocol
///
/// `HttpSource` does **not** implement [`DataSource`]: `read_batch` is
/// synchronous and every fetch here is `async`, and blocking on a future inside
/// a `&mut self` method panics within a Tokio runtime. Acquire first, serve
/// second — [`fetch_source`](HttpSource::fetch_source) returns a
/// [`BatchSource`], which does implement the trait:
///
/// ```no_run
/// # async fn demo() -> Result<(), ix_io::error::IoError> {
/// use ix_io::http::{HttpSource, PayloadFormat};
/// use ix_io::protocol::{drain, DataSource};
///
/// let mut source = HttpSource::get("https://example.invalid/data.csv")
///     .with_format(PayloadFormat::Csv { has_header: true })
///     .fetch_source()
///     .await?;
///
/// let batch = drain(&mut source, 256, 10_000)?;
/// let matrix = batch.to_array2();
/// # let _ = matrix;
/// # Ok(())
/// # }
/// ```
pub struct HttpSource {
    /// Endpoint to fetch.
    pub url: String,
    /// Verb and, for POST, the request body.
    pub method: HttpMethod,
    /// Extra request headers, e.g. authorization.
    pub headers: Vec<(String, String)>,
    /// Advisory poll interval; [`poll`](HttpSource::poll) takes its own.
    pub poll_interval_secs: Option<u64>,
    /// How the response body is parsed into records.
    pub format: PayloadFormat,
    /// Egress bounds for every fetch this source makes.
    pub limits: FetchLimits,
}

/// HTTP verb, carrying the body for POST.
#[derive(Debug, Clone)]
pub enum HttpMethod {
    /// `GET`.
    Get,
    /// `POST` with a literal body.
    Post {
        /// Request body, sent as `application/json`.
        body: String,
    },
}

impl HttpSource {
    /// A `GET` source over JSON, with default bounds.
    pub fn get(url: &str) -> Self {
        Self {
            url: url.to_string(),
            method: HttpMethod::Get,
            headers: Vec::new(),
            poll_interval_secs: None,
            format: PayloadFormat::Json,
            limits: FetchLimits::default(),
        }
    }

    /// Add a request header.
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.push((key.to_string(), value.to_string()));
        self
    }

    /// Set how the response body is parsed.
    pub fn with_format(mut self, format: PayloadFormat) -> Self {
        self.format = format;
        self
    }

    /// Tighten (or loosen) the egress bounds.
    pub fn with_limits(mut self, limits: FetchLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Build the bounded request for this source.
    fn request(&self, client: &reqwest::Client) -> Result<reqwest::RequestBuilder, IoError> {
        let url = checked_url(&self.url)?;
        let mut request = match &self.method {
            HttpMethod::Get => client.get(url),
            HttpMethod::Post { body } => client
                .post(url)
                .header("Content-Type", "application/json")
                .body(body.clone()),
        };
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        Ok(request)
    }

    /// Parse a body according to [`format`](HttpSource::format).
    fn parse(&self, text: &str) -> Result<DataBatch, IoError> {
        match self.format {
            PayloadFormat::Json => crate::json_io::read_json_string(text),
            PayloadFormat::Ndjson => crate::json_io::read_ndjson_string(text),
            PayloadFormat::Csv { has_header } => crate::csv_io::read_csv_string(text, has_header),
        }
    }

    /// Fetch once, bounded, and parse into a batch.
    pub async fn fetch(&self) -> Result<DataBatch, IoError> {
        let client = client(&self.limits)?;
        let body = send_bounded(self.request(&client)?, self.limits.max_bytes).await?;
        let text = String::from_utf8(body)
            .map_err(|e| IoError::Parse(format!("response is not UTF-8: {e}")))?;
        self.parse(&text)
    }

    /// Fetch once and hand back a [`BatchSource`] over the result.
    pub async fn fetch_source(&self) -> Result<BatchSource, IoError> {
        Ok(BatchSource::new(self.fetch().await?))
    }

    /// Poll the endpoint on an interval, sending each parsed batch downstream.
    ///
    /// Each poll is a fully bounded [`fetch`](HttpSource::fetch); a failing
    /// poll is skipped and the loop continues. The task ends when the receiver
    /// is dropped.
    pub async fn poll(
        self,
        interval_secs: u64,
    ) -> Result<tokio::sync::mpsc::Receiver<DataBatch>, IoError> {
        // Validate the URL once, up front: a bad scheme should be an error the
        // caller sees now, not a silently-skipped poll every interval forever.
        checked_url(&self.url)?;

        let (tx, rx) = tokio::sync::mpsc::channel(10);

        tokio::spawn(async move {
            loop {
                if let Ok(batch) = self.fetch().await {
                    if tx.send(batch).await.is_err() {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_secs(interval_secs)).await;
            }
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::DataSource;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// What the throwaway test server does with the one connection it accepts.
    enum Behaviour {
        /// Send a response. `declare_len` controls the `Content-Length` header:
        /// `None` omits it, so the body is delimited by the close and the
        /// client cannot know the size in advance.
        Respond {
            status: &'static str,
            body: Vec<u8>,
            declare_len: Option<usize>,
        },
        /// Accept the connection and never answer.
        Hang,
    }

    /// Bind an ephemeral loopback port, serve one request, return its URL.
    async fn serve_once(behaviour: Behaviour) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            // Drain the request head; we do not care what it says.
            let mut scratch = [0u8; 4096];
            let _ = stream.read(&mut scratch).await;

            match behaviour {
                Behaviour::Hang => tokio::time::sleep(Duration::from_secs(60)).await,
                Behaviour::Respond {
                    status,
                    body,
                    declare_len,
                } => {
                    let mut head = format!("HTTP/1.1 {status}\r\nConnection: close\r\n");
                    if let Some(len) = declare_len {
                        head.push_str(&format!("Content-Length: {len}\r\n"));
                    }
                    head.push_str("\r\n");
                    let _ = stream.write_all(head.as_bytes()).await;
                    let _ = stream.write_all(&body).await;
                    let _ = stream.shutdown().await;
                }
            }
        });

        format!("http://{addr}/")
    }

    fn ok_body(body: &str) -> Behaviour {
        Behaviour::Respond {
            status: "200 OK",
            body: body.as_bytes().to_vec(),
            declare_len: Some(body.len()),
        }
    }

    #[test]
    fn non_http_schemes_are_refused_without_touching_the_network() {
        for url in [
            "file:///etc/passwd",
            "ftp://example.com/data.csv",
            "data:text/plain,hello",
            "javascript:alert(1)",
        ] {
            let err = checked_url(url).expect_err("{url} must be refused");
            assert!(
                err.to_string().contains("http and https only"),
                "unexpected error for {url}: {err}"
            );
        }
    }

    #[test]
    fn http_and_https_are_accepted() {
        assert!(checked_url("http://example.com/a.csv").is_ok());
        assert!(checked_url("https://example.com/a.csv").is_ok());
    }

    #[test]
    fn malformed_urls_are_refused() {
        assert!(checked_url("not a url").is_err());
        assert!(checked_url("").is_err());
    }

    /// Pins the two URL-parser behaviours [`checked_url`] relies on instead of
    /// re-checking itself, so a future parser change is a test failure rather
    /// than a silent hole.
    #[test]
    fn empty_host_is_rejected_by_the_parser_itself() {
        for url in ["http://", "https://"] {
            let err = checked_url(url).expect_err("an empty host must not be fetchable");
            assert!(err.to_string().contains("empty host"), "got: {err}");
        }
        // Extra slashes are normalised away for special schemes, so this is a
        // request for host `nohost` — not a host-less URL.
        assert_eq!(
            checked_url("http:///nohost").unwrap().host_str(),
            Some("nohost")
        );
    }

    #[tokio::test]
    async fn fetches_a_bounded_body() {
        let url = serve_once(ok_body(r#"[{"x":1,"y":2},{"x":3,"y":4}]"#)).await;
        let batch = HttpSource::get(&url).fetch().await.unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.to_array2().unwrap().dim(), (2, 2));
    }

    #[tokio::test]
    async fn fetch_source_feeds_the_synchronous_protocol() {
        let url = serve_once(ok_body("a,b\n1,2\n3,4\n")).await;
        let mut source = HttpSource::get(&url)
            .with_format(PayloadFormat::Csv { has_header: true })
            .fetch_source()
            .await
            .unwrap();

        assert!(source.has_more());
        let first = source.read_batch(1).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(
            first.column_names.as_deref(),
            Some(&["a".to_string(), "b".to_string()][..])
        );
        assert_eq!(source.remaining(), 1);
    }

    /// Guard 1: a declared `Content-Length` above the ceiling is refused before
    /// the body is read at all.
    #[tokio::test]
    async fn declared_oversize_body_is_refused_before_reading() {
        let body = "x".repeat(4096);
        let url = serve_once(Behaviour::Respond {
            status: "200 OK",
            body: body.into_bytes(),
            declare_len: Some(4096),
        })
        .await;

        let limits = FetchLimits::default().with_max_bytes(1024);
        let err = fetch_text_limited(&url, &limits)
            .await
            .expect_err("4096 bytes must not fit under a 1024-byte ceiling");
        assert!(
            err.to_string().contains("declares Content-Length 4096"),
            "unexpected error: {err}"
        );
    }

    /// Guard 2: with no declared length the cap still holds, because the
    /// running byte count — not the header — is what is checked.
    #[tokio::test]
    async fn undeclared_oversize_body_is_refused_while_streaming() {
        let url = serve_once(Behaviour::Respond {
            status: "200 OK",
            body: "y".repeat(64 * 1024).into_bytes(),
            declare_len: None,
        })
        .await;

        let limits = FetchLimits::default().with_max_bytes(1024);
        let err = fetch_text_limited(&url, &limits)
            .await
            .expect_err("an undeclared 64 KiB body must not fit under a 1024-byte ceiling");
        assert!(
            err.to_string().contains("exceeds the 1024-byte ceiling"),
            "unexpected error: {err}"
        );
    }

    /// A body at the ceiling still succeeds — the cap rejects excess, it does
    /// not reject everything, which is what makes the two tests above mean
    /// something.
    #[tokio::test]
    async fn a_body_within_the_ceiling_still_succeeds() {
        let url = serve_once(ok_body("hello")).await;
        let limits = FetchLimits::default().with_max_bytes(1024);
        assert_eq!(fetch_text_limited(&url, &limits).await.unwrap(), "hello");
    }

    #[tokio::test]
    async fn a_silent_server_trips_the_timeout_rather_than_hanging() {
        let url = serve_once(Behaviour::Hang).await;
        let limits = FetchLimits::default().with_timeout(Duration::from_millis(300));

        let err = fetch_text_limited(&url, &limits)
            .await
            .expect_err("a server that never answers must not hang the caller");
        match err {
            IoError::Http(e) => assert!(e.is_timeout(), "expected a timeout, got: {e}"),
            other => panic!("expected a transport timeout, got: {other}"),
        }
    }

    /// An error page must not be scraped into a batch of `NaN`s.
    #[tokio::test]
    async fn a_non_success_status_is_an_error_not_a_parsed_batch() {
        let url = serve_once(Behaviour::Respond {
            status: "404 Not Found",
            body: b"<html>nope</html>".to_vec(),
            declare_len: Some(17),
        })
        .await;

        let err = HttpSource::get(&url)
            .fetch()
            .await
            .expect_err("404 must not parse as data");
        match err {
            IoError::Http(e) => assert_eq!(e.status().map(|s| s.as_u16()), Some(404)),
            other => panic!("expected an HTTP status error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn poll_rejects_a_bad_scheme_up_front() {
        let err = HttpSource::get("file:///etc/passwd")
            .poll(1)
            .await
            .expect_err("poll must validate the URL before spawning");
        assert!(
            err.to_string().contains("http and https only"),
            "got: {err}"
        );
    }
}
