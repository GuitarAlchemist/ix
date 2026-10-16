//! `ix-voice`: the IX side of the local voice service (ADR-0013).
//!
//! The service, `piper-voice`, lives outside this workspace. It speaks text aloud with Piper,
//! sentence by sentence: it plays one sentence while it synthesizes the next. This crate is its
//! client, one method per route of the HTTP contract v0.1:
//!
//! | Method | Route |
//! |---|---|
//! | [`Client::health`] | `GET /health` |
//! | [`Client::voices`] | `GET /v1/voices` |
//! | [`Client::say`] | `POST /v1/say` |
//! | [`Client::stop`] | `POST /v1/stop` |
//!
//! The service listens on [`DEFAULT_URL`]; `IX_VOICE_URL` points elsewhere.
//!
//! ```no_run
//! let voice = ix_voice::Client::from_env()?;
//! voice.say("Bonjour. Le build est vert.", &ix_voice::SayOptions::default())?;
//! # Ok::<(), ix_voice::Error>(())
//! ```
//!
//! The client is blocking. An async caller runs it on a blocking thread, since reqwest's
//! blocking client refuses to run inside an async runtime.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Where the service listens unless `IX_VOICE_URL` says otherwise.
pub const DEFAULT_URL: &str = "http://127.0.0.1:59126";

/// The environment variable that overrides [`DEFAULT_URL`].
pub const URL_ENV: &str = "IX_VOICE_URL";

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const SHORT_TIMEOUT: Duration = Duration::from_secs(5);

/// `GET /health`: the service is up, and what it speaks with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub ok: bool,
    pub version: String,
    pub engine: String,
    /// The default voice.
    pub voice: String,
    pub sample_rate: u32,
}

/// One voice installed in the service, from `GET /v1/voices`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    pub language: String,
    pub quality: String,
    pub sample_rate: u32,
    /// The voice's own licence, which the person must respect (CC-BY 4.0 asks for attribution).
    pub license: String,
}

/// What `POST /v1/say` may set besides the text. The default is the service's voice and pace,
/// answered as soon as the text is queued.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SayOptions {
    /// A voice id from [`Client::voices`]; `None` is the service's default voice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// A speed factor: 1.0 is the voice's own pace, above 1 faster.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<f64>,
    /// Answer only once the utterance has been played.
    pub wait: bool,
}

/// The answer to `POST /v1/say`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Said {
    /// The utterance's id in the service's queue.
    pub id: String,
    /// How many sentences the text was split into.
    pub sentences: u32,
    /// How long the playback took; only an answer to a `wait` request has it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub played_ms: Option<u64>,
}

/// The answer to `POST /v1/stop`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stopped {
    /// How many utterances were cut or dropped from the queue.
    pub stopped: u32,
}

/// Why a call failed.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Nothing listens at the URL.
    NotRunning { url: String },
    /// The service answered with an error status. `message` is its `{"error"}` text.
    Service { status: u16, message: String },
    /// Anything else: a timeout, or an answer that does not match the contract.
    Transport(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotRunning { url } => write!(
                f,
                "the voice service is not running at {url}: start it with `piper-voice serve`, \
                 or point {URL_ENV} at it"
            ),
            Error::Service { status, message } => {
                write!(f, "the voice service refused ({status}): {message}")
            }
            Error::Transport(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Serialize)]
struct SayBody<'a> {
    text: &'a str,
    #[serde(flatten)]
    options: &'a SayOptions,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: String,
}

/// A connection to the voice service.
#[derive(Debug, Clone)]
pub struct Client {
    base: String,
    http: reqwest::blocking::Client,
}

impl Client {
    /// A client of the service at `base_url`, such as `http://127.0.0.1:59126`.
    pub fn new(base_url: impl Into<String>) -> Result<Self, Error> {
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            // The service is local: a system proxy would only get in the way.
            .no_proxy()
            .build()
            .map_err(|e| Error::Transport(format!("cannot build the HTTP client: {e}")))?;
        Ok(Client {
            base: base_url.into().trim_end_matches('/').to_string(),
            http,
        })
    }

    /// A client of the service at `IX_VOICE_URL`, else at [`DEFAULT_URL`].
    pub fn from_env() -> Result<Self, Error> {
        Client::new(base_url_from(std::env::var(URL_ENV).ok()))
    }

    /// The service's URL, without a trailing slash.
    pub fn url(&self) -> &str {
        &self.base
    }

    /// `GET /health`.
    pub fn health(&self) -> Result<Health, Error> {
        self.answer(self.http.get(self.route("/health")).timeout(SHORT_TIMEOUT))
    }

    /// `GET /v1/voices`.
    pub fn voices(&self) -> Result<Vec<Voice>, Error> {
        self.answer(
            self.http
                .get(self.route("/v1/voices"))
                .timeout(SHORT_TIMEOUT),
        )
    }

    /// `POST /v1/say`: queue `text` for playback. With `options.wait`, the call returns once the
    /// text has been played, so its timeout grows with the text ([`say_timeout`]).
    pub fn say(&self, text: &str, options: &SayOptions) -> Result<Said, Error> {
        let body = SayBody { text, options };
        self.answer(
            self.http
                .post(self.route("/v1/say"))
                .json(&body)
                .timeout(say_timeout(text, options)),
        )
    }

    /// `POST /v1/stop`: cut the current sentence and empty the queue.
    pub fn stop(&self) -> Result<Stopped, Error> {
        self.answer(
            self.http
                .post(self.route("/v1/stop"))
                .timeout(SHORT_TIMEOUT),
        )
    }

    fn route(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn answer<T: DeserializeOwned>(
        &self,
        request: reqwest::blocking::RequestBuilder,
    ) -> Result<T, Error> {
        let response = request.send().map_err(|e| self.failed(e))?;
        let status = response.status();
        let body = response.text().map_err(|e| self.failed(e))?;
        if !status.is_success() {
            let message = match serde_json::from_str::<ErrorBody>(&body) {
                Ok(b) => b.error,
                Err(_) if body.trim().is_empty() => status.to_string(),
                Err(_) => body.trim().to_string(),
            };
            return Err(Error::Service {
                status: status.as_u16(),
                message,
            });
        }
        serde_json::from_str(&body).map_err(|e| {
            let shown: String = body.chars().take(200).collect();
            Error::Transport(format!(
                "the voice service's answer does not match the contract ({e}): {shown}"
            ))
        })
    }

    fn failed(&self, e: reqwest::Error) -> Error {
        if e.is_connect() {
            Error::NotRunning {
                url: self.base.clone(),
            }
        } else {
            Error::Transport(format!("the voice service at {}: {e}", self.base))
        }
    }
}

/// How long `say` waits for an answer. Queuing is quick, so 10 s. A `wait` request adds 0.2 s a
/// character at the voice's own pace, about two and a half times as long as reading it aloud
/// takes, scaled by the rate.
pub fn say_timeout(text: &str, options: &SayOptions) -> Duration {
    let mut seconds = 10.0;
    if options.wait {
        let rate = options
            .rate
            .filter(|r| r.is_finite() && *r > 0.0)
            .unwrap_or(1.0)
            .clamp(0.25, 4.0);
        seconds += text.chars().count() as f64 * 0.2 / rate;
    }
    Duration::from_secs_f64(seconds)
}

fn base_url_from(value: Option<String>) -> String {
    value
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_URL.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_url_comes_from_the_environment_value_else_the_default() {
        assert_eq!(base_url_from(None), DEFAULT_URL);
        assert_eq!(base_url_from(Some("  ".into())), DEFAULT_URL);
        assert_eq!(
            base_url_from(Some(" http://127.0.0.1:6000 ".into())),
            "http://127.0.0.1:6000"
        );
    }

    #[test]
    fn a_trailing_slash_is_dropped() {
        let c = Client::new("http://127.0.0.1:6000/").unwrap();
        assert_eq!(c.url(), "http://127.0.0.1:6000");
        assert_eq!(c.route("/health"), "http://127.0.0.1:6000/health");
    }

    #[test]
    fn the_say_timeout_grows_with_the_text_only_when_waiting() {
        let text = "a".repeat(1000);
        let queued = SayOptions::default();
        let waiting = SayOptions {
            wait: true,
            ..SayOptions::default()
        };
        let slow = SayOptions {
            wait: true,
            rate: Some(0.5),
            ..SayOptions::default()
        };
        assert_eq!(say_timeout(&text, &queued), Duration::from_secs(10));
        assert_eq!(say_timeout(&text, &waiting), Duration::from_secs(210));
        assert_eq!(say_timeout(&text, &slow), Duration::from_secs(410));
        // A rate the service will refuse still gets a finite timeout.
        let odd = SayOptions {
            wait: true,
            rate: Some(f64::NAN),
            ..SayOptions::default()
        };
        assert_eq!(say_timeout(&text, &odd), Duration::from_secs(210));
    }

    #[test]
    fn the_not_running_message_says_how_to_start_the_service() {
        let e = Error::NotRunning {
            url: DEFAULT_URL.into(),
        };
        let shown = e.to_string();
        assert!(shown.contains("piper-voice serve"), "{shown}");
        assert!(shown.contains(URL_ENV), "{shown}");
    }
}
