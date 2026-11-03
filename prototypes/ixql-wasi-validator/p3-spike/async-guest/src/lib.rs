//! SPIKE: what WASI 0.3 async actually does, measured from inside one component.
//!
//! `wasmtime run <component> <mode>`, where mode is one of
//! `concurrent | sequential | blocking | stream | stdin | future-intra | all` (default `all`,
//! which runs the first four).
//!
//! - `concurrent`: two `async func wait-for` calls joined in one task. Concurrent means about
//!   max(300, 100) ms in total, with the short one finishing first.
//! - `sequential`: the same two calls awaited one after the other, about 300 + 100 ms.
//! - `blocking`: the 300 ms wait done with `std::thread::sleep`. On `wasm32-wasip2` std lowers
//!   that to a *synchronous* WASI 0.2 poll, which blocks the whole instance, so the 100 ms async
//!   wait cannot even start until it returns.
//! - `stream`: a guest-created `stream<u8>`. The producer writes 3 chunks 100 ms apart and the
//!   consumer logs when each one arrives.
//! - `stdin`: the host-backed pair from `wasi:cli/stdin@0.3.0#read-via-stream`, a `stream<u8>`
//!   plus a `future<result<_, error-code>>` that resolves at end of input. A 100 ms timer runs
//!   concurrently. Feed it slowly, e.g. `(echo a; sleep 0.3; echo b) | wasmtime run … stdin`.
//! - `future-intra`: a guest-created `future<result<_, error-code>>`, written by one task and
//!   read by another in the same instance. Wasmtime 50.0.1 refuses this (see README).
//!
//! The log of each mode is written through `wasi:cli/stdout@0.3.0#write-via-stream` as soon as
//! that mode ends, so a mode that traps does not hide the ones before it.

use std::cell::RefCell;
use std::time::Duration;

use wasip3::cli::types::ErrorCode;
use wasip3::clocks::monotonic_clock::{now, wait_for};
use wasip3::wit_bindgen::StreamResult;

const MS: u64 = 1_000_000;

wasip3::cli::command::export!(Spike);

struct Log {
    t0: u64,
    lines: RefCell<Vec<String>>,
}

impl Log {
    fn new(mode: &str) -> Self {
        Self {
            t0: now(),
            lines: RefCell::new(vec![format!("== {mode}")]),
        }
    }
    fn at(&self, what: impl AsRef<str>) {
        let ms = (now() - self.t0) as f64 / 1e6;
        self.lines
            .borrow_mut()
            .push(format!("  {ms:7.1} ms  {}", what.as_ref()));
    }
}

async fn wait(log: &Log, name: &str, ms: u64) {
    log.at(format!("{name}: start async wait-for({ms} ms)"));
    wait_for(ms * MS).await;
    log.at(format!("{name}: done"));
}

async fn run_mode(mode: &str, log: &Log) {
    match mode {
        "concurrent" => {
            futures::join!(wait(log, "slow", 300), wait(log, "fast", 100));
        }
        "sequential" => {
            wait(log, "slow", 300).await;
            wait(log, "fast", 100).await;
        }
        "blocking" => {
            futures::join!(
                async {
                    log.at("slow: start std::thread::sleep(300 ms) [sync WASI 0.2 poll]");
                    std::thread::sleep(Duration::from_millis(300));
                    log.at("slow: done");
                },
                wait(log, "fast", 100)
            );
        }
        "stream" => {
            let (mut tx, mut rx) = wasip3::wit_stream::new::<u8>();
            futures::join!(
                async {
                    for i in 0..3 {
                        let rest = tx.write_all(format!("chunk{i}").into_bytes()).await;
                        log.at(format!(
                            "producer: wrote chunk{i} (unwritten {})",
                            rest.len()
                        ));
                        wait_for(100 * MS).await;
                    }
                    drop(tx);
                    log.at("producer: dropped writer");
                },
                async {
                    loop {
                        let (res, buf) = rx.read(Vec::with_capacity(64)).await;
                        match res {
                            StreamResult::Complete(n) => log.at(format!(
                                "consumer: read {n} bytes {:?}",
                                String::from_utf8_lossy(&buf)
                            )),
                            other => {
                                log.at(format!("consumer: {other:?}"));
                                break;
                            }
                        }
                    }
                }
            );
        }
        "stdin" => {
            let (mut rx, done) = wasip3::cli::stdin::read_via_stream();
            futures::join!(
                async {
                    loop {
                        let (res, buf) = rx.read(Vec::with_capacity(64)).await;
                        match res {
                            StreamResult::Complete(n) => log.at(format!(
                                "stdin stream: read {n} bytes {:?}",
                                String::from_utf8_lossy(&buf)
                            )),
                            other => {
                                log.at(format!("stdin stream: {other:?}"));
                                break;
                            }
                        }
                    }
                },
                async {
                    log.at("stdin future: awaiting completion");
                    let v = done.await;
                    log.at(format!("stdin future: resolved to {v:?}"));
                },
                wait(log, "timer", 100)
            );
        }
        "future-intra" => {
            let (tx, rx) = wasip3::wit_future::new::<Result<(), ErrorCode>>(|| Ok(()));
            futures::join!(
                async {
                    wait_for(150 * MS).await;
                    log.at("writer: future.write(Err(ErrorCode::Io))");
                    let _ = tx.write(Err(ErrorCode::Io)).await;
                    log.at("writer: write completed");
                },
                async {
                    log.at("reader: awaiting future");
                    let v = rx.await;
                    log.at(format!("reader: future resolved to {v:?}"));
                }
            );
        }
        other => log.at(format!("unknown mode {other}")),
    }
    log.at("total");
}

async fn emit(text: String) -> Result<(), ()> {
    let (mut tx, rx) = wasip3::wit_stream::new();
    let (written, ()) = futures::join!(
        async { wasip3::cli::stdout::write_via_stream(rx).await },
        async {
            tx.write_all(text.into_bytes()).await;
            drop(tx);
        }
    );
    written.map_err(|_| ())
}

struct Spike;

impl wasip3::exports::cli::run::Guest for Spike {
    async fn run() -> Result<(), ()> {
        let args = wasip3::cli::environment::get_arguments();
        let modes: Vec<&str> = match args.get(1).map(String::as_str).unwrap_or("all") {
            "all" => vec!["concurrent", "sequential", "blocking", "stream"],
            one => vec![one],
        };
        for mode in modes {
            let log = Log::new(mode);
            run_mode(mode, &log).await;
            let mut text = log.lines.into_inner().join("\n");
            text.push('\n');
            emit(text).await?;
        }
        Ok(())
    }
}
