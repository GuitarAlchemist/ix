//! A Server-Sent Events feed of the bus, and the page that draws it.
//!
//! `GET /` answers the page. `GET /events` answers `text/event-stream`: the events the feed
//! kept (the last `history`), then each new one, as `id: <run>-<n>`, `event: <kind>` and
//! `data: <json>`. A browser that reconnects sends the last id it saw (`Last-Event-ID`) and
//! gets only what came after, so nothing is counted twice. Each feed has its own `<run>`: a
//! browser that reconnects to a new run at the same address sends an id of the old one, and
//! gets everything the new run kept. `GET /events?replay` answers the
//! kept events and closes: a run seen after the fact, or by a browser with no live
//! connection. A slow browser loses events instead of slowing anything down: the feed only
//! observes. It binds where it is told; give it a loopback address.

use crate::bus::Bus;
use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Events a browser may fall behind by before it loses some.
const CLIENT_QUEUE: usize = 4096;
const MAX_HEAD: usize = 8 << 10;

/// A name no other feed has had: the clock, and a count for two feeds started in the same tick.
fn run_tag() -> Arc<str> {
    static STARTED: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    format!(
        "{:x}",
        nanos.wrapping_add(STARTED.fetch_add(1, Ordering::Relaxed))
    )
    .into()
}

struct Hub {
    history: VecDeque<(u64, Arc<str>)>,
    run: Arc<str>,
    next_id: u64,
    keep: usize,
    clients: Vec<SyncSender<Arc<str>>>,
}

/// A running feed. [`Feed::close`] stops it; dropping it leaves it running until the process
/// ends.
pub struct Feed {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    clients: Arc<AtomicUsize>,
    threads: Vec<JoinHandle<()>>,
}

impl Feed {
    /// Serves `page` and the events of `kinds` (all kinds when empty) at `addr`.
    pub fn serve(
        addr: &str,
        page: &'static str,
        bus: &Bus,
        kinds: &[&str],
        history: usize,
    ) -> io::Result<Feed> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let clients = Arc::new(AtomicUsize::new(0));
        let run = run_tag();
        let hub = Arc::new(Mutex::new(Hub {
            history: VecDeque::new(),
            run: run.clone(),
            next_id: 1,
            keep: history,
            clients: Vec::new(),
        }));

        let events = bus.subscribe(kinds, 1 << 16);
        let pump = {
            let (hub, stop) = (hub.clone(), stop.clone());
            std::thread::spawn(move || loop {
                match events.recv_timeout(Duration::from_millis(100)) {
                    Ok(e) => {
                        let mut h = hub.lock().unwrap_or_else(|e| e.into_inner());
                        let id = h.next_id;
                        h.next_id += 1;
                        let frame: Arc<str> = format!(
                            "id: {}-{id}\nevent: {}\ndata: {}\n\n",
                            h.run,
                            e.kind,
                            e.json()
                        )
                        .into();
                        if h.keep > 0 {
                            if h.history.len() == h.keep {
                                h.history.pop_front();
                            }
                            h.history.push_back((id, frame.clone()));
                        }
                        h.clients.retain(|c| {
                            !matches!(
                                c.try_send(frame.clone()),
                                Err(TrySendError::Disconnected(_))
                            )
                        });
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            })
        };

        let accept = {
            let (hub, stop, clients) = (hub.clone(), stop.clone(), clients.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let (hub, stop, clients, run) =
                                (hub.clone(), stop.clone(), clients.clone(), run.clone());
                            std::thread::spawn(move || {
                                let _ = answer(stream, page, &run, &hub, &stop, &clients);
                            });
                        }
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(25));
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(25)),
                    }
                }
            })
        };
        Ok(Feed {
            addr,
            stop,
            clients,
            threads: vec![pump, accept],
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Browsers reading `/events` now.
    pub fn clients(&self) -> usize {
        self.clients.load(Ordering::Relaxed)
    }

    /// Stops accepting, ends every stream within a second, and waits for the feed's threads.
    pub fn close(self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads {
            let _ = t.join();
        }
    }
}

fn answer(
    mut stream: TcpStream,
    page: &str,
    run: &str,
    hub: &Mutex<Hub>,
    stop: &AtomicBool,
    clients: &AtomicUsize,
) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut head = Vec::new();
    let mut buf = [0u8; 1024];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut buf)?;
        if n == 0 || head.len() > MAX_HEAD {
            return Ok(());
        }
        head.extend_from_slice(&buf[..n]);
    }
    let head = String::from_utf8_lossy(&head);
    let mut first = head.lines().next().unwrap_or("").split(' ');
    let (method, target) = (first.next().unwrap_or(""), first.next().unwrap_or(""));
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let replay = query
        .split('&')
        .any(|q| q == "replay" || q.starts_with("replay="));
    // An id of another run, or of no run, counts as nothing seen.
    let last_seen: u64 = head
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("last-event-id"))
        .and_then(|(_, v)| v.trim().split_once('-'))
        .filter(|(r, _)| *r == run)
        .and_then(|(_, n)| n.parse().ok())
        .unwrap_or(0);
    if method != "GET" {
        return reply(
            &mut stream,
            "405 Method Not Allowed",
            "text/plain",
            "GET only\n",
        );
    }
    match path {
        "/" | "/index.html" => reply(&mut stream, "200 OK", "text/html; charset=utf-8", page),
        "/events" if replay => {
            let backlog: Vec<Arc<str>> = {
                let h = hub.lock().unwrap_or_else(|e| e.into_inner());
                h.history
                    .iter()
                    .filter(|(id, _)| *id > last_seen)
                    .map(|(_, f)| f.clone())
                    .collect()
            };
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
            )?;
            for frame in backlog {
                stream.write_all(frame.as_bytes())?;
            }
            stream.flush()
        }
        "/events" => {
            let (tx, rx) = sync_channel::<Arc<str>>(CLIENT_QUEUE);
            // Taken under the lock that registers the client: no event is both replayed and
            // sent live, and none falls between.
            let backlog: Vec<Arc<str>> = {
                let mut h = hub.lock().unwrap_or_else(|e| e.into_inner());
                h.clients.push(tx);
                h.history
                    .iter()
                    .filter(|(id, _)| *id > last_seen)
                    .map(|(_, f)| f.clone())
                    .collect()
            };
            clients.fetch_add(1, Ordering::Relaxed);
            let result = (|| -> io::Result<()> {
                stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n",
                )?;
                for frame in backlog {
                    stream.write_all(frame.as_bytes())?;
                }
                stream.flush()?;
                loop {
                    match rx.recv_timeout(Duration::from_secs(1)) {
                        Ok(frame) => stream.write_all(frame.as_bytes())?,
                        Err(RecvTimeoutError::Timeout) => stream.write_all(b": ping\n\n")?,
                        Err(RecvTimeoutError::Disconnected) => return Ok(()),
                    }
                    stream.flush()?;
                    if stop.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                }
            })();
            clients.fetch_sub(1, Ordering::Relaxed);
            result
        }
        _ => reply(&mut stream, "404 Not Found", "text/plain", "not here\n"),
    }
}

fn reply(stream: &mut TcpStream, status: &str, kind: &str, body: &str) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}
