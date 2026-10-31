//! The ixrelay adapter (`ix_ctl::relay`) against a fake plane: a loopback `TcpListener` that
//! keeps each request line it reads and answers with whatever the test hands it, including a
//! stall, a cut and an answer too large to read.

use ix_ctl::relay::{self, Outcome, Reply, Request, MAX_REPLY_BYTES};
use ix_ctl::Ctl;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

/// A plane that answers every connection with `answer`, once it has read the request head.
/// The request lines it read are kept, so a test can see what was sent, and that nothing was.
fn fake<F>(answer: F) -> (Ctl, Arc<Mutex<Vec<String>>>)
where
    F: Fn(&mut TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { return };
            conn.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let mut first = String::new();
            r.read_line(&mut first).unwrap();
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
            }
            log.lock().unwrap().push(first.trim_end().to_string());
            answer(&mut conn);
        }
    });
    (Ctl::new(addr, TOKEN), seen)
}

fn answer_json(status: &'static str, body: String) -> impl Fn(&mut TcpStream) + Send + 'static {
    move |c: &mut TcpStream| {
        let _ = write!(
            c,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{body}",
            body.len()
        );
    }
}

fn status_body(contract: Option<&str>, name: &str) -> String {
    let mut v = json!({ "name": name, "epoch": "e-7", "generation": 3, "services": [] });
    if let Some(c) = contract {
        v["contract"] = json!(c);
    }
    v.to_string()
}

fn ping(id: &str, target: Option<&str>, deadline: Duration) -> Request {
    Request {
        id: id.into(),
        op: "ping".into(),
        target: target.map(str::to_string),
        deadline,
    }
}

#[test]
fn test_ping_round_trip_keeps_id_target_and_names_who_answered() {
    let (ctl, seen) = fake(answer_json("200 OK", status_body(Some("run/1"), "desk")));
    let reply = relay::request(&ctl, &ping("req-42", Some("desk"), Duration::from_secs(2)));
    assert_eq!(
        Reply {
            id: "req-42".into(),
            op: "ping".into(),
            target: Some("desk".into()),
            outcome: Outcome::Answered {
                name: "desk".into(),
                epoch: "e-7".into()
            },
        },
        reply
    );
    assert_eq!(
        vec!["GET /v1/status?peer=desk HTTP/1.1".to_string()],
        *seen.lock().unwrap()
    );
}

#[test]
fn test_ping_without_target_asks_the_plane_itself() {
    let (ctl, seen) = fake(answer_json("200 OK", status_body(Some("run/1"), "svc")));
    let reply = relay::request(&ctl, &ping("a", None, Duration::from_secs(2)));
    assert_eq!(None, reply.target);
    assert!(matches!(reply.outcome, Outcome::Answered { ref name, .. } if name == "svc"));
    assert_eq!(
        vec!["GET /v1/status HTTP/1.1".to_string()],
        *seen.lock().unwrap()
    );
}

#[test]
fn test_an_answer_off_contract_is_malformed() {
    for body in [
        "not json".to_string(),
        json!({ "contract": "run/1", "epoch": "e" }).to_string(),
        json!({ "contract": "run/1", "name": "", "epoch": "e" }).to_string(),
        json!({ "contract": "run/1", "name": "x", "epoch": 5 }).to_string(),
    ] {
        let (ctl, _) = fake(answer_json("200 OK", body.clone()));
        let reply = relay::request(&ctl, &ping("b", None, Duration::from_secs(2)));
        assert!(
            matches!(reply.outcome, Outcome::Malformed { .. }),
            "{body}: {:?}",
            reply.outcome
        );
    }
}

#[test]
fn test_another_contract_is_named_not_accepted() {
    let (ctl, _) = fake(answer_json("200 OK", status_body(Some("run/2"), "svc")));
    let reply = relay::request(&ctl, &ping("c", None, Duration::from_secs(2)));
    assert_eq!(
        Outcome::WrongContract {
            got: Some("run/2".into())
        },
        reply.outcome
    );
    let (ctl, _) = fake(answer_json("200 OK", status_body(None, "svc")));
    let reply = relay::request(&ctl, &ping("c", None, Duration::from_secs(2)));
    assert_eq!(Outcome::WrongContract { got: None }, reply.outcome);
}

#[test]
fn test_ops_without_a_backend_are_unsupported_and_send_nothing() {
    let (ctl, seen) = fake(answer_json("200 OK", status_body(Some("run/1"), "svc")));
    for op in ["deliver", "reply", "frobnicate"] {
        let reply = relay::request(
            &ctl,
            &Request {
                id: "d".into(),
                op: op.into(),
                target: Some("ix-winsvc-0c".into()),
                deadline: Duration::from_secs(2),
            },
        );
        assert!(
            matches!(reply.outcome, Outcome::Unsupported { .. }),
            "{op}: {:?}",
            reply.outcome
        );
    }
    std::thread::sleep(Duration::from_millis(50));
    assert!(seen.lock().unwrap().is_empty());
}

#[test]
fn test_an_inadmissible_request_sends_nothing() {
    let (ctl, seen) = fake(answer_json("200 OK", status_body(Some("run/1"), "svc")));
    let bad = [
        ping("", None, Duration::from_secs(1)),
        ping(&"x".repeat(201), None, Duration::from_secs(1)),
        ping("e", None, Duration::ZERO),
        ping("e", None, Duration::from_secs(31)),
        ping("e", Some(""), Duration::from_secs(1)),
    ];
    for req in &bad {
        let reply = relay::request(&ctl, req);
        assert!(
            matches!(reply.outcome, Outcome::Invalid { .. }),
            "{req:?}: {:?}",
            reply.outcome
        );
    }
    // A loopback address is the only one the token goes to.
    let far = Ctl::new("192.0.2.1:9", TOKEN);
    let reply = relay::request(&far, &ping("e", None, Duration::from_secs(1)));
    assert!(
        matches!(reply.outcome, Outcome::Invalid { .. }),
        "{:?}",
        reply.outcome
    );
    std::thread::sleep(Duration::from_millis(50));
    assert!(seen.lock().unwrap().is_empty());
}

#[test]
fn test_a_stalled_plane_is_indeterminate_within_the_deadline() {
    let (ctl, _) = fake(|_c: &mut TcpStream| std::thread::sleep(Duration::from_secs(3)));
    let started = Instant::now();
    let reply = relay::request(&ctl, &ping("f", None, Duration::from_millis(300)));
    assert!(
        matches!(reply.outcome, Outcome::Indeterminate { .. }),
        "{:?}",
        reply.outcome
    );
    assert!(started.elapsed() < Duration::from_millis(1500));
}

#[test]
fn test_a_lost_link_is_indeterminate_and_not_asked_again() {
    let (ctl, seen) = fake(|c: &mut TcpStream| {
        let _ = c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\n{\"name\"");
    });
    let reply = relay::request(&ctl, &ping("g", None, Duration::from_secs(2)));
    assert!(
        matches!(reply.outcome, Outcome::Indeterminate { .. }),
        "{:?}",
        reply.outcome
    );
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(1, seen.lock().unwrap().len());
}

#[test]
fn test_nothing_listening_is_unreachable() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    drop(listener);
    let reply = relay::request(
        &Ctl::new(addr, TOKEN),
        &ping("h", None, Duration::from_secs(2)),
    );
    assert_eq!(Outcome::Unreachable, reply.outcome);
}

#[test]
fn test_an_answer_too_large_is_refused_unread() {
    // Declared by length: refused from the head, the body never sent.
    let (ctl, _) = fake(|c: &mut TcpStream| {
        let _ = write!(
            c,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
            MAX_REPLY_BYTES + 1
        );
        std::thread::sleep(Duration::from_secs(1));
    });
    let started = Instant::now();
    let reply = relay::request(&ctl, &ping("i", None, Duration::from_secs(5)));
    assert_eq!(Outcome::TooLarge, reply.outcome);
    assert!(started.elapsed() < Duration::from_millis(900));
    // Streamed in chunks, and to the end of the stream: refused once past the bound.
    let chunk = "a".repeat(64 * 1024);
    let (ctl, _) = fake(move |c: &mut TcpStream| {
        let _ = c.write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n");
        for _ in 0..20 {
            if write!(c, "{:x}\r\n{chunk}\r\n", chunk.len()).is_err() {
                return;
            }
        }
        let _ = c.write_all(b"0\r\n\r\n");
    });
    let reply = relay::request(&ctl, &ping("j", None, Duration::from_secs(5)));
    assert_eq!(Outcome::TooLarge, reply.outcome);
    let big = "b".repeat(MAX_REPLY_BYTES + 10);
    let (ctl, _) = fake(move |c: &mut TcpStream| {
        let _ = c.write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n");
        let _ = c.write_all(big.as_bytes());
    });
    let reply = relay::request(&ctl, &ping("k", None, Duration::from_secs(5)));
    assert_eq!(Outcome::TooLarge, reply.outcome);
}

#[test]
fn test_a_refusal_is_kept_with_a_bounded_message() {
    let long = "z".repeat(5000);
    let (ctl, _) = fake(answer_json(
        "400 Bad Request",
        json!({ "error": format!("unknown peer {long}") }).to_string(),
    ));
    let reply = relay::request(&ctl, &ping("l", Some("nobody"), Duration::from_secs(2)));
    let Outcome::Refused { status, message } = reply.outcome else {
        panic!("{:?}", reply.outcome)
    };
    assert_eq!(400, status);
    assert!(message.starts_with("unknown peer"), "{message}");
    assert!(message.len() <= relay::MAX_TEXT_BYTES, "{}", message.len());
}

#[test]
fn test_the_reply_json_names_its_contract_and_outcome() {
    let answered = Reply {
        id: "m".into(),
        op: "ping".into(),
        target: Some("desk".into()),
        outcome: Outcome::Answered {
            name: "desk".into(),
            epoch: "e-7".into(),
        },
    };
    assert_eq!(
        json!({
            "contract": "ixrelay.adapter/0", "id": "m", "op": "ping", "target": "desk",
            "outcome": "answered", "name": "desk", "epoch": "e-7"
        }),
        answered.to_json()
    );
    let lost = Reply {
        id: "n".into(),
        op: "ping".into(),
        target: None,
        outcome: Outcome::Indeterminate {
            reason: "timed out".into(),
        },
    };
    assert_eq!(
        json!({
            "contract": "ixrelay.adapter/0", "id": "n", "op": "ping", "target": null,
            "outcome": "indeterminate", "reason": "timed out"
        }),
        lost.to_json()
    );
    let refused = Reply {
        id: "o".into(),
        op: "deliver".into(),
        target: None,
        outcome: Outcome::WrongContract { got: None },
    };
    assert_eq!(
        json!({
            "contract": "ixrelay.adapter/0", "id": "o", "op": "deliver", "target": null,
            "outcome": "wrong_contract", "got": null
        }),
        refused.to_json()
    );
}
