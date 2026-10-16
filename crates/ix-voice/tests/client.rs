//! The client against a fake service: one thread on a `TcpListener` answers each request with a
//! canned reply and records what it was sent, so the tests check both sides of the contract.

use ix_voice::{Client, Error, Health, Said, SayOptions, Stopped, Voice};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::JoinHandle;

#[derive(Debug)]
struct Seen {
    method: String,
    path: String,
    body: String,
}

impl Seen {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("the request body is JSON")
    }
}

/// A fake service that answers the given replies in order, one per connection, then stops.
fn fake(replies: Vec<(u16, &'static str)>) -> (Client, JoinHandle<Vec<Seen>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let mut seen = Vec::new();
        for (status, body) in replies {
            let (stream, _) = listener.accept().unwrap();
            seen.push(answer(stream, status, body));
        }
        seen
    });
    (Client::new(url).unwrap(), handle)
}

fn answer(stream: TcpStream, status: u16, body: &str) -> Seen {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let mut length = 0;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        if header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap();
            }
        }
    }
    let mut sent = vec![0; length];
    reader.read_exact(&mut sent).unwrap();
    let mut stream = stream;
    write!(
        stream,
        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    Seen {
        method,
        path,
        body: String::from_utf8(sent).unwrap(),
    }
}

const HEALTH: &str = r#"{"ok":true,"version":"0.1.0","engine":"piper-cli","voice":"fr_FR-siwis-medium","sample_rate":22050}"#;

#[test]
fn health_reads_the_status_route() {
    let (client, server) = fake(vec![(200, HEALTH)]);
    let health = client.health().unwrap();
    assert_eq!(
        health,
        Health {
            ok: true,
            version: "0.1.0".into(),
            engine: "piper-cli".into(),
            voice: "fr_FR-siwis-medium".into(),
            sample_rate: 22050,
        }
    );
    let seen = server.join().unwrap();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].path.as_str()),
        ("GET", "/health")
    );
}

#[test]
fn voices_lists_each_voice_with_its_licence() {
    let (client, server) = fake(vec![(
        200,
        r#"[{"id":"fr_FR-siwis-medium","language":"fr_FR","quality":"medium","sample_rate":22050,"license":"CC-BY 4.0"},
            {"id":"en_US-ljspeech-medium","language":"en_US","quality":"medium","sample_rate":22050,"license":"public domain","extra":1}]"#,
    )]);
    let voices = client.voices().unwrap();
    assert_eq!(voices.len(), 2);
    assert_eq!(
        voices[0],
        Voice {
            id: "fr_FR-siwis-medium".into(),
            language: "fr_FR".into(),
            quality: "medium".into(),
            sample_rate: 22050,
            license: "CC-BY 4.0".into(),
        }
    );
    // A field the contract does not name is ignored, so the service can add fields.
    assert_eq!(voices[1].license, "public domain");
    let seen = server.join().unwrap();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].path.as_str()),
        ("GET", "/v1/voices")
    );
}

#[test]
fn say_sends_only_the_text_and_wait_by_default() {
    let (client, server) = fake(vec![(200, r#"{"id":"u-1","sentences":2}"#)]);
    let said = client
        .say("Bonjour. Le build est vert.", &SayOptions::default())
        .unwrap();
    assert_eq!(
        said,
        Said {
            id: "u-1".into(),
            sentences: 2,
            played_ms: None,
        }
    );
    let seen = server.join().unwrap();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].path.as_str()),
        ("POST", "/v1/say")
    );
    assert_eq!(
        seen[0].json(),
        json!({ "text": "Bonjour. Le build est vert.", "wait": false })
    );
}

#[test]
fn say_sends_the_contract_s_field_names_for_every_option() {
    let (client, server) = fake(vec![(
        200,
        r#"{"id":"u-2","sentences":1,"played_ms":1830}"#,
    )]);
    let options = SayOptions {
        voice: Some("en_US-ljspeech-medium".into()),
        rate: Some(1.5),
        wait: true,
    };
    let said = client.say("Done.", &options).unwrap();
    assert_eq!(said.played_ms, Some(1830));
    let seen = server.join().unwrap();
    assert_eq!(
        seen[0].json(),
        json!({ "text": "Done.", "voice": "en_US-ljspeech-medium", "rate": 1.5, "wait": true })
    );
}

#[test]
fn stop_posts_to_the_stop_route() {
    let (client, server) = fake(vec![(200, r#"{"stopped":3}"#)]);
    assert_eq!(client.stop().unwrap(), Stopped { stopped: 3 });
    let seen = server.join().unwrap();
    assert_eq!(
        (seen[0].method.as_str(), seen[0].path.as_str()),
        ("POST", "/v1/stop")
    );
}

#[test]
fn the_service_s_error_text_reaches_the_caller() {
    let (client, server) = fake(vec![
        (400, r#"{"error":"text is longer than 20000 characters"}"#),
        (500, "engine crashed\n"),
        (503, ""),
    ]);
    let refused = client.say("x", &SayOptions::default()).unwrap_err();
    assert_eq!(
        refused,
        Error::Service {
            status: 400,
            message: "text is longer than 20000 characters".into(),
        }
    );
    assert!(refused.to_string().contains("20000"), "{refused}");
    // A body that is not the contract's error object is passed on as it is.
    assert_eq!(
        client.stop().unwrap_err(),
        Error::Service {
            status: 500,
            message: "engine crashed".into(),
        }
    );
    // An empty body falls back to the status.
    match client.health().unwrap_err() {
        Error::Service {
            status: 503,
            message,
        } => assert!(message.contains("503"), "{message}"),
        other => panic!("expected a 503, got {other:?}"),
    }
    server.join().unwrap();
}

#[test]
fn an_answer_that_breaks_the_contract_is_an_error_not_a_default() {
    let (client, server) = fake(vec![(200, r#"{"ok":true}"#)]);
    match client.health().unwrap_err() {
        Error::Transport(why) => assert!(why.contains("does not match the contract"), "{why}"),
        other => panic!("expected a contract error, got {other:?}"),
    }
    server.join().unwrap();
}

#[test]
fn a_closed_port_says_the_service_is_not_running() {
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let url = format!("http://127.0.0.1:{port}");
    let client = Client::new(url.clone()).unwrap();
    let e = client.health().unwrap_err();
    assert_eq!(e, Error::NotRunning { url });
    assert!(e.to_string().contains("piper-voice serve"), "{e}");
}
