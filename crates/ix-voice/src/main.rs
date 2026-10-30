//! `ix-voice`: speak through the local voice service, stop it, ask what it has.
//!
//! ```text
//! ix-voice say <text> [--voice V] [--rate R] [--wait]
//! ix-voice stop
//! ix-voice status
//! ix-voice voices
//! ```
//!
//! The service is `IX_VOICE_URL`, else `http://127.0.0.1:59126`. Each answer is printed as JSON.

use ix_voice::{Client, Error, SayOptions};
use serde::Serialize;
use std::process::ExitCode;

const USAGE: &str =
    "usage: ix-voice say <text> [--voice V] [--rate R] [--wait] | stop | status | voices";

/// The words after `say`: the text is every word that is not a flag, joined by spaces.
fn say_args(args: &[String]) -> Result<(String, SayOptions), String> {
    let mut words = Vec::new();
    let mut options = SayOptions::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--voice" => {
                let v = it.next().ok_or("--voice needs a voice id")?;
                options.voice = Some(v.clone());
            }
            "--rate" => {
                let r = it
                    .next()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|r| r.is_finite() && *r > 0.0)
                    .ok_or("--rate needs a number above 0")?;
                options.rate = Some(r);
            }
            "--wait" => options.wait = true,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            word => words.push(word),
        }
    }
    let text = words.join(" ");
    if text.trim().is_empty() {
        return Err("say needs a text".into());
    }
    Ok((text, options))
}

fn print<T: Serialize>(answer: Result<T, Error>) -> ExitCode {
    match answer {
        Ok(v) => {
            println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let client = match Client::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    match (args.first().map(String::as_str), args.len()) {
        (Some("say"), _) => match say_args(&args[1..]) {
            Ok((text, options)) => print(client.say(&text, &options)),
            Err(e) => {
                eprintln!("{e}\n{USAGE}");
                ExitCode::from(2)
            }
        },
        (Some("stop"), 1) => print(client.stop()),
        (Some("status"), 1) => print(client.health()),
        (Some("voices"), 1) => print(client.voices()),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn the_text_is_every_word_that_is_not_a_flag() {
        let (text, options) = say_args(&args(&[
            "Bonjour",
            "--voice",
            "fr_FR-siwis-medium",
            "tout",
            "le",
            "monde.",
            "--rate",
            "1.25",
            "--wait",
        ]))
        .unwrap();
        assert_eq!(text, "Bonjour tout le monde.");
        assert_eq!(
            options,
            SayOptions {
                voice: Some("fr_FR-siwis-medium".into()),
                rate: Some(1.25),
                wait: true,
            }
        );
    }

    #[test]
    fn bad_say_arguments_are_refused() {
        for (words, why) in [
            (&[][..], "say needs a text"),
            (&["   "][..], "say needs a text"),
            (&["hi", "--voice"][..], "--voice needs a voice id"),
            (
                &["hi", "--rate", "fast"][..],
                "--rate needs a number above 0",
            ),
            (&["hi", "--rate", "0"][..], "--rate needs a number above 0"),
            (&["hi", "--loud"][..], "unknown flag --loud"),
        ] {
            assert_eq!(say_args(&args(words)).unwrap_err(), why, "{words:?}");
        }
    }
}
