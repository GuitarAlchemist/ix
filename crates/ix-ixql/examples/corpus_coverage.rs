//! Parse-coverage probe over a directory of `.ixql` files.
//!
//! ```text
//! cargo run -p ix-ixql --example corpus_coverage -- ../Demerzel/pipelines
//! ```
//!
//! Prints one line per file and a summary grouped by *blocker* — the class of
//! the first error — because that is the number worth steering by. "Files that
//! parse" moves in steps of one and stays at its floor until a file's *last*
//! missing feature lands, so it reads as no progress right up until it jumps.
//! The blocker histogram shows which single construct is costing the most
//! files, which is what tells you what to build next.
//!
//! One caveat the histogram cannot hide: a lex error aborts the whole file
//! before parsing starts, so a file whose lexing is fixed can report a *lower*
//! line number afterwards. Line numbers are comparable within a phase, not
//! across one. The phase is shown per file for exactly that reason.

use std::path::PathBuf;

fn main() {
    let dir: PathBuf = match std::env::args().nth(1) {
        Some(d) => d.into(),
        None => {
            eprintln!("usage: corpus_coverage <dir-of-ixql-files>");
            std::process::exit(2);
        }
    };

    let mut files: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "ixql"))
            .collect(),
        Err(e) => {
            eprintln!("cannot read {}: {e}", dir.display());
            std::process::exit(2);
        }
    };
    files.sort();

    let mut parsed = 0usize;
    let mut blockers: std::collections::BTreeMap<String, usize> = Default::default();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                println!("SKIP  {name}  :: unreadable ({e})");
                continue;
            }
        };
        match ix_ixql::parse_program(&src) {
            Ok(_) => {
                parsed += 1;
                println!("OK    {name}");
            }
            Err(e) => {
                // `Lex` aborts before any parsing; `Syntax` means the file
                // tokenized cleanly and got as far as the grammar.
                let phase = match &e {
                    ix_ixql::ParseError::Lex(_) => "lex  ",
                    ix_ixql::ParseError::Syntax { .. } => "parse",
                };
                println!("FAIL  [{phase}] {name}  :: {e}");
                *blockers.entry(classify(&e.to_string())).or_default() += 1;
            }
        }
    }

    println!("\n== {parsed} parse / {} fail / {} total ==", files.len() - parsed, files.len());
    if !blockers.is_empty() {
        let mut ranked: Vec<_> = blockers.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("\nblockers, most files first:");
        for (blocker, count) in ranked {
            println!("  {count:>2}  {blocker}");
        }
    }
}

/// Collapse an error message to the construct behind it, so counts group.
fn classify(message: &str) -> String {
    let tail = message.split_once(": ").map(|(_, t)| t).unwrap_or(message);
    if let Some(rest) = tail.strip_prefix("unexpected character ") {
        return format!("unsupported character {rest}");
    }
    if tail.starts_with("a `→` step must be a call") {
        return "non-call pipe step (`→ {…}`, `→ when …`, `→ name`)".to_string();
    }
    tail.split(", found").next().unwrap_or(tail).to_string()
}
