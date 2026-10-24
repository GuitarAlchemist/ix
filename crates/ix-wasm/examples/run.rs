//! Runs one stage on stdin and prints what it wrote, then a record of the run on stderr.
//!
//! ```text
//! cargo run -p ix-wasm --example run -- stage.wasm [--fuel N] < input.json
//! ```

use ix_wasm::{Limits, Program};
use std::io::{Read, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: run <stage.wasm|.wat> [--fuel N] < input");
        std::process::exit(2);
    };
    let mut limits = Limits::default();
    if let Some(i) = args.iter().position(|a| a == "--fuel") {
        limits.fuel = args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .expect("--fuel needs a number");
    }
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut stdin = Vec::new();
    std::io::stdin().read_to_end(&mut stdin).expect("stdin");

    let loaded = Instant::now();
    let program = match Program::new(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(3);
        }
    };
    let load_ms = loaded.elapsed().as_secs_f64() * 1e3;
    let ran = Instant::now();
    let out = program.run(&stdin, &limits);
    let run_ms = ran.elapsed().as_secs_f64() * 1e3;
    match out {
        Ok(o) => {
            std::io::stdout().write_all(&o.stdout).expect("stdout");
            std::io::stderr().write_all(&o.stderr).expect("stderr");
            eprintln!(
                "exit={} load={load_ms:.1}ms run={run_ms:.1}ms fuel={} imports={} closed={:?}",
                o.exit,
                o.fuel_used,
                program.imports().len(),
                program.closed()
            );
            std::process::exit(o.exit.clamp(0, 255));
        }
        Err(e) => {
            eprintln!("{e} (load={load_ms:.1}ms run={run_ms:.1}ms)");
            std::process::exit(4);
        }
    }
}
