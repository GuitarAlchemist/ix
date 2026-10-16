//! What a stage can and cannot do, on small WAT modules (no toolchain needed).

use ix_wasm::{Error, Limits, Program};

const ECHO: &str = r#"
(module
  (import "wasi_snapshot_preview1" "fd_read" (func $read (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const 4096))
    (drop (call $read (i32.const 0) (i32.const 0) (i32.const 1) (i32.const 16)))
    (i32.store (i32.const 8) (i32.const 64))
    (i32.store (i32.const 12) (i32.load (i32.const 16)))
    (drop (call $write (i32.const 1) (i32.const 8) (i32.const 1) (i32.const 20)))))
"#;

/// Exits with the errno `fd_write` answers for an iovec array at `$iovs`.
fn write_errno(iovs: i32) -> String {
    format!(
        r#"
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (call $exit (call $write (i32.const 1) (i32.const {iovs}) (i32.const 1) (i32.const 20)))))
"#
    )
}

/// Writes eight random bytes, then two clock readings.
const ENTROPY: &str = r#"
(module
  (import "wasi_snapshot_preview1" "random_get" (func $random (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (param i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (drop (call $random (i32.const 64) (i32.const 8)))
    (drop (call $clock (i32.const 0) (i64.const 1) (i32.const 72)))
    (drop (call $clock (i32.const 0) (i64.const 1) (i32.const 80)))
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const 24))
    (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 20)))))
"#;

fn limits() -> Limits {
    Limits::default()
}

#[test]
fn a_stage_reads_stdin_and_writes_stdout() {
    let p = Program::new(ECHO.as_bytes()).unwrap();
    let out = p.run(br#"{"data":[3,1,4]}"#, &limits()).unwrap();
    assert_eq!(out.exit, 0);
    assert_eq!(out.stdout, br#"{"data":[3,1,4]}"#);
    assert!(out.fuel_used > 0);
}

#[test]
fn two_runs_on_one_input_spend_the_same_fuel_and_say_the_same() {
    let p = Program::new(ENTROPY.as_bytes()).unwrap();
    let a = p.run(b"", &limits()).unwrap();
    let b = p.run(b"", &limits()).unwrap();
    assert_eq!(a, b);
    // The clock starts at 1 ns and moves 1 µs per reading.
    let t1 = u64::from_le_bytes(a.stdout[8..16].try_into().unwrap());
    let t2 = u64::from_le_bytes(a.stdout[16..24].try_into().unwrap());
    assert_eq!((t1, t2), (1_001, 2_001));
    let other = p
        .run(
            b"",
            &Limits {
                seed: 7,
                ..limits()
            },
        )
        .unwrap();
    assert_ne!(
        other.stdout[..8],
        a.stdout[..8],
        "the seed moves the random source"
    );
}

#[test]
fn an_endless_loop_stops_on_fuel() {
    let p = Program::new(br#"(module (func (export "_start") (loop $l (br $l))))"#).unwrap();
    let e = p
        .run(
            b"",
            &Limits {
                fuel: 1_000_000,
                ..limits()
            },
        )
        .unwrap_err();
    assert!(
        matches!(e, Error::OutOfFuel { fuel_used } if fuel_used > 0 && fuel_used <= 1_000_000),
        "{e}"
    );
}

#[test]
fn a_module_that_asks_for_files_is_refused_before_it_runs() {
    let wat = r#"
(module
  (import "wasi_snapshot_preview1" "path_open"
    (func (param i32 i32 i32 i32 i32 i64 i64 i32 i32) (result i32)))
  (func (export "_start")))
"#;
    let e = Program::new(wat.as_bytes()).err().unwrap();
    assert!(
        matches!(&e, Error::Refused(m) if m.contains("path_open")),
        "{e}"
    );
    let e = Program::new(br#"(module (import "env" "socket" (func)) (func (export "_start")))"#)
        .err()
        .unwrap();
    assert!(
        matches!(&e, Error::Refused(m) if m.contains("env.socket")),
        "{e}"
    );
}

#[test]
fn a_module_without_start_is_refused() {
    let e = Program::new(br#"(module (memory (export "memory") 1))"#)
        .err()
        .unwrap();
    assert!(
        matches!(&e, Error::Refused(m) if m.contains("_start")),
        "{e}"
    );
}

#[test]
fn bad_bytes_are_not_a_module() {
    assert!(matches!(
        Program::new(b"\0asm\x01\0\0\0garbage"),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn proc_exit_gives_the_exit_status() {
    let wat = r#"
(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (func (export "_start") (call $exit (i32.const 7))))
"#;
    let out = Program::new(wat.as_bytes())
        .unwrap()
        .run(b"", &limits())
        .unwrap();
    assert_eq!(out.exit, 7);
}

#[test]
fn pointers_out_of_range_answer_einval_and_never_reach_the_host() {
    const EINVAL: i32 = 28;
    // Past the end of the one page, and a negative pointer (a huge unsigned one).
    for iovs in [65_532, -8] {
        let p = Program::new(write_errno(iovs).as_bytes()).unwrap();
        assert_eq!(p.run(b"", &limits()).unwrap().exit, EINVAL, "iovs {iovs}");
    }
}

#[test]
fn output_past_the_limit_stops_the_run() {
    // Writes the same 4 KiB forever.
    let wat = r#"
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (i32.store (i32.const 0) (i32.const 64))
    (i32.store (i32.const 4) (i32.const 4096))
    (loop $l
      (drop (call $write (i32.const 1) (i32.const 0) (i32.const 1) (i32.const 20)))
      (br $l))))
"#;
    let e = Program::new(wat.as_bytes())
        .unwrap()
        .run(
            b"",
            &Limits {
                max_output: 64 << 10,
                ..limits()
            },
        )
        .unwrap_err();
    assert_eq!(e, Error::OutputTooLarge { limit: 64 << 10 });
}

#[test]
fn memory_past_the_limit_is_not_given() {
    // Asks for 100 more pages (6.4 MB) and exits with what memory.grow answered.
    let wat = r#"
(module
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start") (call $exit (memory.grow (i32.const 100)))))
"#;
    let p = Program::new(wat.as_bytes()).unwrap();
    let small = Limits {
        max_memory: 1 << 20,
        ..limits()
    };
    assert_eq!(p.run(b"", &small).unwrap().exit, -1, "refused under 1 MB");
    assert_eq!(p.run(b"", &limits()).unwrap().exit, 1, "given under 256 MB");
}

#[test]
fn only_stdout_and_stderr_can_be_written() {
    const EBADF: i32 = 8;
    let wat = r#"
(module
  (import "wasi_snapshot_preview1" "fd_write" (func $write (param i32 i32 i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start")
    (call $exit (call $write (i32.const 3) (i32.const 0) (i32.const 0) (i32.const 20)))))
"#;
    let out = Program::new(wat.as_bytes())
        .unwrap()
        .run(b"", &limits())
        .unwrap();
    assert_eq!(out.exit, EBADF);
}
