//! A Zig stage, built and run. Needs Zig 0.17 (`IX_ZIG`, else `zig` on `PATH`), so it is
//! ignored by default: `cargo test -p ix-wasm -- --ignored`.

use ix_wasm::build::{is_id, Toolchain};
use ix_wasm::{Limits, Program};
use std::time::{Duration, Instant};

/// A stage on raw WASI calls: std.Io would import most of WASI for nothing.
const STATS: &str = r#"const std = @import("std");
const wasi = std.os.wasi;

const In = struct { data: []f64 };

fn readAll(gpa: std.mem.Allocator) ![]u8 {
    var list: std.ArrayList(u8) = .empty;
    var chunk: [4096]u8 = undefined;
    while (true) {
        const iov = [_]wasi.iovec_t{.{ .base = &chunk, .len = chunk.len }};
        var n: usize = 0;
        if (wasi.fd_read(0, &iov, 1, &n) != .SUCCESS) return error.ReadFailed;
        if (n == 0) return list.items;
        try list.appendSlice(gpa, chunk[0..n]);
    }
}

fn writeAll(fd: wasi.fd_t, bytes: []const u8) void {
    var rest = bytes;
    while (rest.len > 0) {
        const iov = [_]wasi.ciovec_t{.{ .base = rest.ptr, .len = rest.len }};
        var n: usize = 0;
        if (wasi.fd_write(fd, &iov, 1, &n) != .SUCCESS or n == 0) return;
        rest = rest[n..];
    }
}

pub fn main() u8 {
    const gpa = std.heap.wasm_allocator;
    const raw = readAll(gpa) catch return 2;
    const x = std.json.parseFromSliceLeaky(In, gpa, raw, .{ .ignore_unknown_fields = true }) catch {
        writeAll(2, "bad input\n");
        return 2;
    };
    var sum: f64 = 0;
    for (x.data) |v| sum += v;
    var out: [128]u8 = undefined;
    const line = std.fmt.bufPrint(&out, "{{\"n\":{d},\"sum\":{d}}}\n", .{ x.data.len, sum }) catch return 1;
    writeAll(1, line);
    return 0;
}
"#;

fn zig() -> (tempfile::TempDir, Toolchain) {
    let cache = tempfile::tempdir().unwrap();
    let exe = std::env::var_os("IX_ZIG").unwrap_or_else(|| "zig".into());
    let build = Toolchain::zig(exe, cache.path());
    (cache, build)
}

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(300)
}

#[test]
#[ignore = "needs the Zig toolchain"]
fn a_zig_stage_is_built_once_and_runs_in_the_sandbox() {
    let (_cache, zig) = zig();
    let first = zig.build(STATS, deadline()).unwrap();
    assert!(
        is_id(&first.id) && first.id.starts_with("zig-"),
        "{first:?}"
    );
    let again = zig.build(STATS, deadline()).unwrap();
    assert!(again.cached);
    let p = Program::new(&std::fs::read(&first.path).unwrap()).unwrap();
    assert!(p.closed().is_empty(), "{:?}", p.closed());
    let out = p
        .run(br#"{"data":[3,1,4,1,5]}"#, &Limits::default())
        .unwrap();
    assert_eq!(out.exit, 0, "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().trim(),
        r#"{"n":5,"sum":14}"#
    );
    assert_eq!(p.run(b"[", &Limits::default()).unwrap().exit, 2);
}

#[test]
#[ignore = "needs the Zig toolchain"]
fn a_zig_stage_that_does_not_compile_answers_the_compiler() {
    let (_cache, zig) = zig();
    let e = zig
        .build("pub fn main() void { undefined_name(); }\n", deadline())
        .unwrap_err();
    assert!(e.contains("undefined_name"), "{e}");
}
