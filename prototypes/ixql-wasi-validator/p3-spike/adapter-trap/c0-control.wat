(module
  (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (param i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "args_sizes_get" (func $args (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start") (local $brk i32) (local $new i32)
    (local.set $brk (i32.mul (memory.size) (i32.const 65536)))
    ;; 1. a WASI call that goes through State::with: the adapter allocates its State now
    (drop (call $clock (i32.const 1) (i64.const 0) (i32.const 0)))
    ;; 2. nothing
    ;; 3. a second State::with call
    (drop (call $args (i32.const 16) (i32.const 20)))
    (call $exit (i32.const 0))))
