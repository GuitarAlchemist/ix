(module
  (import "wasi_snapshot_preview1" "clock_time_get" (func $clock (param i32 i64 i32) (result i32)))
  (import "wasi_snapshot_preview1" "args_sizes_get" (func $args (param i32 i32) (result i32)))
  (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
  (memory (export "memory") 1)
  (func (export "_start") (local $brk i32) (local $p i32) (local $bad i32)
    (local.set $brk (i32.mul (memory.size) (i32.const 65536)))
    (drop (call $clock (i32.const 1) (i64.const 0) (i32.const 0)))
    ;; Go counts [initial data end, brk) as heap, which includes the adapter's stack page:
    ;; put "heap objects" (a 0xAB pattern) in the top 256 bytes of it.
    (memory.fill (i32.sub (local.get $brk) (i32.const 256)) (i32.const 0xAB) (i32.const 256))
    (drop (call $args (i32.const 16) (i32.const 20)))
    ;; count bytes the adapter call changed
    (local.set $p (i32.sub (local.get $brk) (i32.const 256)))
    (block $done (loop $l
      (br_if $done (i32.ge_u (local.get $p) (local.get $brk)))
      (if (i32.ne (i32.load8_u (local.get $p)) (i32.const 0xAB))
        (then (local.set $bad (i32.add (local.get $bad) (i32.const 1)))))
      (local.set $p (i32.add (local.get $p) (i32.const 1)))
      (br $l)))
    (call $exit (local.get $bad))))
