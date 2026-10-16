//! The WASI preview 1 functions a stage is given, and nothing more.
//!
//! Every pointer and length comes from the module, so every read and write of its memory is
//! checked: an out-of-range one answers `EINVAL`, never a panic of the host.

use crate::Limits;
use wasmi::{Caller, Error, Extern, Linker, StoreLimits};

pub(crate) const MODULE: &str = "wasi_snapshot_preview1";

/// The functions [`link`] defines. A module importing any other is refused when loaded.
pub(crate) const GIVEN: [&str; 16] = [
    "proc_exit",
    "sched_yield",
    "args_sizes_get",
    "args_get",
    "environ_sizes_get",
    "environ_get",
    "clock_time_get",
    "random_get",
    "fd_write",
    "fd_read",
    "fd_close",
    "fd_fdstat_get",
    "fd_fdstat_set_flags",
    "fd_prestat_get",
    "fd_prestat_dir_name",
    "poll_oneoff",
];

const ESUCCESS: i32 = 0;
const EBADF: i32 = 8;
const EINVAL: i32 = 28;
/// The most buffers or subscriptions one call may pass, as POSIX's IOV_MAX.
const MAX_VECTORS: u32 = 1024;

pub(crate) struct Ctx {
    stdin: Vec<u8>,
    pos: usize,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    max_output: usize,
    /// Set when a write went past `max_output`; the run then traps.
    pub(crate) overflow: bool,
    rng: u64,
    clock_ns: u64,
    pub(crate) limits: StoreLimits,
}

impl Ctx {
    pub(crate) fn new(stdin: Vec<u8>, limits: &Limits, store_limits: StoreLimits) -> Self {
        Self {
            stdin,
            pos: 0,
            stdout: Vec::new(),
            stderr: Vec::new(),
            max_output: limits.max_output,
            overflow: false,
            rng: limits.seed,
            clock_ns: 1,
            limits: store_limits,
        }
    }

    /// splitmix64.
    fn next(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn mem<'a>(caller: &'a mut Caller<'_, Ctx>) -> Option<(&'a mut [u8], &'a mut Ctx)> {
    let m = caller.get_export("memory").and_then(Extern::into_memory)?;
    Some(m.data_and_store_mut(caller))
}

/// A module's pointer as an offset into its memory. Pointers are unsigned 32-bit.
fn at(p: i32) -> usize {
    p as u32 as usize
}

fn span(start: usize, len: usize) -> Option<std::ops::Range<usize>> {
    Some(start..start.checked_add(len)?)
}

fn rd32(m: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_le_bytes(m.get(span(p, 4)?)?.try_into().ok()?))
}

fn rd64(m: &[u8], p: usize) -> Option<u64> {
    Some(u64::from_le_bytes(m.get(span(p, 8)?)?.try_into().ok()?))
}

fn wr(m: &mut [u8], p: usize, bytes: &[u8]) -> Option<()> {
    m.get_mut(span(p, bytes.len())?)?.copy_from_slice(bytes);
    Some(())
}

fn status(done: Option<()>) -> i32 {
    match done {
        Some(()) => ESUCCESS,
        None => EINVAL,
    }
}

/// The `k`-th buffer of an iovec array: its offset and length.
fn iovec(m: &[u8], iovs: i32, k: usize) -> Option<(usize, usize)> {
    let p = at(iovs).checked_add(8 * k)?;
    Some((rd32(m, p)? as usize, rd32(m, p + 4)? as usize))
}

fn is_stdio(fd: i32) -> bool {
    (0..3).contains(&fd)
}

pub(crate) fn link(linker: &mut Linker<Ctx>) -> Result<(), Error> {
    const W: &str = MODULE;
    linker.func_wrap(
        W,
        "proc_exit",
        |_: Caller<'_, Ctx>, code: i32| -> Result<(), Error> { Err(Error::i32_exit(code)) },
    )?;
    linker.func_wrap(W, "sched_yield", |_: Caller<'_, Ctx>| -> i32 { ESUCCESS })?;
    // One argument, the program's name; no environment.
    linker.func_wrap(
        W,
        "args_sizes_get",
        |mut c: Caller<'_, Ctx>, argc: i32, size: i32| -> i32 {
            let Some((m, _)) = mem(&mut c) else {
                return EINVAL;
            };
            status(wr(m, at(argc), &1u32.to_le_bytes()).and(wr(m, at(size), &6u32.to_le_bytes())))
        },
    )?;
    linker.func_wrap(
        W,
        "args_get",
        |mut c: Caller<'_, Ctx>, argv: i32, buf: i32| -> i32 {
            let Some((m, _)) = mem(&mut c) else {
                return EINVAL;
            };
            status(wr(m, at(argv), &(buf as u32).to_le_bytes()).and(wr(m, at(buf), b"stage\0")))
        },
    )?;
    linker.func_wrap(
        W,
        "environ_sizes_get",
        |mut c: Caller<'_, Ctx>, n: i32, size: i32| -> i32 {
            let Some((m, _)) = mem(&mut c) else {
                return EINVAL;
            };
            status(wr(m, at(n), &0u32.to_le_bytes()).and(wr(m, at(size), &0u32.to_le_bytes())))
        },
    )?;
    linker.func_wrap(
        W,
        "environ_get",
        |_: Caller<'_, Ctx>, _: i32, _: i32| -> i32 { ESUCCESS },
    )?;
    // The clock moves 1 µs per reading, and by the whole timeout of every sleep.
    linker.func_wrap(
        W,
        "clock_time_get",
        |mut c: Caller<'_, Ctx>, _id: i32, _precision: i64, out: i32| -> i32 {
            let Some((m, ctx)) = mem(&mut c) else {
                return EINVAL;
            };
            ctx.clock_ns = ctx.clock_ns.saturating_add(1_000);
            status(wr(m, at(out), &ctx.clock_ns.to_le_bytes()))
        },
    )?;
    linker.func_wrap(
        W,
        "random_get",
        |mut c: Caller<'_, Ctx>, buf: i32, len: i32| -> i32 {
            let Some((m, ctx)) = mem(&mut c) else {
                return EINVAL;
            };
            let Some(dst) = span(at(buf), at(len)).and_then(|r| m.get_mut(r)) else {
                return EINVAL;
            };
            for chunk in dst.chunks_mut(8) {
                let r = ctx.next().to_le_bytes();
                chunk.copy_from_slice(&r[..chunk.len()]);
            }
            ESUCCESS
        },
    )?;
    linker.func_wrap(
        W,
        "fd_write",
        |mut c: Caller<'_, Ctx>, fd: i32, iovs: i32, n: i32, written: i32| -> Result<i32, Error> {
            if fd != 1 && fd != 2 {
                return Ok(EBADF);
            }
            if n as u32 > MAX_VECTORS {
                return Ok(EINVAL);
            }
            let Some((m, ctx)) = mem(&mut c) else {
                return Ok(EINVAL);
            };
            let mut total = 0usize;
            for k in 0..n as u32 as usize {
                let Some((p, l)) = iovec(m, iovs, k) else {
                    return Ok(EINVAL);
                };
                let Some(bytes) = span(p, l).and_then(|r| m.get(r)) else {
                    return Ok(EINVAL);
                };
                if ctx.stdout.len() + ctx.stderr.len() + bytes.len() > ctx.max_output {
                    ctx.overflow = true;
                    return Err(Error::new("output limit reached"));
                }
                let sink = if fd == 1 {
                    &mut ctx.stdout
                } else {
                    &mut ctx.stderr
                };
                sink.extend_from_slice(bytes);
                total += l;
            }
            Ok(status(wr(m, at(written), &(total as u32).to_le_bytes())))
        },
    )?;
    linker.func_wrap(
        W,
        "fd_read",
        |mut c: Caller<'_, Ctx>, fd: i32, iovs: i32, n: i32, read: i32| -> i32 {
            if fd != 0 {
                return EBADF;
            }
            if n as u32 > MAX_VECTORS {
                return EINVAL;
            }
            let Some((m, ctx)) = mem(&mut c) else {
                return EINVAL;
            };
            let mut total = 0usize;
            for k in 0..n as u32 as usize {
                let Some((p, l)) = iovec(m, iovs, k) else {
                    return EINVAL;
                };
                let take = l.min(ctx.stdin.len() - ctx.pos);
                let Some(dst) = span(p, take).and_then(|r| m.get_mut(r)) else {
                    return EINVAL;
                };
                dst.copy_from_slice(&ctx.stdin[ctx.pos..ctx.pos + take]);
                ctx.pos += take;
                total += take;
            }
            status(wr(m, at(read), &(total as u32).to_le_bytes()))
        },
    )?;
    linker.func_wrap(W, "fd_close", |_: Caller<'_, Ctx>, fd: i32| -> i32 {
        if is_stdio(fd) {
            ESUCCESS
        } else {
            EBADF
        }
    })?;
    // stdin, stdout and stderr are character devices; nothing else is open.
    linker.func_wrap(
        W,
        "fd_fdstat_get",
        |mut c: Caller<'_, Ctx>, fd: i32, out: i32| -> i32 {
            if !is_stdio(fd) {
                return EBADF;
            }
            let Some((m, _)) = mem(&mut c) else {
                return EINVAL;
            };
            let mut stat = [0u8; 24];
            stat[0] = 2; // CHARACTER_DEVICE
            stat[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
            status(wr(m, at(out), &stat))
        },
    )?;
    linker.func_wrap(
        W,
        "fd_fdstat_set_flags",
        |_: Caller<'_, Ctx>, fd: i32, _: i32| -> i32 {
            if is_stdio(fd) {
                ESUCCESS
            } else {
                EBADF
            }
        },
    )?;
    // No preopened directory: the program sees no file system.
    linker.func_wrap(
        W,
        "fd_prestat_get",
        |_: Caller<'_, Ctx>, _: i32, _: i32| -> i32 { EBADF },
    )?;
    linker.func_wrap(
        W,
        "fd_prestat_dir_name",
        |_: Caller<'_, Ctx>, _: i32, _: i32, _: i32| -> i32 { EBADF },
    )?;
    // Sleeps complete at once and advance the clock; stdio is always ready.
    linker.func_wrap(
        W,
        "poll_oneoff",
        |mut c: Caller<'_, Ctx>, subs: i32, events: i32, n: i32, count: i32| -> i32 {
            if n as u32 > MAX_VECTORS {
                return EINVAL;
            }
            let Some((m, ctx)) = mem(&mut c) else {
                return EINVAL;
            };
            for k in 0..n as u32 as usize {
                let s = at(subs) + 48 * k;
                let (Some(userdata), Some(tag)) = (rd64(m, s), m.get(s + 8).copied()) else {
                    return EINVAL;
                };
                let mut ev = [0u8; 32];
                ev[0..8].copy_from_slice(&userdata.to_le_bytes());
                ev[10] = tag;
                if tag == 0 {
                    let Some(timeout) = rd64(m, s + 24) else {
                        return EINVAL;
                    };
                    ctx.clock_ns = ctx.clock_ns.saturating_add(timeout);
                } else {
                    let ready = (ctx.stdin.len() - ctx.pos) as u64;
                    ev[16..24].copy_from_slice(&ready.max(1).to_le_bytes());
                }
                if wr(m, at(events) + 32 * k, &ev).is_none() {
                    return EINVAL;
                }
            }
            status(wr(m, at(count), &(n as u32).to_le_bytes()))
        },
    )?;
    Ok(())
}
