//! Spike: FlashAssign-style kernel fusion on IX's wgpu stack.
//!
//! Inspired by Flash-KMeans (arXiv:2603.09229, github.com/svg-project/flash-kmeans),
//! an IO-aware *exact* k-means whose headline trick — "FlashAssign" — fuses the
//! distance computation with an online argmin so the N×K distance matrix is never
//! written to global memory. Flash-KMeans ships as NVIDIA-only Triton kernels, which
//! we will NOT adopt (IX is pure-Rust + wgpu, cross-vendor, f32 WGSL). This spike
//! tests whether the *idea* transfers to WGSL on our actual box.
//!
//! Target: the k-means **assignment step** — assign each of N points to its nearest
//! of K centroids (argmin over K). Two GPU paths, one CPU reference:
//!   - MATERIALIZED: kernel A writes the full N×K squared-distance matrix to a global
//!     buffer; kernel B reads it back row-by-row and argmins. (N×K global write + read.)
//!   - FUSED (FlashAssign): one kernel, one thread per point, loops the K centroids
//!     keeping a running argmin in registers; writes only N assignments + N min-dists.
//!     The N×K matrix is never materialized.
//!
//! We assert all three agree, then report wall-clock + the intermediate memory the
//! fused path avoids. This is a SPIKE: kernels live here, not in the stable API.
//!
//! Run:  cargo run -p ix-gpu --example flash_assign_spike --release
//!       cargo run -p ix-gpu --example flash_assign_spike --release -- 200000 128 64
//!       (args: N points, K centroids, D dims)

use ix_gpu::assign::nearest_centroid_cpu;
use ix_gpu::context::GpuContext;
use std::time::Instant;
use wgpu::*;

const WG: u32 = 256;

// ── WGSL kernels ──────────────────────────────────────────────────────────────

/// FlashAssign: one thread per point; running argmin over K centroids in registers.
/// Writes only `assign[N]` + `mindist[N]` — no N×K materialization.
const FUSED_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;  // [n, k, d]
@group(0) @binding(3) var<storage, read_write> assign: array<u32>;
@group(0) @binding(4) var<storage, read_write> mindist: array<f32>;

@compute @workgroup_size(256)
fn assign_fused(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params[0]; let k = params[1]; let d = params[2];
    let p = gid.x;
    if (p >= n) { return; }
    let p_base = p * d;
    var best: f32 = 3.402823e+38;
    var best_k: u32 = 0u;
    for (var c = 0u; c < k; c++) {
        let c_base = c * d;
        var dist: f32 = 0.0;
        for (var i = 0u; i < d; i++) {
            let diff = points[p_base + i] - centroids[c_base + i];
            dist += diff * diff;
        }
        if (dist < best) { best = dist; best_k = c; }
    }
    assign[p] = best_k;
    mindist[p] = best;
}
"#;

/// Materialized kernel A: one thread per (point, centroid) cell → writes N×K matrix.
const MATMUL_DIST_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;  // [n, k, d]
@group(0) @binding(3) var<storage, read_write> distmat: array<f32>;  // N*K

@compute @workgroup_size(256)
fn dist_matrix(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = params[0]; let k = params[1]; let d = params[2];
    let total = n * k;
    let stride = nwg.x * 256u;     // grid-stride: bounded workgroups cover all N×K cells
    var idx = gid.x;               // flattened p*k + c
    while (idx < total) {
        let p = idx / k;
        let c = idx % k;
        let p_base = p * d; let c_base = c * d;
        var dist: f32 = 0.0;
        for (var i = 0u; i < d; i++) {
            let diff = points[p_base + i] - centroids[c_base + i];
            dist += diff * diff;
        }
        distmat[idx] = dist;
        idx += stride;
    }
}
"#;

/// Materialized kernel B: one thread per point → reads its N×K row, argmins.
const ARGMIN_ROWS_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> distmat: array<f32>;   // N*K
@group(0) @binding(1) var<storage, read> params: array<u32>;    // [n, k, d]
@group(0) @binding(2) var<storage, read_write> assign: array<u32>;
@group(0) @binding(3) var<storage, read_write> mindist: array<f32>;

@compute @workgroup_size(256)
fn argmin_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params[0]; let k = params[1];
    let p = gid.x;
    if (p >= n) { return; }
    let base = p * k;
    var best: f32 = 3.402823e+38;
    var best_k: u32 = 0u;
    for (var c = 0u; c < k; c++) {
        let val = distmat[base + c];
        if (val < best) { best = val; best_k = c; }
    }
    assign[p] = best_k;
    mindist[p] = best;
}
"#;

// CPU correctness oracle: `ix_gpu::assign::nearest_centroid_cpu` (in the library so
// `cargo test --workspace` runs its tests).

// ── GPU plumbing ────────────────────────────────────────────────────────────────

fn u32_buffer(ctx: &GpuContext, label: &str, len: usize) -> Buffer {
    ctx.create_output_buffer(label, (len * std::mem::size_of::<u32>()) as u64)
}

fn params_buffer(ctx: &GpuContext, n: usize, k: usize, d: usize) -> Buffer {
    use wgpu::util::DeviceExt;
    let params = [n as u32, k as u32, d as u32];
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    })
}

fn bind(ctx: &GpuContext, pipe: &ComputePipeline, entries: &[(u32, &Buffer)]) -> BindGroup {
    let layout = pipe.get_bind_group_layout(0);
    let e: Vec<BindGroupEntry> = entries
        .iter()
        .map(|(b, buf)| BindGroupEntry {
            binding: *b,
            resource: buf.as_entire_binding(),
        })
        .collect();
    ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("bind"),
        layout: &layout,
        entries: &e,
    })
}

fn dispatch(ctx: &GpuContext, passes: &[(&ComputePipeline, &BindGroup, u32)]) {
    let mut enc = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: Some("enc") });
    for (pipe, bg, groups) in passes {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, *bg, &[]);
        pass.dispatch_workgroups(*groups, 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    let _ = ctx.device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    });
}

fn read_u32(ctx: &GpuContext, buf: &Buffer, len: usize) -> Vec<u32> {
    let raw = ctx.read_buffer(buf, (len * std::mem::size_of::<u32>()) as u64);
    raw.iter().map(|f| f.to_bits()).collect()
}

fn run_fused(ctx: &GpuContext, pts: &Buffer, cents: &Buffer, prm: &Buffer, n: usize) -> Vec<u32> {
    let assign = u32_buffer(ctx, "assign_fused", n);
    let mindist = ctx.create_output_buffer("mindist_fused", (n * 4) as u64);
    let pipe = ctx.create_compute_pipeline("fused", FUSED_SHADER, "assign_fused");
    let bg = bind(ctx, &pipe, &[(0, pts), (1, cents), (2, prm), (3, &assign), (4, &mindist)]);
    let groups = (n as u32).div_ceil(WG);
    dispatch(ctx, &[(&pipe, &bg, groups)]);
    read_u32(ctx, &assign, n)
}

fn run_materialized(
    ctx: &GpuContext,
    pts: &Buffer,
    cents: &Buffer,
    prm: &Buffer,
    n: usize,
    k: usize,
) -> Vec<u32> {
    let distmat = ctx.create_output_buffer("distmat", (n * k * 4) as u64); // the N×K spill
    let assign = u32_buffer(ctx, "assign_mat", n);
    let mindist = ctx.create_output_buffer("mindist_mat", (n * 4) as u64);

    let pipe_a = ctx.create_compute_pipeline("distmat", MATMUL_DIST_SHADER, "dist_matrix");
    let bg_a = bind(ctx, &pipe_a, &[(0, pts), (1, cents), (2, prm), (3, &distmat)]);
    let pipe_b = ctx.create_compute_pipeline("argmin", ARGMIN_ROWS_SHADER, "argmin_rows");
    let bg_b = bind(ctx, &pipe_b, &[(0, &distmat), (1, prm), (2, &assign), (3, &mindist)]);

    // Cap kernel-A workgroups at the wgpu per-dimension limit (65535); the
    // grid-stride loop in dist_matrix covers the rest. Argmin/fused stay 1:1
    // with N (our N keeps them well under the cap).
    let groups_a = ((n * k) as u32).div_ceil(WG).min(65_535);
    let groups_b = (n as u32).div_ceil(WG);
    dispatch(ctx, &[(&pipe_a, &bg_a, groups_a), (&pipe_b, &bg_b, groups_b)]);
    read_u32(ctx, &assign, n)
}

/// Median wall-clock of `f` over `reps` (after one warmup), in milliseconds.
fn time_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warmup (pipeline compile + caches)
    let mut samples: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let d: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let reps = 5;

    println!("FlashAssign spike — k-means assignment (N={n}, K={k}, D={d})");
    let spill_mb = (n * k * 4) as f64 / (1024.0 * 1024.0);
    println!("  N×K distance matrix the fused path avoids: {spill_mb:.1} MB\n");

    // Deterministic synthetic data (no RNG dep): cheap hash-ish spread.
    let gen = |seed: usize, len: usize| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) % 10_000;
                x as f32 / 10_000.0
            })
            .collect()
    };
    let points = gen(1, n * d);
    let centroids = gen(7, k * d);

    let cpu = nearest_centroid_cpu(&points, &centroids, n, k, d);

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("No GPU adapter ({e}). CPU-only on this box:");
            let cpu_ms = time_ms(reps, || {
                std::hint::black_box(nearest_centroid_cpu(&points, &centroids, n, k, d));
            });
            println!("  CPU assign: {cpu_ms:.2} ms");
            println!("\n(Run on a GPU box to compare materialized vs fused.)");
            return;
        }
    };
    println!("GPU: {} ({:?})\n", ctx.gpu_name(), ctx.backend());

    let pts = ctx.create_buffer_init("points", &points);
    let cents = ctx.create_buffer_init("centroids", &centroids);
    let prm = params_buffer(&ctx, n, k, d);

    // Correctness first — both GPU paths must match the CPU oracle.
    let mat = run_materialized(&ctx, &pts, &cents, &prm, n, k);
    let fused = run_fused(&ctx, &pts, &cents, &prm, n);
    let mat_ok = mat == cpu;
    let fused_ok = fused == cpu;
    println!("Correctness vs CPU oracle:  materialized={mat_ok}  fused={fused_ok}");
    if !mat_ok || !fused_ok {
        // Hard-fail: never report a performance conclusion for incorrect results.
        let mat_mism = mat.iter().zip(&cpu).filter(|(a, b)| a != b).count();
        let fused_mism = fused.iter().zip(&cpu).filter(|(a, b)| a != b).count();
        eprintln!("  ✗ mismatches vs oracle — materialized: {mat_mism}/{n}, fused: {fused_mism}/{n}");
        eprintln!("  Refusing to benchmark incorrect kernels.");
        std::process::exit(1);
    }

    // Timing.
    let mat_ms = time_ms(reps, || {
        std::hint::black_box(run_materialized(&ctx, &pts, &cents, &prm, n, k));
    });
    let fused_ms = time_ms(reps, || {
        std::hint::black_box(run_fused(&ctx, &pts, &cents, &prm, n));
    });

    println!("\nMedian over {reps} reps (lower is better):");
    println!("  materialized (N×K spill + argmin): {mat_ms:.2} ms");
    println!("  fused (FlashAssign, no spill):     {fused_ms:.2} ms");
    if fused_ms > 0.0 {
        println!("  → fused speedup: {:.2}×", mat_ms / fused_ms);
    }
    println!("  → intermediate memory avoided: {spill_mb:.1} MB");
}

// Oracle tests live in the library (`ix_gpu::assign`), so `cargo test --workspace`
// actually runs them — example-local `#[cfg(test)]` modules are only built, not run.
