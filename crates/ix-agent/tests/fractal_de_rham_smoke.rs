//! Smoke tests for the `ix_fractal { operation: "de_rham_1d" }` MCP operation (ix#203).
//!
//! De Rham curve generation takes a `&mut impl Rng`, so the tool layer owns the seeding.
//! That makes reproducibility a *correctness* property, not a quality one — and the
//! callee's silent cap at depth 20 (1,048,577 samples) makes the loud depth cap the other
//! load-bearing behaviour. Both are asserted here through `ToolRegistry::call`, so a
//! regression in either the handler or the schema fails here rather than in a notebook.

//!
//! Two call paths on purpose. `call` goes through `ToolRegistry::call`, which is the real
//! MCP surface *including* the loop-detection middleware — and that middleware trips a
//! circuit breaker after 10 calls to the same tool within 300s, process-wide. So the
//! end-to-end path is spent only on wiring proofs, and the exhaustive boundary sweeps go
//! through `op`, calling `handlers::fractal` directly below the governance instrument.
//! Trading breadth for the breaker would be the wrong way round: the breaker is a feature.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

/// End-to-end: handler *and* schema, through the governance middleware. Budgeted — see the
/// module note; keep the total number of `call` sites in this file well under 10.
fn call(params: serde_json::Value) -> Result<serde_json::Value, String> {
    ToolRegistry::new().call("ix_fractal", params)
}

/// The handler alone, bypassing loop detection so boundary sweeps can be exhaustive.
fn op(params: serde_json::Value) -> Result<serde_json::Value, String> {
    ix_agent::handlers::fractal(params)
}

/// Same (depth, roughness, seed) ⇒ identical samples; a different seed ⇒ a different curve.
/// (Determinism is per-build: `StdRng` is not stable across `rand` major versions.)
#[test]
fn de_rham_1d_is_reproducible_and_seed_sensitive() {
    let params = json!({"operation": "de_rham_1d", "depth": 8, "roughness": 0.3, "seed": 42});
    let a = call(params.clone()).expect("de_rham_1d should succeed");
    let b = call(params).expect("de_rham_1d should succeed");
    assert_eq!(
        a["points"], b["points"],
        "same seed must give the same curve"
    );

    let c = call(json!({"operation": "de_rham_1d", "depth": 8, "roughness": 0.3, "seed": 43}))
        .expect("de_rham_1d should succeed");
    assert_ne!(
        a["points"], c["points"],
        "a different seed must give a different curve (guards a silently-ignored seed)"
    );
}

/// Output size is bounded and exactly `2^depth + 1`, with `t` spanning [0, 1].
#[test]
fn de_rham_1d_output_is_bounded() {
    let r = call(json!({"operation": "de_rham_1d", "depth": 12, "roughness": 0.3, "seed": 42}))
        .expect("depth 12 is at the cap and must succeed");
    assert_eq!(r["n_samples"], json!(4097), "2^12 + 1 samples");
    let points = r["points"].as_array().expect("points array");
    assert_eq!(points.len(), 4097);
    assert_eq!(points[0][0], json!(0.0));
    assert!((points[4096][0].as_f64().unwrap() - 1.0).abs() < 1e-12);
    assert!(
        points
            .iter()
            .all(|p| p[1].as_f64().is_some_and(f64::is_finite)),
        "finite roughness must not produce NaN/Inf samples"
    );
}

/// Depth above the cap is a loud error, never the callee's silent cap-at-20.
#[test]
fn de_rham_1d_rejects_excessive_depth() {
    let err = call(json!({"operation": "de_rham_1d", "depth": 13, "roughness": 0.3, "seed": 42}))
        .expect_err("depth 13 must be rejected");
    assert!(
        err.contains("depth must be <= 12"),
        "unexpected error: {err}"
    );
}

/// Roughness must be finite and non-negative. Non-finite is unreachable through JSON
/// (`serde_json::Value` cannot hold NaN/Infinity), so the reachable case is the negative one;
/// a missing or non-numeric `roughness` is rejected by the parse step.
#[test]
fn de_rham_1d_rejects_invalid_roughness() {
    let err = call(json!({"operation": "de_rham_1d", "depth": 4, "roughness": -0.1, "seed": 42}))
        .expect_err("negative roughness must be rejected");
    assert!(
        err.contains("roughness must be finite and >= 0"),
        "unexpected error: {err}"
    );

    let err = call(json!({"operation": "de_rham_1d", "depth": 4, "seed": 42}))
        .expect_err("missing roughness must be rejected");
    assert!(err.contains("roughness"), "unexpected error: {err}");
}

/// Roughness has an upper bound too. Midpoint displacement compounds multiplicatively over
/// `depth` levels, so a large *finite* roughness overflows `f64` — and `serde_json` renders
/// a non-finite `f64` as JSON `null`. Without the cap, `roughness: 1e100` returns a
/// **successful** response whose `points` are silently null-corrupted. Assert the boundary
/// from both sides, and assert that nothing above it can ever come back as a success.
#[test]
fn de_rham_1d_bounds_roughness() {
    let at_cap = op(json!({
        "operation": "de_rham_1d", "depth": 12, "roughness": 1.0e6, "seed": 42
    }))
    .expect("roughness exactly at the cap must remain computable");
    let points = at_cap["points"].as_array().expect("points array");
    assert_eq!(points.len(), 4097);
    assert!(
        points
            .iter()
            .all(|p| p[1].as_f64().is_some_and(f64::is_finite)),
        "at the cap every sample must still be a finite number, never a JSON null"
    );
    assert_eq!(
        at_cap["max_roughness"],
        json!(1.0e6),
        "the cap is self-describing"
    );

    for over in [1.000001e6, 1.0e10, 1.0e100, 1.0e300, f64::MAX] {
        let err = op(json!({
            "operation": "de_rham_1d", "depth": 12, "roughness": over, "seed": 42
        }))
        .expect_err("roughness above the cap must be an explicit error, never null points");
        assert!(
            err.contains("roughness must be <= 1000000"),
            "unexpected error for roughness={over}: {err}"
        );
    }
}

/// The success path never emits a JSON `null` inside `points`, across the whole accepted
/// (depth, roughness) envelope — the property the two guards above exist to protect.
#[test]
fn de_rham_1d_never_emits_null_points() {
    for depth in [0, 1, 6, 12] {
        for roughness in [0.0, 0.3, 1.0, 1.0e3, 1.0e6] {
            let r = op(json!({
                "operation": "de_rham_1d", "depth": depth, "roughness": roughness, "seed": 7
            }))
            .unwrap_or_else(|e| panic!("depth={depth} roughness={roughness} must succeed: {e}"));
            for (i, p) in r["points"].as_array().unwrap().iter().enumerate() {
                assert!(
                    p[0].as_f64().is_some_and(f64::is_finite)
                        && p[1].as_f64().is_some_and(f64::is_finite),
                    "depth={depth} roughness={roughness}: point {i} = {p} is not a finite pair"
                );
            }
        }
    }
}

/// Zero roughness degenerates to the straight line value == t — the cheapest end-to-end
/// check that the depth/seed plumbing reaches the real callee and not a stub.
#[test]
fn de_rham_1d_zero_roughness_is_a_line() {
    let r = call(json!({"operation": "de_rham_1d", "depth": 6, "roughness": 0.0, "seed": 42}))
        .expect("de_rham_1d should succeed");
    for p in r["points"].as_array().unwrap() {
        let (t, v) = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
        assert!(
            (v - t).abs() < 1e-10,
            "expected value == t at t={t}, got {v}"
        );
    }
}
