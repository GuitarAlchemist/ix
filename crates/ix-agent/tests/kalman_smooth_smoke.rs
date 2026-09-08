//! Smoke tests for the `ix_kalman { operation: "smooth_1d" }` MCP operation (ix#193).
//!
//! `ix-signal`'s Kalman filter has been correct and tested since long before this tool
//! existed; what is new here is the *boundary*. So these tests target the three things
//! the boundary owns and the callee does not:
//!
//! 1. **Real filtered values**, not shapes. A tool that returned the input unchanged, or
//!    returned zeros, would satisfy any length/shape assertion. The ramp test below pins
//!    actual estimates to 1e-9, so a wiring regression that silently swapped the model,
//!    the parameter order, or position-for-velocity fails here.
//! 2. **Loud bounds.** The callee caps nothing and the SQL surface caps nothing; an MCP
//!    payload must not materialise an unbounded series, and `dt` overflows to NaN — which
//!    `serde_json` renders as JSON `null` — well inside the range of a finite `f64`.
//! 3. **Agreement with the SQL surface**, so `ix_kalman` and `ix_kalman_smooth` cannot
//!    drift into being two different filters wearing one name.
//!
//! Two call paths on purpose, following `fractal_de_rham_smoke.rs`: `call` goes through
//! `ToolRegistry::call` — the real MCP surface *including* loop-detection middleware,
//! whose circuit breaker trips after 10 calls to one tool within 300s, process-wide. So
//! end-to-end calls are budgeted for wiring proofs and the boundary sweeps go through
//! `op`, below the middleware. The breaker is a feature; the tests bend around it.

use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

/// End-to-end: handler *and* schema, through the governance middleware. Budgeted — see
/// the module note; keep the total number of `call` sites in this file well under 10.
fn call(params: Value) -> Result<Value, String> {
    ToolRegistry::new().call("ix_kalman", params)
}

/// The handler alone, bypassing loop detection so boundary sweeps can be exhaustive.
fn op(params: Value) -> Result<Value, String> {
    ix_agent::handlers::kalman(params)
}

fn nums(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("response['{key}'] is not an array: {v}"))
        .iter()
        .map(|x| {
            x.as_f64()
                .unwrap_or_else(|| panic!("response['{key}'] holds a non-number (JSON null?): {v}"))
        })
        .collect()
}

/// A clean unit-rate ramp is exactly the constant-velocity motion model, so from a cold
/// `[0, 0]` start the filter locks on: position converges to the final measurement and
/// velocity climbs toward the slope of 1.
///
/// The four pinned values are the *measured* output of `constant_velocity_1d(0.01, 0.5,
/// 1.0)` over `[0..8)`, asserted to 1e-9. They are a regression oracle, not a derivation:
/// they would change if the callee's filter changed, and that is the point — this test
/// then fails loudly instead of the tool quietly returning different numbers. Asserting
/// only "length == 8" would pass for a handler that returned the input untouched.
#[test]
fn kalman_smooth_1d_tracks_a_ramp() {
    let series: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let r = call(json!({
        "operation": "smooth_1d",
        "series": series,
        "process_noise": 0.01,
        "measurement_noise": 0.5
    }))
    .expect("smooth_1d should succeed on a clean ramp");

    let position = nums(&r, "position");
    let velocity = nums(&r, "velocity");
    assert_eq!(position.len(), 8, "one position estimate per input sample");
    assert_eq!(velocity.len(), 8, "one velocity estimate per input sample");
    assert_eq!(r["n_samples"], json!(8));
    assert_eq!(r["model"], json!("constant_velocity_1d"));
    assert_eq!(
        r["dt"],
        json!(1.0),
        "dt must default to the SQL surface's 1.0"
    );

    // Cold start: the filter has seen nothing, so its first estimate is the zero state.
    assert!(
        position[0].abs() < 1e-12,
        "cold start must begin at 0, got {}",
        position[0]
    );

    // Measured values — the actual regression oracle.
    for (i, expected) in [
        (1usize, 0.738_419_732_417_f64),
        (3, 2.773_858_278_624),
        (7, 6.910_711_316_174),
    ] {
        assert!(
            (position[i] - expected).abs() < 1e-9,
            "position[{i}] = {}, expected {expected} (±1e-9)",
            position[i]
        );
    }
    assert!(
        (velocity[7] - 0.975_253_232_261).abs() < 1e-9,
        "velocity[7] = {}, expected 0.975253232261 (±1e-9)",
        velocity[7]
    );

    // The estimates are genuinely *filtered*, not echoed: every one lags the measurement
    // it saw, because the filter is still converging from a cold start on a rising ramp.
    for (i, (&p, &z)) in position.iter().zip(series.iter()).enumerate() {
        assert!(
            p < z || (p - z).abs() < 1e-12,
            "position[{i}] = {p} should not overshoot measurement {z} on a cold-start ramp"
        );
    }
    assert!(
        position[7] > position[3] && position[3] > position[1],
        "position must increase along a rising ramp"
    );
    assert!(
        velocity[7] > velocity[1],
        "velocity must climb toward the ramp slope as the filter converges"
    );
}

/// The two surfaces over `ix-signal`'s filter must stay the same filter.
///
/// `ix_duck::graphsig`'s `kalman_smooth_tracks_ramp` asserts that
/// `ix_kalman_smooth('[0..7]', 0.01, 0.5)` emits 8 rows whose last `value` lands in
/// `5.0..=9.0`. This re-asserts that exact contract against the MCP `position` series, so
/// the two cannot silently become different models. It duplicates the SQL test's
/// *contract* rather than calling DuckDB: `ix-duck`'s UDF surface sits behind optional
/// `duck`/`udf` features that are never compiled in CI, so a real cross-surface call
/// would be a test that does not run.
#[test]
fn kalman_agrees_with_sql_surface_contract() {
    let series: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let r = op(json!({
        "operation": "smooth_1d",
        "series": series,
        "process_noise": 0.01,
        "measurement_noise": 0.5
    }))
    .expect("smooth_1d should succeed");

    let position = nums(&r, "position");
    assert_eq!(
        position.len(),
        8,
        "ix_kalman_smooth emits one row per measurement"
    );
    let last = position[7];
    assert!(
        (5.0..=9.0).contains(&last),
        "SQL surface asserts the final estimate tracks the ramp to ~7; MCP gave {last}"
    );
}

/// A constant signal must be smoothed toward its own level, and the velocity estimate
/// must decay toward zero — the opposite behaviour from the ramp, which is what proves
/// the filter is responding to the data rather than to its parameters.
#[test]
fn kalman_smooths_a_constant_toward_its_level() {
    let r = op(json!({
        "operation": "smooth_1d",
        "series": vec![5.0f64; 10],
        "process_noise": 0.01,
        "measurement_noise": 1.0
    }))
    .expect("smooth_1d should succeed");

    let position = nums(&r, "position");
    let velocity = nums(&r, "velocity");
    assert!(
        (position[9] - 5.0).abs() < 0.5,
        "a constant-5 series must settle near 5.0, got {}",
        position[9]
    );
    assert!(
        velocity[9].abs() < velocity[0].abs(),
        "velocity must decay toward 0 on a constant series: got {} then {}",
        velocity[0],
        velocity[9]
    );
}

/// The sample cap is loud from both sides. The callee has no cap at all, so this bound
/// exists only at the MCP boundary and nothing below would catch its removal.
#[test]
fn kalman_sample_cap_is_enforced_from_both_sides() {
    let at_cap: Vec<f64> = (0..4096).map(|i| i as f64).collect();
    let r = op(json!({
        "operation": "smooth_1d",
        "series": at_cap,
        "process_noise": 0.01,
        "measurement_noise": 0.5
    }))
    .expect("exactly at the cap must be accepted");
    let position = nums(&r, "position");
    assert_eq!(position.len(), 4096);
    assert_eq!(r["max_samples"], json!(4096), "the cap must be echoed back");
    assert!(
        (position[4095] - 4095.0).abs() < 1e-6,
        "a long clean ramp must be tracked exactly by the end, got {}",
        position[4095]
    );

    let over: Vec<f64> = (0..4097).map(|i| i as f64).collect();
    let err = op(json!({
        "operation": "smooth_1d",
        "series": over,
        "process_noise": 0.01,
        "measurement_noise": 0.5
    }))
    .expect_err("one sample over the cap must be a loud error");
    assert!(
        err.contains("series length must be <= 4096"),
        "unexpected error: {err}"
    );
}

/// `dt` is the parameter that can turn a successful response into one full of JSON
/// `null`. `dt = 1e110` was measured to produce NaN estimates; the cap sits at 1e6.
#[test]
fn kalman_dt_cap_is_enforced_and_output_stays_finite() {
    let series: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let base = |dt: f64| {
        json!({
            "operation": "smooth_1d",
            "series": series,
            "process_noise": 0.01,
            "measurement_noise": 1.0,
            "dt": dt
        })
    };

    let r = op(base(1.0e6)).expect("exactly at the dt cap must be accepted");
    assert_eq!(r["max_dt"], json!(1.0e6), "the cap must be echoed back");
    for key in ["position", "velocity"] {
        assert!(
            nums(&r, key).iter().all(|x| x.is_finite()),
            "{key} must be entirely finite at the dt cap"
        );
    }

    let err = op(base(1.0e6 + 1.0)).expect_err("above the dt cap must be a loud error");
    assert!(err.contains("dt must be <= "), "unexpected error: {err}");

    // The value that actually produces NaN in the callee is rejected long before reaching
    // it — the cap is not merely decorative.
    let err = op(base(1.0e110)).expect_err("the measured NaN threshold must be rejected");
    assert!(err.contains("dt must be <= "), "unexpected error: {err}");
}

/// No accepted (dt, noise) combination may yield a `null` coordinate. This sweeps the
/// accepted envelope rather than trusting the single measured threshold — the same shape
/// as the sibling de Rham sweep.
#[test]
fn kalman_never_emits_null_estimates() {
    let series: Vec<f64> = (0..64)
        .map(|i| (i as f64).sin() * 10.0 + i as f64)
        .collect();
    for dt in [1e-6, 1e-3, 1.0, 10.0, 1e3, 1e6] {
        for q in [1e-6, 0.01, 1.0, 1e6] {
            for r_noise in [1e-6, 0.5, 1.0, 1e6] {
                let resp = op(json!({
                    "operation": "smooth_1d",
                    "series": series,
                    "process_noise": q,
                    "measurement_noise": r_noise,
                    "dt": dt
                }))
                .unwrap_or_else(|e| panic!("dt={dt} q={q} r={r_noise} should succeed: {e}"));

                for key in ["position", "velocity"] {
                    let arr = resp[key].as_array().expect("array");
                    assert!(
                        arr.iter().all(|x| x.as_f64().is_some_and(f64::is_finite)),
                        "dt={dt} q={q} r={r_noise} produced a null/non-finite {key}"
                    );
                }
            }
        }
    }
}

/// Degenerate inputs are rejected with a reason, matching the SQL surface's domain.
#[test]
fn kalman_rejects_degenerate_input() {
    let ok_series = vec![1.0, 2.0, 3.0];

    // Non-positive covariances — the same domain `ix_kalman_smooth` enforces.
    for (field, bad) in [
        ("process_noise", 0.0),
        ("process_noise", -1.0),
        ("measurement_noise", 0.0),
        ("measurement_noise", -1.0),
    ] {
        let mut p = json!({
            "operation": "smooth_1d",
            "series": ok_series,
            "process_noise": 0.01,
            "measurement_noise": 1.0
        });
        p[field] = json!(bad);
        let err = op(p).expect_err("a non-positive covariance must be rejected");
        assert!(
            err.contains(&format!("{field} must be a finite number > 0")),
            "unexpected error for {field}={bad}: {err}"
        );
    }

    // dt <= 0.
    let err = op(json!({
        "operation": "smooth_1d",
        "series": ok_series,
        "process_noise": 0.01,
        "measurement_noise": 1.0,
        "dt": 0.0
    }))
    .expect_err("dt = 0 must be rejected");
    assert!(err.contains("dt must be a finite number > 0"), "got: {err}");

    // An empty series has no estimate to return.
    let err = op(json!({
        "operation": "smooth_1d",
        "series": Vec::<f64>::new(),
        "process_noise": 0.01,
        "measurement_noise": 1.0
    }))
    .expect_err("an empty series must be rejected");
    assert!(err.contains("at least one sample"), "got: {err}");

    // Unknown operation names must not silently fall through to smoothing.
    let err = op(json!({
        "operation": "smooth_nd",
        "series": ok_series,
        "process_noise": 0.01,
        "measurement_noise": 1.0
    }))
    .expect_err("an unknown operation must be rejected");
    assert!(err.contains("Unknown kalman operation"), "got: {err}");
}

/// The tool is reachable under its registered name with its schema attached — the actual
/// subject of ix#193. Before this change `ix_kalman` was not in `ToolRegistry::list()` at
/// all, so an agent could not call the filter however correct `ix-signal` was.
#[test]
fn kalman_is_exposed_on_the_mcp_surface() {
    let listing = ToolRegistry::new().list();
    let tools = listing["tools"].as_array().expect("tools array");

    let tool = tools
        .iter()
        .find(|t| t["name"] == json!("ix_kalman"))
        .expect("ix_kalman must be reachable through ToolRegistry::list()");

    let schema = serde_json::to_string(&tool["inputSchema"]).expect("schema serializes");
    for key in [
        "operation",
        "series",
        "process_noise",
        "measurement_noise",
        "dt",
    ] {
        assert!(
            schema.contains(key),
            "input schema must document '{key}': {schema}"
        );
    }
}
