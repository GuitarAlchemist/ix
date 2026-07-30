//! `ix-baml` — the boundary between IXQL's dynamic dataflow and BAML's typed
//! LLM functions.
//!
//! IXQL values are [`serde_json::Value`] throughout, and BAML's generated Rust
//! client speaks typed structs that are themselves `serde`-derived. That makes
//! the seam a pure JSON round-trip:
//!
//! ```text
//! Value --serde_json::from_value--> BamlInput --client--> BamlOutput --to_value--> Value
//! ```
//!
//! This crate owns only the *dynamic* half: the [`BamlOperation`] trait, the
//! name → operation [`BamlRegistry`] the evaluator dispatches through, and
//! deterministic stand-ins ([`StaticResponse`], [`FnOperation`]) so a pipeline
//! containing LLM steps runs offline in tests.
//!
//! # Why the generated client is not linked here yet
//!
//! The design spec (`Demerzel/ixql_executor_design_spec.md` §5) proposes
//! `#[path = "…/Demerzel/clients/rust/baml_client/mod.rs"] pub mod baml_client;`.
//! That is deliberately **not** done:
//!
//! - the generated client currently exists only on Demerzel's unmerged
//!   `feat/baml-adoption` branch (PR #908), so it is not a stable artifact yet;
//! - a `#[path]` escape into a sibling clone makes `cargo check --workspace`
//!   depend on a checkout that CI and most contributors do not have — the
//!   workspace would fail to build for everyone but the author.
//!
//! When #908 lands, the client is generated *into this crate* by
//! `baml generate` (a vendored, committed `src/generated/` tree) and registered
//! as `BamlOperation` impls. Nothing outside this crate changes: the evaluator
//! already dispatches through the trait.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

/// Why a BAML step could not produce a value.
#[derive(Debug, thiserror::Error)]
pub enum BamlError {
    /// The pipeline named a function that no registered operation provides.
    #[error("no BAML function registered under `{0}`")]
    UnknownFunction(String),

    /// The dynamic input did not fit the function's declared input type.
    #[error("input for `{function}` did not match the BAML schema: {source}")]
    InputShape {
        function: String,
        #[source]
        source: serde_json::Error,
    },

    /// The provider, or the stand-in standing for it, failed.
    #[error("BAML function `{function}` failed: {message}")]
    Invocation { function: String, message: String },
}

/// One BAML function, seen dynamically.
///
/// Implementors deserialize `input` into the generated input type, call the
/// generated client, and serialize the result back. Errors are reported rather
/// than panicked: a failed LLM step must be able to fail its pipeline cleanly.
pub trait BamlOperation: Send + Sync {
    /// The BAML function name as written in `baml_src` — this is the key the
    /// IXQL evaluator looks up.
    fn name(&self) -> &str;

    /// Run the function over a dynamic input.
    fn invoke(&self, input: &Value) -> Result<Value, BamlError>;
}

/// The set of BAML functions an IXQL run may call.
///
/// Empty by default: a pipeline that reaches a BAML step without a registered
/// operation fails with [`BamlError::UnknownFunction`] rather than silently
/// evaluating to null. An LLM step that quietly no-ops is the worst outcome —
/// downstream governance would then validate a hollow value.
#[derive(Default, Clone)]
pub struct BamlRegistry {
    ops: BTreeMap<String, Arc<dyn BamlOperation>>,
}

impl BamlRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an operation under its own [`BamlOperation::name`].
    pub fn register(&mut self, op: Arc<dyn BamlOperation>) -> &mut Self {
        self.ops.insert(op.name().to_string(), op);
        self
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn BamlOperation>> {
        self.ops.get(name)
    }

    /// Dispatch by name, or fail loudly if nothing is registered.
    pub fn invoke(&self, name: &str, input: &Value) -> Result<Value, BamlError> {
        self.ops
            .get(name)
            .ok_or_else(|| BamlError::UnknownFunction(name.to_string()))?
            .invoke(input)
    }

    /// Function names currently registered, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.ops.keys().map(String::as_str).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl std::fmt::Debug for BamlRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BamlRegistry")
            .field("functions", &self.names())
            .finish()
    }
}

/// A function that always returns the same value, ignoring its input.
///
/// This is the offline fixture the tracer-bullet test uses: it exercises the
/// full dispatch and JSON round-trip while keeping the run deterministic and
/// provider-free.
pub struct StaticResponse {
    name: String,
    response: Value,
}

impl StaticResponse {
    pub fn new(name: impl Into<String>, response: Value) -> Self {
        Self {
            name: name.into(),
            response,
        }
    }
}

impl BamlOperation for StaticResponse {
    fn name(&self) -> &str {
        &self.name
    }

    fn invoke(&self, _input: &Value) -> Result<Value, BamlError> {
        Ok(self.response.clone())
    }
}

/// A function backed by a closure — for tests that need the response to depend
/// on the input (e.g. asserting the evaluator passed the right payload).
pub struct FnOperation<F> {
    name: String,
    func: F,
}

impl<F> FnOperation<F>
where
    F: Fn(&Value) -> Result<Value, String> + Send + Sync,
{
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

impl<F> BamlOperation for FnOperation<F>
where
    F: Fn(&Value) -> Result<Value, String> + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn invoke(&self, input: &Value) -> Result<Value, BamlError> {
        (self.func)(input).map_err(|message| BamlError::Invocation {
            function: self.name.clone(),
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unknown_function_fails_rather_than_returning_null() {
        let registry = BamlRegistry::new();
        let err = registry
            .invoke("EvaluateSignalSwarm", &json!({}))
            .unwrap_err();
        assert!(matches!(err, BamlError::UnknownFunction(name) if name == "EvaluateSignalSwarm"));
    }

    #[test]
    fn static_response_round_trips_through_the_registry() {
        let mut registry = BamlRegistry::new();
        registry.register(Arc::new(StaticResponse::new(
            "EvaluateSignalSwarm",
            json!({"vote": "TrueVal", "confidence": 0.9}),
        )));

        let out = registry
            .invoke("EvaluateSignalSwarm", &json!({"thd": 0.02}))
            .unwrap();
        assert_eq!(out["vote"], json!("TrueVal"));
        assert_eq!(registry.names(), vec!["EvaluateSignalSwarm"]);
    }

    #[test]
    fn closure_operation_sees_the_input_the_pipeline_passed() {
        let mut registry = BamlRegistry::new();
        registry.register(Arc::new(FnOperation::new("Echo", |input: &Value| {
            Ok(json!({"saw": input.clone()}))
        })));

        let out = registry.invoke("Echo", &json!({"k": 1})).unwrap();
        assert_eq!(out["saw"]["k"], json!(1));
    }

    #[test]
    fn invocation_failure_names_the_function() {
        let mut registry = BamlRegistry::new();
        registry.register(Arc::new(FnOperation::new("Fails", |_: &Value| {
            Err("provider timeout".to_string())
        })));

        let err = registry.invoke("Fails", &json!(null)).unwrap_err();
        assert!(err.to_string().contains("Fails"));
        assert!(err.to_string().contains("provider timeout"));
    }
}
