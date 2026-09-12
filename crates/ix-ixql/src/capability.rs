//! The port a pipeline reaches its federation peers through.
//!
//! `tars.research(…)`, `alert(…)`, `ga.query(…)` name capabilities that live in
//! other processes and other repositories. The evaluator must not know how any
//! of them is reached — MCP, an F# computation expression in tars, a native IX
//! function — which is the distinction Demerzel's grammar already records after
//! the fact as `runtime_binding ::= mcp_binding | ce_binding | native_binding`.
//! A [`Capability`] is where that binding is chosen *before* the call instead:
//! the host registers an adapter under the name the pipeline uses, and the
//! evaluator dispatches to it without a `match` on peer names.
//!
//! # What stays in the evaluator
//!
//! The language's own primitives — `ix.io.read`, `ix.io.write`, `default`,
//! `tars.validate`, `now_utc` — are not capabilities. `ix.io.write` in
//! particular must pass the schema gate, and letting an adapter be registered
//! under that name would let it route around the gate. Registering over a
//! built-in, or under `baml.`, is therefore refused.
//!
//! # Verdicts
//!
//! A capability may attach a [`Verdict`] to what it returns. That is the only
//! way a verdict enters a run, and it is what a `→ when T >= 0.7:` step reads.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value;

pub use crate::ast::Verdict;

/// Names the evaluator implements itself. Nothing may be registered over them.
pub(crate) const BUILTINS: &[&str] = &[
    "now_utc_iso8601",
    "now_utc",
    "ix.io.read",
    "ix.io.write",
    "default",
    "tars.validate",
];

/// The arguments a capability is called with, already evaluated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CallArgs {
    /// The value flowing in, when the call is a `→` step.
    pub piped: Option<Value>,
    pub positional: Vec<Value>,
    pub named: BTreeMap<String, Value>,
}

/// What a capability returned.
#[derive(Debug, Clone, PartialEq)]
pub struct Produced {
    pub value: Value,
    pub verdict: Option<Verdict>,
}

impl Produced {
    /// A value with no verdict attached.
    pub fn plain(value: Value) -> Self {
        Self {
            value,
            verdict: None,
        }
    }

    /// A value carrying a verdict.
    pub fn judged(value: Value, verdict: Verdict) -> Self {
        Self {
            value,
            verdict: Some(verdict),
        }
    }
}

/// An adapter for one peer operation.
pub trait Capability: Send + Sync {
    /// `Err` carries a message; the evaluator attributes it to the call.
    fn call(&self, args: CallArgs) -> Result<Produced, String>;
}

impl<F> Capability for F
where
    F: Fn(CallArgs) -> Result<Produced, String> + Send + Sync,
{
    fn call(&self, args: CallArgs) -> Result<Produced, String> {
        self(args)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistrationError {
    #[error(
        "`{0}` is implemented by the evaluator itself; an adapter registered under it \
         could bypass the schema gate or the validation it performs"
    )]
    Builtin(String),
    #[error("`{0}` is under `baml.`, which the BAML registry owns")]
    BamlNamespace(String),
    #[error("`{0}` is already registered")]
    Duplicate(String),
}

/// Name → adapter.
#[derive(Default, Clone)]
pub struct Capabilities {
    adapters: BTreeMap<String, Arc<dyn Capability>>,
}

impl Capabilities {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `adapter` under the dotted name pipelines call it by.
    pub fn register(
        &mut self,
        name: &str,
        adapter: impl Capability + 'static,
    ) -> Result<(), RegistrationError> {
        if BUILTINS.contains(&name) {
            return Err(RegistrationError::Builtin(name.to_string()));
        }
        if name.starts_with("baml.") {
            return Err(RegistrationError::BamlNamespace(name.to_string()));
        }
        if self.adapters.contains_key(name) {
            return Err(RegistrationError::Duplicate(name.to_string()));
        }
        self.adapters.insert(name.to_string(), Arc::new(adapter));
        Ok(())
    }

    pub(crate) fn get(&self, name: &str) -> Option<&Arc<dyn Capability>> {
        self.adapters.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo(args: CallArgs) -> Result<Produced, String> {
        Ok(Produced::plain(args.piped.unwrap_or(Value::Null)))
    }

    #[test]
    fn a_builtin_name_cannot_be_taken_over() {
        let mut caps = Capabilities::new();
        assert_eq!(
            caps.register("ix.io.write", echo),
            Err(RegistrationError::Builtin("ix.io.write".into()))
        );
    }

    #[test]
    fn the_baml_namespace_and_duplicates_are_refused() {
        let mut caps = Capabilities::new();
        assert!(matches!(
            caps.register("baml.Summarise", echo),
            Err(RegistrationError::BamlNamespace(_))
        ));
        caps.register("tars.research", echo).unwrap();
        assert_eq!(
            caps.register("tars.research", echo),
            Err(RegistrationError::Duplicate("tars.research".into()))
        );
    }
}
