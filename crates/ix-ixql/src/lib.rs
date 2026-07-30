//! `ix-ixql` — parser and evaluator for IXQL, the language Demerzel writes its
//! governance pipelines in (`Demerzel/pipelines/*.ixql`).
//!
//! A pipeline is a sequence of bindings and `→` steps over dynamic JSON:
//!
//! ```text
//! trigger_context <- ix.io.read("state/qa-architect/trigger.json")
//!   → default({ kind: "scheduled_sweep", repo: "guitar-alchemist/ga" })
//!
//! verdict <- { schema_version: 1, target: trigger_context, verdict: "informational" }
//!
//! ix.io.write("state/quality/verdicts/{{verdict.target.repo}}/v.json", verdict)
//!   → tars.validate(check: "verdict.schema_version == 1",
//!                   reject_message: "Schema mismatch — refusing to persist.")
//!   → compound:
//!       harvest verdict.followups
//!       log qa_architect_cycle to "state/evolution/"
//! ```
//!
//! # Where correctness comes from
//!
//! Two gates, at two different moments:
//!
//! - **In flight** — LLM steps cross the [`ix_baml`] seam, where BAML's typed
//!   functions decide whether a model's answer has the shape it claimed.
//! - **At rest** — every `ix.io.write` passes through [`SchemaGate`], so a
//!   value that violates its canonical JSON Schema stops the run instead of
//!   landing on disk.
//!
//! # Running one
//!
//! ```
//! use std::sync::Arc;
//! use ix_ixql::{Executor, MemoryHost};
//!
//! let host = Arc::new(MemoryHost::frozen());
//! let executor = Executor::new(host);
//! let outcome = executor
//!     .run_source("stamp <- now_utc(\"yyyy-MM-dd\")")
//!     .unwrap();
//!
//! assert_eq!("2026-01-02", outcome.binding("stamp").unwrap());
//! ```
//!
//! Everything effectful is injected — I/O, the clock, the schemas, the BAML
//! functions — so a pipeline is fully executable offline, which is how the
//! fixture tests run Demerzel's real `qa-architect-cycle.ixql` with no
//! provider and no filesystem.
//!
//! # Scope
//!
//! This covers the binding/record dialect the governance pipelines use. The
//! ML-pipeline dialect in `Demerzel/tree-sitter-ixql/grammar.js` (`csv(…) →
//! train(…)`) is a different surface and is not parsed here; see
//! [`parser`] for the distinction.

pub mod ast;
pub mod eval;
pub mod host;
pub mod lexer;
pub mod parser;
pub mod schema;

pub use ast::{Block, CompoundOp, Expr, Literal, PipeStep, Statement};
pub use eval::{CompoundRecord, EvalError, Executor, RunError, RunOutcome, WriteRecord};
pub use host::{FsHost, Host, HostError, MemoryHost};
pub use parser::{parse_expression, parse_program, ParseError};
pub use schema::{SchemaGate, SchemaViolation};
