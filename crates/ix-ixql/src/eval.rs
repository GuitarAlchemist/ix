//! The IXQL evaluator.
//!
//! Values are [`serde_json::Value`] end to end: that is what makes the BAML
//! seam a plain `from_value`/`to_value` round-trip and lets the JSON-Schema
//! gate inspect any value a pipeline is about to persist without a conversion
//! layer in between.
//!
//! Everything effectful is reached through three injected pieces — the
//! [`Host`](crate::Host) (I/O + clock), the [`SchemaGate`] (at-rest
//! validation), and the [`BamlRegistry`] (LLM steps) — so a whole pipeline
//! runs offline and deterministically in tests.

use std::collections::BTreeMap;
use std::sync::Arc;

use ix_baml::{BamlError, BamlRegistry};
use serde_json::{Map, Value};

use crate::ast::{BinaryOp, Block, CompoundOp, Expr, Literal, PipeStep, Statement, UnaryOp};
use crate::host::{dotnet_format_to_chrono, Host, HostError};
use crate::parser::{parse_expression, parse_program, ParseError};
use crate::schema::{SchemaGate, SchemaViolation};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("unknown name `{0}`")]
    UnknownVariable(String),

    #[error("`{field}` is not a field of {found}")]
    NoSuchField { field: String, found: &'static str },

    #[error("{context} expected {expected}, found {found}")]
    Type {
        context: String,
        expected: &'static str,
        found: &'static str,
    },

    #[error("record key `{0}` given twice")]
    DuplicateKey(String),

    #[error("`{0}` is not a function this executor provides")]
    UnknownFunction(String),

    #[error("`{function}` takes {expected} positional argument(s), got {found}")]
    Arity {
        function: &'static str,
        expected: usize,
        found: usize,
    },

    #[error("`{function}` requires the named argument `{name}`")]
    MissingArgument {
        function: &'static str,
        name: &'static str,
    },

    #[error("`{0}` is only meaningful as a `→` step — it consumes the piped value")]
    NeedsPipedInput(&'static str),

    #[error("validation failed: {message} (check: `{check}`)")]
    ValidationFailed { message: String, check: String },

    #[error("`{{{{…}}}}` interpolation cannot render a {found}")]
    NotInterpolable { found: &'static str },

    #[error(transparent)]
    Host(#[from] HostError),

    #[error(transparent)]
    Schema(#[from] SchemaViolation),

    #[error(transparent)]
    Baml(#[from] BamlError),

    #[error("in a `check:` predicate — {0}")]
    Predicate(#[from] ParseError),
}

/// Parse-then-run failure, for [`Executor::run_source`].
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Eval(#[from] EvalError),
}

/// One `ix.io.write` that got past the schema gate.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteRecord {
    pub path: String,
    pub value: Value,
}

/// A compound-phase effect, with its operands already evaluated.
///
/// The AST's [`CompoundOp`] holds unevaluated expressions; the run needs the
/// values, so the two are deliberately different types.
#[derive(Debug, Clone, PartialEq)]
pub enum CompoundRecord {
    Harvested {
        value: Value,
    },
    Promoted {
        id: String,
    },
    Logged {
        id: String,
        destination: String,
        value: Value,
    },
    Taught {
        id: String,
        target: String,
    },
}

/// What a completed run produced.
#[derive(Debug, Clone, Default)]
pub struct RunOutcome {
    /// Every binding, in the state it ended in.
    pub env: BTreeMap<String, Value>,
    /// Writes in the order they happened.
    pub writes: Vec<WriteRecord>,
    /// Compound-phase effects in the order they were declared.
    pub compound: Vec<CompoundRecord>,
}

impl RunOutcome {
    pub fn binding(&self, name: &str) -> Option<&Value> {
        self.env.get(name)
    }
}

/// Runs IXQL programs.
pub struct Executor {
    host: Arc<dyn Host>,
    gate: SchemaGate,
    baml: BamlRegistry,
}

impl Executor {
    pub fn new(host: Arc<dyn Host>) -> Self {
        Self {
            host,
            gate: SchemaGate::new(),
            baml: BamlRegistry::new(),
        }
    }

    /// The at-rest gate — register the schemas this run must satisfy.
    pub fn schema_gate(&mut self) -> &mut SchemaGate {
        &mut self.gate
    }

    /// The BAML functions this run may call.
    pub fn baml_registry(&mut self) -> &mut BamlRegistry {
        &mut self.baml
    }

    pub fn run_source(&self, src: &str) -> Result<RunOutcome, RunError> {
        Ok(self.run(&parse_program(src)?)?)
    }

    pub fn run(&self, program: &Block) -> Result<RunOutcome, EvalError> {
        let mut outcome = RunOutcome::default();
        for statement in program {
            self.exec_statement(statement, &mut outcome)?;
        }
        Ok(outcome)
    }

    fn exec_statement(
        &self,
        statement: &Statement,
        outcome: &mut RunOutcome,
    ) -> Result<(), EvalError> {
        match statement {
            Statement::Assign(name, expr) => {
                let value = self.eval(expr, outcome)?;
                outcome.env.insert(name.clone(), value);
            }
            Statement::Do(expr) => {
                self.eval(expr, outcome)?;
            }
            Statement::When(condition, body) => {
                if self.expect_bool(self.eval(condition, outcome)?, "a `when` condition")? {
                    for inner in body.iter() {
                        self.exec_statement(inner, outcome)?;
                    }
                }
            }
        }
        Ok(())
    }

    // ---- expressions ---------------------------------------------------

    fn eval(&self, expr: &Expr, outcome: &mut RunOutcome) -> Result<Value, EvalError> {
        match expr {
            Expr::Lit(lit) => Ok(literal_value(lit)),

            Expr::Var(name) => outcome
                .env
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::UnknownVariable(name.clone())),

            Expr::Member(base, field) => {
                let value = self.eval(base, outcome)?;
                match value {
                    Value::Object(map) => {
                        map.get(field)
                            .cloned()
                            .ok_or_else(|| EvalError::NoSuchField {
                                field: field.clone(),
                                found: "that record",
                            })
                    }
                    other => Err(EvalError::NoSuchField {
                        field: field.clone(),
                        found: type_name(&other),
                    }),
                }
            }

            Expr::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.eval(item, outcome)?);
                }
                Ok(Value::Array(out))
            }

            Expr::Record(fields) => {
                let mut map = Map::new();
                for (key, value_expr) in fields {
                    let value = self.eval(value_expr, outcome)?;
                    if map.insert(key.clone(), value).is_some() {
                        return Err(EvalError::DuplicateKey(key.clone()));
                    }
                }
                Ok(Value::Object(map))
            }

            Expr::Interpolation(parts) => {
                let mut out = String::new();
                for part in parts {
                    let value = self.eval(part, outcome)?;
                    out.push_str(&render(&value)?);
                }
                Ok(Value::String(out))
            }

            Expr::BinOp(left, op, right) => self.eval_binop(left, *op, right, outcome),

            Expr::Unary(op, operand) => {
                let value = self.eval(operand, outcome)?;
                Ok(match op {
                    UnaryOp::IsEmpty => Value::Bool(is_empty(&value)),
                    UnaryOp::IsNotEmpty => Value::Bool(!is_empty(&value)),
                    UnaryOp::Not => Value::Bool(!self.expect_bool(value, "`!`")?),
                })
            }

            Expr::Call {
                target,
                positional,
                named,
            } => {
                let name = callee_path(target)?;
                let args = self.eval_args(positional, named, outcome)?;
                self.call(&name, args, None, outcome)
            }

            Expr::Pipeline(head, steps) => {
                let mut value = self.eval(head, outcome)?;
                for step in steps {
                    value = self.apply_step(step, value, outcome)?;
                }
                Ok(value)
            }
        }
    }

    fn eval_binop(
        &self,
        left: &Expr,
        op: BinaryOp,
        right: &Expr,
        outcome: &mut RunOutcome,
    ) -> Result<Value, EvalError> {
        // `&&` / `||` short-circuit, so the right side is only evaluated when
        // it can change the answer.
        if matches!(op, BinaryOp::And | BinaryOp::Or) {
            let lhs = self.expect_bool(self.eval(left, outcome)?, op.as_str())?;
            return Ok(Value::Bool(match op {
                BinaryOp::And if !lhs => false,
                BinaryOp::Or if lhs => true,
                _ => self.expect_bool(self.eval(right, outcome)?, op.as_str())?,
            }));
        }

        let lhs = self.eval(left, outcome)?;
        let rhs = self.eval(right, outcome)?;

        Ok(Value::Bool(match op {
            BinaryOp::Eq => values_equal(&lhs, &rhs),
            BinaryOp::Neq => !values_equal(&lhs, &rhs),
            BinaryOp::Gt | BinaryOp::Gte | BinaryOp::Lt | BinaryOp::Lte => {
                let ordering = compare(&lhs, &rhs, op.as_str())?;
                match op {
                    BinaryOp::Gt => ordering.is_gt(),
                    BinaryOp::Gte => ordering.is_ge(),
                    BinaryOp::Lt => ordering.is_lt(),
                    _ => ordering.is_le(),
                }
            }
            BinaryOp::In => contains(&rhs, &lhs)?,
            BinaryOp::NotIn => !contains(&rhs, &lhs)?,
            BinaryOp::And | BinaryOp::Or => unreachable!("handled above"),
        }))
    }

    fn eval_args(
        &self,
        positional: &[Expr],
        named: &BTreeMap<String, Expr>,
        outcome: &mut RunOutcome,
    ) -> Result<Args, EvalError> {
        let mut args = Args::default();
        for expr in positional {
            args.positional.push(self.eval(expr, outcome)?);
        }
        for (key, expr) in named {
            let value = self.eval(expr, outcome)?;
            args.named.insert(key.clone(), value);
        }
        Ok(args)
    }

    // ---- pipeline steps ------------------------------------------------

    fn apply_step(
        &self,
        step: &PipeStep,
        input: Value,
        outcome: &mut RunOutcome,
    ) -> Result<Value, EvalError> {
        match step {
            PipeStep::CallStep {
                target,
                positional,
                named,
            } => {
                let name = callee_path(target)?;
                let args = self.eval_args(positional, named, outcome)?;
                self.call(&name, args, Some(input), outcome)
            }
            PipeStep::Compound(ops) => {
                for op in ops {
                    self.apply_compound(op, &input, outcome)?;
                }
                // The compound phase observes; it does not rewrite the value.
                Ok(input)
            }
        }
    }

    fn apply_compound(
        &self,
        op: &CompoundOp,
        input: &Value,
        outcome: &mut RunOutcome,
    ) -> Result<(), EvalError> {
        let record = match op {
            CompoundOp::Harvest(expr) => CompoundRecord::Harvested {
                value: self.eval(expr, outcome)?,
            },
            CompoundOp::Promote { id, condition } => {
                if let Some(condition) = condition {
                    let value = self.eval(condition, outcome)?;
                    if !self.expect_bool(value, "a `promote … when` condition")? {
                        return Ok(());
                    }
                }
                CompoundRecord::Promoted { id: id.clone() }
            }
            CompoundOp::Log { id, destination } => {
                let destination = self.eval(destination, outcome)?;
                CompoundRecord::Logged {
                    id: id.clone(),
                    destination: expect_string(destination, "a `log … to` destination")?,
                    value: input.clone(),
                }
            }
            CompoundOp::Teach { id, target } => CompoundRecord::Taught {
                id: id.clone(),
                target: target.clone(),
            },
        };
        outcome.compound.push(record);
        Ok(())
    }

    // ---- host functions ------------------------------------------------

    fn call(
        &self,
        name: &str,
        args: Args,
        piped: Option<Value>,
        outcome: &mut RunOutcome,
    ) -> Result<Value, EvalError> {
        match name {
            "now_utc_iso8601" => {
                args.expect_positional("now_utc_iso8601", 0)?;
                Ok(Value::String(
                    self.host
                        .now()
                        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                ))
            }

            "now_utc" => {
                args.expect_positional("now_utc", 1)?;
                let fmt = expect_string(args.positional[0].clone(), "`now_utc` format")?;
                Ok(Value::String(
                    self.host
                        .now()
                        .format(&dotnet_format_to_chrono(&fmt))
                        .to_string(),
                ))
            }

            "ix.io.read" => {
                args.expect_positional("ix.io.read", 1)?;
                let path = expect_string(args.positional[0].clone(), "`ix.io.read` path")?;
                // Absence becomes null so `→ default(…)` can supply a value.
                Ok(self.host.read(&path)?.unwrap_or(Value::Null))
            }

            "ix.io.write" => {
                args.expect_positional("ix.io.write", 2)?;
                let path = expect_string(args.positional[0].clone(), "`ix.io.write` path")?;
                let value = args.positional[1].clone();
                // Gate first: a value that fails its schema must never reach
                // the host, not even to be deleted afterwards.
                self.gate.check(&path, &value)?;
                self.host.write(&path, &value)?;
                outcome.writes.push(WriteRecord {
                    path,
                    value: value.clone(),
                });
                Ok(value)
            }

            "default" => {
                let input = piped.ok_or(EvalError::NeedsPipedInput("default"))?;
                args.expect_positional("default", 1)?;
                Ok(if input.is_null() {
                    args.positional[0].clone()
                } else {
                    input
                })
            }

            "tars.validate" => {
                let input = piped.ok_or(EvalError::NeedsPipedInput("tars.validate"))?;
                let check = args.named.get("check").cloned().ok_or({
                    EvalError::MissingArgument {
                        function: "tars.validate",
                        name: "check",
                    }
                })?;
                let check = expect_string(check, "`tars.validate` check")?;

                // The predicate is authored as a string, so it is parsed and
                // evaluated here against the run's bindings.
                let predicate = parse_expression(&check)?;
                let holds = self.expect_bool(
                    self.eval(&predicate, outcome)?,
                    "a `tars.validate` predicate",
                )?;
                if holds {
                    return Ok(input);
                }

                let message = args
                    .named
                    .get("reject_message")
                    .and_then(Value::as_str)
                    .unwrap_or("predicate did not hold")
                    .to_string();
                Err(EvalError::ValidationFailed { message, check })
            }

            _ => {
                if let Some(function) = name.strip_prefix("baml.") {
                    // The piped value is the input; a bare call may pass it
                    // positionally instead.
                    let input = match (piped, args.positional.first()) {
                        (Some(value), _) => value,
                        (None, Some(value)) => value.clone(),
                        (None, None) => Value::Null,
                    };
                    return Ok(self.baml.invoke(function, &input)?);
                }
                Err(EvalError::UnknownFunction(name.to_string()))
            }
        }
    }

    fn expect_bool(&self, value: Value, context: &str) -> Result<bool, EvalError> {
        match value {
            Value::Bool(b) => Ok(b),
            other => Err(EvalError::Type {
                context: context.to_string(),
                expected: "a boolean",
                found: type_name(&other),
            }),
        }
    }
}

#[derive(Default)]
struct Args {
    positional: Vec<Value>,
    named: BTreeMap<String, Value>,
}

impl Args {
    fn expect_positional(&self, function: &'static str, expected: usize) -> Result<(), EvalError> {
        if self.positional.len() == expected {
            Ok(())
        } else {
            Err(EvalError::Arity {
                function,
                expected,
                found: self.positional.len(),
            })
        }
    }
}

/// Flatten a callee expression back into its dotted spelling (`ix.io.read`).
///
/// Only names and member chains can be called; anything else is a value, and
/// calling a value is a program error rather than a lookup miss.
fn callee_path(target: &Expr) -> Result<String, EvalError> {
    match target {
        Expr::Var(name) => Ok(name.clone()),
        Expr::Member(base, field) => Ok(format!("{}.{field}", callee_path(base)?)),
        _ => Err(EvalError::UnknownFunction(
            "a computed expression".to_string(),
        )),
    }
}

fn literal_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Bool(b) => Value::Bool(*b),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Number(n) => number_value(*n),
    }
}

/// Numbers are parsed as `f64`, but an integer literal must serialize back as
/// an integer — `"schema_version": 1.0` would be a gratuitous difference from
/// the JSON every other producer of these artifacts writes.
fn number_value(n: f64) -> Value {
    if n.is_finite() && n.fract() == 0.0 && n.abs() <= i64::MAX as f64 {
        Value::from(n as i64)
    } else {
        serde_json::Number::from_f64(n).map_or(Value::Null, Value::Number)
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "a record",
    }
}

fn expect_string(value: Value, context: &str) -> Result<String, EvalError> {
    match value {
        Value::String(s) => Ok(s),
        other => Err(EvalError::Type {
            context: context.to_string(),
            expected: "a string",
            found: type_name(&other),
        }),
    }
}

/// How a value reads inside `"{{…}}"`.
///
/// Containers are refused: interpolated values become path segments and ids,
/// and `[object]` in a filename is a bug that would only surface much later.
fn render(value: &Value) -> Result<String, EvalError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(EvalError::NotInterpolable {
            found: type_name(other),
        }),
    }
}

/// Equality that does not care whether a number arrived as `1` or `1.0`.
///
/// A literal in a pipeline is parsed as `f64`; the same field read back from
/// disk is a JSON integer. Without this, `verdict.schema_version == 1` would
/// depend on where the value came from.
fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => a == b,
        },
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|other| values_equal(v, other)))
        }
        _ => left == right,
    }
}

fn compare(left: &Value, right: &Value, op: &str) -> Result<std::cmp::Ordering, EvalError> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (
                a.as_f64().unwrap_or(f64::NAN),
                b.as_f64().unwrap_or(f64::NAN),
            );
            a.partial_cmp(&b).ok_or_else(|| EvalError::Type {
                context: format!("`{op}`"),
                expected: "comparable numbers",
                found: "NaN",
            })
        }
        (Value::String(a), Value::String(b)) => Ok(a.cmp(b)),
        (other, _) => Err(EvalError::Type {
            context: format!("`{op}`"),
            expected: "two numbers or two strings",
            found: type_name(other),
        }),
    }
}

fn contains(haystack: &Value, needle: &Value) -> Result<bool, EvalError> {
    match haystack {
        Value::Array(items) => Ok(items.iter().any(|item| values_equal(item, needle))),
        Value::Object(map) => match needle {
            Value::String(key) => Ok(map.contains_key(key)),
            other => Err(EvalError::Type {
                context: "`in` over a record".to_string(),
                expected: "a string key",
                found: type_name(other),
            }),
        },
        Value::String(text) => match needle {
            Value::String(part) => Ok(text.contains(part.as_str())),
            other => Err(EvalError::Type {
                context: "`in` over a string".to_string(),
                expected: "a string",
                found: type_name(other),
            }),
        },
        other => Err(EvalError::Type {
            context: "`in`".to_string(),
            expected: "an array, record or string on the right",
            found: type_name(other),
        }),
    }
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(map) => map.is_empty(),
        Value::String(text) => text.is_empty(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::MemoryHost;
    use ix_baml::StaticResponse;
    use serde_json::json;

    fn executor() -> (Executor, Arc<MemoryHost>) {
        let host = Arc::new(MemoryHost::frozen());
        (Executor::new(host.clone()), host)
    }

    #[test]
    fn bindings_records_and_member_access() {
        let (exec, _) = executor();
        let out = exec
            .run_source("a <- { x: 1, y: [2, 3] }\nb <- a.y")
            .unwrap();
        assert_eq!(&json!({"x": 1, "y": [2, 3]}), out.binding("a").unwrap());
        assert_eq!(&json!([2, 3]), out.binding("b").unwrap());
    }

    #[test]
    fn integer_literals_stay_integers_in_the_emitted_json() {
        let (exec, _) = executor();
        let out = exec.run_source("a <- { schema_version: 1 }").unwrap();
        assert_eq!(
            "{\"schema_version\":1}",
            serde_json::to_string(out.binding("a").unwrap()).unwrap()
        );
    }

    #[test]
    fn default_fires_only_when_the_read_found_nothing() {
        let (exec, host) = executor();
        let src = "t <- ix.io.read(\"state/t.json\")\n  → default({ kind: \"sweep\" })";
        assert_eq!(
            &json!({"kind": "sweep"}),
            exec.run_source(src).unwrap().binding("t").unwrap()
        );

        host.seed("state/t.json", json!({"kind": "pull_request"}));
        assert_eq!(
            &json!({"kind": "pull_request"}),
            exec.run_source(src).unwrap().binding("t").unwrap()
        );
    }

    #[test]
    fn interpolation_builds_ids_from_bindings() {
        let (exec, _) = executor();
        let out = exec
            .run_source("k <- \"sweep\"\nid <- \"{{k}}-skeleton\"")
            .unwrap();
        assert_eq!(&json!("sweep-skeleton"), out.binding("id").unwrap());
    }

    #[test]
    fn interpolating_a_record_is_refused_rather_than_stringified() {
        let (exec, _) = executor();
        let err = exec
            .run_source("a <- { x: 1 }\np <- \"path/{{a}}.json\"")
            .unwrap_err();
        assert!(err.to_string().contains("interpolation"));
    }

    #[test]
    fn a_number_compares_equal_whether_it_came_from_disk_or_source() {
        let (exec, host) = executor();
        // Seeded as a JSON integer; the literal is parsed as f64.
        host.seed("state/v.json", json!({"schema_version": 1}));
        let out = exec
            .run_source("v <- ix.io.read(\"state/v.json\")\nok <- v.schema_version == 1")
            .unwrap();
        assert_eq!(&json!(true), out.binding("ok").unwrap());
    }

    #[test]
    fn tars_validate_passes_the_value_through_when_the_check_holds() {
        let (exec, _) = executor();
        let out = exec
            .run_source(
                "v <- { schema_version: 1 }\nw <- v\n  → tars.validate(check: \"v.schema_version == 1\", reject_message: \"no\")",
            )
            .unwrap();
        assert_eq!(&json!({"schema_version": 1}), out.binding("w").unwrap());
    }

    #[test]
    fn tars_validate_stops_the_run_and_reports_the_authors_message() {
        let (exec, _) = executor();
        let err = exec
            .run_source(
                "v <- { schema_version: 2 }\nw <- v\n  → tars.validate(check: \"v.schema_version == 1\", reject_message: \"Schema mismatch\")",
            )
            .unwrap_err();
        assert!(err.to_string().contains("Schema mismatch"));
    }

    #[test]
    fn a_write_that_fails_its_schema_never_reaches_the_host() {
        let host = Arc::new(MemoryHost::frozen());
        let mut exec = Executor::new(host.clone());
        exec.schema_gate()
            .register(
                "state/quality/",
                "verdict",
                &json!({"type": "object", "required": ["verdict"]}),
            )
            .unwrap();

        let err = exec
            .run_source("ix.io.write(\"state/quality/v.json\", { other: 1 })")
            .unwrap_err();
        assert!(err.to_string().contains("violates schema"));
        assert!(host.files().is_empty(), "nothing may be persisted");
    }

    #[test]
    fn compound_ops_are_recorded_with_evaluated_operands() {
        let (exec, _) = executor();
        let out = exec
            .run_source(
                "v <- { followups: [\"a\"] }\nix.io.write(\"state/v.json\", v)\n  → compound:\n      harvest v.followups\n      log cycle to \"state/evolution/\"",
            )
            .unwrap();

        assert_eq!(
            vec![
                CompoundRecord::Harvested {
                    value: json!(["a"])
                },
                CompoundRecord::Logged {
                    id: "cycle".into(),
                    destination: "state/evolution/".into(),
                    value: json!({"followups": ["a"]}),
                },
            ],
            out.compound
        );
    }

    #[test]
    fn a_baml_step_crosses_the_seam_and_returns_its_value() {
        let host = Arc::new(MemoryHost::frozen());
        let mut exec = Executor::new(host);
        exec.baml_registry().register(Arc::new(StaticResponse::new(
            "EvaluateSignalSwarm",
            json!({"vote": "TrueVal"}),
        )));

        let out = exec
            .run_source(
                "telemetry <- { thd: 0.02 }\nvote <- telemetry\n  → baml.EvaluateSignalSwarm()",
            )
            .unwrap();
        assert_eq!(&json!({"vote": "TrueVal"}), out.binding("vote").unwrap());
    }

    #[test]
    fn an_unregistered_baml_function_fails_the_run() {
        let (exec, _) = executor();
        let err = exec
            .run_source("x <- { a: 1 }\ny <- x\n  → baml.Missing()")
            .unwrap_err();
        assert!(err.to_string().contains("no BAML function registered"));
    }

    #[test]
    fn unknown_names_and_functions_are_reported_not_defaulted() {
        let (exec, _) = executor();
        assert!(exec
            .run_source("a <- b")
            .unwrap_err()
            .to_string()
            .contains("unknown name"));
        assert!(exec
            .run_source("a <- nope.thing(1)")
            .unwrap_err()
            .to_string()
            .contains("not a function"));
    }

    #[test]
    fn is_empty_and_in_operate_on_containers() {
        let (exec, _) = executor();
        let out = exec
            .run_source(
                "xs <- []\nempty <- xs is empty\nhas <- \"a\" in [\"a\", \"b\"]\nmissing <- \"c\" not in [\"a\"]",
            )
            .unwrap();
        assert_eq!(&json!(true), out.binding("empty").unwrap());
        assert_eq!(&json!(true), out.binding("has").unwrap());
        assert_eq!(&json!(true), out.binding("missing").unwrap());
    }

    #[test]
    fn the_clock_is_the_hosts_clock() {
        let (exec, _) = executor();
        let out = exec
            .run_source("iso <- now_utc_iso8601()\nsafe <- now_utc(\"yyyy-MM-ddTHH-mm-ssZ\")")
            .unwrap();
        assert_eq!(&json!("2026-01-02T03:04:05Z"), out.binding("iso").unwrap());
        assert_eq!(&json!("2026-01-02T03-04-05Z"), out.binding("safe").unwrap());
    }
}
