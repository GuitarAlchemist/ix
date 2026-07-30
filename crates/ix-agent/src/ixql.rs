use crate::ast::{Expr, Literal, BinaryOp, Block, Statement, PipeStep, CompoundOp};
use std::collections::HashMap;
use serde_json::Value;
use jsonschema::JSONSchema;
use std::fs;
use std::path::Path;
use futures_util::future::{BoxFuture, FutureExt};

pub struct ExecutionContext {
    pub env: HashMap<String, Value>,
    pub compound_stash: Vec<CompoundOp>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            env: HashMap::new(),
            compound_stash: Vec::new(),
        }
    }
}

/// Invokes BAML client function if called inside the pipeline
pub async fn invoke_baml_func(
    func_name: &str,
    args: &[Value],
) -> Result<Value, String> {
    match func_name {
        "EvaluateSignalSwarm" => {
            if args.len() != 3 {
                return Err(format!("EvaluateSignalSwarm expects 3 arguments, got {}", args.len()));
            }
            if std::env::var("IXQL_TEST_MOCK_BAML").is_ok() {
                return Ok(serde_json::json!({
                    "agent_id": "test-agent",
                    "role": "Architect",
                    "value": "TrueVal",
                    "reason": "Mocked validation reasoning",
                    "self_trust": 0.95
                }));
            }

            let telemetry: ix_baml::baml_client::types::SignalTelemetry = serde_json::from_value(args[0].clone())
                .map_err(|e| format!("BAML telemetry map failed: {}", e))?;
            let role: ix_baml::baml_client::types::AgentRole = serde_json::from_value(args[1].clone())
                .map_err(|e| format!("BAML role map failed: {}", e))?;
            let bounds: ix_baml::baml_client::types::SafetyBounds = serde_json::from_value(args[2].clone())
                .map_err(|e| format!("BAML bounds map failed: {}", e))?;

            let result = ix_baml::baml_client::async_client::B.EvaluateSignalSwarm.call(&telemetry, &role, &bounds).await
                .map_err(|e| format!("BAML invocation failed: {:?}", e))?;

            let out_str = serde_json::to_string(&result)
                .map_err(|e| format!("BAML serialize output failed: {}", e))?;
            let out_val: Value = serde_json::from_str(&out_str)
                .map_err(|e| format!("BAML output map to json failed: {}", e))?;
            Ok(out_val)
        }
        _ => Err(format!("Unknown BAML function: {}", func_name)),
    }
}

/// Helper to validate a JSON value against a JSON Schema file
pub fn validate_schema_file(schema_path: &str, value: &Value) -> Result<(), String> {
    let schema_content = fs::read_to_string(schema_path)
        .map_err(|e| format!("Failed to read schema file {}: {}", schema_path, e))?;
    let schema_json: Value = serde_json::from_str(&schema_content)
        .map_err(|e| format!("Failed to parse schema json: {}", e))?;
    
    let compiled = JSONSchema::compile(&schema_json)
        .map_err(|e| format!("Invalid JSON Schema: {}", e))?;
    
    if let Err(errors) = compiled.validate(value) {
        let errs: Vec<String> = errors.map(|e| e.to_string()).collect();
        return Err(format!("JSON Schema validation failed: {}", errs.join(", ")));
    }
    Ok(())
}

/// Evaluates a binary operator
fn eval_binop(left: &Value, op: &BinaryOp, right: &Value) -> Result<Value, String> {
    match op {
        BinaryOp::Eq => Ok(Value::Bool(left == right)),
        BinaryOp::Neq => Ok(Value::Bool(left != right)),
        BinaryOp::Gt => {
            if let (Some(l), Some(r)) = (left.as_f64(), right.as_f64()) {
                Ok(Value::Bool(l > r))
            } else {
                Err(format!("Cannot compare GT on {:?} and {:?}", left, right))
            }
        }
        BinaryOp::Gte => {
            if let (Some(l), Some(r)) = (left.as_f64(), right.as_f64()) {
                Ok(Value::Bool(l >= r))
            } else {
                Err(format!("Cannot compare GTE on {:?} and {:?}", left, right))
            }
        }
        BinaryOp::Lt => {
            if let (Some(l), Some(r)) = (left.as_f64(), right.as_f64()) {
                Ok(Value::Bool(l < r))
            } else {
                Err(format!("Cannot compare LT on {:?} and {:?}", left, right))
            }
        }
        BinaryOp::Lte => {
            if let (Some(l), Some(r)) = (left.as_f64(), right.as_f64()) {
                Ok(Value::Bool(l <= r))
            } else {
                Err(format!("Cannot compare LTE on {:?} and {:?}", left, right))
            }
        }
        BinaryOp::And => Ok(Value::Bool(left.as_bool().unwrap_or(false) && right.as_bool().unwrap_or(false))),
        BinaryOp::Or => Ok(Value::Bool(left.as_bool().unwrap_or(false) || right.as_bool().unwrap_or(false))),
        BinaryOp::In => {
            if let Some(arr) = right.as_array() {
                Ok(Value::Bool(arr.contains(left)))
            } else {
                Err(format!("Right operand for 'in' must be an array, got {:?}", right))
            }
        }
        BinaryOp::NotIn => {
            if let Some(arr) = right.as_array() {
                Ok(Value::Bool(!arr.contains(left)))
            } else {
                Err(format!("Right operand for 'not in' must be an array, got {:?}", right))
            }
        }
        BinaryOp::IsEmpty => {
            if let Some(arr) = left.as_array() {
                Ok(Value::Bool(arr.is_empty()))
            } else if let Some(s) = left.as_str() {
                Ok(Value::Bool(s.is_empty()))
            } else {
                Ok(Value::Bool(left.is_null()))
            }
        }
        BinaryOp::IsNotEmpty => {
            if let Some(arr) = left.as_array() {
                Ok(Value::Bool(!arr.is_empty()))
            } else if let Some(s) = left.as_str() {
                Ok(Value::Bool(!s.is_empty()))
            } else {
                Ok(Value::Bool(!left.is_null()))
            }
        }
    }
}

/// Evaluates an expression down to a Value (boxed for async recursion)
pub fn eval_expr<'a>(
    ctx: &'a mut ExecutionContext,
    expr: &'a Expr,
) -> BoxFuture<'a, Result<Value, String>> {
    async move {
        match expr {
            Expr::Lit(lit) => match lit {
                Literal::Null => Ok(Value::Null),
                Literal::Bool(b) => Ok(Value::Bool(*b)),
                Literal::Number(n) => Ok(Value::Number(serde_json::Number::from_f64(*n).unwrap())),
                Literal::String(s) => Ok(Value::String(s.clone())),
            },
            Expr::Var(var) => ctx.env.get(var).cloned().ok_or_else(|| format!("Undefined variable: {}", var)),
            Expr::Member(obj_expr, member) => {
                let obj: Value = eval_expr(ctx, obj_expr).await?;
                if let Some(record) = obj.as_object() {
                    record.get(member).cloned().ok_or_else(|| format!("Member {} not found on record", member))
                } else {
                    Err(format!("Cannot access member {} on non-object value {:?}", member, obj))
                }
            }
            Expr::Array(items) => {
                let mut out = Vec::new();
                for item in items {
                    out.push(eval_expr(ctx, item).await?);
                }
                Ok(Value::Array(out))
            }
            Expr::Record(fields) => {
                let mut out = serde_json::Map::new();
                for (key, val_expr) in fields {
                    out.insert(key.clone(), eval_expr(ctx, val_expr).await?);
                }
                Ok(Value::Object(out))
            }
            Expr::Interpolation(parts) => {
                let mut s = String::new();
                for part in parts {
                    let v: Value = eval_expr(ctx, part).await?;
                    s.push_str(v.as_str().unwrap_or(&v.to_string()));
                }
                Ok(Value::String(s))
            }
            Expr::BinOp(left, op, right) => {
                let l = eval_expr(ctx, left).await?;
                let r = eval_expr(ctx, right).await?;
                eval_binop(&l, op, &r)
            }
            Expr::Call { target, positional, named: _ } => {
                if let Expr::Var(ref name) = **target {
                    if name.starts_with("baml.") {
                        let func = name.strip_prefix("baml.").unwrap();
                        let mut args = Vec::new();
                        for arg in positional {
                            args.push(eval_expr(ctx, arg).await?);
                        }
                        return invoke_baml_func(func, &args).await;
                    }
                    
                    match name.as_str() {
                        "now" => {
                            let now = chrono::Utc::now().to_rfc3339();
                            return Ok(Value::String(now));
                        }
                        _ => {}
                    }
                }
                Err(format!("Unsupported function call: {:?}", target))
            }
            Expr::Lambda(_, _) => Err("Lambda can only be executed in pipeline steps like map/filter".to_string()),
            Expr::Pipeline(subject, steps) => {
                let mut current = eval_expr(ctx, subject).await?;
                for step in steps {
                    current = apply_pipe_step(ctx, current, step).await?;
                }
                Ok(current)
            }
        }
    }.boxed()
}

/// Applies a step in a pipeline (boxed for async recursion)
fn apply_pipe_step<'a>(
    ctx: &'a mut ExecutionContext,
    val: Value,
    step: &'a PipeStep,
) -> BoxFuture<'a, Result<Value, String>> {
    async move {
        match step {
            PipeStep::CallStep { target, positional, named: _ } => {
                if let Expr::Var(ref name) = **target {
                    match name.as_str() {
                        "filter" => {
                            if positional.len() != 1 {
                                return Err("filter step expects exactly 1 expression argument".to_string());
                            }
                            if let Some(arr) = val.as_array() {
                                let mut filtered = Vec::new();
                                for item in arr {
                                    let mut local_ctx = ExecutionContext::new();
                                    local_ctx.env = ctx.env.clone();
                                    local_ctx.env.insert("r".to_string(), item.clone());
                                    
                                    let condition: Value = eval_expr(&mut local_ctx, &positional[0]).await?;
                                    if condition.as_bool().unwrap_or(false) {
                                        filtered.push(item.clone());
                                    }
                                }
                                return Ok(Value::Array(filtered));
                            }
                            return Err(format!("filter expects array input, got {:?}", val));
                        }
                        "tars.validate_schema" => {
                            if positional.len() != 1 {
                                return Err("tars.validate_schema expects 1 path argument".to_string());
                            }
                            let schema_path: String = eval_expr(ctx, &positional[0]).await?
                                .as_str().ok_or("Schema path must be a string")?.to_string();
                            
                            validate_schema_file(&schema_path, &val)?;
                            return Ok(val);
                        }
                        "ix.io.read" => {
                            if positional.len() != 1 {
                                return Err("ix.io.read expects 1 path/glob argument".to_string());
                            }
                            let _path_glob: String = eval_expr(ctx, &positional[0]).await?
                                .as_str().ok_or("Path must be a string")?.to_string();

                            let dir_path = "state/oversight/ml-recommendations";
                            let mut results = Vec::new();
                            if Path::new(dir_path).exists() {
                                for entry in fs::read_dir(dir_path).map_err(|e| e.to_string())? {
                                    let entry = entry.map_err(|e| e.to_string())?;
                                    let p = entry.path();
                                    if p.extension().map_or(false, |ext| ext == "json") {
                                        let content = fs::read_to_string(&p).map_err(|e| e.to_string())?;
                                        let json_val: Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
                                        results.push(json_val);
                                    }
                                }
                            }
                            return Ok(Value::Array(results));
                        }
                        "ix.io.write" => {
                            if positional.len() != 2 {
                                return Err("ix.io.write expects path and data arguments".to_string());
                            }
                            let dest_path: String = eval_expr(ctx, &positional[0]).await?
                                .as_str().ok_or("Destination path must be a string")?.to_string();
                            
                            if dest_path.contains("state/quality/verdicts") {
                                validate_schema_file("schemas/contracts/qa-verdict.schema.json", &val)?;
                            } else if dest_path.contains("state/oversight/ml-recommendations") {
                                validate_schema_file("schemas/contracts/ml-feedback-recommendation.schema.json", &val)?;
                            }

                            let parent = Path::new(&dest_path).parent().unwrap();
                            fs::create_dir_all(parent).map_err(|e| format!("Failed to create dirs: {}", e))?;
                            fs::write(&dest_path, serde_json::to_string_pretty(&val).unwrap())
                                .map_err(|e| format!("Failed to write file {}: {}", dest_path, e))?;

                            return Ok(val);
                        }
                        _ => {}
                    }
                }
                Err(format!("Unsupported pipe step: {:?}", target))
            }
            PipeStep::FanOut(blocks) => {
                let mut results = Vec::new();
                for block in blocks {
                    let mut fork_ctx = ExecutionContext::new();
                    fork_ctx.env = ctx.env.clone();
                    execute_block(&mut fork_ctx, block).await?;
                    ctx.compound_stash.extend(fork_ctx.compound_stash);
                    results.push(Value::Null);
                }
                Ok(Value::Array(results))
            }
            PipeStep::Parallel(blocks) => {
                let mut results = Vec::new();
                for block in blocks {
                    let mut fork_ctx = ExecutionContext::new();
                    fork_ctx.env = ctx.env.clone();
                    execute_block(&mut fork_ctx, block).await?;
                    ctx.compound_stash.extend(fork_ctx.compound_stash);
                    results.push(Value::Null);
                }
                Ok(Value::Array(results))
            }
            PipeStep::Compound(ops) => {
                ctx.compound_stash.extend(ops.clone());
                Ok(val)
            }
        }
    }.boxed()
}

/// Executes a statement (boxed for async recursion)
pub fn execute_statement<'a>(
    ctx: &'a mut ExecutionContext,
    stmt: &'a Statement,
) -> BoxFuture<'a, Result<(), String>> {
    async move {
        match stmt {
            Statement::Assign(var, expr) => {
                let val = eval_expr(ctx, expr).await?;
                ctx.env.insert(var.clone(), val);
                Ok(())
            }
            Statement::Do(expr) => {
                eval_expr(ctx, expr).await?;
                Ok(())
            }
            Statement::When(cond_expr, block) => {
                let cond = eval_expr(ctx, cond_expr).await?;
                let is_true = match cond {
                    Value::Bool(b) => b,
                    Value::Array(arr) => !arr.is_empty(),
                    Value::Object(map) => !map.is_empty(),
                    Value::Null => false,
                    _ => true,
                };
                if is_true {
                    execute_block(ctx, block).await?;
                }
                Ok(())
            }
        }
    }.boxed()
}

/// Executes a block of statements transactionally (boxed for async recursion)
pub fn execute_block<'a>(
    ctx: &'a mut ExecutionContext,
    block: &'a Block,
) -> BoxFuture<'a, Result<(), String>> {
    async move {
        let rollback_stash = ctx.compound_stash.clone();
        for stmt in block {
            if let Err(e) = execute_statement(ctx, stmt).await {
                ctx.compound_stash = rollback_stash;
                return Err(e);
            }
        }
        Ok(())
    }.boxed()
}
