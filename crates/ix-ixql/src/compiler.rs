//! Pure IXQL compilation and static program metadata.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::ast::{BinaryOp, Block, CompoundOp, Expr, Literal, PipeStep, Statement, UnaryOp};
use crate::parser::{parse_expression, parse_program};

/// A host capability required by a compiled program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Clock,
    ReadArtifact,
    WriteArtifact,
    Validate,
    InvokeModel,
    Compound,
}

/// The useful, deliberately small type lattice available before evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Unknown,
    Null,
    Bool,
    Number,
    String,
    Array,
    Record,
}

/// One deterministic compiler diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub message: String,
}

/// One artifact read discovered without executing the program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRead {
    /// `None` means the path is dynamic and cannot be freshness-verified statically.
    pub path: Option<String>,
}

/// Static contract attached to one `ix.io.write` effect site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteEffectDeclaration {
    pub idempotency_key: Option<String>,
    pub expected_state: Option<String>,
    pub compensation: Option<String>,
    pub authority: Option<String>,
}

/// Compilation failed before a runnable program existed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("IXQL compilation failed with {count} diagnostic(s)", count = .0.len())]
pub struct CompileDiagnostics(pub Vec<Diagnostic>);

/// Lowered statement IR with the type inferred for the statement's value.
#[derive(Debug, Clone)]
pub enum TypedStatement {
    Assign {
        name: String,
        expression: Expr,
        value_type: ValueType,
    },
    Do {
        expression: Expr,
        value_type: ValueType,
    },
    When {
        condition: Expr,
        body: TypedBlock,
    },
}

/// Typed, scope-aware IR produced after parsing and before policy verification.
#[derive(Debug, Clone, Default)]
pub struct TypedBlock {
    statements: Vec<TypedStatement>,
}

impl TypedBlock {
    pub fn statements(&self) -> &[TypedStatement] {
        &self.statements
    }
}

/// Parsed IXQL plus the metadata a verifier needs without executing it.
#[derive(Debug, Clone)]
pub struct TypedProgram {
    ast: Block,
    typed_ir: TypedBlock,
    source_digest: String,
    required_capabilities: BTreeSet<Capability>,
    binding_types: BTreeMap<String, ValueType>,
    artifact_reads: Vec<ArtifactRead>,
    write_effects: Vec<WriteEffectDeclaration>,
    statement_count: usize,
    effect_count: usize,
}

impl TypedProgram {
    pub fn ast(&self) -> &Block {
        &self.ast
    }

    pub fn typed_ir(&self) -> &TypedBlock {
        &self.typed_ir
    }

    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    pub fn required_capabilities(&self) -> &BTreeSet<Capability> {
        &self.required_capabilities
    }

    pub fn binding_type(&self, name: &str) -> Option<ValueType> {
        self.binding_types.get(name).copied()
    }

    pub fn statement_count(&self) -> usize {
        self.statement_count
    }

    pub fn artifact_reads(&self) -> &[ArtifactRead] {
        &self.artifact_reads
    }

    pub fn write_effects(&self) -> &[WriteEffectDeclaration] {
        &self.write_effects
    }

    pub fn effect_count(&self) -> usize {
        self.effect_count
    }
}

/// Pure compiler entry point.
pub struct Compiler;

impl Compiler {
    pub fn compile(source: &str) -> Result<TypedProgram, CompileDiagnostics> {
        let ast = parse_program(source).map_err(|error| {
            CompileDiagnostics(vec![Diagnostic {
                code: "IXQL_PARSE",
                message: error.to_string(),
            }])
        })?;

        let mut analysis = Analysis::default();
        let typed_ir = analysis.block(&ast);
        if !analysis.diagnostics.is_empty() {
            return Err(CompileDiagnostics(analysis.diagnostics));
        }
        let source_digest = format!("{:x}", Sha256::digest(source.as_bytes()));

        Ok(TypedProgram {
            ast,
            typed_ir,
            source_digest,
            required_capabilities: analysis.capabilities,
            binding_types: analysis.bindings,
            artifact_reads: analysis.artifact_reads,
            write_effects: analysis.write_effects,
            statement_count: analysis.statement_count,
            effect_count: analysis.effect_count,
        })
    }
}

#[derive(Default)]
struct Analysis {
    capabilities: BTreeSet<Capability>,
    bindings: BTreeMap<String, ValueType>,
    artifact_reads: Vec<ArtifactRead>,
    write_effects: Vec<WriteEffectDeclaration>,
    statement_count: usize,
    effect_count: usize,
    diagnostics: Vec<Diagnostic>,
}

impl Analysis {
    fn block(&mut self, block: &Block) -> TypedBlock {
        let mut statements = Vec::with_capacity(block.len());
        for statement in block {
            self.statement_count += 1;
            match statement {
                Statement::Assign(name, expression) => {
                    let value_type = self.expression(expression);
                    self.bindings.insert(name.clone(), value_type);
                    statements.push(TypedStatement::Assign {
                        name: name.clone(),
                        expression: (**expression).clone(),
                        value_type,
                    });
                }
                Statement::Do(expression) => {
                    let value_type = self.expression(expression);
                    statements.push(TypedStatement::Do {
                        expression: (**expression).clone(),
                        value_type,
                    });
                }
                Statement::When(condition, body) => {
                    let condition_type = self.expression(condition);
                    self.require_type(condition_type, ValueType::Bool, "a `when` condition");
                    let outer_bindings = self.bindings.clone();
                    let body = self.block(body);
                    self.bindings = outer_bindings;
                    statements.push(TypedStatement::When {
                        condition: (**condition).clone(),
                        body,
                    });
                }
            }
        }
        TypedBlock { statements }
    }

    fn expression(&mut self, expression: &Expr) -> ValueType {
        match expression {
            Expr::Lit(literal) => match literal {
                Literal::Null => ValueType::Null,
                Literal::Bool(_) => ValueType::Bool,
                Literal::Number(_) => ValueType::Number,
                Literal::String(_) => ValueType::String,
            },
            Expr::Var(name) => self
                .bindings
                .get(name)
                .copied()
                .unwrap_or(ValueType::Unknown),
            Expr::Member(base, _) => {
                self.expression(base);
                ValueType::Unknown
            }
            Expr::Array(items) => {
                for item in items {
                    self.expression(item);
                }
                ValueType::Array
            }
            Expr::Record(fields) => {
                for (_, value) in fields {
                    self.expression(value);
                }
                ValueType::Record
            }
            Expr::Interpolation(parts) => {
                for part in parts {
                    self.expression(part);
                }
                ValueType::String
            }
            Expr::BinOp(left, operator, right) => {
                let left_type = self.expression(left);
                let right_type = self.expression(right);
                match operator {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                        self.require_type(left_type, ValueType::Number, operator.as_str());
                        self.require_type(right_type, ValueType::Number, operator.as_str());
                        ValueType::Number
                    }
                    BinaryOp::Concat => ValueType::Array,
                    BinaryOp::And | BinaryOp::Or => {
                        self.require_type(left_type, ValueType::Bool, operator.as_str());
                        self.require_type(right_type, ValueType::Bool, operator.as_str());
                        ValueType::Bool
                    }
                    _ => ValueType::Bool,
                }
            }
            Expr::Unary(operator, operand) => {
                let operand_type = self.expression(operand);
                match operator {
                    UnaryOp::IsEmpty | UnaryOp::IsNotEmpty => ValueType::Bool,
                    UnaryOp::Not => {
                        self.require_type(operand_type, ValueType::Bool, "`!`");
                        ValueType::Bool
                    }
                }
            }
            Expr::Call {
                target,
                positional,
                named,
            } => {
                let positional_types: Vec<_> = positional
                    .iter()
                    .map(|argument| self.expression(argument))
                    .collect();
                for argument in named.values() {
                    self.expression(argument);
                }
                self.call(target, positional, &positional_types, named, None)
            }
            Expr::Pipeline(head, steps) => {
                let mut current = self.expression(head);
                for step in steps {
                    current = match step {
                        PipeStep::CallStep {
                            target,
                            positional,
                            named,
                        } => {
                            let positional_types: Vec<_> = positional
                                .iter()
                                .map(|argument| self.expression(argument))
                                .collect();
                            for argument in named.values() {
                                self.expression(argument);
                            }
                            self.call(target, positional, &positional_types, named, Some(current))
                        }
                        PipeStep::Compound(operations) => {
                            self.capabilities.insert(Capability::Compound);
                            self.effect_count += operations.len();
                            for operation in operations {
                                self.compound(operation);
                            }
                            current
                        }
                    };
                }
                current
            }
            Expr::Lambda { body, .. } => {
                self.expression(body);
                ValueType::Unknown
            }
        }
    }

    fn call(
        &mut self,
        target: &Expr,
        positional: &[Expr],
        positional_types: &[ValueType],
        named: &BTreeMap<String, Expr>,
        piped: Option<ValueType>,
    ) -> ValueType {
        let Some(name) = callee_path(target) else {
            return ValueType::Unknown;
        };
        match name.as_str() {
            "now_utc" | "now_utc_iso8601" => {
                self.capabilities.insert(Capability::Clock);
                ValueType::String
            }
            "ix.io.read" => {
                self.capabilities.insert(Capability::ReadArtifact);
                self.effect_count += 1;
                let path = positional
                    .first()
                    .and_then(literal_string)
                    .and_then(|path| match crate::path::normalize(&path) {
                        Ok(path) => Some(path),
                        Err(error) => {
                            self.diagnostics.push(Diagnostic {
                                code: "IXQL_INVALID_PATH",
                                message: error.to_string(),
                            });
                            None
                        }
                    });
                self.artifact_reads.push(ArtifactRead { path });
                ValueType::Unknown
            }
            "ix.io.write" => {
                self.capabilities.insert(Capability::WriteArtifact);
                self.effect_count += 1;
                self.write_effects.push(WriteEffectDeclaration {
                    idempotency_key: literal_named(named, "idempotency_key"),
                    expected_state: literal_named(named, "expected_state"),
                    compensation: literal_named(named, "compensation"),
                    authority: literal_named(named, "authority"),
                });
                positional_types
                    .get(1)
                    .copied()
                    .unwrap_or(ValueType::Unknown)
            }
            "tars.validate" => {
                self.capabilities.insert(Capability::Validate);
                match literal_named(named, "check") {
                    Some(check) => match parse_expression(&check) {
                        Ok(predicate) => {
                            let predicate_type = self.expression(&predicate);
                            self.require_type(
                                predicate_type,
                                ValueType::Bool,
                                "a `tars.validate` predicate",
                            );
                        }
                        Err(error) => self.diagnostics.push(Diagnostic {
                            code: "IXQL_INVALID_VALIDATE_PREDICATE",
                            message: error.to_string(),
                        }),
                    },
                    None => self.diagnostics.push(Diagnostic {
                        code: "IXQL_DYNAMIC_VALIDATE_PREDICATE",
                        message:
                            "tars.validate requires a literal `check` under static verification"
                                .into(),
                    }),
                }
                piped.unwrap_or(ValueType::Unknown)
            }
            name if name.starts_with("baml.") => {
                self.capabilities.insert(Capability::InvokeModel);
                self.effect_count += 1;
                ValueType::Unknown
            }
            "default" => piped.unwrap_or(ValueType::Unknown),
            _ => ValueType::Unknown,
        }
    }

    fn compound(&mut self, operation: &CompoundOp) {
        match operation {
            CompoundOp::Harvest(expression) => {
                self.expression(expression);
            }
            CompoundOp::Promote { condition, .. } => {
                if let Some(condition) = condition {
                    self.expression(condition);
                }
            }
            CompoundOp::Log { destination, .. } => {
                self.expression(destination);
            }
            CompoundOp::Teach { .. } => {}
        }
    }

    fn require_type(&mut self, found: ValueType, expected: ValueType, context: &str) {
        if found != ValueType::Unknown && found != expected {
            self.diagnostics.push(Diagnostic {
                code: "IXQL_TYPE_MISMATCH",
                message: format!("{context} expected {expected:?}, found {found:?}"),
            });
        }
    }
}

fn literal_string(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Lit(Literal::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn literal_named(named: &BTreeMap<String, Expr>, name: &str) -> Option<String> {
    named.get(name).and_then(literal_string)
}

fn callee_path(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Var(name) => Some(name.clone()),
        Expr::Member(base, field) => Some(format!("{}.{}", callee_path(base)?, field)),
        _ => None,
    }
}
