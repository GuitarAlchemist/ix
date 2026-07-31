//! IXQL abstract syntax.
//!
//! Shape follows `Demerzel/ixql_executor_design_spec.md` §2. Recursive
//! positions are boxed; a program is a [`Block`].

use std::collections::BTreeMap;

/// Binary operators, in the surface spellings IXQL uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    And,
    Or,
    In,
    NotIn,
    Add,
    Sub,
    Mul,
    Div,
    /// `++` — sequence concatenation, not numeric.
    Concat,
}

impl BinaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Concat => "++",
            BinaryOp::Eq => "==",
            BinaryOp::Neq => "!=",
            BinaryOp::Gt => ">",
            BinaryOp::Gte => ">=",
            BinaryOp::Lt => "<",
            BinaryOp::Lte => "<=",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::In => "in",
            BinaryOp::NotIn => "not in",
        }
    }
}

/// Postfix predicates. `is empty` / `is not empty` read as unary in the surface
/// syntax, so they are modelled that way rather than forced into [`BinaryOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    IsEmpty,
    IsNotEmpty,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    /// Held as a `serde_json::Number` rather than an `f64` so an integer
    /// literal keeps its exact value all the way into the emitted JSON. These
    /// end up in ids and counters, where rounding `9007199254740993` down to
    /// `…992` would be a silent corruption.
    Number(serde_json::Number),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit(Literal),
    /// A name resolved against the environment.
    Var(String),
    /// `base.field`. Also how namespaced callees (`ix.io.read`) are spelled —
    /// the evaluator flattens such a chain back to a dotted path before
    /// looking for a host function.
    Member(Box<Expr>, String),
    Array(Vec<Expr>),
    /// Record literal, kept as ordered pairs so the AST mirrors the source and
    /// a duplicate key is detectable. (The emitted JSON is still key-sorted —
    /// `serde_json::Map` is a `BTreeMap` here.)
    Record(Vec<(String, Expr)>),
    /// A `"…{{expr}}…"` string: alternating literal and embedded pieces,
    /// concatenated at evaluation time.
    Interpolation(Vec<Expr>),
    BinOp(Box<Expr>, BinaryOp, Box<Expr>),
    Unary(UnaryOp, Box<Expr>),
    Call {
        target: Box<Expr>,
        positional: Vec<Expr>,
        named: BTreeMap<String, Expr>,
    },
    /// `source → step → step`.
    Pipeline(Box<Expr>, Vec<PipeStep>),
    /// `x => body`, or `(acc, item) => body` for the fold-shaped callers.
    ///
    /// Only ever an *argument* to a higher-order host function — the corpus
    /// never binds one to a name or returns one. It is therefore not a value:
    /// there is no `Value::Lambda`, and the evaluator matches this node
    /// syntactically at the call site rather than building a closure. That
    /// keeps the value domain exactly JSON, which is what every artifact this
    /// language writes has to be.
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
    },
}

/// One `→` stage. The value flowing in is the previous stage's output.
#[derive(Debug, Clone, PartialEq)]
pub enum PipeStep {
    /// `→ tars.validate(check: "…")`, `→ default({…})`, `→ baml.Fn()`.
    CallStep {
        target: Box<Expr>,
        positional: Vec<Expr>,
        named: BTreeMap<String, Expr>,
    },
    /// `→ compound:` followed by an indented op list.
    Compound(Vec<CompoundOp>),
}

/// Compound-phase operations — the "what did this run teach us" tail of a
/// pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum CompoundOp {
    /// `harvest <expr>` — collect learnings from the value.
    Harvest(Box<Expr>),
    /// `promote <id> [when <expr>]` — raise a candidate to durable state.
    Promote {
        id: String,
        condition: Option<Box<Expr>>,
    },
    /// `log <id> to <expr>` — append a record under a destination path.
    Log { id: String, destination: Box<Expr> },
    /// `teach <id> to <target>` — hand a learning to a named consumer.
    Teach { id: String, target: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// `name <- expr`
    Assign(String, Box<Expr>),
    /// A bare expression evaluated for its effects (a write, a validate).
    Do(Box<Expr>),
    /// `when <expr>: <block>`
    When(Box<Expr>, Box<Block>),
}

/// A sequence of statements — both a whole program and a `when` body.
pub type Block = Vec<Statement>;
