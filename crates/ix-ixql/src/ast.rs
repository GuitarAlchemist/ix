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
    /// One or more consecutive `→ when <truth-value> [op N]: <call>` steps.
    ///
    /// Consecutive arms are grouped because the corpus uses them as a case
    /// analysis over *one* incoming verdict, not as filters applied in turn:
    ///
    /// ```text
    /// tars.assess(regret, question: "Has this regret been addressed?")
    ///   → when T >= 0.8: …archive…
    ///   → when F: …escalate…
    ///   → when U: …keep active…
    /// ```
    ///
    /// Read one at a time, a verdict of `F` would fail the first arm and stop
    /// the chain before the second was ever consulted.
    VerdictMatch(Vec<VerdictArm>),
}

/// One arm of a [`PipeStep::VerdictMatch`].
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictArm {
    pub guard: VerdictGuard,
    /// The step this arm runs: always a call (see [`PipeStep::CallStep`]).
    pub target: Box<Expr>,
    pub positional: Vec<Expr>,
    pub named: BTreeMap<String, Expr>,
}

/// `T`, `U`, `T >= 0.7` — the tree-sitter grammar's `membership_test`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerdictGuard {
    pub truth: ix_types::Hexavalent,
    /// A bound on the verdict's confidence, when the guard states one.
    pub confidence: Option<(ConfidenceOp, f64)>,
}

/// The comparisons `membership_test` allows — deliberately no `==`, which the
/// grammar does not offer for a float confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceOp {
    Gte,
    Gt,
    Lte,
    Lt,
}

impl ConfidenceOp {
    pub fn as_str(self) -> &'static str {
        match self {
            ConfidenceOp::Gte => ">=",
            ConfidenceOp::Gt => ">",
            ConfidenceOp::Lte => "<=",
            ConfidenceOp::Lt => "<",
        }
    }

    pub fn holds(self, confidence: f64, bound: f64) -> bool {
        match self {
            ConfidenceOp::Gte => confidence >= bound,
            ConfidenceOp::Gt => confidence > bound,
            ConfidenceOp::Lte => confidence <= bound,
            ConfidenceOp::Lt => confidence < bound,
        }
    }
}

impl VerdictGuard {
    /// Whether a verdict satisfies this guard.
    pub fn matches(&self, verdict: &Verdict) -> bool {
        verdict.truth == self.truth
            && match self.confidence {
                Some((op, bound)) => op.holds(verdict.confidence, bound),
                None => true,
            }
    }

    /// Source spelling, e.g. `T>=0.7` — used as a compiled stage's op.
    pub fn render(&self) -> String {
        match self.confidence {
            Some((op, bound)) => format!("{}{}{bound}", self.truth.symbol(), op.as_str()),
            None => self.truth.symbol().to_string(),
        }
    }
}

/// What a peer attached to a value: a hexavalent truth and a confidence in
/// `[0, 1]`.
///
/// It is carried *beside* a value, never inside it. Values stay exactly JSON
/// because everything IXQL produces is written to disk as JSON; a verdict is
/// the context a value flows through a pipeline in — the `M` of `M<Value>` —
/// which is how a computation expression keeps its effects out of its data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    pub truth: ix_types::Hexavalent,
    pub confidence: f64,
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
