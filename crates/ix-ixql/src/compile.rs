//! Compile an IXQL program to a stage DAG.
//!
//! IXQL is executed by [`crate::Executor`], a tree-walking interpreter that
//! evaluates statements in source order. That is the right shape for running a
//! pipeline and the wrong shape for *reasoning* about one: nothing in the
//! evaluator can answer "which of these steps are independent", "what does
//! `verdict` transitively depend on", or "what is the critical path", because
//! the dependency structure is never materialised.
//!
//! This module materialises it. A program becomes an ordered list of
//! [`Stage`]s plus the edges between them, which is exactly the shape
//! [`ix_pipeline::dag::Dag`] already knows how to schedule — so cycle
//! rejection, `topological_sort`, `parallel_levels` and `critical_path` are
//! reused rather than rewritten.
//!
//! # What this is not
//!
//! Compiling is not lowering. [`ix_pipeline::lower`] resolves every stage's
//! `skill` against `ix-registry` and fails on an unknown one; IXQL's callees
//! are *host* functions (`ix.io.read`, `tars.validate`, `now_utc`), which are
//! not registry skills. A [`CompiledPlan`] can be rendered as a
//! [`PipelineSpec`] and shape-validated, but `lower()` on it would fail
//! `UnknownSkill` until someone defines an IXQL-host-to-registry-skill
//! mapping. That mapping is not invented here.
//!
//! # Element scope is not decided here (issue #281)
//!
//! Higher-order calls — `map`, `filter`, `any`, `reduce`, `find`, `transform`
//! — need a scoping rule that covers both spellings the corpus uses
//! (`filter(p => …)` and the binder-less `group.filter(signal_type == "pain")`).
//! That decision is open and reserved. Rather than guess it, the compiler
//! refuses any program containing one, with [`CompileError::RequiresElementScope`]
//! naming the function and the stage. This mirrors `EvalError::LambdaIsNotAValue`:
//! a loud refusal keeps the gap measurable, where quietly compiling the call as
//! an opaque node would bury it.

use std::collections::{BTreeMap, BTreeSet};

use ix_pipeline::dag::{Dag, DagError};
use ix_pipeline::spec::{PipelineSpec, StageSpec};
use serde_json::{json, Value};

use crate::ast::{Block, Expr, PipeStep, Statement};
use crate::parser::{parse_program, ParseError};

/// Higher-order callees whose argument is evaluated in *element* scope.
///
/// Taken verbatim from the corpus census in issue #281 (88 sites). Membership
/// here means "compiling this requires a scoping rule nobody has picked yet",
/// not "this is unsupported forever".
pub const ELEMENT_SCOPED_FUNCTIONS: &[&str] =
    &["map", "filter", "any", "reduce", "find", "transform"];

/// What a stage does, structurally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StageKind {
    /// `name <- expr` — a value binding with no pipe steps.
    Bind,
    /// The head of a `a → b` chain: the expression the pipeline flows out of.
    Source,
    /// One `→ call(...)` stage.
    Pipe,
    /// One `→ compound:` stage.
    Compound,
    /// A bare expression statement evaluated for its effects.
    Effect,
    /// A `when <cond>:` guard. Statements in its block depend on it.
    When,
}

impl StageKind {
    /// Lowercase tag used in the emitted relation. Stable — the DuckDB macros
    /// validate against exactly these spellings.
    pub fn as_str(self) -> &'static str {
        match self {
            StageKind::Bind => "bind",
            StageKind::Source => "source",
            StageKind::Pipe => "pipe",
            StageKind::Compound => "compound",
            StageKind::Effect => "effect",
            StageKind::When => "when",
        }
    }
}

/// One node of the compiled plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// Unique id. A binding's final stage takes the binding's name, so other
    /// stages reference it by the name the source uses. Intermediate stages of
    /// a `→` chain are suffixed `~0`, `~1`, …
    pub id: String,
    /// Position in the compiled order. Assigned in source order, so the
    /// emitted relation is stable across runs.
    pub ordinal: usize,
    /// Structural role.
    pub kind: StageKind,
    /// Dotted callee (`ix.io.read`, `tars.validate`, `now_utc`), or the empty
    /// string when the stage is a pure expression with no call at its root.
    pub op: String,
    /// Upstream stage ids, sorted and de-duplicated.
    pub deps: Vec<String>,
}

/// A compiled IXQL program: stages plus their scheduling.
#[derive(Debug, Clone)]
pub struct CompiledPlan {
    stages: Vec<Stage>,
    /// Stage id -> 0-based parallel level, from [`Dag::parallel_levels`].
    levels: BTreeMap<String, usize>,
    /// Stage ids in topological order.
    topo: Vec<String>,
}

/// Why a program could not be compiled.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    /// The program uses a higher-order call whose argument is evaluated in
    /// element scope. Compiling it would require picking the scoping rule that
    /// issue #281 reserves for a maintainer.
    #[error(
        "stage '{stage}' calls '{function}', whose argument is evaluated in element scope. \
         The scoping rule for higher-order calls is an open design decision (issue #281) \
         and is deliberately not settled by the compiler. Programs without \
         map/filter/any/reduce/find/transform compile today."
    )]
    RequiresElementScope {
        /// The higher-order callee, e.g. `filter`.
        function: String,
        /// The stage whose expression contains it.
        stage: String,
    },

    /// A lambda appeared. Same reservation as [`Self::RequiresElementScope`],
    /// reported separately because the lambda may sit under a callee this
    /// compiler does not recognise as higher-order.
    #[error(
        "stage '{stage}' contains a lambda ({params}). Lambda bodies need the element-scope \
         rule from issue #281, which is deliberately not settled by the compiler."
    )]
    LambdaRequiresElementScope {
        /// Comma-joined parameter names.
        params: String,
        /// The stage whose expression contains it.
        stage: String,
    },

    /// The dependency graph is cyclic. Structurally unreachable today — deps
    /// only point at already-bound names — but the DAG is the enforcement
    /// point, not an assumption.
    #[error("stage graph is cyclic: {0}")]
    Cycle(String),
}

/// Compile IXQL source text to a plan.
pub fn compile(source: &str) -> Result<CompiledPlan, CompileError> {
    compile_program(&parse_program(source)?)
}

/// Compile an already-parsed program.
pub fn compile_program(program: &Block) -> Result<CompiledPlan, CompileError> {
    let mut builder = Builder::default();
    builder.block(program, &[])?;
    builder.finish()
}

impl CompiledPlan {
    /// Stages in compiled (source) order.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Stage ids in topological order.
    pub fn topological_order(&self) -> &[String] {
        &self.topo
    }

    /// 0-based parallel level of a stage: every stage at level `n` can run once
    /// every stage at levels `< n` has finished.
    pub fn level_of(&self, stage_id: &str) -> Option<usize> {
        self.levels.get(stage_id).copied()
    }

    /// Stage ids grouped by parallel level, each group sorted by stage id.
    pub fn schedule(&self) -> Vec<Vec<String>> {
        let depth = self.levels.values().copied().max().map_or(0, |m| m + 1);
        let mut out = vec![Vec::new(); depth];
        for (id, level) in &self.levels {
            out[*level].push(id.clone());
        }
        for group in &mut out {
            group.sort();
        }
        out
    }

    /// The compiled structure as a deterministic CSV relation.
    ///
    /// Deliberately carries **no** scheduling column. The schedule is what the
    /// Rust and DuckDB surfaces each derive from this relation independently;
    /// shipping it here would let the SQL half echo an answer instead of
    /// computing one.
    pub fn to_plan_csv(&self) -> String {
        let mut out = String::from("ordinal,stage_id,kind,op,deps\n");
        for stage in &self.stages {
            out.push_str(&format!(
                "{},{},{},{},{}\n",
                stage.ordinal,
                stage.id,
                stage.kind.as_str(),
                stage.op,
                stage.deps.join(";")
            ));
        }
        out
    }

    /// The derived execution schedule as a deterministic CSV relation: one row
    /// per parallel level.
    pub fn to_schedule_csv(&self) -> String {
        let mut out = String::from("level,stage_count,stages\n");
        for (level, group) in self.schedule().iter().enumerate() {
            out.push_str(&format!(
                "{},{},{}\n",
                level,
                group.len(),
                group.join(";")
            ));
        }
        out
    }

    /// Render as the workspace's canonical pipeline IR.
    ///
    /// The result round-trips through YAML and passes
    /// [`PipelineSpec::validate_shape`], but see the module docs: its `skill`
    /// fields are IXQL host functions, so [`ix_pipeline::lower`] would reject
    /// it as `UnknownSkill`. This is the compiled shape, not an executable
    /// pipeline.
    pub fn to_pipeline_spec(&self) -> PipelineSpec {
        let mut stages = BTreeMap::new();
        for stage in &self.stages {
            stages.insert(
                stage.id.clone(),
                StageSpec {
                    skill: if stage.op.is_empty() {
                        format!("ixql.{}", stage.kind.as_str())
                    } else {
                        stage.op.clone()
                    },
                    args: json!({ "ixql_kind": stage.kind.as_str() }),
                    deps: stage.deps.clone(),
                    cache: None,
                },
            );
        }
        PipelineSpec {
            version: "1".into(),
            params: BTreeMap::new(),
            stages,
            x_editor: Value::Null,
        }
    }
}

/// Accumulates stages while walking the program.
#[derive(Default)]
struct Builder {
    stages: Vec<Stage>,
    /// Binding name -> the stage id currently holding that name's value.
    bound: BTreeMap<String, String>,
    /// Every id issued so far, for uniquifying a rebound name.
    issued: BTreeSet<String>,
}

impl Builder {
    /// Compile a block. `inherited` deps are added to every top-level stage of
    /// the block — that is how a `when` guard reaches its body.
    fn block(&mut self, block: &Block, inherited: &[String]) -> Result<(), CompileError> {
        for (index, statement) in block.iter().enumerate() {
            match statement {
                Statement::Assign(name, expr) => {
                    self.chain(Some(name.clone()), index, expr, inherited)?;
                }
                Statement::Do(expr) => {
                    self.chain(None, index, expr, inherited)?;
                }
                Statement::When(condition, body) => {
                    let id = self.unique(&format!("s{index:02}_when"));
                    reject_element_scope(condition, &id)?;
                    let mut deps = self.refs(condition);
                    deps.extend(inherited.iter().cloned());
                    self.push(id.clone(), StageKind::When, "when".into(), deps);

                    let mut nested = inherited.to_vec();
                    nested.push(id);
                    self.block(body, &nested)?;
                }
            }
        }
        Ok(())
    }

    /// Compile one statement's expression, expanding a `→` chain into one
    /// stage per link.
    ///
    /// The chain's *last* stage takes the binding name, so a later reference to
    /// `trigger_context` resolves to the value after `→ default(…)` ran — which
    /// is what the source means.
    fn chain(
        &mut self,
        name: Option<String>,
        index: usize,
        expr: &Expr,
        inherited: &[String],
    ) -> Result<(), CompileError> {
        let base = name
            .clone()
            .unwrap_or_else(|| format!("s{index:02}"));

        let (head, steps) = match expr {
            Expr::Pipeline(head, steps) => (head.as_ref(), steps.as_slice()),
            other => (other, &[][..]),
        };

        // Head of the chain. With no pipe steps it *is* the whole statement, so
        // it takes the binding name directly.
        let head_id = if steps.is_empty() {
            self.unique(&base)
        } else {
            self.unique(&format!("{base}~0"))
        };
        reject_element_scope(head, &head_id)?;
        let mut head_deps = self.refs(head);
        head_deps.extend(inherited.iter().cloned());
        let head_kind = if steps.is_empty() {
            if name.is_some() {
                StageKind::Bind
            } else {
                StageKind::Effect
            }
        } else {
            StageKind::Source
        };
        self.push(head_id.clone(), head_kind, callee_of(head), head_deps);

        let mut previous = head_id;
        for (position, step) in steps.iter().enumerate() {
            let last = position + 1 == steps.len();
            let id = if last {
                self.unique(&base)
            } else {
                self.unique(&format!("{base}~{}", position + 1))
            };

            let (kind, op, mut deps) = match step {
                PipeStep::CallStep {
                    target,
                    positional,
                    named,
                } => {
                    // Callee before arguments: `map(x => …)` trips both checks,
                    // and naming `map` is more use to a reader than naming `x`.
                    if let Some(function) = element_scoped_callee(target) {
                        return Err(CompileError::RequiresElementScope { function, stage: id });
                    }
                    reject_element_scope(target, &id)?;
                    let mut deps = Vec::new();
                    for argument in positional.iter().chain(named.values()) {
                        reject_element_scope(argument, &id)?;
                        deps.extend(self.refs(argument));
                    }
                    (StageKind::Pipe, dotted(target).unwrap_or_default(), deps)
                }
                PipeStep::Compound(ops) => {
                    let mut deps = Vec::new();
                    for expression in compound_exprs(ops) {
                        reject_element_scope(expression, &id)?;
                        deps.extend(self.refs(expression));
                    }
                    (StageKind::Compound, "compound".to_string(), deps)
                }
                // One stage for the whole match: which arm runs is decided by a
                // verdict at run time, so the plan cannot split it into arms
                // that would all look schedulable. Its op spells every arm, e.g.
                // `T>=0.8:ix.io.write|C:alert`.
                PipeStep::VerdictMatch(arms) => {
                    let mut deps = Vec::new();
                    let mut spelled = Vec::with_capacity(arms.len());
                    for arm in arms {
                        if let Some(function) = element_scoped_callee(&arm.target) {
                            return Err(CompileError::RequiresElementScope {
                                function,
                                stage: id,
                            });
                        }
                        reject_element_scope(&arm.target, &id)?;
                        for argument in arm.positional.iter().chain(arm.named.values()) {
                            reject_element_scope(argument, &id)?;
                            deps.extend(self.refs(argument));
                        }
                        spelled.push(format!(
                            "{}:{}",
                            arm.guard.render(),
                            dotted(&arm.target).unwrap_or_default()
                        ));
                    }
                    (StageKind::When, spelled.join("|"), deps)
                }
            };

            deps.push(previous.clone());
            deps.extend(inherited.iter().cloned());
            self.push(id.clone(), kind, op, deps);
            previous = id;
        }

        if let Some(name) = name {
            self.bound.insert(name, previous);
        }
        Ok(())
    }

    fn push(&mut self, id: String, kind: StageKind, op: String, deps: Vec<String>) {
        let mut deps: Vec<String> = deps.into_iter().collect::<BTreeSet<_>>().into_iter().collect();
        deps.retain(|d| d != &id);
        self.stages.push(Stage {
            ordinal: self.stages.len(),
            id,
            kind,
            op,
            deps,
        });
    }

    /// Issue an id, suffixing on collision so a rebound name cannot silently
    /// overwrite an earlier stage.
    fn unique(&mut self, wanted: &str) -> String {
        if self.issued.insert(wanted.to_string()) {
            return wanted.to_string();
        }
        let mut n = 2;
        loop {
            let candidate = format!("{wanted}#{n}");
            if self.issued.insert(candidate.clone()) {
                return candidate;
            }
            n += 1;
        }
    }

    /// Stage ids an expression reads from: every free variable that names a
    /// currently-bound value.
    ///
    /// A callee path like `ix.io.read` is rooted at `Var("ix")`, which is not a
    /// binding, so it contributes no edge — namespaces fall out for free rather
    /// than needing a keyword list.
    fn refs(&self, expr: &Expr) -> Vec<String> {
        let mut found = Vec::new();
        for name in free_vars(expr) {
            if let Some(stage) = self.bound.get(&name) {
                found.push(stage.clone());
            }
        }
        found
    }

    fn finish(self) -> Result<CompiledPlan, CompileError> {
        let mut dag: Dag<()> = Dag::new();
        for stage in &self.stages {
            dag.add_node(stage.id.clone(), ())
                .map_err(|e: DagError| CompileError::Cycle(e.to_string()))?;
        }
        for stage in &self.stages {
            for dep in &stage.deps {
                dag.add_edge(dep.clone(), stage.id.clone())
                    .map_err(|e: DagError| CompileError::Cycle(e.to_string()))?;
            }
        }

        let topo: Vec<String> = dag
            .topological_sort()
            .into_iter()
            .map(|id| id.to_string())
            .collect();
        let mut levels = BTreeMap::new();
        for (level, group) in dag.parallel_levels().into_iter().enumerate() {
            for id in group {
                levels.insert(id.to_string(), level);
            }
        }

        Ok(CompiledPlan {
            stages: self.stages,
            levels,
            topo,
        })
    }
}

/// Flatten `a.b.c` to `"a.b.c"`. `None` when the chain is not a plain path.
fn dotted(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Var(name) => Some(name.clone()),
        Expr::Member(base, field) => Some(format!("{}.{field}", dotted(base)?)),
        _ => None,
    }
}

/// The callee of an expression whose root is a call, as a dotted path.
fn callee_of(expr: &Expr) -> String {
    match expr {
        Expr::Call { target, .. } => dotted(target).unwrap_or_default(),
        _ => String::new(),
    }
}

/// If `target` names a higher-order function, return its final segment.
fn element_scoped_callee(target: &Expr) -> Option<String> {
    let path = dotted(target)?;
    let last = path.rsplit('.').next()?;
    ELEMENT_SCOPED_FUNCTIONS
        .contains(&last)
        .then(|| last.to_string())
}

/// Refuse anything whose meaning depends on the open element-scope decision.
///
/// Two triggers, because they can occur apart: an explicit lambda anywhere in
/// the tree, and a call to a known higher-order function (which the corpus also
/// spells with a bare binder-less predicate, so the lambda check alone would
/// miss it).
fn reject_element_scope(expr: &Expr, stage: &str) -> Result<(), CompileError> {
    match expr {
        Expr::Lambda { params, .. } => {
            return Err(CompileError::LambdaRequiresElementScope {
                params: params.join(", "),
                stage: stage.to_string(),
            })
        }
        Expr::Call { target, .. } => {
            if let Some(function) = element_scoped_callee(target) {
                return Err(CompileError::RequiresElementScope {
                    function,
                    stage: stage.to_string(),
                });
            }
        }
        _ => {}
    }
    for child in children(expr) {
        reject_element_scope(child, stage)?;
    }
    Ok(())
}

/// Every variable name mentioned anywhere in an expression.
fn free_vars(expr: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    if let Expr::Var(name) = expr {
        out.push(name.clone());
    }
    for child in children(expr) {
        out.extend(free_vars(child));
    }
    out
}

/// Direct sub-expressions, so the walkers above stay in one place.
fn children(expr: &Expr) -> Vec<&Expr> {
    match expr {
        Expr::Lit(_) | Expr::Var(_) => Vec::new(),
        Expr::Member(base, _) => vec![base.as_ref()],
        Expr::Array(items) => items.iter().collect(),
        Expr::Record(fields) => fields.iter().map(|(_, value)| value).collect(),
        Expr::Interpolation(parts) => parts.iter().collect(),
        Expr::BinOp(left, _, right) => vec![left.as_ref(), right.as_ref()],
        Expr::Unary(_, inner) => vec![inner.as_ref()],
        Expr::Lambda { body, .. } => vec![body.as_ref()],
        Expr::Call {
            target,
            positional,
            named,
        } => {
            let mut out = vec![target.as_ref()];
            out.extend(positional.iter());
            out.extend(named.values());
            out
        }
        Expr::Pipeline(head, steps) => {
            let mut out = vec![head.as_ref()];
            for step in steps {
                match step {
                    PipeStep::CallStep {
                        target,
                        positional,
                        named,
                    } => {
                        out.push(target.as_ref());
                        out.extend(positional.iter());
                        out.extend(named.values());
                    }
                    PipeStep::Compound(ops) => out.extend(compound_exprs(ops)),
                    PipeStep::VerdictMatch(arms) => {
                        for arm in arms {
                            out.push(arm.target.as_ref());
                            out.extend(arm.positional.iter());
                            out.extend(arm.named.values());
                        }
                    }
                }
            }
            out
        }
    }
}

/// Expressions reachable from a compound block's operations.
fn compound_exprs(ops: &[crate::ast::CompoundOp]) -> Vec<&Expr> {
    use crate::ast::CompoundOp;
    let mut out = Vec::new();
    for op in ops {
        match op {
            CompoundOp::Harvest(expr) => out.push(expr.as_ref()),
            CompoundOp::Promote { condition, .. } => {
                if let Some(condition) = condition {
                    out.push(condition.as_ref());
                }
            }
            CompoundOp::Log { destination, .. } => out.push(destination.as_ref()),
            CompoundOp::Teach { .. } => {}
        }
    }
    out
}
