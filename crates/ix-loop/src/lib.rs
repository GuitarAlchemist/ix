//! Agentic loops written in sandboxed Lua, that can run, judge and write other loops.
//!
//! A **loop** is a Lua chunk. It runs in a fresh sandbox and reaches the world only through the
//! capabilities the [`Host`] gives it, each charged to its [`Budget`]:
//!
//! - `claude(prompt, opts)` asks the [`SemanticNode`] (Claude Code headless, on the person's
//!   plan; a script in tests);
//! - `run(src, opts)` runs a child loop in its own sandbox, with a budget carved out of this one
//!   and given back unspent. Only JSON data crosses between the two: no function, no state;
//! - `tasks.get` / `tasks.draw` hand out [oracles](oracle) the host owns, and `tasks.judge` runs a
//!   loop on instances the host draws and no loop has seen;
//! - `loops.propose` / `loops.list` / `loops.load` reach the [`Store`]. A loop can propose; only
//!   a person, through `ix-loop accept`, moves a proposal into the library;
//! - `say(text, opts)` / `voice.stop()` speak aloud, and `gh.prs` / `gh.checks` / `gh.runs` /
//!   `gh.log` read GitHub. Both are [off](world) unless the person turns them on;
//! - `wasm.go` / `wasm.wat` build a stage in another language and `wasm.run` runs it in a WASI
//!   sandbox, JSON in and JSON out. [Off](wasm) unless the person turns it on.
//!
//! The model drafts, the host's oracles judge, and a person promotes (ADR-0011).

pub mod budget;
pub mod ledger;
pub mod node;
pub mod oracle;
pub mod sandbox;
pub mod store;
pub mod wasm;
pub mod world;

pub use budget::{Budget, Grant, MAX_DEPTH};
pub use ledger::Ledger;
pub use node::{Answer, Ask, ClaudeCode, Scripted, SemanticNode};
pub use oracle::{Check, Task, FAMILIES};
pub use sandbox::{Engine, Job, Lua54};
pub use store::{Draft, Store};
pub use wasm::Stages;
pub use world::{GhCli, GitHub, Voice, VoiceService};

use serde_json::{json, Value as Json};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::rc::Rc;
use std::time::Instant;

/// Proposals one root run may write.
pub const MAX_PROPOSALS_PER_RUN: u32 = 10;
/// Instances one judgement may draw.
pub const MAX_JUDGED: usize = 20;

/// What a run spent and returned.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The loop's value, as JSON, or why it failed.
    pub value: Result<Json, String>,
    /// Semantic-node calls the loop and its children spent.
    pub calls_used: i64,
    pub fuel_used: i64,
    pub seconds: f64,
    /// At list price; nothing is billed per call on a plan.
    pub notional_usd: f64,
}

/// One instance of a judgement.
#[derive(Debug, Clone, PartialEq)]
pub struct Judged {
    pub task: String,
    pub ok: bool,
    pub why: String,
    pub calls_used: i64,
}

/// A loop judged on instances drawn by the host.
#[derive(Debug, Clone, PartialEq)]
pub struct Judgement {
    pub passed: usize,
    pub results: Vec<Judged>,
}

/// What loops share: the semantic node, the store, the ledger, the engine and the host's draws.
pub struct Host {
    node: Rc<dyn SemanticNode>,
    store: Store,
    ledger: Ledger,
    engine: Box<dyn Engine>,
    proposals: Cell<u32>,
    rng: Cell<u64>,
    /// Every instance handed to a loop in this run. A judgement draws outside it.
    seen: RefCell<BTreeSet<String>>,
    voice: Box<dyn Voice>,
    says: Cell<u32>,
    github: Box<dyn GitHub>,
    /// The owners whose repositories `gh` may read.
    owners: Vec<String>,
    gh_reads: Cell<u32>,
    stages: Option<Stages>,
}

pub(crate) fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}

/// The answer a loop gives: its value if a string, else its `answer` field.
fn answer_of(v: &Json) -> Option<&str> {
    v.as_str()
        .or_else(|| v.get("answer").and_then(Json::as_str))
}

impl Host {
    /// A host on `node` and `store`, with no ledger and the Lua 5.4 engine.
    pub fn new(node: Rc<dyn SemanticNode>, store: Store) -> Self {
        Self {
            node,
            store,
            ledger: Ledger::none(),
            engine: Box::new(Lua54),
            proposals: Cell::new(0),
            rng: Cell::new(store::now_ms()),
            seen: RefCell::new(BTreeSet::new()),
            voice: Box::new(world::VOICE_OFF),
            says: Cell::new(0),
            github: Box::new(world::GITHUB_OFF),
            owners: Vec::new(),
            gh_reads: Cell::new(0),
            stages: None,
        }
    }

    /// Turns stages in other languages on.
    pub fn with_wasm(mut self, stages: Stages) -> Self {
        self.stages = Some(stages);
        self
    }

    /// Turns the voice on.
    pub fn with_voice(mut self, voice: Box<dyn Voice>) -> Self {
        self.voice = voice;
        self
    }

    /// Turns GitHub reads on, for repositories of `owners` only.
    pub fn with_github(mut self, github: Box<dyn GitHub>, owners: Vec<String>) -> Self {
        self.github = github;
        self.owners = owners;
        self
    }

    pub fn with_ledger(mut self, ledger: Ledger) -> Self {
        self.ledger = ledger;
        self
    }

    pub fn with_engine(mut self, engine: Box<dyn Engine>) -> Self {
        self.engine = engine;
        self
    }

    /// Fixes the host's draws, for tests.
    pub fn with_seed(self, seed: u64) -> Self {
        self.rng.set(seed);
        self
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Runs `src` as a root loop with `grant`.
    pub fn run(self: &Rc<Self>, src: &str, name: &str, grant: &Grant) -> Outcome {
        let started = Instant::now();
        let budget = Rc::new(Budget::root(grant));
        let job = Job {
            src,
            name,
            task: None,
            input: &Json::Null,
        };
        let value = self.engine.eval(self, &budget, &job);
        Outcome {
            value,
            calls_used: budget.calls_used(),
            fuel_used: grant.fuel.max(0) - budget.fuel(),
            seconds: started.elapsed().as_secs_f64(),
            notional_usd: budget.notional_usd(),
        }
    }

    /// Runs a child loop on a budget carved out of `parent`'s, and gives back what it left.
    pub fn child(self: &Rc<Self>, parent: &Budget, want: &Grant, job: &Job<'_>) -> Outcome {
        let started = Instant::now();
        let Some(budget) = parent.carve(want) else {
            return Outcome {
                value: Err(format!(
                    "budget: too deep (a loop at depth {MAX_DEPTH} cannot start another)"
                )),
                calls_used: 0,
                fuel_used: 0,
                seconds: 0.0,
                notional_usd: 0.0,
            };
        };
        let budget = Rc::new(budget);
        let granted = budget.fuel();
        let value = self.engine.eval(self, &budget, job);
        parent.refund(&budget);
        self.ledger.record(json!({
            "kind": "run", "depth": budget.depth(), "ok": value.is_ok(),
            "calls_used": budget.calls_used(), "fuel_used": granted - budget.fuel(),
        }));
        Outcome {
            value,
            calls_used: budget.calls_used(),
            fuel_used: granted - budget.fuel(),
            seconds: started.elapsed().as_secs_f64(),
            notional_usd: budget.notional_usd(),
        }
    }

    /// Asks the semantic node, charging one call to `budget`.
    pub fn ask(&self, budget: &Budget, ask: &Ask) -> Result<Json, String> {
        budget.take_call()?;
        let started = Instant::now();
        let answer = self.node.ask(ask, budget.deadline());
        if let Ok(a) = &answer {
            budget.add_notional(a.notional_usd);
        }
        self.ledger.record(json!({
            "kind": "ask", "depth": budget.depth(), "model": ask.model,
            "ms": started.elapsed().as_millis() as u64, "prompt_chars": ask.prompt.chars().count(),
            "notional_usd": answer.as_ref().map(|a| a.notional_usd).unwrap_or(0.0),
            "ok": answer.is_ok(),
        }));
        answer.map(|a| a.value)
    }

    /// Speaks `text`; at most [`world::MAX_SAYS_PER_RUN`] utterances per run.
    pub fn say(&self, budget: &Budget, text: &str, wait: bool) -> Result<Json, String> {
        let chars = text.chars().count();
        if chars > world::MAX_SAY_CHARS {
            return Err(format!(
                "say: {chars} characters; one say speaks at most {}",
                world::MAX_SAY_CHARS
            ));
        }
        if self.says.get() >= world::MAX_SAYS_PER_RUN {
            return Err("say: too many utterances in one run".into());
        }
        self.says.set(self.says.get() + 1);
        let out = self.voice.say(text, wait);
        self.ledger.record(json!({
            "kind": "say", "depth": budget.depth(), "chars": chars, "ok": out.is_ok(),
        }));
        out
    }

    /// Cuts what the voice is saying.
    pub fn voice_stop(&self) -> Result<Json, String> {
        self.voice.stop()
    }

    /// Runs one `gh` read: `read` gets the checked repository and a counted GET.
    pub fn gh(
        &self,
        budget: &Budget,
        what: &str,
        repo: &str,
        read: impl FnOnce(&dyn Fn(&str) -> Result<Json, String>, &str) -> Result<Json, String>,
    ) -> Result<Json, String> {
        if self.owners.is_empty() {
            return Err(world::GITHUB_OFF.0.to_string());
        }
        let repo = world::checked_repo(repo, &self.owners)?;
        let get = |path: &str| {
            if self.gh_reads.get() >= world::MAX_GH_PER_RUN {
                return Err("gh: too many reads in one run".to_string());
            }
            self.gh_reads.set(self.gh_reads.get() + 1);
            self.github.get(path, budget.deadline())
        };
        let out = read(&get, &repo);
        self.ledger.record(json!({
            "kind": "gh", "what": what, "repo": repo, "depth": budget.depth(), "ok": out.is_ok(),
        }));
        out
    }

    fn stages(&self) -> Result<&Stages, String> {
        self.stages
            .as_ref()
            .ok_or_else(|| wasm::WASM_OFF.to_string())
    }

    /// Builds Go source into a stage: its id, and whether an earlier build was reused.
    pub fn wasm_go(&self, budget: &Budget, source: &str) -> Result<(String, bool), String> {
        let started = Instant::now();
        let out = self.stages()?.go(budget, source);
        self.ledger.record(json!({
            "kind": "wasm.go", "depth": budget.depth(), "ok": out.is_ok(),
            "id": out.as_ref().ok().map(|(id, _)| id.clone()),
            "cached": out.as_ref().ok().map(|(_, c)| *c),
            "ms": started.elapsed().as_millis() as u64,
        }));
        out
    }

    /// Checks WAT text and stores it as a stage: its id.
    pub fn wasm_wat(&self, budget: &Budget, text: &str) -> Result<String, String> {
        let out = self.stages()?.wat(text);
        self.ledger.record(json!({
            "kind": "wasm.wat", "depth": budget.depth(), "ok": out.is_ok(),
            "id": out.as_ref().ok(),
        }));
        out
    }

    /// Runs stage `id` on `input`, charging its fuel to `budget`.
    pub fn wasm_run(
        &self,
        budget: &Budget,
        id: &str,
        input: &Json,
        fuel: Option<u64>,
    ) -> Result<wasm::Ran, String> {
        let started = Instant::now();
        let out = self.stages()?.run(budget, id, input, fuel);
        self.ledger.record(json!({
            "kind": "wasm.run", "depth": budget.depth(), "id": id,
            "ok": out.as_ref().is_ok_and(|r| r.value.is_ok()),
            "fuel_used": out.as_ref().map(|r| r.fuel_used).unwrap_or(0),
            "ms": started.elapsed().as_millis() as u64,
        }));
        out
    }

    /// Stores a loop's proposal; at most [`MAX_PROPOSALS_PER_RUN`] per run.
    pub fn propose(&self, budget: &Budget, draft: &Draft) -> Result<String, String> {
        if self.proposals.get() >= MAX_PROPOSALS_PER_RUN {
            return Err("propose: too many proposals in one run".into());
        }
        let id = self
            .store
            .propose(draft, &format!("loop at depth {}", budget.depth()))?;
        self.proposals.set(self.proposals.get() + 1);
        self.ledger
            .record(json!({ "kind": "propose", "id": id, "name": draft.name }));
        Ok(id)
    }

    /// Hands instance `spec` to a loop, and remembers that a loop has seen it.
    pub fn task(&self, spec: &str) -> Option<Task> {
        let task = Task::parse(spec)?;
        self.seen.borrow_mut().insert(task.spec());
        Some(task)
    }

    /// Draws an instance of `family` no loop has seen in this run, and hands it out.
    pub fn draw(&self, family: &str) -> Option<Task> {
        for _ in 0..256 {
            let seed = self.rng.get().wrapping_add(1);
            self.rng.set(seed);
            let task = Task::draw(family, seed)?;
            if self.seen.borrow_mut().insert(task.spec()) {
                return Some(task);
            }
        }
        None
    }

    /// Runs `src` once on each of `n` instances of `family` drawn by the host, each as a child of
    /// `parent` with `each`, and checks each answer with the host's oracle. The loop's own claims
    /// are never read: only its answer (its value, or its `answer` field).
    pub fn judge(
        self: &Rc<Self>,
        parent: &Budget,
        src: &str,
        family: &str,
        n: usize,
        each: &Grant,
    ) -> Result<Judgement, String> {
        if !(1..=MAX_JUDGED).contains(&n) {
            return Err(format!("judge: draw 1 to {MAX_JUDGED} instances"));
        }
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            let task = self
                .draw(family)
                .ok_or_else(|| format!("judge: cannot draw an unseen {family} instance"))?;
            let job = Job {
                src,
                name: "draft",
                task: Some(&task),
                input: &Json::Null,
            };
            let out = self.child(parent, each, &job);
            let (ok, why) = match &out.value {
                Ok(v) => match answer_of(v) {
                    Some(a) => {
                        let c = task.check(a);
                        (c.ok, c.why)
                    }
                    None => (false, "the loop gave no answer".to_string()),
                },
                Err(e) => (false, format!("the loop failed: {}", clip(e, 200))),
            };
            results.push(Judged {
                task: task.spec(),
                ok,
                why,
                calls_used: out.calls_used,
            });
        }
        let passed = results.iter().filter(|r| r.ok).count();
        self.ledger.record(json!({
            "kind": "judge", "family": family, "passed": passed, "total": n,
        }));
        Ok(Judgement { passed, results })
    }
}
