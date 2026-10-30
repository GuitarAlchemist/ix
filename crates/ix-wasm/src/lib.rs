//! Pipeline stages in any language that targets WASI, run sandboxed.
//!
//! A **stage** is a WASI command module (`_start`): it reads its input on stdin and writes its
//! output on stdout, by convention one JSON value each way. It runs in [wasmi], an interpreter
//! written in Rust, and sees only what [the shim](wasi) gives it:
//!
//! - stdin, stdout and stderr, and nothing else: no preopened directory, so no file system, and
//!   no socket. Every preview 1 function is linked, so a module from any toolchain loads, but
//!   the file and socket functions answer an error ([`Program::closed`] lists those a module
//!   imports). A module importing anything outside preview 1 is refused when it is loaded;
//! - one argument (`stage`) and an empty environment;
//! - a clock that starts at 1 ns and moves 1 µs per reading and by the whole timeout of every
//!   sleep, and a random source seeded by [`Limits::seed`]. Two runs on the same input give the
//!   same output and spend the same fuel.
//!
//! Every run is bounded by [`Limits`]: fuel (about one unit per instruction), memory, and output.
//!
//! [`Program`] loads a module once (binary, or WAT text) and runs it many times. [`build`]
//! builds Go or Zig source into a stage, in a cache keyed by the source and the toolchain.
//! [`skill`] makes a stage one node of an `ix.yaml` pipeline (`wasm.run`).

pub mod build;
pub mod skill;
mod wasi;

use std::fmt;
use wasmi::{
    CompilationMode, Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder,
    TrapCode,
};

/// What one run may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Fuel: about one unit per instruction. 2e9 is about 1.5 s on a desktop.
    pub fuel: u64,
    /// The bytes a run may write to stdout and stderr together.
    pub max_output: usize,
    /// The bytes of linear memory a module may hold.
    pub max_memory: usize,
    /// Seeds the random source.
    pub seed: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            fuel: 2_000_000_000,
            max_output: 16 << 20,
            max_memory: 256 << 20,
            seed: 0x001D_5EED,
        }
    }
}

/// The largest module [`Program::new`] reads.
pub const MAX_MODULE_BYTES: usize = 64 << 20;

/// What a run wrote, and what it spent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    /// The exit status: 0 when `_start` returned, else what `proc_exit` was given.
    pub exit: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub fuel_used: u64,
    /// The closed functions (files, sockets, signals) the run called on anything but stdio, and
    /// was told no: what it tried and did not get.
    pub refused: Vec<String>,
}

/// Why a module did not load or a run did not finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Not a module: bad bytes, bad WAT, or too large.
    Invalid(String),
    /// The module asks for something the sandbox does not give.
    Refused(String),
    /// The run spent all its fuel.
    OutOfFuel { fuel_used: u64 },
    /// The run wrote more than [`Limits::max_output`].
    OutputTooLarge { limit: usize },
    /// The run trapped: an unreachable, a bad memory access, a stack overflow, memory refused.
    Trap { message: String, stderr: Vec<u8> },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(m) => write!(f, "wasm: not a module: {m}"),
            Error::Refused(m) => write!(f, "wasm: refused: {m}"),
            Error::OutOfFuel { fuel_used } => write!(f, "wasm: out of fuel after {fuel_used}"),
            Error::OutputTooLarge { limit } => write!(f, "wasm: wrote more than {limit} bytes"),
            Error::Trap { message, .. } => write!(f, "wasm: trap: {message}"),
        }
    }
}

impl std::error::Error for Error {}

/// A loaded module, ready to run any number of times. Each run gets a fresh instance.
pub struct Program {
    engine: Engine,
    module: Module,
    linker: Linker<wasi::Ctx>,
    closed: Vec<String>,
}

impl Program {
    /// Loads a module from its binary form or its WAT text, and refuses it if it imports
    /// anything the sandbox does not give or has no `_start`.
    pub fn new(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > MAX_MODULE_BYTES {
            return Err(Error::Invalid(format!(
                "{} bytes; a module is at most {MAX_MODULE_BYTES}",
                bytes.len()
            )));
        }
        let mut config = Config::default();
        config.consume_fuel(true);
        // Translated whole when loaded: a lazy translation charges its work to the first run's
        // fuel, and two runs on one input would not spend the same.
        config.compilation_mode(CompilationMode::Eager);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, bytes).map_err(|e| Error::Invalid(e.to_string()))?;
        let mut closed = Vec::new();
        for import in module.imports() {
            let (m, n) = (import.module(), import.name());
            if m == wasi::MODULE && wasi::CLOSED.contains(&n) {
                closed.push(n.to_string());
            } else if m != wasi::MODULE || !wasi::GIVEN.contains(&n) {
                return Err(Error::Refused(format!(
                    "it imports {m}.{n}, which the sandbox does not give (it gives stdin, stdout, \
                     stderr, args, a clock and a random source: no files, no network)"
                )));
            }
        }
        if !module.exports().any(|e| e.name() == "_start") {
            return Err(Error::Refused(
                "it has no _start: a stage is a WASI command".into(),
            ));
        }
        let mut linker = Linker::new(&engine);
        wasi::link(&mut linker).map_err(|e| Error::Invalid(e.to_string()))?;
        Ok(Self {
            engine,
            module,
            linker,
            closed,
        })
    }

    /// The preview 1 functions this module imports that answer only an error (files, sockets,
    /// signals). A stage that calls one is told no; the list is for the record of a run.
    pub fn closed(&self) -> &[String] {
        &self.closed
    }

    /// The module's imports, as `module.name`.
    pub fn imports(&self) -> Vec<String> {
        self.module
            .imports()
            .map(|i| format!("{}.{}", i.module(), i.name()))
            .collect()
    }

    /// Runs the module once on `stdin`.
    pub fn run(&self, stdin: &[u8], limits: &Limits) -> Result<Output, Error> {
        let ctx = wasi::Ctx::new(stdin.to_vec(), limits, store_limits(limits));
        let mut store = Store::new(&self.engine, ctx);
        store.limiter(|ctx| &mut ctx.limits);
        store
            .set_fuel(limits.fuel)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let result = self
            .linker
            .instantiate_and_start(&mut store, &self.module)
            .and_then(|instance| {
                instance
                    .get_typed_func::<(), ()>(&store, "_start")?
                    .call(&mut store, ())
            });
        let fuel_used = limits.fuel - store.get_fuel().unwrap_or(0);
        let ctx = store.into_data();
        let exit = match result {
            Ok(()) => 0,
            Err(e) => {
                if let Some(code) = e.i32_exit_status() {
                    code
                } else if ctx.overflow {
                    return Err(Error::OutputTooLarge {
                        limit: limits.max_output,
                    });
                } else if e.as_trap_code() == Some(TrapCode::OutOfFuel) {
                    return Err(Error::OutOfFuel { fuel_used });
                } else {
                    return Err(Error::Trap {
                        message: e.to_string(),
                        stderr: ctx.stderr,
                    });
                }
            }
        };
        Ok(Output {
            exit,
            stdout: ctx.stdout,
            stderr: ctx.stderr,
            fuel_used,
            refused: ctx.refused.iter().map(|f| f.to_string()).collect(),
        })
    }
}

fn store_limits(limits: &Limits) -> StoreLimits {
    StoreLimitsBuilder::new()
        .memory_size(limits.max_memory)
        .memories(1)
        .instances(1)
        .build()
}
