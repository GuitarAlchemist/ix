//! The engine seam, and the Lua 5.4 sandbox behind it.
//!
//! A sandbox loads only `table`, `string`, `math`, `coroutine` and `utf8` (and the base
//! functions, without `dofile`, `loadfile` and `collectgarbage`). There is no `io`, `os`,
//! `require` or `debug`. `load` takes text only, never a precompiled chunk, and `string.dump` is
//! gone. A state may use 64 MB.
//!
//! Fuel is counted by an instruction hook. Four things could hide a spent budget, and each is
//! closed: a caught error (`pcall`, `xpcall`, `coroutine.resume`, `coroutine.close` raise it again
//! once the budget is spent), a message handler (Lua runs `xpcall`'s handler with the hooks still
//! off when the hook raised the error, so the handler is skipped once the budget is spent), a
//! coroutine (`mlua` hooks one thread at a time, so the host hooks a coroutine while it runs), and
//! a finalizer, whose errors Lua swallows (`__gc` is refused).
//!
//! The hook runs only between Lua instructions. A single costly call into C, such as a string
//! pattern that backtracks a long time, runs to its end before the budget can stop it.

use crate::budget::{Budget, Grant};
use crate::node::Ask;
use crate::oracle::{Task, FAMILIES};
use crate::store::Draft;
use crate::{clip, world, Host};
use mlua::{
    ChunkMode, Debug, HookTriggers, Lua, LuaOptions, StdLib, Table, Thread, Value, Variadic,
    VmState,
};
use serde_json::{json, Value as Json};
use std::rc::Rc;

/// Instructions between two fuel checks.
pub const FUEL_STEP: u32 = 1000;
/// Memory one Lua state may use.
pub const MEMORY_LIMIT: usize = 64 << 20;

/// One loop for an engine to run.
#[derive(Debug, Clone, Copy)]
pub struct Job<'a> {
    pub src: &'a str,
    /// The chunk's name, in error messages.
    pub name: &'a str,
    /// The instance a child is given, as the global `task`.
    pub task: Option<&'a Task>,
    /// Data a child is given, as the global `input`.
    pub input: &'a Json,
}

/// Runs one loop in a fresh sandbox, charged to `budget`, with `host`'s capabilities. Only the
/// JSON value it returns leaves it.
pub trait Engine {
    fn eval(&self, host: &Rc<Host>, budget: &Rc<Budget>, job: &Job<'_>) -> Result<Json, String>;
}

/// Lua 5.4, through `mlua`, compiled from vendored sources.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lua54;

impl Engine for Lua54 {
    fn eval(&self, host: &Rc<Host>, budget: &Rc<Budget>, job: &Job<'_>) -> Result<Json, String> {
        let lua = sandbox(host, budget, job).map_err(|e| clip(&e.to_string(), 600))?;
        let v = lua
            .load(job.src)
            .set_name(job.name)
            .set_mode(ChunkMode::Text)
            .eval::<Value>()
            .map_err(|e| clip(&e.to_string(), 600))?;
        to_json(&v, 0)
    }
}

/// A Lua value as JSON. Functions, threads and userdata cannot leave a loop.
pub fn to_json(v: &Value, depth: usize) -> Result<Json, String> {
    if depth > 32 {
        return Err("a value nested too deep cannot leave a loop".into());
    }
    Ok(match v {
        Value::Nil => Json::Null,
        Value::Boolean(b) => Json::Bool(*b),
        Value::Integer(i) => json!(i),
        Value::Number(n) => json!(n),
        Value::String(s) => Json::String(s.to_string_lossy()),
        Value::Table(t) => {
            let len = t.raw_len();
            let mut pairs = Vec::new();
            for kv in t.clone().pairs::<Value, Value>() {
                pairs.push(kv.map_err(|e| e.to_string())?);
            }
            if len > 0 && pairs.len() == len {
                let mut out = Vec::with_capacity(len);
                for i in 1..=len {
                    let x = t.raw_get::<Value>(i).map_err(|e| e.to_string())?;
                    out.push(to_json(&x, depth + 1)?);
                }
                Json::Array(out)
            } else {
                let mut out = serde_json::Map::new();
                for (k, x) in pairs {
                    let key = match &k {
                        Value::String(s) => s.to_string_lossy(),
                        Value::Integer(i) => i.to_string(),
                        other => {
                            return Err(format!("a {} key cannot leave a loop", other.type_name()))
                        }
                    };
                    out.insert(key, to_json(&x, depth + 1)?);
                }
                Json::Object(out)
            }
        }
        other => return Err(format!("a {} cannot leave a loop", other.type_name())),
    })
}

/// JSON as a Lua value.
pub fn to_lua(lua: &Lua, j: &Json) -> mlua::Result<Value> {
    Ok(match j {
        Json::Null => Value::Nil,
        Json::Bool(b) => Value::Boolean(*b),
        Json::Number(n) => match n.as_i64() {
            Some(i) => Value::Integer(i),
            None => Value::Number(n.as_f64().unwrap_or(f64::NAN)),
        },
        Json::String(s) => Value::String(lua.create_string(s)?),
        Json::Array(a) => {
            let t = lua.create_table()?;
            for (i, x) in a.iter().enumerate() {
                t.raw_set(i + 1, to_lua(lua, x)?)?;
            }
            Value::Table(t)
        }
        Json::Object(o) => {
            let t = lua.create_table()?;
            for (k, x) in o {
                t.raw_set(k.as_str(), to_lua(lua, x)?)?;
            }
            Value::Table(t)
        }
    })
}

fn triggers() -> HookTriggers {
    HookTriggers::new().every_nth_instruction(FUEL_STEP)
}

fn fuel_hook(b: Rc<Budget>) -> impl Fn(&Lua, Debug<'_>) -> mlua::Result<VmState> + 'static {
    move |_, _| {
        b.burn(FUEL_STEP as i64)
            .map(|()| VmState::Continue)
            .map_err(mlua::Error::runtime)
    }
}

fn opt_num(opts: Option<&Table>, key: &str) -> mlua::Result<Option<f64>> {
    match opts {
        Some(o) => o.raw_get::<Option<f64>>(key),
        None => Ok(None),
    }
}

fn opt_str(opts: Option<&Table>, key: &str) -> mlua::Result<Option<String>> {
    match opts {
        Some(o) => o.raw_get::<Option<String>>(key),
        None => Ok(None),
    }
}

/// `{ calls, fuel, seconds }` from a loop's options; what is left out means "all that is left".
fn grant_of(opts: Option<&Table>) -> mlua::Result<Grant> {
    Ok(Grant {
        calls: opt_num(opts, "calls")?.map_or(i64::MAX, |x| x as i64),
        fuel: opt_num(opts, "fuel")?.map_or(i64::MAX, |x| x as i64),
        seconds: opt_num(opts, "seconds")?.unwrap_or(f64::INFINITY),
    })
}

fn task_table(lua: &Lua, task: &Task) -> mlua::Result<Table> {
    let t = lua.create_table()?;
    t.raw_set("name", task.spec())?;
    t.raw_set("family", task.family())?;
    t.raw_set("describe", task.describe())?;
    match task {
        Task::Lipogram { words, letter } => {
            t.raw_set("words", *words)?;
            t.raw_set("letter", letter.to_string())?;
        }
        Task::Acrostic { word, letter } => {
            t.raw_set("word", word.as_str())?;
            t.raw_set("letter", letter.to_string())?;
        }
        Task::Origami { faces, ply } => {
            t.raw_set("faces", *faces)?;
            t.raw_set("ply", *ply)?;
            t.raw_set("grammar", ix_origami::GRAMMAR)?;
        }
    }
    let oracle = task.clone();
    t.raw_set(
        "check",
        lua.create_function(move |_, answer: String| {
            let c = oracle.check(&answer);
            Ok((c.ok, c.why))
        })?,
    )?;
    Ok(t)
}

/// Rewires the functions that could hide a spent budget. It receives three host functions as
/// arguments and keeps them as upvalues, out of the loop's reach: `hook_thread(co)` hooks a
/// coroutine, `hook_here()` hooks the running thread again, and `spent()` says why the budget is
/// gone, or nothing.
const PRELUDE: &str = r#"
local hook_thread, hook_here, spent = ...
local pack, unpack, error, type, rawget = table.pack, table.unpack, error, type, rawget
local pcall0, xpcall0, setmetatable0 = pcall, xpcall, setmetatable
local resume0, close0, create0 = coroutine.resume, coroutine.close, coroutine.create

local function pass(r)
  local why = spent()
  if why then error(why, 0) end
  return unpack(r, 1, r.n)
end

pcall = function(...) return pass(pack(pcall0(...))) end
-- Lua runs a message handler before the protected call unwinds, and when the hook raised the
-- error the hooks are still off: once the budget is spent, the loop's handler is skipped.
xpcall = function(f, handler, ...)
  if type(handler) ~= "function" then return pass(pack(xpcall0(f, handler, ...))) end
  return pass(pack(xpcall0(f, function(m)
    if spent() then return m end
    return handler(m)
  end, ...)))
end

local function resume(co, ...)
  hook_thread(co)
  local r = pack(resume0(co, ...))
  hook_here()
  return pass(r)
end
coroutine.resume = resume

coroutine.close = function(co)
  hook_thread(co)
  local r = pack(close0(co))
  hook_here()
  return pass(r)
end

coroutine.wrap = function(f)
  local co = create0(f)
  return function(...)
    local r = pack(resume(co, ...))
    if r[1] then return unpack(r, 2, r.n) end
    error(r[2], 0)
  end
end

setmetatable = function(t, mt)
  if type(mt) == "table" and rawget(mt, "__gc") ~= nil then
    error("a loop cannot set a finalizer (__gc)", 2)
  end
  return setmetatable0(t, mt)
end
"#;

fn sandbox(host: &Rc<Host>, budget: &Rc<Budget>, job: &Job<'_>) -> mlua::Result<Lua> {
    let lua = Lua::new_with(
        StdLib::TABLE | StdLib::STRING | StdLib::MATH | StdLib::COROUTINE | StdLib::UTF8,
        LuaOptions::default(),
    )?;
    lua.set_memory_limit(MEMORY_LIMIT)?;
    let g = lua.globals();
    for name in ["dofile", "loadfile", "collectgarbage"] {
        g.raw_set(name, Value::Nil)?;
    }
    g.raw_get::<Table>("string")?.raw_set("dump", Value::Nil)?;

    // Text only: a precompiled chunk can break the VM's invariants.
    g.raw_set(
        "load",
        lua.create_function(
            |lua, (chunk, name, _mode, env): (mlua::String, Option<String>, Value, Option<Table>)| {
                let mut c = lua
                    .load(chunk.as_bytes().to_vec())
                    .set_mode(ChunkMode::Text);
                if let Some(n) = name {
                    c = c.set_name(n);
                }
                if let Some(e) = env {
                    c = c.set_environment(e);
                }
                match c.into_function() {
                    Ok(f) => Ok((Value::Function(f), Value::Nil)),
                    Err(e) => Ok((Value::Nil, Value::String(lua.create_string(e.to_string())?))),
                }
            },
        )?,
    )?;

    lua.set_hook(triggers(), fuel_hook(budget.clone()));
    let b = budget.clone();
    let hook_thread = lua.create_function(move |_, co: Thread| {
        co.set_hook(triggers(), fuel_hook(b.clone()));
        Ok(())
    })?;
    let b = budget.clone();
    let hook_here = lua.create_function(move |lua, ()| {
        lua.current_thread()
            .set_hook(triggers(), fuel_hook(b.clone()));
        Ok(())
    })?;
    let b = budget.clone();
    let spent = lua.create_function(move |_, ()| Ok(b.exhausted()))?;
    lua.load(PRELUDE)
        .set_name("=prelude")
        .set_mode(ChunkMode::Text)
        .call::<()>((hook_thread, hook_here, spent))?;

    let prefix = format!(
        "{}[d{}] ",
        "    ".repeat(budget.depth() as usize),
        budget.depth()
    );
    g.raw_set(
        "print",
        lua.create_function(move |_, args: Variadic<Value>| {
            let parts: Vec<String> = args
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.to_string_lossy(),
                    other => to_json(other, 0)
                        .map(|j| j.to_string())
                        .unwrap_or_else(|_| other.type_name().to_string()),
                })
                .collect();
            // stderr: stdout may be a protocol channel (an MCP server), or the run's summary.
            eprintln!("{prefix}{}", parts.join("\t"));
            Ok(())
        })?,
    )?;

    let b = budget.clone();
    g.raw_set(
        "budget",
        lua.create_function(move |lua, ()| {
            let t = lua.create_table()?;
            t.raw_set("calls", b.calls())?;
            t.raw_set("fuel", b.fuel())?;
            t.raw_set("seconds", b.seconds_left())?;
            t.raw_set("depth", b.depth())?;
            Ok(t)
        })?,
    )?;

    let (h, b) = (host.clone(), budget.clone());
    g.raw_set(
        "claude",
        lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
            let o = opts.as_ref();
            let schema = match o {
                Some(t) => match t.raw_get::<Value>("schema")? {
                    Value::Nil => None,
                    s => Some(to_json(&s, 0).map_err(mlua::Error::runtime)?),
                },
                None => None,
            };
            let ask = Ask {
                prompt,
                model: opt_str(o, "model")?.unwrap_or_else(|| "haiku".into()),
                schema,
                system: opt_str(o, "system")?,
            };
            let value = h.ask(&b, &ask).map_err(mlua::Error::runtime)?;
            to_lua(lua, &value)
        })?,
    )?;

    let (h, b) = (host.clone(), budget.clone());
    g.raw_set(
        "run",
        lua.create_function(move |lua, (src, opts): (String, Option<Table>)| {
            let o = opts.as_ref();
            let want = grant_of(o)?;
            let task = match opt_str(o, "task")? {
                Some(spec) => match h.task(&spec) {
                    Some(t) => Some(t),
                    None => return Err(mlua::Error::runtime(format!("run: no task named {spec}"))),
                },
                None => None,
            };
            let input = match o {
                Some(t) => to_json(&t.raw_get::<Value>("input")?, 0)
                    .map_err(|e| mlua::Error::runtime(format!("run: input: {e}")))?,
                None => Json::Null,
            };
            let job = Job {
                src: &src,
                name: "loop",
                task: task.as_ref(),
                input: &input,
            };
            let out = h.child(&b, &want, &job);
            let stats = lua.create_table()?;
            stats.raw_set("calls_used", out.calls_used)?;
            stats.raw_set("fuel_used", out.fuel_used)?;
            stats.raw_set("seconds", out.seconds)?;
            match out.value {
                Ok(j) => Ok((true, to_lua(lua, &j)?, stats)),
                Err(e) => Ok((
                    false,
                    Value::String(lua.create_string(clip(&e, 600))?),
                    stats,
                )),
            }
        })?,
    )?;

    // The voice: off unless the person turned it on.
    let (h, b) = (host.clone(), budget.clone());
    g.raw_set(
        "say",
        lua.create_function(move |lua, (text, opts): (String, Option<Table>)| {
            let wait = match opts.as_ref() {
                Some(t) => t.raw_get::<Option<bool>>("wait")?.unwrap_or(false),
                None => false,
            };
            let v = h.say(&b, &text, wait).map_err(mlua::Error::runtime)?;
            to_lua(lua, &v)
        })?,
    )?;
    let voice = lua.create_table()?;
    let h = host.clone();
    voice.raw_set(
        "stop",
        lua.create_function(move |lua, ()| {
            let v = h.voice_stop().map_err(mlua::Error::runtime)?;
            to_lua(lua, &v)
        })?,
    )?;
    g.raw_set("voice", voice)?;

    // GitHub, read only: off unless the person turned it on, and only for owners they allowed.
    let gh = lua.create_table()?;
    let (h, b) = (host.clone(), budget.clone());
    gh.raw_set(
        "prs",
        lua.create_function(move |lua, repo: String| {
            let v = h
                .gh(&b, "prs", &repo, |get, r| world::prs(get, r))
                .map_err(mlua::Error::runtime)?;
            to_lua(lua, &v)
        })?,
    )?;
    let (h, b) = (host.clone(), budget.clone());
    gh.raw_set(
        "checks",
        lua.create_function(move |lua, (repo, number): (String, u64)| {
            let v = h
                .gh(&b, "checks", &repo, |get, r| world::checks(get, r, number))
                .map_err(mlua::Error::runtime)?;
            to_lua(lua, &v)
        })?,
    )?;
    let (h, b) = (host.clone(), budget.clone());
    gh.raw_set(
        "runs",
        lua.create_function(move |lua, (repo, opts): (String, Option<Table>)| {
            let branch = opt_str(opts.as_ref(), "branch")?;
            let v = h
                .gh(&b, "runs", &repo, |get, r| {
                    world::runs(get, r, branch.as_deref())
                })
                .map_err(mlua::Error::runtime)?;
            to_lua(lua, &v)
        })?,
    )?;
    let (h, b) = (host.clone(), budget.clone());
    gh.raw_set(
        "log",
        lua.create_function(
            move |lua, (repo, job_id, lines): (String, u64, Option<usize>)| {
                let v = h
                    .gh(&b, "log", &repo, |get, r| {
                        world::log(get, r, job_id, lines.unwrap_or(40))
                    })
                    .map_err(mlua::Error::runtime)?;
                to_lua(lua, &v)
            },
        )?,
    )?;
    g.raw_set("gh", gh)?;

    // Stages in other languages: off unless the person turned them on. Each answers like `run`:
    // `ok, value or why, stats`.
    let wasm = lua.create_table()?;
    for lang in ["go", "zig"] {
        let (h, b) = (host.clone(), budget.clone());
        wasm.raw_set(
            lang,
            lua.create_function(move |lua, source: String| {
                match h.wasm_build(&b, lang, &source) {
                    Ok((id, cached)) => {
                        let info = lua.create_table()?;
                        info.raw_set("cached", cached)?;
                        Ok((
                            true,
                            Value::String(lua.create_string(&id)?),
                            Value::Table(info),
                        ))
                    }
                    Err(e) => Ok((false, Value::String(lua.create_string(&e)?), Value::Nil)),
                }
            })?,
        )?;
    }
    let (h, b) = (host.clone(), budget.clone());
    wasm.raw_set(
        "wat",
        lua.create_function(move |lua, text: String| {
            let r = h.wasm_wat(&b, &text);
            let (ok, s) = match r {
                Ok(id) => (true, id),
                Err(e) => (false, e),
            };
            Ok((ok, Value::String(lua.create_string(&s)?)))
        })?,
    )?;
    let (h, b) = (host.clone(), budget.clone());
    wasm.raw_set(
        "run",
        lua.create_function(
            move |lua, (id, input, opts): (String, Value, Option<Table>)| {
                let input = to_json(&input, 0)
                    .map_err(|e| mlua::Error::runtime(format!("wasm.run: input: {e}")))?;
                let fuel = opt_num(opts.as_ref(), "fuel")?.map(|f| f.max(0.0) as u64);
                match h.wasm_run(&b, &id, &input, fuel) {
                    Ok(ran) => {
                        let stats = to_lua(lua, &crate::wasm::stats(&ran))?;
                        match ran.value {
                            Ok(v) => Ok((true, to_lua(lua, &v)?, stats)),
                            Err(e) => Ok((false, Value::String(lua.create_string(&e)?), stats)),
                        }
                    }
                    Err(e) => Ok((false, Value::String(lua.create_string(&e)?), Value::Nil)),
                }
            },
        )?,
    )?;
    g.raw_set("wasm", wasm)?;

    // The store. A loop can propose a draft; it cannot accept one.
    let loops = lua.create_table()?;
    let (h, b) = (host.clone(), budget.clone());
    loops.raw_set(
        "propose",
        lua.create_function(
            move |_, (name, kind, body, note): (String, String, String, Option<String>)| {
                let draft = Draft {
                    name,
                    kind,
                    body,
                    note,
                };
                h.propose(&b, &draft).map_err(mlua::Error::runtime)
            },
        )?,
    )?;
    let h = host.clone();
    loops.raw_set(
        "list",
        lua.create_function(move |lua, ()| {
            let names = h.store().library().into_iter().map(Json::String).collect();
            to_lua(lua, &Json::Array(names))
        })?,
    )?;
    let h = host.clone();
    loops.raw_set(
        "load",
        lua.create_function(move |_, file: String| Ok(h.store().load(&file)))?,
    )?;
    g.raw_set("loops", loops)?;

    // The host's oracles. `get` and `draw` hand an instance out; `judge` draws its own.
    let tasks = lua.create_table()?;
    tasks.raw_set("families", to_lua(&lua, &json!(FAMILIES))?)?;
    let h = host.clone();
    tasks.raw_set(
        "get",
        lua.create_function(move |lua, spec: String| match h.task(&spec) {
            Some(t) => Ok(Some(task_table(lua, &t)?)),
            None => Ok(None),
        })?,
    )?;
    let h = host.clone();
    tasks.raw_set(
        "draw",
        lua.create_function(move |lua, family: String| match h.draw(&family) {
            Some(t) => Ok(Some(task_table(lua, &t)?)),
            None => Ok(None),
        })?,
    )?;
    let (h, b) = (host.clone(), budget.clone());
    tasks.raw_set(
        "judge",
        lua.create_function(
            move |lua, (src, family, opts): (String, String, Option<Table>)| {
                let o = opts.as_ref();
                let n = opt_num(o, "n")?.map_or(3, |x| x.max(0.0) as usize);
                let j = h
                    .judge(&b, &src, &family, n, &grant_of(o)?)
                    .map_err(mlua::Error::runtime)?;
                let results: Vec<Json> = j
                    .results
                    .iter()
                    .map(|r| {
                        json!({ "task": r.task, "ok": r.ok, "why": r.why, "calls_used": r.calls_used })
                    })
                    .collect();
                to_lua(
                    lua,
                    &json!({ "passed": j.passed, "total": j.results.len(), "results": results }),
                )
            },
        )?,
    )?;
    g.raw_set("tasks", tasks)?;

    if let Some(task) = job.task {
        g.raw_set("task", task_table(&lua, task)?)?;
    }
    g.raw_set("input", to_lua(&lua, job.input)?)?;
    Ok(lua)
}
