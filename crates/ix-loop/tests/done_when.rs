//! The slice's acceptance tests, one per "done when" line of the spec. None needs `claude`:
//! the semantic node is a [`Scripted`] one.

use ix_loop::{Ask, ClaudeCode, Grant, Host, Scripted, Store, Task, MAX_DEPTH};
use serde_json::{json, Value as Json};
use std::rc::Rc;

fn grant(calls: i64, fuel: i64) -> Grant {
    Grant {
        calls,
        fuel,
        seconds: 60.0,
    }
}

fn host_with(node: Rc<Scripted>, dir: &tempfile::TempDir) -> Rc<Host> {
    Rc::new(Host::new(node, Store::at(dir.path())).with_seed(7))
}

fn run_with(src: &str, fuel: i64) -> Json {
    let dir = tempfile::tempdir().unwrap();
    let host = host_with(Rc::new(Scripted::always("hi")), &dir);
    host.run(src, "test", &grant(5, fuel))
        .value
        .unwrap_or_else(|e| panic!("the loop failed: {e}"))
}

fn run(src: &str) -> Json {
    run_with(src, 10_000_000)
}

fn text(v: &Json) -> &str {
    v.as_str().unwrap_or_default()
}

#[test]
fn done_1_a_sandboxed_loop_has_no_io_os_require_dofile_or_loadfile() {
    let v = run(r#"
        local names = { "io", "os", "require", "dofile", "loadfile", "debug", "package",
                        "collectgarbage" }
        local present = {}
        for _, n in ipairs(names) do
          if _G[n] ~= nil then present[#present + 1] = n end
        end
        local fn, err = load(string.char(27) .. "Lua")
        local ok_gc = pcall(setmetatable, {}, { __gc = function() end })
        return { present = present, dump = string.dump == nil, binary = fn == nil,
                 binary_err = err, gc = ok_gc, text = load("return 1 + 1")() }
    "#);
    assert_eq!(v["present"], json!({}), "globals that should be gone: {v}");
    assert_eq!(v["dump"], json!(true));
    assert_eq!(v["binary"], json!(true), "a precompiled chunk was loaded");
    assert!(text(&v["binary_err"]).contains("binary"), "{v}");
    assert_eq!(v["gc"], json!(false), "a finalizer was accepted");
    assert_eq!(v["text"], json!(2));
}

#[test]
fn done_2_an_endless_loop_is_stopped_by_its_fuel_and_the_parent_gets_the_rest_back() {
    let v = run_with(
        r#"
        local before = budget().fuel
        local ok, err, stats = run("while true do end", { fuel = 100000 })
        local mid = budget().fuel
        local ok2, v2 = run("return 1", { fuel = 1000000 })
        local after = budget().fuel
        local hidden = {
          caught = "while true do pcall(function() while true do end end) end",
          handled = "xpcall(function() while true do end end, function() while true do end end)",
          handled_after = "xpcall(function() error('x') end, function() while true do end end)",
          coroutine = "coroutine.wrap(function() while true do end end)()",
          resumed = "while true do coroutine.resume(coroutine.create(function() while true do end end)) end",
          closed = [[
            local co = coroutine.create(function()
              local x <close> = setmetatable({}, { __close = function() while true do end end })
              coroutine.yield()
            end)
            coroutine.resume(co)
            coroutine.close(co)
          ]],
        }
        local escaped = {}
        for name, src in pairs(hidden) do
          local ok3, err3 = run(src, { fuel = 100000 })
          if ok3 or not tostring(err3):find("out of fuel", 1, true) then
            escaped[#escaped + 1] = name .. ": " .. tostring(err3)
          end
        end
        local ok4, err4 = run("while true do end", { seconds = 0.2, fuel = 1e9 })
        return { ok = ok, err = err, used = stats.fuel_used, spent = before - mid,
                 back = mid - after, ok2 = ok2, v2 = v2, escaped = escaped,
                 timed = not ok4 and err4:find("out of time", 1, true) ~= nil }
    "#,
        2_000_000_000,
    );
    assert_eq!(v["ok"], json!(false));
    assert!(text(&v["err"]).contains("out of fuel"), "{v}");
    assert_eq!(v["used"], json!(100_000));
    let spent = v["spent"].as_i64().unwrap();
    assert!((100_000..=101_000).contains(&spent), "{v}");
    // The short child spent less than one fuel step, and its whole grant came back.
    let back = v["back"].as_i64().unwrap();
    assert!((0..=1_000).contains(&back), "{v}");
    assert_eq!((v["ok2"].clone(), v["v2"].clone()), (json!(true), json!(1)));
    assert_eq!(v["escaped"], json!({}), "{v}");
    assert_eq!(v["timed"], json!(true), "{v}");
}

#[test]
fn done_3_a_child_cannot_spend_more_calls_than_it_was_given_nor_the_tree_more_than_the_root() {
    let dir = tempfile::tempdir().unwrap();
    let node = Rc::new(Scripted::always("hi"));
    let host = host_with(node.clone(), &dir);
    let greedy = r#"
        local n = 0
        while pcall(claude, "again") do n = n + 1 end
        local ok, v = run(input.src, { calls = 100, input = input })
        return { mine = n, child = ok and v or v }
    "#;
    let src = format!(
        r#"
        local greedy = {greedy:?}
        local ok, v, stats = run(greedy, {{ calls = 2, input = {{ src = greedy }} }})
        local ok2, v2, stats2 = run(greedy, {{ calls = 100, input = {{ src = greedy }} }})
        local tail = 0
        while pcall(claude, "more") do tail = tail + 1 end
        return {{ first = v, first_used = stats.calls_used, second = v2,
                 second_used = stats2.calls_used, tail = tail, left = budget().calls }}
        "#
    );
    let out = host.run(&src, "test", &grant(3, 10_000_000));
    let v = out.value.unwrap();
    assert_eq!(v["first"]["mine"], json!(2), "{v}");
    assert_eq!(v["first"]["child"]["mine"], json!(0), "{v}");
    assert_eq!(v["first_used"], json!(2));
    assert_eq!(v["second"]["mine"], json!(1), "{v}");
    assert_eq!(v["second_used"], json!(1));
    assert_eq!((v["tail"].clone(), v["left"].clone()), (json!(0), json!(0)));
    assert_eq!(
        node.calls(),
        3,
        "the tree asked more than the root's budget"
    );
    assert_eq!(out.calls_used, 3);
}

#[test]
fn done_4_the_depth_limit_holds_and_a_function_cannot_leave_a_loop() {
    let v = run(r#"
        local me = [[
          local ok, v = run(input.src, { input = input })
          return { depth = budget().depth, child = v }
        ]]
        local ok, chain = run(me, { input = { src = me } })
        local okf, errf = run("return function() end")
        local okt, errt = run("return coroutine.create(function() end)")
        local oki, erri = pcall(run, "return 1", { input = { f = print } })
        return { chain = chain, okf = okf, errf = errf, okt = okt, errt = errt,
                 oki = oki, erri = tostring(erri) }
    "#);
    let mut node = &v["chain"];
    for depth in 1..=MAX_DEPTH {
        assert_eq!(node["depth"], json!(depth), "{v}");
        node = &node["child"];
    }
    assert!(text(node).contains("too deep"), "{v}");
    assert_eq!(v["okf"], json!(false));
    assert!(
        text(&v["errf"]).contains("function cannot leave a loop"),
        "{v}"
    );
    assert_eq!(v["okt"], json!(false));
    assert!(text(&v["errt"]).contains("cannot leave a loop"), "{v}");
    assert_eq!(v["oki"], json!(false));
    assert!(text(&v["erri"]).contains("cannot leave a loop"), "{v}");
}

#[test]
fn done_5_a_loop_can_propose_and_nothing_reachable_from_lua_accepts() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_with(Rc::new(Scripted::always("hi")), &dir);
    let out = host.run(
        r#"
        local id = loops.propose("count-up", "lua", "return 1", "a first loop")
        local bad = pcall(loops.propose, "../library/x", "lua", "return 1")
        local names, found = {}, {}
        local function walk(t, path, depth)
          for k, v in pairs(t) do
            local name = path .. tostring(k)
            if tostring(k):lower():find("accept", 1, true) then found[#found + 1] = name end
            if type(v) == "table" and depth < 2 and v ~= _G then walk(v, name .. ".", depth + 1) end
          end
        end
        walk(_G, "", 0)
        for k in pairs(loops) do names[#names + 1] = k end
        table.sort(names)
        return { id = id, bad = bad, loops = names, found = found, library = loops.list() }
        "#,
        "test",
        &grant(5, 10_000_000),
    );
    let v = out.value.unwrap();
    let id = text(&v["id"]).to_string();
    assert!(dir
        .path()
        .join("proposals")
        .join(format!("{id}.json"))
        .exists());
    assert_eq!(v["bad"], json!(false), "a name left the store");
    assert_eq!(v["loops"], json!(["list", "load", "propose"]));
    assert_eq!(v["found"], json!({}), "{v}");
    assert_eq!(v["library"], json!({}));
    assert!(host.store().library().is_empty());

    // A person accepts; the next loop can load it.
    host.store().accept(&id).unwrap();
    let out = host.run(
        r#"return { list = loops.list(), body = loops.load("count-up.lua") }"#,
        "test",
        &grant(5, 10_000_000),
    );
    let v = out.value.unwrap();
    assert_eq!(v["list"], json!(["count-up.lua"]));
    assert_eq!(v["body"], json!("return 1"));
}

#[test]
fn done_6_a_draft_that_hardcodes_the_answer_it_was_shown_fails_on_hidden_instances() {
    // Run 1 of the experiment: the writer saw one instance and returned its answer, not a loop.
    let hardcoded =
        r#"return { answer = "A big cat with soft fur sat on a warm mat today", ok = true }"#;
    let general = r#"
        for _, w in ipairs({ "go", "up", "my", "by", "it" }) do
          if not w:find(task.letter, 1, true) then
            return { answer = string.rep(w, task.words, " ") }
          end
        end
    "#;
    let meta = r#"
        local shown = tasks.get("lipogram:12:e")
        local draft = claude("Write a Lua loop for this task. " .. shown.describe)
        local ok, v = run(draft, { task = shown.name })
        local on_shown = ok and shown.check(v.answer)
        local judged = tasks.judge(draft, "lipogram", { n = 5 })
        return { on_shown = on_shown, judged = judged }
    "#;
    let mut outcomes = Vec::new();
    for draft in [hardcoded, general] {
        let dir = tempfile::tempdir().unwrap();
        let reply = draft.to_string();
        let node = Rc::new(Scripted::new(move |_: &Ask| {
            Ok(Json::String(reply.clone()))
        }));
        let host = host_with(node.clone(), &dir);
        let v = host.run(meta, "meta", &grant(5, 10_000_000)).value.unwrap();
        assert_eq!(node.calls(), 1);
        assert!(node.asks()[0].prompt.contains("exactly 12 words"));
        for r in v["judged"]["results"].as_array().unwrap() {
            assert_ne!(
                r["task"],
                json!("lipogram:12:e"),
                "a hidden instance was the shown one"
            );
            assert!(Task::parse(text(&r["task"])).is_some());
        }
        outcomes.push(v);
    }
    let (cheat, honest) = (&outcomes[0], &outcomes[1]);
    assert_eq!(cheat["on_shown"], json!(true), "{cheat}");
    assert_eq!(cheat["judged"]["total"], json!(5));
    assert_eq!(cheat["judged"]["passed"], json!(0), "{cheat}");
    assert_eq!(honest["on_shown"], json!(true), "{honest}");
    assert_eq!(honest["judged"]["passed"], json!(5), "{honest}");
}

#[test]
fn done_7_the_claude_code_command_line_has_the_flags_and_never_bare() {
    let cc = ClaudeCode {
        bin: "claude".into(),
        work_dir: ".".into(),
    };
    let mut ask = Ask {
        prompt: "the prompt goes on stdin".into(),
        model: "haiku".into(),
        schema: None,
        system: None,
    };
    let plain = cc.args(&ask);
    let head: Vec<&str> = plain.iter().map(String::as_str).collect();
    assert_eq!(
        head,
        [
            "-p",
            "--safe-mode",
            "--tools",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--model",
            "haiku"
        ]
    );
    assert!(!plain.iter().any(|a| a.contains("json-schema")));
    ask.schema = Some(json!({ "type": "object" }));
    ask.system = Some("Answer in English.".into());
    let full = cc.args(&ask);
    let at = full.iter().position(|a| a == "--json-schema").unwrap();
    assert_eq!(full[at + 1], r#"{"type":"object"}"#);
    assert!(full.iter().any(|a| a == "--system-prompt"));
    for args in [&plain, &full] {
        assert!(!args.iter().any(|a| a == "--bare"), "{args:?}");
        assert!(
            !args.iter().any(|a| a.contains("the prompt")),
            "the prompt is on argv"
        );
    }
}
