//! What crosses a sandbox's edge: JSON in and out of a loop, and what a loop may write to the
//! store.

use ix_loop::store::{MAX_BODY_BYTES, MAX_NOTE_BYTES};
use ix_loop::{Ask, Grant, Host, Scripted, Store};
use serde_json::{json, Value as Json};
use std::rc::Rc;

fn grant() -> Grant {
    Grant {
        calls: 5,
        fuel: 100_000_000,
        seconds: 60.0,
    }
}

#[test]
fn a_nested_json_null_survives_the_trip_through_a_loop() {
    let dir = tempfile::tempdir().unwrap();
    let reply = json!({ "answer": null, "list": [1, null, 3], "deep": { "x": null } });
    let sent = reply.clone();
    let node = Rc::new(Scripted::new(move |_: &Ask| Ok(sent.clone())));
    let host = Rc::new(Host::new(node, Store::at(dir.path())));
    let out = host.run(
        r#"
        local r = claude("x", { schema = { type = "object" } })
        local ok, child = run("return input", { input = r })
        return { r = r, child = child, is_null = r.answer == null, n = #r.list,
                 built = { a = null, b = { null } } }
        "#,
        "test",
        &grant(),
    );
    let v = out.value.unwrap();
    assert_eq!(v["r"], reply, "{v}");
    assert_eq!(v["child"], reply, "{v}");
    assert_eq!(v["is_null"], json!(true));
    assert_eq!(v["n"], json!(3));
    assert_eq!(v["built"], json!({ "a": null, "b": [null] }));
}

#[test]
fn a_loop_cannot_propose_an_oversized_field() {
    let dir = tempfile::tempdir().unwrap();
    let host = Rc::new(Host::new(
        Rc::new(Scripted::always("hi")),
        Store::at(dir.path()),
    ));
    let src = format!(
        r#"
        local function try(...)
          local ok, err = pcall(loops.propose, ...)
          return ok or tostring(err)
        end
        return {{
          name = try(string.rep("a", 1000000), "lua", "return 1"),
          kind = try("k", string.rep("lua", 1000000), "return 1"),
          body = try("b", "lua", string.rep("x", {body})),
          note = try("n", "lua", "return 1", string.rep("x", {note})),
          fits = try("f", "lua", string.rep("x", {fit_body}), string.rep("x", {fit_note})),
        }}
        "#,
        body = MAX_BODY_BYTES + 1,
        note = MAX_NOTE_BYTES + 1,
        fit_body = MAX_BODY_BYTES,
        fit_note = MAX_NOTE_BYTES,
    );
    let v = host.run(&src, "test", &grant()).value.unwrap();
    for field in ["name", "kind", "body", "note"] {
        let err = v[field].as_str().unwrap_or_default();
        assert!(
            err.contains(&format!("a {field} holds at most")),
            "{field}: {v}"
        );
    }
    assert_eq!(v["fits"], json!(true), "{v}");
    let proposals: Vec<Json> = host.store().proposals();
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0]["name"], json!("f"));
}
