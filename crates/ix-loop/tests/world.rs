//! The nodes that reach past the model: the voice and GitHub, read only. Scripted stand-ins
//! replace the voice service and `gh`, so no test speaks or reaches the network.

use ix_loop::world::{ScriptedGitHub, ScriptedVoice, MAX_SAYS_PER_RUN};
use ix_loop::{Grant, Host, Scripted, Store};
use serde_json::{json, Value as Json};
use std::rc::Rc;

fn run(host: Host, src: &str) -> Json {
    let host = Rc::new(host);
    let grant = Grant {
        calls: 5,
        fuel: 10_000_000,
        seconds: 60.0,
    };
    host.run(src, "test", &grant)
        .value
        .unwrap_or_else(|e| panic!("the loop failed: {e}"))
}

fn host(dir: &tempfile::TempDir) -> Host {
    Host::new(Rc::new(Scripted::always("hi")), Store::at(dir.path()))
}

/// A canned GitHub: two pull requests, the checks on #2's head, two runs, a log.
fn github() -> ScriptedGitHub {
    ScriptedGitHub::new(|path| {
        let p = path.split('?').next().unwrap_or(path);
        Ok(match p {
            "repos/GuitarAlchemist/ix/pulls" => json!([
                { "number": 1, "title": "one", "draft": false, "user": { "login": "a" },
                  "head": { "ref": "feat/one", "sha": "aaaaaaa1" }, "updated_at": "t1",
                  "html_url": "u1", "body": "not passed on" },
                { "number": 2, "title": "two", "draft": true, "user": { "login": "b" },
                  "head": { "ref": "feat/two", "sha": "bbbbbbb2" }, "updated_at": "t2",
                  "html_url": "u2" },
            ]),
            "repos/GuitarAlchemist/ix/pulls/2" => json!({ "head": { "sha": "bbbbbbb2" } }),
            "repos/GuitarAlchemist/ix/commits/bbbbbbb2/check-runs" => json!({ "check_runs": [
                { "name": "build", "status": "completed", "conclusion": "success", "id": 11,
                  "html_url": "c11" },
                { "name": "risk-report", "status": "completed", "conclusion": "failure",
                  "id": 12, "html_url": "c12" },
            ]}),
            "repos/GuitarAlchemist/ix/actions/runs" => json!({ "workflow_runs": [
                { "id": 7, "name": "CI", "status": "completed", "conclusion": "failure",
                  "head_branch": "feat/two", "head_sha": "bbbbbbb2", "event": "pull_request",
                  "created_at": "t", "html_url": "r7" },
            ]}),
            "repos/GuitarAlchemist/ix/actions/jobs/12/logs" => {
                Json::String("line 1\nline 2\nverdict: fail\n".into())
            }
            other => return Err(format!("gh: HTTP 404 for {other}")),
        })
    })
}

#[test]
fn both_nodes_are_off_unless_the_person_turns_them_on() {
    let dir = tempfile::tempdir().unwrap();
    let v = run(
        host(&dir),
        r#"
        local s_ok, s_err = pcall(say, "bonjour")
        local v_ok, v_err = pcall(voice.stop)
        local g_ok, g_err = pcall(gh.prs, "GuitarAlchemist/ix")
        return { s_ok, tostring(s_err), v_ok, tostring(v_err), g_ok, tostring(g_err) }
    "#,
    );
    assert_eq!(
        (v[0].as_bool(), v[2].as_bool(), v[4].as_bool()),
        (Some(false), Some(false), Some(false))
    );
    assert!(v[1].as_str().unwrap().contains("--voice"), "{v}");
    assert!(v[3].as_str().unwrap().contains("--voice"), "{v}");
    assert!(v[5].as_str().unwrap().contains("--github"), "{v}");
}

#[test]
fn say_speaks_and_is_capped_per_run_children_included() {
    let dir = tempfile::tempdir().unwrap();
    let voice = ScriptedVoice::default();
    let v = run(
        host(&dir).with_voice(Box::new(voice.clone())),
        r#"
        local first = say("Bonjour.", { wait = true })
        local child_ok = run([[ for i = 1, 5 do say("enfant " .. i) end return true ]])
        local spoken, refused = 6, nil
        while true do
          local ok, err = pcall(say, "encore")
          if not ok then refused = tostring(err) break end
          spoken = spoken + 1
        end
        local long_ok, long_err = pcall(say, string.rep("a", 2001))
        return { first = first, child_ok = child_ok, spoken = spoken, refused = refused,
                 long_ok = long_ok, long_err = tostring(long_err) }
    "#,
    );
    assert_eq!(v["first"]["id"], "u1");
    assert_eq!(v["child_ok"], true);
    assert_eq!(v["spoken"], MAX_SAYS_PER_RUN);
    assert!(v["refused"].as_str().unwrap().contains("too many"), "{v}");
    assert_eq!(v["long_ok"], false);
    assert!(
        v["long_err"].as_str().unwrap().contains("2001 characters"),
        "{v}"
    );
    let said = voice.said.borrow();
    assert_eq!(said.len() as u32, MAX_SAYS_PER_RUN);
    assert_eq!(
        (said[0].as_str(), said[1].as_str()),
        ("Bonjour.", "enfant 1")
    );
}

#[test]
fn gh_reads_pull_requests_checks_runs_and_log_tails() {
    let dir = tempfile::tempdir().unwrap();
    let gh = github();
    let v = run(
        host(&dir).with_github(Box::new(gh.clone()), vec!["GuitarAlchemist".into()]),
        r#"
        local repo = "GuitarAlchemist/ix"
        return { prs = gh.prs(repo), checks = gh.checks(repo, 2),
                 runs = gh.runs(repo, { branch = "feat/two" }), log = gh.log(repo, 12, 2) }
    "#,
    );
    assert_eq!(v["prs"].as_array().unwrap().len(), 2);
    assert_eq!(
        v["prs"][1],
        json!({ "number": 2, "title": "two", "draft": true, "author": "b", "branch": "feat/two",
                "sha": "bbbbbbb2", "updated_at": "t2", "url": "u2" })
    );
    let failing: Vec<&Json> = v["checks"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["conclusion"] == "failure")
        .collect();
    assert_eq!(failing.len(), 1);
    assert_eq!(
        (&failing[0]["name"], &failing[0]["job_id"]),
        (&json!("risk-report"), &json!(12))
    );
    assert_eq!(v["runs"][0]["conclusion"], "failure");
    assert_eq!(v["log"], json!(["line 2", "verdict: fail"]));
    let paths = gh.paths.borrow();
    assert_eq!(
        *paths,
        vec![
            "repos/GuitarAlchemist/ix/pulls?state=open&per_page=50",
            "repos/GuitarAlchemist/ix/pulls/2",
            "repos/GuitarAlchemist/ix/commits/bbbbbbb2/check-runs?per_page=100",
            "repos/GuitarAlchemist/ix/actions/runs?per_page=20&branch=feat/two",
            "repos/GuitarAlchemist/ix/actions/jobs/12/logs",
        ]
    );
}

#[test]
fn gh_refuses_other_owners_and_malformed_names_before_any_read() {
    let dir = tempfile::tempdir().unwrap();
    let gh = github();
    let v = run(
        host(&dir).with_github(Box::new(gh.clone()), vec!["GuitarAlchemist".into()]),
        r#"
        local out = {}
        for _, call in ipairs({
          function() return gh.prs("someone-else/repo") end,
          function() return gh.prs("GuitarAlchemist/..") end,
          function() return gh.prs("GuitarAlchemist/ix/pulls/1/merge") end,
          function() return gh.prs("Guitar Alchemist/ix") end,
          function() return gh.runs("GuitarAlchemist/ix", { branch = "a..b" }) end,
          function() return gh.runs("GuitarAlchemist/ix", { branch = "x&per_page=100" }) end,
        }) do
          local ok, err = pcall(call)
          out[#out + 1] = { ok = ok, err = tostring(err) }
        end
        return out
    "#,
    );
    let errs: Vec<&str> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            assert_eq!(r["ok"], false, "{r}");
            r["err"].as_str().unwrap()
        })
        .collect();
    assert!(errs[0].contains("not an allowed owner"), "{errs:?}");
    for e in &errs[1..4] {
        assert!(e.contains("not a repository"), "{errs:?}");
    }
    assert!(
        errs[4].contains("not a branch name") && errs[5].contains("not a branch name"),
        "{errs:?}"
    );
    assert!(
        gh.paths.borrow().is_empty(),
        "a refused call reached GitHub: {:?}",
        gh.paths.borrow()
    );
}
