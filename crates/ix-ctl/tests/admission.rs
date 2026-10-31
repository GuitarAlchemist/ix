//! Plan admission (`ix_ctl::admission`), one test per guarantee of Gaia's IX-PLAN-ADMISSION v1:
//! a recording runner stands in for the worker, so "nothing launched" is a count of zero.

use ix_ctl::admission::{
    admit, apply, resolve, Admitted, ArtifactRef, Code, DirStore, Limits, Mandate, Runner, Store,
    World,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const DIGEST: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";

/// An in-memory store: one artifact, `echo-worker` 1.4.2 for windows-x86_64.
struct OneArtifact(String);

impl Store for OneArtifact {
    fn find(&self, a: &ArtifactRef) -> Option<(PathBuf, String)> {
        (a.name == "echo-worker" && a.version == "1.4.2" && a.platform == "windows-x86_64")
            .then(|| (PathBuf::from("store/echo-worker"), self.0.clone()))
    }
}

#[derive(Default)]
struct Recorder(Vec<String>);

impl Runner for Recorder {
    fn launch(&mut self, digest: &str, _artifact: &Path, _plan: &Value) {
        self.0.push(digest.to_string());
    }
}

fn proposal() -> Value {
    json!({
        "schema": "ix-admission-proposal/0",
        "service": "echo",
        "artifact": { "name": "echo-worker", "version": "1.4.2", "platform": "windows-x86_64" },
        "contract": "run/1",
        "config": {
            "PORT": { "env": "ECHO_PORT", "type": "int" },
            "VERBOSE": { "env": "ECHO_VERBOSE", "type": "bool" },
            "API": { "env": "ECHO_API_KEY", "type": "string", "secret": true }
        },
        "limits": { "wall_ms": 60000, "memory_mb": 256 },
        "needs": ["process.spawn"]
    })
}

fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn good_env() -> BTreeMap<String, String> {
    env(&[
        ("ECHO_PORT", "8080"),
        ("ECHO_VERBOSE", "true"),
        ("ECHO_API_KEY", "s3cr3t-value-do-not-print"),
    ])
}

fn world<'a>(store: &'a dyn Store, env: BTreeMap<String, String>) -> World<'a> {
    World {
        platform: "windows-x86_64".into(),
        contract: "run/1".into(),
        env,
        store,
    }
}

fn mandate(scopes: &[&str], wall_ms: u64, memory_mb: u64) -> Mandate {
    Mandate {
        scopes: scopes
            .iter()
            .map(|s| s.to_string())
            .collect::<BTreeSet<_>>(),
        max: Limits { wall_ms, memory_mb },
    }
}

fn codes(findings: &[ix_ctl::admission::Finding]) -> Vec<(Code, String)> {
    findings.iter().map(|f| (f.code, f.path.clone())).collect()
}

/// The whole path, as a caller would run it; the runner sees only what passes every step.
fn run(
    proposal: &Value,
    world: &World,
    mandate: &Mandate,
    runner: &mut Recorder,
) -> Result<String, Vec<ix_ctl::admission::Finding>> {
    let plan = resolve(proposal, world)?;
    let admitted = admit(&plan, mandate)?;
    apply(&admitted, admitted.digest(), mandate, world, runner)
}

#[test]
fn test_missing_or_mistyped_variables_are_named_and_nothing_launches() {
    let store = OneArtifact(DIGEST.into());
    let w = world(
        &store,
        env(&[("ECHO_VERBOSE", "yes"), ("ECHO_API_KEY", "k")]),
    );
    let mut runner = Recorder::default();
    let findings = run(
        &proposal(),
        &w,
        &mandate(&["process.spawn"], 60000, 256),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![
            (Code::MissingConfig, "$.config.PORT".to_string()),
            (Code::ConfigType, "$.config.VERBOSE".to_string()),
        ],
        codes(&findings)
    );
    assert!(findings[0].message.contains("ECHO_PORT"), "{}", findings[0]);
    assert!(runner.0.is_empty());
}

#[test]
fn test_a_malformed_proposal_is_a_schema_finding() {
    let store = OneArtifact(DIGEST.into());
    let w = world(&store, good_env());
    for (bad, path) in [
        (
            {
                let mut p = proposal();
                p["schema"] = json!("ix-admission-proposal/9");
                p
            },
            "$.schema",
        ),
        (
            {
                let mut p = proposal();
                p["surprise"] = json!(1);
                p
            },
            "$",
        ),
        (
            {
                let mut p = proposal();
                p["config"]["PORT"]["type"] = json!("float");
                p
            },
            "$",
        ),
    ] {
        let findings = resolve(&bad, &w).unwrap_err();
        assert_eq!(
            vec![(Code::Schema, path.to_string())],
            codes(&findings),
            "{bad}"
        );
    }
}

#[test]
fn test_floating_or_absent_artifacts_and_other_contracts_are_refused_before_launch() {
    let store = OneArtifact(DIGEST.into());
    let m = mandate(&["process.spawn"], 60000, 256);
    type Case = (fn(&mut Value), Code, &'static str);
    let cases: [Case; 4] = [
        (
            |p| p["artifact"]["version"] = json!("latest"),
            Code::FloatingVersion,
            "$.artifact.version",
        ),
        (
            |p| p["artifact"]["version"] = json!("^1.4"),
            Code::FloatingVersion,
            "$.artifact.version",
        ),
        (
            |p| p["artifact"]["version"] = json!("1.4.3"),
            Code::ArtifactNotFound,
            "$.artifact",
        ),
        (
            |p| p["contract"] = json!("run/2"),
            Code::ContractMismatch,
            "$.contract",
        ),
    ];
    for (edit, code, path) in cases {
        let mut p = proposal();
        edit(&mut p);
        let mut runner = Recorder::default();
        let findings = run(&p, &world(&store, good_env()), &m, &mut runner).unwrap_err();
        assert_eq!(vec![(code, path.to_string())], codes(&findings), "{p}");
        assert!(runner.0.is_empty());
    }
    // An artifact for another platform than the one it would run on.
    let mut p = proposal();
    p["artifact"]["platform"] = json!("linux-x86_64");
    let findings = resolve(&p, &world(&store, good_env())).unwrap_err();
    assert_eq!(
        vec![(Code::PlatformMismatch, "$.artifact.platform".to_string())],
        codes(&findings)
    );
}

#[test]
fn test_the_corrected_plan_is_admitted_and_exactly_that_document_runs() {
    let store = OneArtifact(DIGEST.into());
    let m = mandate(&["process.spawn"], 60000, 256);
    let w = world(&store, good_env());
    let plan = resolve(&proposal(), &w).unwrap();
    let admitted = admit(&plan, &m).unwrap();
    let digest = admitted.digest().to_string();
    assert!(
        digest.starts_with("sha256:") && digest.len() == 71,
        "{digest}"
    );
    // The same inputs name the same document; a different input names another one.
    assert_eq!(
        digest,
        admit(&resolve(&proposal(), &w).unwrap(), &m)
            .unwrap()
            .digest()
    );
    let mut other = good_env();
    other.insert("ECHO_PORT".into(), "8081".into());
    let moved = admit(&resolve(&proposal(), &world(&store, other)).unwrap(), &m).unwrap();
    assert_ne!(digest, moved.digest());

    // The admitted document crosses a process boundary as JSON and still applies.
    let back = Admitted::from_json(&admitted.to_json()).unwrap();
    let mut runner = Recorder::default();
    assert_eq!(digest, apply(&back, &digest, &m, &w, &mut runner).unwrap());
    assert_eq!(vec![digest.clone()], runner.0);

    // What the plan pins: the artifact digest, the typed values, the budget.
    let p = admitted.plan();
    assert_eq!(json!(DIGEST), p["artifact"]["sha256"]);
    assert_eq!(json!(8080), p["config"]["PORT"]["value"]);
    assert_eq!(json!(true), p["config"]["VERBOSE"]["value"]);
    assert_eq!(json!({ "wall_ms": 60000, "memory_mb": 256 }), p["limits"]);
}

#[test]
fn test_a_secret_is_checked_but_never_written_down() {
    let store = OneArtifact(DIGEST.into());
    let m = mandate(&["process.spawn"], 60000, 256);
    let secret = "s3cr3t-value-do-not-print";
    let admitted = admit(
        &resolve(&proposal(), &world(&store, good_env())).unwrap(),
        &m,
    )
    .unwrap();
    let text = admitted.to_json().to_string();
    assert!(!text.contains(secret), "{text}");
    assert!(
        !text.contains(&ix_recover::sha256_hex(secret.as_bytes())),
        "{text}"
    );
    assert_eq!(
        json!({ "env": "ECHO_API_KEY", "type": "string", "secret": true }),
        admitted.plan()["config"]["API"]
    );
    // Absent, it is refused like any declared variable, by name only.
    let mut no_key = good_env();
    no_key.remove("ECHO_API_KEY");
    let findings = resolve(&proposal(), &world(&store, no_key)).unwrap_err();
    assert_eq!(
        vec![(Code::MissingConfig, "$.config.API".to_string())],
        codes(&findings)
    );
}

#[test]
fn test_changes_after_admission_are_refused_and_nothing_launches() {
    let store = OneArtifact(DIGEST.into());
    let m = mandate(&["process.spawn"], 60000, 256);
    let admitted = admit(
        &resolve(&proposal(), &world(&store, good_env())).unwrap(),
        &m,
    )
    .unwrap();
    let mut runner = Recorder::default();

    // A variable moved.
    let mut moved = good_env();
    moved.insert("ECHO_PORT".into(), "9090".into());
    let f = apply(
        &admitted,
        admitted.digest(),
        &m,
        &world(&store, moved),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::ConfigChanged, "$.config.PORT".to_string())],
        codes(&f)
    );

    // A variable gone.
    let mut gone = good_env();
    gone.remove("ECHO_VERBOSE");
    let f = apply(
        &admitted,
        admitted.digest(),
        &m,
        &world(&store, gone),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::MissingConfig, "$.config.VERBOSE".to_string())],
        codes(&f)
    );

    // The artifact's bytes changed under the same name and version.
    let rebuilt = OneArtifact(
        "sha256:2222222222222222222222222222222222222222222222222222222222222222".into(),
    );
    let f = apply(
        &admitted,
        admitted.digest(),
        &m,
        &world(&rebuilt, good_env()),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::ArtifactChanged, "$.artifact".to_string())],
        codes(&f)
    );

    // The plane now keeps another contract.
    let mut w = world(&store, good_env());
    w.contract = "run/2".into();
    let f = apply(&admitted, admitted.digest(), &m, &w, &mut runner).unwrap_err();
    assert_eq!(
        vec![(Code::ContractChanged, "$.contract".to_string())],
        codes(&f)
    );

    // The document was edited after admission: its budget widened, its digest kept.
    let mut doc = admitted.to_json();
    doc["plan"]["limits"]["memory_mb"] = json!(4096);
    let edited = Admitted::from_json(&doc).unwrap();
    let f = apply(
        &edited,
        admitted.digest(),
        &m,
        &world(&store, good_env()),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::RevisionTampered, "$.plan".to_string())],
        codes(&f)
    );

    assert!(runner.0.is_empty(), "{:?}", runner.0);
}

#[test]
fn test_a_rebuilt_binary_on_disk_is_refused() {
    let root = std::env::temp_dir().join(format!("ix-admission-{}", std::process::id()));
    let dir = root
        .join("echo-worker")
        .join("1.4.2")
        .join("windows-x86_64");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("echo-worker.exe"), b"build one").unwrap();
    let store = DirStore::new(root.clone());
    let m = mandate(&["process.spawn"], 60000, 256);
    let admitted = admit(
        &resolve(&proposal(), &world(&store, good_env())).unwrap(),
        &m,
    )
    .unwrap();
    assert_eq!(
        json!(ix_recover::sha256_hex(b"build one")),
        admitted.plan()["artifact"]["sha256"]
    );

    std::fs::write(dir.join("echo-worker.exe"), b"build two").unwrap();
    let mut runner = Recorder::default();
    let f = apply(
        &admitted,
        admitted.digest(),
        &m,
        &world(&store, good_env()),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::ArtifactChanged, "$.artifact".to_string())],
        codes(&f)
    );

    // Two files where one artifact is expected: not found, not a guess.
    std::fs::write(dir.join("other.exe"), b"x").unwrap();
    let f = resolve(&proposal(), &world(&store, good_env())).unwrap_err();
    assert_eq!(
        vec![(Code::ArtifactNotFound, "$.artifact".to_string())],
        codes(&f)
    );
    assert!(runner.0.is_empty());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn test_an_insufficient_mandate_or_invalid_budget_is_refused_never_widened() {
    let store = OneArtifact(DIGEST.into());
    let w = world(&store, good_env());
    let mut runner = Recorder::default();

    let mut p = proposal();
    p["needs"] = json!(["process.spawn", "net.loopback"]);
    let f = run(
        &p,
        &w,
        &mandate(&["process.spawn"], 60000, 256),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::MandateInsufficient, "$.needs".to_string())],
        codes(&f)
    );
    assert!(f[0].message.contains("net.loopback"), "{}", f[0]);

    let f = run(
        &proposal(),
        &w,
        &mandate(&["process.spawn"], 60000, 128),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::BudgetExceedsMandate, "$.limits.memory_mb".to_string())],
        codes(&f)
    );

    // A zero budget is ixctl's "no limit": not admissible.
    let mut p = proposal();
    p["limits"]["wall_ms"] = json!(0);
    let f = resolve(&p, &w).unwrap_err();
    assert_eq!(
        vec![(Code::InvalidBudget, "$.limits.wall_ms".to_string())],
        codes(&f)
    );

    // Applying under another mandate than the admitting one, even a wider one, is refused.
    let m = mandate(&["process.spawn"], 60000, 256);
    let admitted = admit(&resolve(&proposal(), &w).unwrap(), &m).unwrap();
    let wider = mandate(&["process.spawn", "net.any"], 600000, 4096);
    let f = apply(&admitted, admitted.digest(), &wider, &w, &mut runner).unwrap_err();
    assert_eq!(
        vec![(Code::MandateChanged, "$.mandate".to_string())],
        codes(&f)
    );
    assert!(runner.0.is_empty());
}

#[test]
fn test_findings_serialise_with_code_path_and_message() {
    let store = OneArtifact(DIGEST.into());
    let mut p = proposal();
    p["artifact"]["version"] = json!("latest");
    let f = resolve(&p, &world(&store, good_env())).unwrap_err();
    let v = f[0].to_json();
    assert_eq!(json!("FloatingVersion"), v["code"]);
    assert_eq!(json!("$.artifact.version"), v["path"]);
    assert!(v["message"].as_str().unwrap().contains("latest"));
}

#[test]
fn test_a_document_forged_with_a_recomputed_digest_is_refused() {
    let store = OneArtifact(DIGEST.into());
    let m = mandate(&["process.spawn"], 60000, 256);
    let admitted = admit(
        &resolve(&proposal(), &world(&store, good_env())).unwrap(),
        &m,
    )
    .unwrap();
    let mut runner = Recorder::default();
    let forge = |edit: &dyn Fn(&mut Value)| {
        let mut doc = admitted.to_json();
        edit(&mut doc["plan"]);
        doc["digest"] = json!(ix_recover::canonical_sha256(&doc["plan"]));
        Admitted::from_json(&doc).unwrap()
    };

    // A pinned value edited to match a changed environment, its digest recomputed: the
    // digest the caller admitted is not this one.
    let mut moved = good_env();
    moved.insert("ECHO_PORT".into(), "9090".into());
    let edited = forge(&|p| p["config"]["PORT"]["value"] = json!(9090));
    let f = apply(
        &edited,
        admitted.digest(),
        &m,
        &world(&store, moved),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::RevisionTampered, "$.digest".to_string())],
        codes(&f)
    );

    // Even taken at its own word, a plan stripped of its budget and scopes is not a plan.
    let stripped = forge(&|p| {
        p.as_object_mut().unwrap().remove("limits");
        p["needs"] = json!("process.spawn");
    });
    let f = apply(
        &stripped,
        stripped.digest(),
        &m,
        &world(&store, good_env()),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(vec![(Code::Schema, "$.plan".to_string())], codes(&f));

    // Nor is one whose budget grew past the mandate, or that needs a scope it lacks.
    let widened = forge(&|p| p["limits"]["memory_mb"] = json!(4096));
    let f = apply(
        &widened,
        widened.digest(),
        &m,
        &world(&store, good_env()),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::BudgetExceedsMandate, "$.limits.memory_mb".to_string())],
        codes(&f)
    );
    let needier = forge(&|p| p["needs"] = json!(["process.spawn", "net.any"]));
    let f = apply(
        &needier,
        needier.digest(),
        &m,
        &world(&store, good_env()),
        &mut runner,
    )
    .unwrap_err();
    assert_eq!(
        vec![(Code::MandateInsufficient, "$.needs".to_string())],
        codes(&f)
    );
    assert!(runner.0.is_empty(), "{:?}", runner.0);
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ix-admission-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_the_store_reads_nothing_too_large() {
    let root = scratch("large");
    let dir = root
        .join("echo-worker")
        .join("1.4.2")
        .join("windows-x86_64");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("echo-worker.exe"), vec![7u8; 4096]).unwrap();
    let mut store = DirStore::new(root.clone());
    store.max_bytes = 4095;
    let f = resolve(&proposal(), &world(&store, good_env())).unwrap_err();
    assert_eq!(
        vec![(Code::ArtifactNotFound, "$.artifact".to_string())],
        codes(&f)
    );
    store.max_bytes = 4096;
    assert!(resolve(&proposal(), &world(&store, good_env())).is_ok());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn test_the_store_never_leaves_its_root() {
    let root = scratch("root");
    let outside = scratch("outside");
    let dir = outside.join("1.4.2").join("windows-x86_64");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("echo-worker.exe"), b"not from the store").unwrap();
    let store = DirStore::new(root.clone());
    // `<root>/echo-worker` is a link to a directory outside the store.
    let link = root.join("echo-worker");
    #[cfg(windows)]
    let made = std::process::Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(&link)
        .arg(&outside)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    #[cfg(unix)]
    let made = std::os::unix::fs::symlink(&outside, &link).is_ok();
    assert!(made, "could not make the link");
    let f = resolve(&proposal(), &world(&store, good_env())).unwrap_err();
    assert_eq!(
        vec![(Code::ArtifactNotFound, "$.artifact".to_string())],
        codes(&f)
    );
    // Segments that name the store's own directories are refused too: `.` would read
    // `<root>/1.4.2/windows-x86_64` as if it were `echo-worker`'s.
    let aliased = root.join("1.4.2").join("windows-x86_64");
    std::fs::create_dir_all(&aliased).unwrap();
    std::fs::write(aliased.join("echo-worker.exe"), b"aliased").unwrap();
    let mut p = proposal();
    p["artifact"]["name"] = json!(".");
    let f = resolve(&p, &world(&store, good_env())).unwrap_err();
    assert_eq!(
        vec![(Code::ArtifactNotFound, "$.artifact".to_string())],
        codes(&f)
    );
    let _ = std::fs::remove_dir(&link);
    std::fs::remove_dir_all(&root).unwrap();
    std::fs::remove_dir_all(&outside).unwrap();
}

#[test]
fn test_versions_that_are_not_exact_semver_are_floating() {
    let store = OneArtifact(DIGEST.into());
    for v in [
        "1.2.3-",
        "1.2.3+",
        "01.2.3",
        "1.02.3",
        "1.2",
        "v1.2.3",
        "1.2.3-a..b",
    ] {
        let mut p = proposal();
        p["artifact"]["version"] = json!(v);
        let f = resolve(&p, &world(&store, good_env())).unwrap_err();
        assert_eq!(
            vec![(Code::FloatingVersion, "$.artifact.version".to_string())],
            codes(&f),
            "{v}"
        );
    }
}
