//! A refused plan, the corrected one admitted and applied, then a change after admission
//! refused, all against an in-memory store and a runner that only prints. No process starts.
//!
//! `cargo run -p ix-ctl --example admission`

use ix_ctl::admission::{
    admit, apply, resolve, ArtifactRef, Limits, Mandate, Runner, Store, World,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

struct Fixture;

impl Store for Fixture {
    fn find(&self, a: &ArtifactRef) -> Option<(PathBuf, String)> {
        (a.name == "echo-worker" && a.version == "1.4.2").then(|| {
            (
                PathBuf::from("store/echo-worker/1.4.2/windows-x86_64/echo-worker.exe"),
                ix_recover::sha256_hex(b"echo-worker 1.4.2"),
            )
        })
    }
}

struct Print;

impl Runner for Print {
    fn launch(&mut self, digest: &str, artifact: &Path, _plan: &Value) {
        println!("launch {digest} from {}", artifact.display());
    }
}

fn findings(f: &[ix_ctl::admission::Finding]) -> Value {
    json!(f.iter().map(|f| f.to_json()).collect::<Vec<_>>())
}

fn main() {
    let proposal = json!({
        "schema": "ix-admission-proposal/0",
        "service": "echo",
        "artifact": { "name": "echo-worker", "version": "latest", "platform": "windows-x86_64" },
        "contract": "run/1",
        "config": { "PORT": { "env": "ECHO_PORT", "type": "int" } },
        "limits": { "wall_ms": 60000, "memory_mb": 256 },
        "needs": ["process.spawn"]
    });
    let mandate = Mandate {
        scopes: ["process.spawn".to_string()].into(),
        max: Limits {
            wall_ms: 60000,
            memory_mb: 256,
        },
    };
    let mut world = World {
        platform: "windows-x86_64".into(),
        contract: "run/1".into(),
        env: BTreeMap::new(),
        store: &Fixture,
    };

    println!("-- refused: floating version, variable unset");
    let refused = resolve(&proposal, &world).unwrap_err();
    println!("{:#}", findings(&refused));

    println!("-- corrected: exact version, variable set");
    let mut corrected = proposal.clone();
    corrected["artifact"]["version"] = json!("1.4.2");
    world.env.insert("ECHO_PORT".into(), "8080".into());
    let admitted = admit(&resolve(&corrected, &world).unwrap(), &mandate).unwrap();
    println!("{:#}", admitted.to_json());
    apply(&admitted, &mandate, &world, &mut Print).unwrap();

    println!("-- the variable moved after admission");
    world.env.insert("ECHO_PORT".into(), "9090".into());
    let moved = apply(&admitted, &mandate, &world, &mut Print).unwrap_err();
    println!("{:#}", findings(&moved));
}
