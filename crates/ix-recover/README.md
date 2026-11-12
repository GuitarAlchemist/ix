# ix-recover

**Provisional.** Publish a run's result once under a run id the caller chooses. After a process kill, a new call with the same id reconciles from what is on disk before doing any work.

This carries the semantics of the Indigo recovery canary (2026-10-17, 228/228 checks, reviewed by Codex) over into a library. The layout, the three record schemas (`ix-recover-intent/0`, `ix-recover-manifest/0`, `ix-recover-receipt/0`) and the error set are spike choices, not a contract. They live on the `exp/ix-recover` branch until the review settles who owns recovery for good: an IX crate, or Gaia's continuity store.

## Use

```rust
use ix_recover::{Begin, Store};
use serde_json::json;

let store = Store::open("state/pipeline-runs")?;
let bindings = json!({ "spec_sha256": spec_digest, "ix_exe_sha256": exe_digest });
match store.begin("nightly-2026-10-17", &bindings)? {
    Begin::Fresh(attempt) => {
        let output = do_the_work();
        attempt.publish(&[("summary.json", output.as_slice())])?;
    }
    Begin::Done(record) | Begin::Repaired(record) => {
        // Already published: use record.file("summary.json"), do not recompute.
    }
}
```

`ix pipeline run --run-id ID [--runs-dir DIR]` is the first caller. It binds the run to two digests: the bound spec, params included, and the `ix` binary. Those two fix the result only when every stage computes from its resolved args alone, so the command runs only stages whose skill was audited as pure (`stats`, `distance`) and refuses any other spec before it opens the store. It publishes three files, the bytes it printed, the summary JSON and the `ix.lock`, and writes nothing else: no `ix.lock` beside the spec and no provenance line. A later run under the same id prints the recorded bytes and executes nothing.

## How it decides

`Store::begin(run_id, bindings)` takes the run's owner lock (`File::try_lock`), then goes through these steps in order:

| On disk | Result |
|---|---|
| lock held by another owner | `Error::Unknown` |
| no intent | write the intent first; a commit or receipt without an intent is `Error::Corrupt` |
| intent with other bindings | `Error::Conflict`, naming the bindings that differ |
| receipt | verify the receipt, the commit and every file → `Begin::Done` |
| commit, no receipt | verify, write the receipt → `Begin::Repaired` |
| neither | `Begin::Fresh`, a new attempt |

`Attempt::publish` writes each file, then `manifest.json`, under `attempts/<attempt_id>/`, each created new and synced. Then it commits with `hard_link(manifest.json, committed.json)`. A hard link fails when its target exists, on every platform, so a run id has at most one committed attempt. The receipt comes last. Verification compares every byte against a digest: the receipt binds the committed manifest, the manifest binds each file, and the attempt directory must hold exactly the files the manifest lists.

The records parse into structs that refuse unknown and duplicated keys. Bindings must be a JSON object without floating-point numbers, because JSON float parsing is not exact and a re-read intent could then differ from the one written.

## Tests

- `tests/recover.rs` runs in-process: replay, conflict, a held lock, a missing receipt, and simulated evidence damage (tampered, extra or missing files; unknown or duplicated fields; a receipt for another manifest; an intent that does not match its digest; a commit without an intent; a foreign commit during an attempt). It also covers unsafe run ids and file names.
- `tests/kill.rs` uses real process kills. The test binary re-runs itself as a child, which stops at a publish point and is killed there (TerminateProcess on Windows, SIGKILL on Unix). A kill after the commit is repaired without a second publication. A kill before the commit leaves its residue and a fresh attempt. A guard kills and reaps the child on every way out of the parent, a failed wait included, and a child left waiting exits by itself after its deadline; both are tested, the first by checking that the child's process id is gone.
- `crates/ix-cli/tests/pipeline_kill.rs` kills the `ix` process itself before and after the commit, at a pause point that exists in debug builds only (`IX_RECOVER_PAUSE`).

## Scope

Process-kill recovery only:

- **Power loss.** On Unix each directory is fsynced after the link that changes it. On Windows no directory is flushed, so the crate makes no power-loss claim there.
- **Filesystems.** Hard links need NTFS, ext4, APFS or similar, not FAT.
- **Residue.** A killed attempt's files stay under `attempts/` and are never cleaned.
- **Owners.** The OS releases the lock when its owner dies, and that is the only proof of a dead owner this crate uses, so an attempt's work must run inside the owning process. A subprocess can outlive its killed parent and overlap the next attempt, and an effect outside the attempt's files is neither undone nor reconciled, so callers keep both out of an attempt.
- **Toolchain.** The crate needs Rust 1.89, for `File::try_lock`.
