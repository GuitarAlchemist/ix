# Provisional Node native Claude relay

Source reference for IX #473; the Go ixrelay binary remains the intended replacement.
This is not the Go release, a cloud receiver or a WMUX input transport.

The resident Node process launches Claude only for an explicitly authorized,
fresh ACK-only request to a unique IX / Indigo IX target. It uses the existing
Claude account through its CLI, safe-mode, empty setting sources and strict
empty MCP. Only ListAgents/SendMessage are enabled. No token is read by this code.
No socket listener, tunnel, global hook or idle model turn is added.

Native user/peer metadata is verified against the claimed source, target and
nonce. The worker sends the proof over owned fork IPC; the supervisor commits
native_returns in SQLite before acknowledging durable reception. Request completion
still needs worker closure and correlated stdout; uncertain work remains UNKNOWN.
Request timeout is 90s; inbox inactivity TTL is 300s, including after an ACK.

## Requirements and checks

Windows in the operator's interactive user session, Node >=24.12 (built-in SQLite
is experimental), and an authenticated Claude CLI at $HOME/.local/bin/claude.exe.
Run node --test tests/*.test.mjs. Tests use local Node fixtures; they send nothing
to Claude. Live operator validation of the configured variant stored one native
ACK 216ms after its native timestamp and 2ms after local observation. This is one
example, not a reliability or latency guarantee. Receiver model usage is separate.

The published working directory uses account-relative paths. The historical live
validation used an operator-configured install; these portable path substitutions
are validated by local tests, not another live model call.

## Explicit installation and removal

Review manage.ps1 -Action Plan. Then, only with operator approval:

    ./manage.ps1 -Action Install -ApprovedPersistentActivation -StartNow
    ./manage.ps1 -Action Status
    ./manage.ps1 -Action Stop
    ./manage.ps1 -Action Start
    ./manage.ps1 -Action Uninstall

The task is Gaia.NativeClaudeRelay, current user, Interactive/Limited, at logon,
with Node resident and Claude on demand. Installation refuses an existing task or
directory; do not install beside another relay for the same route. Uninstall keeps
state. Never restore an old database over new requests/receipts during rollback.
The same request ID with different content is refused; no automatic UNKNOWN retry.

Request schema: gaia-native-ack-request/1 with id, targetName, nonce, issuedAt,
expiresAt, budgetUsd (<=0.05), authorizationRef. Supply fresh UUIDs and current
Unix milliseconds; expiry is at most one hour after issuance. Enqueue with
node agent.mjs enqueue --root <installed-state-path> --request <request.json>.
An authorization reference records provenance; it grants no new permission.
The CLI budget is a soft threshold and can overshoot by a completed model turn;
the receiving session's model usage is separate and is not capped by this value.

The first Gaia-to-PC hop still requires the existing authorized executor. Automatic
cloud Gaia receive/wakeup has not been implemented. Migrate to Go explicitly and
preserve UNKNOWN requests; never run both effect executors concurrently.

Only source, synthetic tests and documentation belong in Git. Do not commit state,
native transcripts, logs, request identities, local account paths or credentials.
No GitHub release or automatic deployment is performed by this package.
