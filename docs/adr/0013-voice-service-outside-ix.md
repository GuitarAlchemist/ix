# IX speaks to a local voice service over HTTP; Piper stays outside the workspace

Status: proposed (2026-10-16)

## Context

The person wants loops, Claude sessions and tools to speak to them aloud, as the text arrives,
with Piper, a local neural text-to-speech engine. Piper's voices are ONNX models run by
onnxruntime (C++), and its phonemes come from espeak-ng, which is GPL-3.0. The original
`rhasspy/piper` (MIT) was archived in 2025; the maintained fork, `OHF-Voice/piper1-gpl`, is
GPL-3.0. IX's convention is pure Rust, with no external ML frameworks.

A local service, `piper-voice` (Go, shipped with a Claude Code plugin in the `guitaralchemist`
marketplace), keeps Piper warm and plays speech on the person's speakers, sentence by sentence.
Its one audio queue serves every client: Claude Code, IX, and anything else that speaks HTTP.

## Decision

1. **IX reaches the voice only through the service's HTTP contract v0.1**: `GET /health`,
   `GET /v1/voices`, `POST /v1/say`, `POST /v1/stop`, on `127.0.0.1:59126`, or wherever
   `IX_VOICE_URL` points. The crate `ix-voice` is the client, and a CLI of the same name.
2. **Nothing in the workspace links, vendors or downloads Piper, onnxruntime or espeak-ng.** The
   service's `setup` installs them on the person's machine, and the service talks to Piper over a
   pipe, so no GPL code reaches an IX binary.
3. **The client is blocking** (reqwest's blocking client, with no TLS, since the service is
   local). An async caller, such as an ix-agent tool handler, runs it on a blocking thread.
4. **Next consumers, each a change of its own**: a `say()` capability in ix-loop, and an
   `ix_voice_say` MCP tool once the open tool PRs that bump `parity.rs` have merged.

## Consequences

- IX gains no native dependency, and its CI needs neither Piper nor a sound card: the tests run
  against a fake service.
- Speech needs the service running. Without it, every call fails with a message that says how to
  start it.
- The contract is shared with a repository outside this workspace, so a change to it is made in
  both.

## Reversibility

A two-way door: the crate is a client of a versioned contract, and removing it removes nothing
else. Revisit if a pure-Rust Piper (ONNX inference plus a permissively licensed phonemizer)
becomes good enough to run in-process, or when the contract moves to v1.
