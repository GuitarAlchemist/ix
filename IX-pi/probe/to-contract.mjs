// Fold one Pi probe's raw JSONL event stream into a single artifact.
//
// Kept deliberately separate from run-probe.sh: the raw stream is the evidence,
// and a bug in here must never be able to destroy it. Re-running this step is
// free; re-running the probe costs tokens.
//
// The artifact's job is to make a probe auditable by someone who did not watch
// it run: what was asked, against which commit, by which harness, what it
// looked at, what it cost, and -- the part that matters -- whether the run is
// admissible as evidence at all.
//
// usage: node probe/to-contract.mjs <run-id>
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";

const runId = process.argv[2];
if (!runId) {
  console.error("usage: node probe/to-contract.mjs <run-id>");
  process.exit(2);
}

const here = dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const ixPi = join(here, "..");
const rawPath = join(ixPi, "state", "raw", `${runId}.events.jsonl`);
const metaPath = join(ixPi, "state", "raw", `${runId}.meta.json`);
const outPath = join(ixPi, "state", `${runId}.second-opinion.json`);

const meta = JSON.parse(readFileSync(metaPath, "utf8"));
const lines = readFileSync(rawPath, "utf8").split("\n").filter((l) => l.trim());

let unparseable = 0;
const events = [];
for (const line of lines) {
  try {
    events.push(JSON.parse(line));
  } catch {
    unparseable++;
  }
}

const typeCounts = {};
const toolCalls = [];
let lastUsage = null;
let agentEnd = null;

for (const e of events) {
  typeCounts[e.type] = (typeCounts[e.type] ?? 0) + 1;
  if (e.type === "tool_execution_start") {
    toolCalls.push({ tool: e.toolName, args: e.args, is_error: null });
  }
  if (e.type === "tool_execution_end") {
    // Attribute the outcome to the most recent call of the same tool that is
    // still open. Pi interleaves calls, so matching on toolName alone would
    // mislabel; the open-call scan keeps the pairing honest.
    const open = [...toolCalls].reverse().find((c) => c.tool === e.toolName && c.is_error === null);
    if (open) open.is_error = Boolean(e.isError);
  }
  if (e.type === "agent_end") agentEnd = e;
  const usage = e.usage ?? e.message?.usage;
  if (usage) lastUsage = usage;
}

// The answer is the last assistant text in the terminal message list.
const messages = agentEnd?.messages ?? [];
const assistant = messages.filter((m) => m.role === "assistant");
const answer = (assistant.at(-1)?.content ?? [])
  .filter((b) => b.type === "text")
  .map((b) => b.text)
  .join("\n")
  .trim();

const toolsUsed = [...new Set(toolCalls.map((c) => c.tool))].sort();

// Admissibility, not decoration. A probe that answered without reading anything
// reported the model's prior, not a finding about this repository -- and a run
// that mutated the worktree was not the read-only observation it claims to be.
// Either way the artifact must say so in a field a consumer can filter on,
// rather than leaving a plausible-looking answer to be taken at face value.
const checks = {
  answer_nonempty: answer.length > 0,
  used_tools: toolsUsed.length > 0,
  agent_completed: Boolean(agentEnd),
  stream_fully_parsed: unparseable === 0,
  worktree_unmodified: meta.worktree_dirty_paths === 0,
  no_tool_errors: toolCalls.every((c) => c.is_error !== true),
};
// A tool error is worth surfacing but does not by itself invalidate a probe --
// agents recover from a failed grep. The other five do invalidate it.
const admissible = Object.entries(checks)
  .filter(([k]) => k !== "no_tool_errors")
  .every(([, v]) => v);

const artifact = {
  schema: "ix-pi.second-opinion/v0.1",
  run_id: meta.run_id,
  admissible,
  checks,
  probe: {
    prompt_file: meta.prompt_file,
    prompt_sha256: meta.prompt_sha256,
    subject_revision: meta.base_sha,
  },
  harness: meta.harness,
  run: {
    ...meta.run,
    duration_s:
      (Date.parse(meta.run.ended_at) - Date.parse(meta.run.started_at)) / 1000,
    events: events.length,
    unparseable_lines: unparseable,
    event_types: typeCounts,
  },
  cost: lastUsage
    ? {
        input_tokens: lastUsage.input,
        output_tokens: lastUsage.output,
        reasoning_tokens: lastUsage.reasoning,
        cache_read_tokens: lastUsage.cacheRead,
        total_tokens: lastUsage.totalTokens,
        usd: lastUsage.cost?.total,
      }
    : null,
  evidence: {
    tools_used: toolsUsed,
    tool_call_count: toolCalls.length,
    // Paths the agent actually opened: the cheapest way to judge whether an
    // answer was grounded or guessed.
    paths_read: [
      ...new Set(toolCalls.filter((c) => c.tool === "read").map((c) => c.args?.path).filter(Boolean)),
    ].sort(),
    commands_run: toolCalls
      .filter((c) => c.tool === "bash")
      .map((c) => String(c.args?.command ?? "").replace(/\s+/g, " ").slice(0, 200)),
  },
  answer,
  answer_sha256: createHash("sha256").update(answer, "utf8").digest("hex"),
};

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");

console.log(`artifact   : ${outPath}`);
console.log(`admissible : ${admissible}`);
for (const [k, v] of Object.entries(checks)) console.log(`  ${v ? "ok  " : "FAIL"} ${k}`);
console.log(`tools      : ${toolsUsed.join(", ") || "(none)"} over ${toolCalls.length} call(s)`);
console.log(`cost       : ${artifact.cost?.total_tokens ?? "?"} tokens, $${artifact.cost?.usd?.toFixed(4) ?? "?"}`);
if (!admissible) process.exit(1);
