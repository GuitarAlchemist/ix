// Deterministic tests of .claude/workflows/micro-unblock.js: the real script runs with stub agents.
// Run: node --test .claude/workflow-tests/
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const src = readFileSync(new URL('../workflows/micro-unblock.js', import.meta.url), 'utf8')
  .replace(/^export const meta/m, 'const meta')
const AsyncFunction = (async () => {}).constructor

const DOC = 'a'.repeat(64)
const ARGS = {
  caseId: 'c', repo: 'r', base: 'b', worktreeRoot: 'w', resultDir: 'd', pkgDir: 'p', oracleCmd: 'o',
  mutantFile: 'm', mutantLine: 'l', counterCase: 'k', allowedFiles: ['f'], blocker: 'x', task: 't', hints: 'h',
  resume: 'read node c with the README alone', frozenDocSha256: DOC,
}
const QUALIFY = { blockerReal: true, mutantResult: 'pass', evidence: 'e', restored: true }
const CANDIDATE = { status: 'candidate', hypothesis: 'h', change: 'c', filesChanged: ['f'], commandsRun: [], leftRunning: 'none' }
const ACCEPTED = { accepted: true, oracleOnCandidate: 'pass', newTestRan: true, mutantOnCandidate: 'fail-assertion',
  docCheck: 'pass', counterCase: 'holds', scopeClean: true, mutantRestored: true, diagnosis: '', evidence: 'e' }
const RESUMED = { stepDone: true, step: 'one sample read', docSha256: DOC, requests: ['GET /v1/events'],
  eventsTotal: 5, eventsReplayed: 4, eventsFresh: 1, helpReceived: [], gaps: [], evidence: 'e' }

async function run(args, answers) {
  const calls = []
  const agent = async (prompt, opts) => {
    calls.push({ label: opts.label, opts })
    const a = answers[opts.label.split(':')[0]]
    return typeof a === 'function' ? a(prompt, opts) : a
  }
  const result = await new AsyncFunction('agent', 'phase', 'log', 'args', src)(agent, () => {}, () => {}, args)
  return { result, calls }
}
const answers = (over = {}) => ({ qualify: QUALIFY, intervene: CANDIDATE, verify: ACCEPTED, resume: RESUMED, recipe: 'd/c-recipe.md', ...over })

test('a validated fix with a passing structured resume is Useful, both verdicts true', async () => {
  const { result } = await run(ARGS, answers())
  assert.equal(result.fixValidated, true)
  assert.equal(result.taskUnblocked, true)
  assert.equal(result.decision, 'Useful')
})

test('the resume agent answers a schema, and its reader identity is set by the script', async () => {
  const { result, calls } = await run(ARGS, answers({ resume: { ...RESUMED, reader: 'Gaia' } }))
  const resume = calls.find(c => c.label.startsWith('resume:'))
  assert.ok(resume.opts.schema, 'resume must be structured')
  assert.equal(result.resume.reader, 'workflow-agent resume:c')
  assert.equal(result.resume.readerIndependent, false)
})

test('without a resume step the fix is validated but the task is not unblocked', async () => {
  const { result, calls } = await run({ ...ARGS, resume: undefined }, answers())
  assert.equal(result.fixValidated, true)
  assert.equal(result.taskUnblocked, false)
  assert.notEqual(result.decision, 'Useful')
  assert.ok(!calls.some(c => c.label.startsWith('resume:')))
})

for (const [name, over, why] of [
  ['help was received', { helpReceived: ['asked where the token is'] }, /help/],
  ['the reader found doc gaps', { gaps: ['no example of the id: line'] }, /gap/],
  ['every event was replayed history', { eventsReplayed: 5, eventsFresh: 0 }, /fresh/],
  ['the parent step was not done', { stepDone: false }, /step/],
  ['the counts do not add up', { eventsTotal: 9 }, /count/],
  ['the doc read is not the frozen doc (a change after the resume)', { docSha256: 'b'.repeat(64) }, /frozen/],
]) {
  test(`the task is not unblocked when ${name}`, async () => {
    const { result } = await run(ARGS, answers({ resume: { ...RESUMED, ...over } }))
    assert.equal(result.fixValidated, true)
    assert.equal(result.taskUnblocked, false)
    assert.notEqual(result.decision, 'Useful')
    assert.match(result.resume.reasons.join('; '), why)
  })
}

test('a resume with no frozen doc proves nothing', async () => {
  const { result } = await run({ ...ARGS, frozenDocSha256: undefined }, answers())
  assert.equal(result.taskUnblocked, false)
  assert.match(result.resume.reasons.join('; '), /frozen/)
})

test('a dead resume agent proves nothing', async () => {
  const { result } = await run(ARGS, answers({ resume: null }))
  assert.equal(result.fixValidated, true)
  assert.equal(result.taskUnblocked, false)
})

test('a rejected fix never reaches the resume', async () => {
  const { result, calls } = await run({ ...ARGS, maxAttempts: 1 }, answers({ verify: { ...ACCEPTED, newTestRan: false } }))
  assert.equal(result.fixValidated, false)
  assert.equal(result.taskUnblocked, false)
  assert.equal(result.decision, 'Deferred')
  assert.ok(!calls.some(c => c.label.startsWith('resume:')))
})

test('an external reader report is gated without spawning a resume agent', async () => {
  const report = { ...RESUMED, reader: 'Gaia executor' }
  const { result, calls } = await run({ ...ARGS, resumeReport: report }, answers())
  assert.ok(!calls.some(c => c.label.startsWith('resume:')))
  assert.equal(result.resume.reader, 'Gaia executor')
  assert.equal(result.resume.readerIndependent, true)
  assert.equal(result.taskUnblocked, true)
  assert.equal(result.decision, 'Useful')
})

test('an external report with no reader named proves nothing', async () => {
  const { result } = await run({ ...ARGS, resumeReport: { ...RESUMED } }, answers())
  assert.equal(result.taskUnblocked, false)
  assert.match(result.resume.reasons.join('; '), /reader/)
})
