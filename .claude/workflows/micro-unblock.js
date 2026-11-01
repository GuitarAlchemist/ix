export const meta = {
  name: 'micro-unblock',
  description: 'Bounded search for a micro-fix to one observed blocker: qualify, intervene, verify independently, decide, resume the main task, extract a recipe',
  whenToUse: 'A small real blocker in your own lane with a fixed, runnable oracle (a mutant that should die, a doc that should exist)',
  phases: [
    { title: 'Qualify', detail: 'confirm the blocker on the clean base with the fixed oracle' },
    { title: 'Intervene', detail: 'one leaf, one fresh worktree per attempt, smallest change' },
    { title: 'Verify', detail: 'independent leaf: before/after, mutant, counter-case, scope' },
    { title: 'Resume', detail: 'the real step of the main task the fix unblocks' },
    { title: 'Recipe', detail: 'candidate recipe with provenance' },
  ],
}

const A = args
const MAX = Math.min(A.maxAttempts || 3, 3)

const ORACLE = `FIXED ORACLE (set by the supervisor; nobody may change it):
- Package dir: <worktree>/${A.pkgDir}. Oracle command, run there: ${A.oracleCmd}
- Mutant M: in <worktree>/${A.mutantFile}, delete the one line containing exactly: ${A.mutantLine}
- Doc check: ${A.docCheck || 'none'}
- Blocker is REAL on the base iff the oracle command PASSES with M applied (nothing binds the line)${A.docCheck ? ' or the doc check fails' : ''}.
- A candidate is ACCEPTED iff: oracle passes on the candidate, with -v showing at least one NEW test that RAN (not SKIP); the oracle FAILS with M applied on the candidate by a test assertion or a refused request (not a compile error); the doc check passes; the counter-case holds: ${A.counterCase}.
- Allowed files: ${A.allowedFiles.join(', ')}. No production file may change.`

const QUAL = { type: 'object', properties: {
  blockerReal: { type: 'boolean' }, mutantResult: { type: 'string', enum: ['pass', 'fail-assertion', 'fail-compile', 'not-run'] },
  docCheck: { type: 'string' }, evidence: { type: 'string' }, restored: { type: 'boolean' } },
  required: ['blockerReal', 'mutantResult', 'evidence', 'restored'] }

const INTERVENE = { type: 'object', properties: {
  status: { type: 'string', enum: ['candidate', 'needs-production-change', 'gave-up'] },
  hypothesis: { type: 'string' }, change: { type: 'string' }, newTests: { type: 'array', items: { type: 'string' } },
  filesChanged: { type: 'array', items: { type: 'string' } }, patchPath: { type: 'string' },
  commandsRun: { type: 'array', items: { type: 'string' } }, oracleOutputTail: { type: 'string' }, leftRunning: { type: 'string' } },
  required: ['status', 'hypothesis', 'change', 'filesChanged', 'commandsRun', 'leftRunning'] }

const VERDICT = { type: 'object', properties: {
  accepted: { type: 'boolean' },
  oracleOnCandidate: { type: 'string', enum: ['pass', 'fail-assertion', 'fail-compile', 'not-run'] },
  newTestRan: { type: 'boolean' },
  mutantOnCandidate: { type: 'string', enum: ['pass', 'fail-assertion', 'fail-compile', 'not-run'] },
  docCheck: { type: 'string', enum: ['pass', 'fail', 'n/a'] },
  counterCase: { type: 'string', enum: ['holds', 'broken', 'not-checked'] },
  scopeClean: { type: 'boolean' }, mutantRestored: { type: 'boolean' },
  diagnosis: { type: 'string' }, evidence: { type: 'string' } },
  required: ['accepted', 'oracleOnCandidate', 'newTestRan', 'mutantOnCandidate', 'docCheck', 'counterCase', 'scopeClean', 'mutantRestored', 'diagnosis', 'evidence'] }

const RULES = `Rules: do not spawn agents; do not commit, push, or touch any other checkout; never print a token or credential; never run "git clean"; one heavy build at a time (you are the only one running now). Restore a mutated file with "git checkout -- <file>" and confirm "git diff --stat -- <file>" is empty.`

phase('Qualify')
const qPath = `${A.worktreeRoot}-${A.caseId}-q`
const q = await agent(`Qualification of a blocker, case "${A.caseId}". You are read-only on the code except for applying and then undoing the mutant.
Create a fresh detached worktree: git -C ${A.repo} worktree add --detach ${qPath} ${A.base}
${ORACLE.replace(/<worktree>/g, qPath)}
Do: (1) run the oracle command with M applied on this clean base and classify the result; (2) restore M; (3) run the doc check if any. Report blockerReal per the definition above, with the exact last lines of output as evidence.
${RULES}`, { label: `qualify:${A.caseId}`, phase: 'Qualify', schema: QUAL, effort: 'low' })
if (!q || !q.blockerReal) {
  log(`Blocker not confirmed for ${A.caseId}: stop, no intervention.`)
  return { caseId: A.caseId, decision: 'Rejected', reason: 'blocker not confirmed by the fixed oracle', qualify: q }
}
log(`Blocker confirmed (${q.mutantResult} with the mutant on the base).`)

const attempts = []
let useful = null
let diagnosis = A.priorDiagnosis || 'none: first attempt'
for (let n = 1; n <= MAX && !useful; n++) {
  const wt = `${A.worktreeRoot}-${A.caseId}-a${n}`
  const patch = `${A.resultDir}/${A.caseId}-a${n}.patch`
  const iv = await agent(`You are leaf "intervene-${A.caseId}-${n}" (attempt ${n} of ${MAX}). Produce the SMALLEST change that removes this blocker, then stop.
BLOCKER: ${A.blocker}
TASK: ${A.task}
FACTS ALREADY ESTABLISHED (do not re-derive; start writing quickly): ${A.hints}
DIAGNOSIS FROM PREVIOUS ATTEMPTS (change your hypothesis accordingly): ${diagnosis}
Create a fresh detached worktree: git -C ${A.repo} worktree add --detach ${wt} ${A.base}  — work ONLY there.
${ORACLE.replace(/<worktree>/g, wt)}
Run the oracle command yourself on your candidate (and the mutant if you wish, restoring it). Your new test's name must match the oracle's -run pattern. If a production file must change, stop with status needs-production-change.
At the end write your diff: git -C ${wt} diff > ${patch}  (include new untracked files with "git add -N" first) and return patchPath=${patch}.
${RULES}`, { label: `intervene:${A.caseId}#${n}`, phase: 'Intervene', schema: INTERVENE })
  if (!iv || iv.status !== 'candidate') {
    attempts.push({ n, worktree: wt, intervene: iv, verdict: null, decision: 'Rejected' })
    diagnosis = `attempt ${n}: ${iv ? iv.status + ' — ' + iv.change : 'leaf died'}`
    log(`Attempt ${n}: no candidate (${iv ? iv.status : 'null'}).`)
    continue
  }
  const v = await agent(`You are an INDEPENDENT verifier for case "${A.caseId}", attempt ${n}. You did not write this change and you do not defend it; default to rejecting when evidence is missing. An implementer's summary, a skipped test, an ACK or a live process is not evidence: run the commands yourself.
BLOCKER: ${A.blocker}
Candidate worktree: ${wt} (based on ${A.base}); its diff: ${patch}. Implementer claims new tests: ${JSON.stringify(iv.newTests || [])}.
${ORACLE.replace(/<worktree>/g, wt)}
Check, in order: (1) git -C ${wt} status / diff --stat: only allowed files changed, no production file, no test deleted or skipped, no token in any file; (2) the oracle command on the candidate, with -v, and that a NEW test RAN; (3) the mutant on the candidate: must fail by assertion or refused request, classify it (assertion vs compile vs not run); restore it and confirm the mutant file's diff is empty; (4) the doc check; (5) the counter-case. Distinguish an assertion failure, a compile failure and a test that did not run.
${RULES}`, { label: `verify:${A.caseId}#${n}`, phase: 'Verify', schema: VERDICT, effort: 'high' })
  const ok = v && v.accepted && v.oracleOnCandidate === 'pass' && v.newTestRan && v.mutantOnCandidate === 'fail-assertion'
    && v.docCheck !== 'fail' && v.counterCase === 'holds' && v.scopeClean && v.mutantRestored
  attempts.push({ n, worktree: wt, patch, intervene: iv, verdict: v, decision: ok ? 'Useful' : 'Rejected' })
  if (ok) { useful = attempts[attempts.length - 1]; break }
  diagnosis = `attempt ${n} rejected by the verifier: ${v ? v.diagnosis : 'verifier died'}`
  log(`Attempt ${n} rejected: ${v ? v.diagnosis : 'verifier died'}`)
}
if (!useful) {
  return { caseId: A.caseId, decision: 'Deferred', reason: `no accepted candidate in ${attempts.length} attempt(s)`, qualify: q, attempts }
}

let resumed = null
if (A.resume) {
  phase('Resume')
  resumed = await agent(`Main-task resumption for case "${A.caseId}". The accepted fix is in ${useful.worktree} (patch ${useful.patch}). Real step to perform, read-only, no mutation of any running service: ${A.resume}
Report what you did, the exact request(s) and the observed fields; never print a token.`, { label: `resume:${A.caseId}`, phase: 'Resume', effort: 'low' })
}

phase('Recipe')
const recipe = await agent(`Write a candidate recipe (markdown, English) to ${A.resultDir}/${A.caseId}-recipe.md for this bounded search, from the run's facts below. Sections: name + version (micro-unblock/0.1), trigger, preconditions, inputs/outputs, permitted actions, fixed oracle, budget and stop rules (state plainly that the runtime gives no per-agent time cap and no token cap unless the user sets one), refusal cases, limits, cleanup/rollback, provenance (this case, attempts and verdicts). Do not generalise beyond what was shown: mark transfer to other cases "unverified" unless a second case is listed. Facts: ${JSON.stringify({ caseId: A.caseId, blocker: A.blocker, oracle: ORACLE, qualify: q, attempts: attempts.map(a => ({ n: a.n, decision: a.decision, hypothesis: a.intervene && a.intervene.hypothesis, verdict: a.verdict })), resumed })}
Return the path written.`, { label: `recipe:${A.caseId}`, phase: 'Recipe', effort: 'low' })

return { caseId: A.caseId, decision: 'Useful', qualify: q, attempts, useful: { worktree: useful.worktree, patch: useful.patch }, resumed, recipe }
