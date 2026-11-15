import test from 'node:test';
import assert from 'node:assert/strict';
import {TYPES, toolState, displayMode} from '../web/catalog.js';

test('the catalog names every widget type the client can show, with a visible label and what it needs', () => {
  assert.deepEqual(TYPES.map(x => x.kind), ['checklist', 'counter', 'queries']);
  assert.deepEqual(TYPES.filter(x => x.create).map(x => x.kind), ['checklist', 'counter'], 'queries are imported, not created by hand');
  assert.deepEqual(TYPES.filter(x => !x.runtime).map(x => x.kind), ['queries'], 'a list of queries runs no IXLET');
  for (const x of TYPES) assert.ok(x.labelKey && x.helpKey && x.icon);
});

const base = {kind: 'counter', hasDefinition: true, local: true, saved: false, status: 'local'};

test('a tool on a device that was never paired is a draft: step 1, blocker "not paired"', () => {
  assert.deepEqual(toolState(base), {state: 'draft', step: 1, steps: 3, blocker: 'notPaired', progressKnown: true});
});

test('a paired device whose tool the Desktop does not have yet is sending: step 2, with the real blocker when there is one', () => {
  const sending = {...base, status: 'connected'};
  assert.deepEqual(toolState(sending), {state: 'sending', step: 2, steps: 3, blocker: null, progressKnown: true});
  assert.equal(toolState({...sending, status: 'offline'}).blocker, 'desktopUnreachable');
  assert.equal(toolState({...sending, status: 'device_pending'}).blocker, 'awaitingApproval');
  assert.equal(toolState({...sending, status: 'device_refused'}).blocker, 'accessRefused');
  assert.equal(toolState({...sending, status: 'device_revoked'}).blocker, 'accessRefused');
});

test('progress is "unknown", never a guessed number, while the connection state itself is unknown', () => {
  const t = toolState({...base, status: 'connecting'});
  assert.equal(t.progressKnown, false);
  assert.ok(!('percent' in t), 'no percentage is ever produced: the milestones are discrete steps');
});

test('a tool hosted by the Desktop is shared (step 3); offline it is the last known value, with the blocker named', () => {
  const hosted = {...base, local: false, status: 'connected'};
  assert.deepEqual(toolState(hosted), {state: 'shared', step: 3, steps: 3, blocker: null, progressKnown: true});
  assert.deepEqual(toolState({...hosted, status: 'offline'}), {state: 'lastKnown', step: 3, steps: 3, blocker: 'desktopUnreachable', progressKnown: true});
});

test('a list or counter without an IXLET definition is not connected to any runtime, and says so', () => {
  assert.deepEqual(toolState({...base, hasDefinition: false}), {state: 'noRuntime', step: 0, steps: 3, blocker: 'noDefinition', progressKnown: true});
});

test('a list of queries needs no runtime: it is shared once the Desktop saved its project', () => {
  const queries = {kind: 'queries', hasDefinition: false, local: true, saved: false, status: 'connected'};
  assert.equal(toolState(queries).state, 'sending');
  assert.equal(toolState({...queries, saved: true}).state, 'shared');
  assert.equal(toolState({...queries, status: 'local'}).state, 'draft');
});

test('the display mode is reported as measured, and installation is never claimed', () => {
  const env = (standalone, ios) => ({matchMedia: q => ({matches: q === '(display-mode: standalone)' && standalone}), navigator: {standalone: ios}});
  assert.equal(displayMode(env(true)), 'standalone');
  assert.equal(displayMode(env(false, true)), 'standalone', 'iOS home-screen apps report navigator.standalone');
  assert.equal(displayMode(env(false, false)), 'browser');
  assert.equal(displayMode({}), 'unknown', 'no matchMedia: unknown, not a guess');
});
