// What the Widgets and IXlets views show, as plain functions: the widget types that exist, the traits each one
// implements, the state of one tool, and the display mode. No DOM and no network here, so a test can state each rule.

// Traits, as in Rust: a trait names a capability, and a widget type states which ones it implements. The client asks
// "does this type implement T?" instead of testing a kind by name.
//   Creatable   the person can make one by hand (a list of queries is imported from the repo, never typed).
//   Runnable    it runs an IXLET on the Desktop (a list of queries runs nothing).
//   Collapsible it can be shown folded to one line; its `summary` method is the optional content shown while folded.
export const TRAITS = ['Creatable', 'Runnable', 'Collapsible'];

// summary(facts) returns an i18n descriptor {key, ...params}, or null when there is nothing to say while folded.
const IMPL = {
  checklist: {Creatable: {}, Runnable: {}, Collapsible: {summary: ({total = 0, done = 0}) => total ? {key: 'sumList', done, total} : {key: 'sumListEmpty'}}},
  counter: {Creatable: {}, Runnable: {}, Collapsible: {summary: ({value = 0}) => ({key: 'sumCounter', value})}},
  queries: {Collapsible: {summary: ({count = 0}) => ({key: 'sumQueries', count})}}
};

export const TYPES = [
  {kind: 'checklist', icon: 'list', labelKey: 'typeList', helpKey: 'typeListHelp'},
  {kind: 'counter', icon: 'counter', labelKey: 'typeCounter', helpKey: 'typeCounterHelp'},
  {kind: 'queries', icon: 'database', labelKey: 'typeQueries', helpKey: 'typeQueriesHelp'}
].map(type => ({...type, traits: TRAITS.filter(name => name in IMPL[type.kind])}));

export const implementsTrait = (kind, trait) => Boolean(IMPL[kind]?.[trait]);

// summarize is the Collapsible method of a type; null when the type does not implement the trait or has no content.
export const summarize = (kind, facts) => IMPL[kind]?.Collapsible?.summary?.(facts ?? {}) ?? null;

const STEPS = 3; // 1 on this device, 2 on its way to the Desktop, 3 held by the Desktop

const BLOCKERS = {offline: 'desktopUnreachable', device_pending: 'awaitingApproval', device_refused: 'accessRefused', device_revoked: 'accessRefused'};

// toolState says where one tool stands and what stops it, from facts the client already has:
//   kind, hasDefinition (an IXLET definition exists), local (this device holds the only copy), saved (the Desktop saved the
//   project, for tools without a runtime), status (the connection pill: local | connecting | connected | offline | device_*).
// The steps are real milestones, so they are counted ("2 of 3"); no percentage is produced. While the connection itself is
// unknown the progress is unknown too.
export function toolState({kind, hasDefinition, local, saved, status}) {
  const needsRuntime = implementsTrait(kind, 'Runnable');
  const progressKnown = status !== 'connecting';
  const done = (state, step, blocker = null) => ({state, step, steps: STEPS, blocker, progressKnown});
  if (needsRuntime && !hasDefinition) return done('noRuntime', 0, 'noDefinition');
  if (status === 'local') return done('draft', 1, 'notPaired');
  const held = needsRuntime ? !local : saved;
  if (held) return status === 'connected' ? done('shared', 3) : done('lastKnown', 3, BLOCKERS[status] || null);
  return done('sending', 2, BLOCKERS[status] || null);
}

// displayMode reports what the browser says about how the page is shown. It cannot tell whether the app is installed:
// a browser tab can belong to an installed app, and "standalone" only means no browser chrome around the page.
export function displayMode(env = globalThis) {
  if (typeof env.matchMedia !== 'function') return 'unknown';
  if (env.matchMedia('(display-mode: standalone)').matches || env.navigator?.standalone === true) return 'standalone';
  return 'browser';
}
