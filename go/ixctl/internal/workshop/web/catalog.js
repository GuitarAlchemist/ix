// What the Widgets and IXlets views show, as plain functions: the widget types that exist, the state of one tool, and
// the display mode. No DOM and no network here, so a test can state each rule.

// The types the client can show. `runtime`: it runs an IXLET on the Desktop. `create`: the person can make one by hand
// (a list of queries is imported from the repo by cmd/ixforge-queries, never typed).
export const TYPES = [
  {kind: 'checklist', icon: 'list', labelKey: 'typeList', helpKey: 'typeListHelp', runtime: true, create: true},
  {kind: 'counter', icon: 'counter', labelKey: 'typeCounter', helpKey: 'typeCounterHelp', runtime: true, create: true},
  {kind: 'queries', icon: 'database', labelKey: 'typeQueries', helpKey: 'typeQueriesHelp', runtime: false, create: false}
];

const STEPS = 3; // 1 on this device, 2 on its way to the Desktop, 3 held by the Desktop

const BLOCKERS = {offline: 'desktopUnreachable', device_pending: 'awaitingApproval', device_refused: 'accessRefused', device_revoked: 'accessRefused'};

// toolState says where one tool stands and what stops it, from facts the client already has:
//   kind, hasDefinition (an IXLET definition exists), local (this device holds the only copy), saved (the Desktop saved the
//   project, for tools without a runtime), status (the connection pill: local | connecting | connected | offline | device_*).
// The steps are real milestones, so they are counted ("2 of 3"); no percentage is produced. While the connection itself is
// unknown the progress is unknown too.
export function toolState({kind, hasDefinition, local, saved, status}) {
  const needsRuntime = kind !== 'queries';
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
