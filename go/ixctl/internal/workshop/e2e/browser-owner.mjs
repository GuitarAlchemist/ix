// Proof that the browser the test drives is the one it started, before any interception. Pure, tested on fixtures
// (browser-owner.test.mjs).
//
// Chrome's own mechanism is used: started with --remote-debugging-port=0, it picks a free port and writes it, with the
// path of its browser endpoint, to DevToolsActivePort in its profile. The path holds a random identifier of that
// browser's incarnation. The test accepts a browser only if all of these hold:
//   - the profile is a fresh, empty, throwaway folder the test made (checkFreshProfile);
//   - DevToolsActivePort exists in that profile, written by the Chrome it holds;
//   - the endpoint answering on that port, on the loopback, names the same incarnation;
//   - the browser process that endpoint reports is the process the test started, still alive.
// A port alone, a PID alone, or the first browser answering, prove nothing.

import { basename, dirname, resolve, sep } from "node:path";

export const PROFILE_PREFIX = "workshop-e2e-chrome-";
const refuse = (why) => new Error("navigateur de test non prouvé, aucune interception : " + why);

export function readActivePort(text) {
  const [first, second] = String(text ?? "").split(/\r?\n/);
  const port = /^\d+$/.test(first ?? "") ? Number(first) : NaN;
  if (!(port >= 1 && port <= 65535) || !/^\/devtools\/browser\/[0-9a-f-]{36}$/i.test(second ?? "")) {
    throw new Error("DevToolsActivePort illisible : " + JSON.stringify(text));
  }
  return { port, path: second };
}

// checkOwnership returns the browser endpoint to drive, or throws without anything having been intercepted.
export function checkOwnership({ activePort, version, browserPid, childPid, childAlive }) {
  const endpoint = checkEndpoint({ activePort, version });
  if (!Number.isInteger(childPid) || !Number.isInteger(browserPid) || browserPid !== childPid) {
    throw refuse(`le processus du navigateur (${browserPid}) n'est pas celui lancé par le test (${childPid})`);
  }
  if (!childAlive) throw refuse("le navigateur lancé par le test est terminé");
  return endpoint;
}

// checkEndpoint: the first half, before the test even connects: the endpoint answering on the profile's port names the
// incarnation the profile's Chrome wrote. Only then may the test connect to it, to ask for its process.
export function checkEndpoint({ activePort, version }) {
  if (activePort === null || activePort === undefined) throw refuse("DevToolsActivePort absent du profil de test");
  let own;
  try { own = readActivePort(activePort); } catch (e) { throw refuse(e.message); }
  if (!version || typeof version.webSocketDebuggerUrl !== "string") throw refuse("le point d'accès du profil ne répond pas");
  let u;
  try { u = new URL(version.webSocketDebuggerUrl); } catch { throw refuse("point d'accès illisible"); }
  if (u.protocol !== "ws:" || u.hostname !== "127.0.0.1" || Number(u.port) !== own.port) {
    throw refuse(`autre point d'accès que celui du profil : ${u.host}`);
  }
  if (u.pathname !== own.path) throw refuse("autre incarnation du navigateur sur ce port");
  return `ws://127.0.0.1:${own.port}${own.path}`;
}

// checkFreshProfile: the profile folder is the test's own, made just now, empty before Chrome starts.
export function checkFreshProfile(profile, tmp, entries) {
  const p = resolve(profile);
  const fold = (x) => (sep === "\\" ? x.toLowerCase() : x);
  if (fold(dirname(p)) !== fold(resolve(tmp)) || !basename(p).startsWith(PROFILE_PREFIX)) {
    throw new Error("profil étranger : " + profile);
  }
  if (entries.length) throw new Error("profil non vide : " + profile);
  return true;
}

