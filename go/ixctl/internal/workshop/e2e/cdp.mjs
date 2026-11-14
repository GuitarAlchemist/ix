// The one DevTools connection the test drives Chrome through: the proof that the browser is the test's own goes over
// it (browser-owner.mjs), and every later action goes over it too. Tested on fixtures (cdp.test.mjs).
//
// The proof binds this connection, not the port: a port can be taken again by another browser once the proved one is
// dead, and asking that port for a new tab (HTTP /json/new) would create and drive a tab of the replacement. So tabs
// are created (Target.createTarget) and attached (Target.attachToTarget, flat sessions) over the proved connection, and
// driven by their session on it. When that connection ends, the browser is lost: every pending and later command is
// refused, and nothing reconnects, by the port or otherwise.

import { checkEndpoint, checkOwnership } from "./browser-owner.mjs";

// proveBrowser joins the endpoint named by the profile's DevToolsActivePort, asks the browser at the end of that very
// connection for its process, and keeps the connection only if it is the process the test started, still alive.
// proof: { activePort, version, childPid, alive() }. Returns the browser session, or throws with the connection closed.
export async function proveBrowser({ activePort, version, childPid, alive }) {
  const endpoint = checkEndpoint({ activePort, version });
  const browser = await connect(endpoint);
  try {
    const info = await browser.send("SystemInfo.getProcessInfo");
    const browserPid = info.result?.processInfo?.find((p) => p.type === "browser")?.id;
    checkOwnership({ activePort, version, browserPid, childPid, childAlive: alive() });
  } catch (e) {
    browser.close();
    throw e;
  }
  return browser;
}

async function connect(endpoint) {
  const ws = new WebSocket(endpoint);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = () => j(new Error("point d'accès du navigateur injoignable")); });
  let id = 0;
  let lost = null;
  const waiting = new Map();
  const listeners = new Set();
  const end = () => {
    if (lost) return;
    lost = new Error("navigateur de test perdu : sa connexion est fermée, plus aucune commande ni aucun onglet");
    for (const { reject } of waiting.values()) reject(lost);
    waiting.clear();
  };
  ws.onclose = end;
  ws.onerror = end;
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && waiting.has(msg.id)) {
      waiting.get(msg.id).resolve(msg);
      waiting.delete(msg.id);
    } else for (const l of listeners) l(msg);
  };

  // send: one command, to the browser or, with sessionId, to one of its tabs; refused once the connection has ended.
  const send = (method, params = {}, sessionId) => {
    if (lost) return Promise.reject(lost);
    return new Promise((resolve, reject) => {
      id++;
      waiting.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  };

  // openTab: a new tab of this browser, attached over this connection, never over HTTP or another socket.
  async function openTab() {
    const created = await send("Target.createTarget", { url: "about:blank" });
    if (created.error || !created.result?.targetId) throw new Error("onglet impossible : " + (created.error?.message || "pas de cible"));
    const { targetId } = created.result;
    const attached = await send("Target.attachToTarget", { targetId, flatten: true });
    if (attached.error || !attached.result?.sessionId) throw new Error("onglet non attaché : " + (attached.error?.message || "pas de session"));
    const { sessionId } = attached.result;
    return {
      targetId,
      sessionId,
      send: (method, params = {}) => send(method, params, sessionId),
      close: () => send("Target.closeTarget", { targetId }).catch(() => {}),
    };
  }

  return {
    send,
    openTab,
    on: (listener) => listeners.add(listener), // events: Fetch.requestPaused and the like
    get lost() { return lost; },
    close: () => { end(); ws.close(); },
  };
}
