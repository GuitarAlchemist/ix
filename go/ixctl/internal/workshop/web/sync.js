const copy = value => structuredClone(value);
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export const initialState = () => ({token: '', storeID: '', cursor: 0, projects: {}, outbox: [], transcript: '', selected: null, tool: null, surface: [], language: 'fr', theme: 'system', deviceName: null, pairedName: '', xlets: {}});
export const xletKey = (projectID, definitionID) => projectID + '/' + definitionID;

// A tool is one widget of a project, with the IXLET definition it presents (same kind and title, the template's
// pairing). Its key is "projectID/widgetID".
export function tools(state) {
  const list = [];
  for (const [projectID, p] of Object.entries(state.projects)) for (const workspace of p.document.workspaces) for (const widget of workspace.widgets) {
    const definition = workspace.ixlet_definitions.find(d => d.kind === widget.kind && d.title === widget.title);
    list.push({key: projectID + '/' + widget.id, projectID, project: p, workspace, widget, definition});
  }
  return list;
}

// While the Desktop holds no instance of a draft IXLET, this device runs it by itself, under the Desktop's own rules
// (xlet.go): the state lives in this replica only (localXlets) and nothing is sent. Once the project is saved on the
// Desktop, the sync activates the IXLET there and the local state goes with the activation as its seed.
export const localXlet = (state, projectID, definition, initialValue = 0) =>
  (state.localXlets || {})[xletKey(projectID, definition.id)] || {kind: definition.kind, value: definition.kind === 'counter' ? initialValue : 0, version: 0, items: []};
export function applyLocal(x, action, fields = {}) {
  const next = copy(x);
  if (x.kind === 'counter') {
    if (!['increment', 'decrement', 'reset'].includes(action)) throw new Error('invalid_document');
    next.value = action === 'reset' ? 0 : x.value + (action === 'increment' ? 1 : -1);
    if (!Number.isSafeInteger(next.value)) throw new Error('invalid_document');
  } else if (action === 'add') {
    const text = String(fields.text ?? '');
    if (!text.trim() || new TextEncoder().encode(text).length > 240 || x.items.length >= 200) throw new Error('invalid_document');
    next.items.push({id: crypto.randomUUID(), text, done: false});
  } else {
    const at = next.items.findIndex(i => i.id === fields.item_id);
    if (at < 0) throw new Error('ixlet_item_missing');
    if (action === 'remove') next.items.splice(at, 1);
    else if (action === 'check' || action === 'uncheck') next.items[at].done = action === 'check';
    else throw new Error('invalid_document');
  }
  if (x.kind === 'checklist') next.value = next.items.length;
  next.version = x.version + 1;
  return next;
}

// The local state of one IXLET as the seed of its activation on the Desktop, or undefined when there is nothing local.
export function seedOf(x) {
  if (!x) return undefined;
  return x.kind === 'counter' ? {value: x.value} : {items: x.items.map(i => ({text: i.text, done: i.done}))};
}
// Whether the Desktop's instance already holds what this device had locally.
const sameContent = (local, hosted) => local.kind === 'counter' ? local.value === hosted.value
  : equal(local.items.map(i => [i.text, i.done]), (hosted.items || []).map(i => [i.text, i.done]));
const enqueueInto = (state, id) => {
  const p = state.projects[id];
  if (!p || p.conflict) throw new Error('revision_conflict');
  const pending = state.outbox.filter(op => op.project.id === id);
  if (pending.length && equal(pending.at(-1).project, p.document)) return;
  if (!p.dirty) return;
  state.outbox.push({op_id: crypto.randomUUID(), base_revision: p.revision + pending.length, project: copy(p.document)});
};

export class Replica {
  constructor(db) { this.db = db; this.onCommit = () => {}; }
  static async open() {
    const db = await new Promise((resolve, reject) => {
      const req = indexedDB.open('ix-workshop-replica', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('state');
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return new Replica(db);
  }
  read() { return this.change(() => {}, 'readonly'); }
  change(update, mode = 'readwrite') {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('state', mode), store = tx.objectStore('state');
      const req = store.get('current'); let state;
      req.onsuccess = () => {
        try { state = req.result || initialState(); update(state); if (mode === 'readwrite') store.put(state, 'current'); }
        catch (error) { tx.abort(); reject(error); }
      };
      tx.oncomplete = () => { if (mode === 'readwrite') this.onCommit(); resolve(copy(state)); };
      tx.onabort = tx.onerror = () => reject(tx.error || new Error('local_storage_unavailable'));
    });
  }
  async newProject(title) {
    return this.change(state => {
      const document = {id: crypto.randomUUID(), title, workspaces: []};
      state.projects[document.id] = {document, revision: 0, dirty: true};
      state.selected = document.id;
      state.outbox.push({op_id: crypto.randomUUID(), base_revision: 0, project: copy(document)});
    });
  }
  async rename(id, title) {
    return this.change(state => {
      const project = state.projects[id];
      if (!project) throw new Error('project_missing');
      project.document.title = title;
      project.dirty = true;
    });
  }
  async enqueue(id) {
    return this.change(state => enqueueInto(state, id));
  }
  // create makes a new project holding one tool, from a command (interpret) or a name and a kind. A seed (an imported
  // widget's items or value) becomes the tool's local state until the Desktop runs it.
  async create(intent, seed) {
    return this.change(state => {
      const project = createDraft(intent);
      const workspace = project.workspaces[0], widget = workspace.widgets[0], definition = workspace.ixlet_definitions[0];
      state.projects[project.id] = {document: copy(project), revision: 0, dirty: true};
      state.selected = project.id; state.tool = project.id + '/' + widget.id;
      state.outbox.push({op_id: crypto.randomUUID(), base_revision: 0, project: copy(project)});
      if (seed && definition) {
        state.localXlets ??= {};
        state.localXlets[xletKey(project.id, definition.id)] = widget.kind === 'counter'
          ? {kind: 'counter', value: seed.value ?? 0, version: 1, items: []}
          : {kind: 'checklist', value: seed.items.length, version: 1, items: seed.items.map(i => ({id: crypto.randomUUID(), text: i.text, done: i.done}))};
      }
    });
  }
  // renameTool renames a widget with its IXLET definition (they pair by title), and the project and its workspace too
  // when the tool is alone in it. An empty or too long name is not applied: the Desktop would refuse it.
  async renameTool(key, title) {
    if (!title.trim() || new TextEncoder().encode(title).length > 240) return this.read();
    return this.change(state => {
      const tool = tools(state).find(x => x.key === key);
      if (!tool) throw new Error('project_missing');
      const p = state.projects[tool.projectID], workspace = p.document.workspaces.find(w => w.id === tool.workspace.id);
      const widget = workspace.widgets.find(w => w.id === tool.widget.id);
      const definition = tool.definition && workspace.ixlet_definitions.find(d => d.id === tool.definition.id);
      widget.title = title; if (definition) definition.title = title;
      if (p.document.workspaces.length === 1 && workspace.widgets.length === 1) { p.document.title = title; workspace.title = title; }
      p.dirty = true;
    });
  }
  async pin(key, pinned) {
    return this.change(state => {
      state.surface = (state.surface || []).filter(k => k !== key);
      if (pinned) state.surface.push(key);
    });
  }
  async actLocal(projectID, definition, initialValue, action, fields) {
    return this.change(state => {
      state.localXlets ??= {};
      state.localXlets[xletKey(projectID, definition.id)] = applyLocal(localXlet(state, projectID, definition, initialValue), action, fields);
    });
  }
  async resolve(id, keepLocal) {
    await this.change(state => {
      const p = state.projects[id]; if (!p?.remote) throw new Error('remote_revision_required');
      state.outbox = state.outbox.filter(op => op.project.id !== id);
      p.revision = p.remote.revision;
      if (!keepLocal) p.document = p.remote.project;
      delete p.remote; delete p.conflict; p.dirty = keepLocal;
    });
    if (keepLocal) return this.enqueue(id);
    return this.read();
  }
}

export class SyncEngine {
  constructor(replica, onChange = () => {}) { this.replica = replica; this.onChange = onChange; this.status = 'notConnected'; this.deniedStatus = ''; this.running = null; this.admin = false; this.terminal = false; }
  // The device the owner linked (or granted) administers the others from the app (admin.go).
  devices() { return this.api('devices'); }
  invite() { return this.api('devices/invite', {method: 'POST', body: '{}'}); }
  decide(id, status) { return this.api('devices/' + encodeURIComponent(id) + '/' + status, {method: 'POST', body: '{}'}); }
  // The terminal (terminal.go) runs a command on the Desktop's PC. A command is not repeatable: an uncertain reply is
  // reported as such and never sent again.
  runTerminal(command) {
    return this.api('terminal', {method: 'POST', body: JSON.stringify({command}), timeout: 45000})
      .catch(error => { if (error instanceof HTTPError && error.status < 500) throw error; throw new Error('terminal_uncertain'); });
  }
  async api(path, {timeout = 15000, ...options} = {}) {
    const state = await this.replica.read();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch('api/' + path, {...options, signal: controller.signal, cache: 'no-store', headers: {'Content-Type': 'application/json', ...(state.token ? {'Authorization': 'Bearer ' + state.token} : {}), ...options.headers}});
      let data;
      try { data = await response.json(); }
      catch { if (response.ok) throw new Error('connection_uncertain'); data = {}; }
      if (!response.ok) throw new HTTPError(response.status, data.code || 'connection_uncertain');
      return data;
    } finally { clearTimeout(timer); }
  }
  async pair(code, name, replaceAccess = false) {
    const state = await this.replica.read();
    if (state.token && !(replaceAccess && ['device_refused', 'device_revoked'].includes(this.deniedStatus))) throw new Error('access_already_present');
    const data = await this.api('pair', {method: 'POST', body: JSON.stringify({code, name, store_id: state.storeID})});
    // An ordinary pairing answers "pending"; the owner linking the Desktop's own PC answers "approved".
    if (!data.token || !data.store_id || !['pending', 'approved'].includes(data.device?.status)) throw new Error('connection_uncertain');
    if (state.storeID && data.store_id !== state.storeID) throw new Error('store_changed');
    // The name is sent only here: keep what Desktop registered, and drop the draft that was just sent.
    await this.replica.change(s => { s.token = data.token; s.storeID = data.store_id; s.pairedName = typeof data.device.name === 'string' ? data.device.name : ''; s.deviceName = null; });
    return this.sync();
  }
  // One request to a counter IXLET hosted by the Desktop. It is sent once: a lost or unclear reply is reported as
  // uncertain and the next sync rereads the Desktop's value; nothing is resent automatically.
  async act(projectID, definitionID, action, fields = {}) {
    const state = await this.replica.read();
    if (!state.token) throw new Error('ixlet_not_connected');
    let result;
    try {
      result = await this.api('xlets/actions', {method: 'POST', body: JSON.stringify({op_id: crypto.randomUUID(), project_id: projectID, definition_id: definitionID, action, ...fields})});
    } catch (error) {
      if (!(error instanceof HTTPError) || error.status >= 500 || error.message === 'connection_uncertain') { this.sync(); throw new Error('ixlet_uncertain'); }
      throw error;
    }
    if (!validXlet(result) || result.project_id !== projectID || result.definition_id !== definitionID) { this.sync(); throw new Error('ixlet_uncertain'); }
    await this.replica.change(s => { keepNewer(s, result); });
    this.onChange();
    return result;
  }
  sync() {
    if (this.running) return this.running;
    const run = () => this.run();
    this.running = (navigator.locks ? navigator.locks.request('ix-workshop-sync', run) : run()).finally(() => { this.running = null; this.onChange(); });
    return this.running;
  }
  async run() {
    try {
      let state = await this.replica.read();
      if (!state.token) { this.status = 'notConnected'; return; }
      if (this.status !== 'connected') { this.status = 'connecting'; this.onChange(); }
      const me = await this.api('me');
      if (me.protocol !== 'workshop/1' || typeof me.store_id !== 'string') throw new Error('connection_uncertain');
      if (me.store_id !== state.storeID) throw new Error('store_changed');
      this.status = 'connected'; this.deniedStatus = ''; this.admin = me.admin === true; this.terminal = me.terminal === true;
      if (typeof me.device?.name === 'string' && me.device.name !== state.pairedName) await this.replica.change(s => { s.pairedName = me.device.name; });
      // Nothing to press: every changed project without a conflict is queued for the Desktop.
      state = await this.replica.change(s => { for (const [id, p] of Object.entries(s.projects)) if (p.dirty && !p.conflict) enqueueInto(s, id); });
      // Retain each exact operation until its committed receipt has been durably recorded locally.
      for (const op of state.outbox) {
        const current = await this.replica.read();
        if (current.projects[op.project.id]?.conflict) continue;
        try {
          const receipt = await this.api('operations', {method: 'POST', body: JSON.stringify(op)});
          if (receipt.op_id !== op.op_id || receipt.project_id !== op.project.id || receipt.revision !== op.base_revision + 1 || !Number.isSafeInteger(receipt.sequence) || receipt.sequence < 1) throw new Error('connection_uncertain');
          await this.replica.change(s => {
            s.outbox = s.outbox.filter(item => item.op_id !== op.op_id);
            const p = s.projects[op.project.id]; p.revision = Math.max(p.revision, receipt.revision);
            p.dirty = !equal(p.document, op.project) || s.outbox.some(item => item.project.id === op.project.id);
          });
        } catch (error) {
          if (error.status !== 409) throw error;
          await this.replica.change(s => { s.projects[op.project.id].conflict = error.message; });
        }
      }
      do {
        state = await this.replica.read();
        const changes = await this.api('changes?since=' + state.cursor);
        if (typeof changes.store_id !== 'string' || !Array.isArray(changes.events) || !Number.isSafeInteger(changes.cursor) || changes.cursor < state.cursor) throw new Error('connection_uncertain');
        if (changes.store_id !== state.storeID) throw new Error('store_changed');
        await this.replica.change(s => {
          for (const event of changes.events) {
            const p = s.projects[event.project.id];
            if (p?.dirty || s.outbox.some(op => op.project.id === event.project.id)) {
              if (event.revision > p.revision) { p.remote = event; p.conflict = 'revision_conflict'; }
            } else s.projects[event.project.id] = {document: event.project, revision: event.revision, dirty: false};
          }
          s.cursor = changes.cursor;
          if (s.selected === null) s.selected = Object.keys(s.projects)[0] || null;
        });
        if (changes.events.length < 100) break;
      } while (true);
      const hosted = await this.api('xlets');
      if (hosted.store_id !== state.storeID || !Array.isArray(hosted.xlets) || !hosted.xlets.every(validXlet)) throw new Error('connection_uncertain');
      // The Desktop's list says which instances exist; per instance, the higher version wins (an action reply may be newer).
      await this.replica.change(s => {
        const previous = s.xlets || {}; s.xlets = {};
        for (const x of hosted.xlets) {
          const key = xletKey(x.project_id, x.definition_id); s.xlets[key] = previous[key]?.version > x.version ? previous[key] : x;
          // The Desktop's instance already holds the local state (a seeded activation whose reply was lost): drop the copy.
          const local = (s.localXlets || {})[key];
          if (local && sameContent(local, s.xlets[key])) delete s.localXlets[key];
        }
      });
      await this.activateSaved();
    } catch (error) {
      this.status = ['device_pending', 'device_refused', 'device_revoked', 'store_changed'].includes(error.message) ? error.message : 'notConnected';
      if (['device_refused', 'device_revoked'].includes(this.status)) this.deniedStatus = this.status;
      // A 401, network error or an explicit refusal never deletes a draft, access token or outbox entry.
    }
  }
}

// activateSaved starts on the Desktop every IXLET of a saved, settled project that has no instance yet, carrying
// what this device did locally. Each activation is sent once: an unclear reply ends this turn, and the next reread
// of the Desktop's instances decides (a seeded instance then matches the local copy, which is dropped).
SyncEngine.prototype.activateSaved = async function () {
  const state = await this.replica.read();
  for (const [projectID, p] of Object.entries(state.projects)) {
    if (!p.revision || p.dirty || p.conflict || state.outbox.some(op => op.project.id === projectID)) continue;
    for (const workspace of p.document.workspaces) for (const definition of workspace.ixlet_definitions) {
      const key = xletKey(projectID, definition.id);
      if ((state.xlets || {})[key] || (state.refusedActivations || {})[key]) continue;
      const local = (state.localXlets || {})[key], seed = seedOf(local);
      let result;
      try {
        result = await this.api('xlets/actions', {method: 'POST', body: JSON.stringify({op_id: crypto.randomUUID(), project_id: projectID, definition_id: definition.id, action: 'activate', ...(seed ? {seed} : {})})});
      } catch (error) {
        if (error instanceof HTTPError && [401, 403].includes(error.status)) throw error;
        // A definite refusal (an invalid seed) is not retried every five seconds; the local state stays as it is.
        if (error instanceof HTTPError && error.status === 400) await this.replica.change(s => { s.refusedActivations ??= {}; s.refusedActivations[key] = error.message; });
        return;
      }
      if (!validXlet(result) || result.project_id !== projectID || result.definition_id !== definition.id) return;
      await this.replica.change(s => {
        keepNewer(s, result);
        if (local && (result.seeded || sameContent(local, result))) delete s.localXlets[key];
      });
    }
  }
};

const validItem = i => i && typeof i.id === 'string' && typeof i.text === 'string' && typeof i.done === 'boolean';
const validXlet = x => x && typeof x.project_id === 'string' && typeof x.definition_id === 'string' && Number.isSafeInteger(x.value) && Number.isSafeInteger(x.version) && x.version >= 1 &&
  (x.kind === 'counter' || (x.kind === 'checklist' && (x.items === undefined || (Array.isArray(x.items) && x.items.every(validItem)))));
function keepNewer(state, x) {
  state.xlets ??= {};
  const key = xletKey(x.project_id, x.definition_id), known = state.xlets[key];
  if (!known || known.version < x.version) state.xlets[key] = x;
}

class HTTPError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

// A name is quoted "…", «…» or “…” (phone keyboards rarely type straight quotes), or bare, as a keyboard's voice
// typing writes it. Quoted forms are tried first, so a quoted name may contain the words that follow it.
const NAME = String.raw`("[^"\n]{1,120}"|«[^«»\n]{1,120}»|“[^“”\n]{1,120}”|[^"«»“”\n]{1,120}?)`;
const COMMANDS = {
  fr: new RegExp(String.raw`^(?:créer|crée|creer|cree) un workspace ${NAME} avec (un compteur|une liste(?: à cocher| a cocher)?) ${NAME}[.!]?$`, 'i'),
  en: new RegExp(String.raw`^create a workspace ${NAME} with a (counter|checklist|check list|list) ${NAME}[.!]?$`, 'i'),
};
const unquote = name => name.replace(/^["«“]|["»”]$/g, '').trim();
export function interpret(text, language) {
  const match = text.replace(/\s+/g, ' ').trim().match(COMMANDS[language === 'fr' ? 'fr' : 'en']);
  const workspaceTitle = match && unquote(match[1]), title = match && unquote(match[3]);
  if (!workspaceTitle || !title) throw new Error('command_unrecognized');
  return /^(?:un compteur|counter)$/i.test(match[2]) ? {workspaceTitle, counterTitle: title} : {workspaceTitle, checklistTitle: title};
}
export function createDraft(intent, existing) {
  const project = existing ? copy(existing) : {id: crypto.randomUUID(), title: intent.workspaceTitle, workspaces: []};
  // A list of queries runs nothing: its queries are in the widget, and it has no IXLET.
  if (intent.queriesTitle) {
    project.workspaces.push({id: crypto.randomUUID(), title: intent.workspaceTitle,
      widgets: [{id: crypto.randomUUID(), title: intent.queriesTitle, kind: 'queries', initial_value: 0, entries: copy(intent.entries)}], ixlet_definitions: []});
    return project;
  }
  const [kind, title] = intent.checklistTitle ? ['checklist', intent.checklistTitle] : ['counter', intent.counterTitle];
  project.workspaces.push({id: crypto.randomUUID(), title: intent.workspaceTitle,
    widgets: [{id: crypto.randomUUID(), title, kind, initial_value: 0}],
    ixlet_definitions: [{id: crypto.randomUUID(), title, kind, status: 'draft'}]});
  return project;
}
