const copy = value => structuredClone(value);
const equal = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export const initialState = () => ({token: '', storeID: '', cursor: 0, projects: {}, outbox: [], transcript: '', selected: null, language: 'fr', theme: 'system'});

export class Replica {
  constructor(db) { this.db = db; }
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
      tx.oncomplete = () => resolve(copy(state));
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
    return this.change(state => {
      const p = state.projects[id];
      if (!p || p.conflict) throw new Error('revision_conflict');
      const pending = state.outbox.filter(op => op.project.id === id);
      if (pending.length && equal(pending.at(-1).project, p.document)) return;
      if (!p.dirty) return;
      state.outbox.push({op_id: crypto.randomUUID(), base_revision: p.revision + pending.length, project: copy(p.document)});
    });
  }
  async create(intent, projectID) {
    return this.change(state => {
      const previous = state.projects[projectID];
      if (previous?.conflict) throw new Error('revision_conflict');
      const project = createDraft(intent, previous?.document);
      const revision = previous?.revision || 0;
      const pending = state.outbox.filter(op => op.project.id === project.id);
      state.projects[project.id] = {...previous, document: copy(project), revision, dirty: true};
      state.selected = project.id;
      state.outbox.push({op_id: crypto.randomUUID(), base_revision: revision + pending.length, project: copy(project)});
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
  constructor(replica, onChange = () => {}) { this.replica = replica; this.onChange = onChange; this.status = 'notConnected'; this.deniedStatus = ''; this.running = null; }
  async api(path, options = {}) {
    const state = await this.replica.read();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch('/workshop/api/' + path, {...options, signal: controller.signal, cache: 'no-store', headers: {'Content-Type': 'application/json', ...(state.token ? {'Authorization': 'Bearer ' + state.token} : {}), ...options.headers}});
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
    if (!data.token || !data.store_id || data.device?.status !== 'pending') throw new Error('connection_uncertain');
    if (state.storeID && data.store_id !== state.storeID) throw new Error('store_changed');
    await this.replica.change(s => { s.token = data.token; s.storeID = data.store_id; });
    return this.sync();
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
      this.status = 'connecting'; this.onChange();
      const me = await this.api('me');
      if (me.protocol !== 'workshop/1' || typeof me.store_id !== 'string') throw new Error('connection_uncertain');
      if (me.store_id !== state.storeID) throw new Error('store_changed');
      this.status = 'connected'; this.deniedStatus = '';
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
    } catch (error) {
      this.status = ['device_pending', 'device_refused', 'device_revoked', 'store_changed'].includes(error.message) ? error.message : 'notConnected';
      if (['device_refused', 'device_revoked'].includes(this.status)) this.deniedStatus = this.status;
      // A 401, network error or an explicit refusal never deletes a draft, access token or outbox entry.
    }
  }
}

class HTTPError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export function interpret(text, language) {
  const pattern = language === 'fr'
    ? /^(?:créer|crée|creer|cree) un workspace "([^"\n]{1,120})" avec un compteur "([^"\n]{1,120})"[.!]?$/i
    : /^create a workspace "([^"\n]{1,120})" with a counter "([^"\n]{1,120})"[.!]?$/i;
  const match = text.trim().match(pattern);
  if (!match) throw new Error('command_unrecognized');
  return {workspaceTitle: match[1], counterTitle: match[2]};
}
export function createDraft(intent, existing) {
  const project = existing ? copy(existing) : {id: crypto.randomUUID(), title: intent.workspaceTitle, workspaces: []};
  project.workspaces.push({id: crypto.randomUUID(), title: intent.workspaceTitle,
    widgets: [{id: crypto.randomUUID(), title: intent.counterTitle, kind: 'counter', initial_value: 0}],
    ixlet_definitions: [{id: crypto.randomUUID(), title: intent.counterTitle, kind: 'counter', status: 'draft'}]});
  return project;
}
