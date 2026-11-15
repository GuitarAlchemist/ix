const CACHE = 'ix-workshop-shell-50b92c4c68f85bea';
// Paths are relative to this worker's folder, so the client runs from any folder (the Desktop's, or a static host's).
const SHELL = ['./', './style.css', './app.js', './sync.js', './voice.js', './updates.js', './widgets.js', './ai.js', './vendor/lit-all.min.js', './branding.json', './vendor/i18next.min.js', './manifest.webmanifest', './icon-192.png', './icon-512.png'];
const SHELL_PATHS = SHELL.map(path => new URL(path, self.location).pathname);
// Static hosts let browsers keep files for minutes (GitHub Pages: max-age=600): a new version must fetch its files from
// the network, never the browser's copies of the previous version.
self.addEventListener('install', event => event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL.map(path => new Request(path, {cache: 'reload'}))))));
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const key of await caches.keys()) if (key.startsWith('ix-workshop-shell-') && key !== CACHE) await caches.delete(key);
  await self.clients.claim();
})()));
let negotiation = null;
const workshopClients = async () => (await self.clients.matchAll({type: 'window', includeUncontrolled: true})).filter(client => client.url.startsWith(self.registration.scope));
async function negotiate() {
  if (negotiation) return negotiation.promise;
  const id = crypto.randomUUID();
  let complete;
  const promise = new Promise(resolve => { complete = resolve; });
  const cycle = {id, clients: [], ready: new Set(), promise, finish: null};
  negotiation = cycle;
  let timer;
  cycle.finish = async activate => {
    if (negotiation !== cycle) return;
    negotiation = null; clearTimeout(timer);
    if (activate) await self.skipWaiting();
    else for (const client of cycle.clients) client.postMessage({type: 'workshop-update-deferred', id});
    complete();
  };
  try {
    cycle.clients = await workshopClients();
    if (!cycle.clients.length) { await cycle.finish(true); return promise; }
    timer = setTimeout(() => cycle.finish(false), 2500);
    for (const client of cycle.clients) client.postMessage({type: 'workshop-prepare-update', id});
  } catch { await cycle.finish(false); }
  return promise;
}
self.addEventListener('message', event => {
  if (event.data?.type === 'workshop-check-update') event.waitUntil(negotiate());
  if (event.data?.type === 'workshop-update-readiness') event.waitUntil((async () => {
    const cycle = negotiation;
    if (!cycle || cycle.id !== event.data.id || !cycle.clients.some(client => client.id === event.source?.id)) return;
    if (!event.data.ready) return cycle.finish(false);
    cycle.ready.add(event.source.id);
    if (cycle.ready.size === cycle.clients.length) {
      const current = await workshopClients();
      return cycle.finish(current.length === cycle.clients.length && current.every(client => cycle.ready.has(client.id)));
    }
  })());
});
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin || !SHELL_PATHS.includes(url.pathname) || url.search) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try { const response = await fetch(event.request); if (!response.ok) throw new Error('shell_unavailable'); await cache.put(event.request, response.clone()); return response; }
    catch (error) { const cached = await cache.match(event.request); if (cached) return cached; throw error; }
  })());
});
