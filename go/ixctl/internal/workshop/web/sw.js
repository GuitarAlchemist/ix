const CACHE = 'ix-workshop-shell-670f8c1a14e2c4aa';
const SHELL = ['/workshop/', '/workshop/style.css', '/workshop/app.js', '/workshop/sync.js', '/workshop/voice.js', '/workshop/updates.js', '/workshop/branding.json', '/workshop/vendor/i18next.min.js', '/workshop/manifest.webmanifest', '/workshop/icon-192.png', '/workshop/icon-512.png'];
self.addEventListener('install', event => event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(SHELL))));
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const key of await caches.keys()) if (key.startsWith('ix-workshop-shell-') && key !== CACHE) await caches.delete(key);
  await self.clients.claim();
})()));
let negotiation = null;
const workshopClients = async () => (await self.clients.matchAll({type: 'window', includeUncontrolled: true})).filter(client => new URL(client.url).pathname.startsWith('/workshop/'));
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
  if (event.request.method !== 'GET' || url.origin !== self.location.origin || !SHELL.includes(url.pathname) || url.search) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try { const response = await fetch(event.request); if (!response.ok) throw new Error('shell_unavailable'); await cache.put(event.request, response.clone()); return response; }
    catch (error) { const cached = await cache.match(event.request); if (cached) return cached; throw error; }
  })());
});
