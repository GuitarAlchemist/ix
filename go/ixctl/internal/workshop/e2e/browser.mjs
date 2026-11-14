import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtempSync, readdirSync, readFileSync, rmSync, mkdirSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {request as httpRequest} from 'node:https';
import {X509Certificate} from 'node:crypto';
import {proveBrowser} from './cdp.mjs';
import {checkFreshProfile, readActivePort, PROFILE_PREFIX} from './browser-owner.mjs';

const endpoint = new URL(process.env.WORKSHOP_FIXTURE_URL);
assert.equal(endpoint.protocol, 'https:'); assert.equal(endpoint.hostname, '127.0.0.1');
assert.ok(Number(endpoint.port) > 1024 && ![8737,8738,8739,8740].includes(Number(endpoint.port)));
const key = process.env.WORKSHOP_FIXTURE_KEY; assert.match(key, /^[0-9a-f]{48}$/);
const ORIGIN = process.env.WORKSHOP_FIXTURE_ORIGIN;
assert.equal(new URL(ORIGIN).hostname,'workshop.invalid'); assert.equal(new URL(ORIGIN).port,endpoint.port);
const certificate=new X509Certificate(Buffer.from(process.env.WORKSHOP_FIXTURE_CERT,'base64')).toString();
const pin=process.env.WORKSHOP_FIXTURE_PIN; assert.match(pin,/^[A-Za-z0-9+/]{43}=$/);
const chromePath = process.env.CHROME || 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const owners = [];
const artifacts = process.env.WORKSHOP_ARTIFACT_DIR;
if (artifacts) mkdirSync(artifacts, {recursive: true});
const log = text => console.log('PASS ' + text);

async function fixture(action, id) {
  const response=await forward({url:ORIGIN+'/fixture',method:'POST',headers:{'X-Workshop-Fixture':key,'Content-Type':'application/json'},postData:JSON.stringify({action,id})});
  assert.equal(response.responseCode,200);return JSON.parse(Buffer.from(response.body,'base64').toString());
}
assert.equal((await fixture('identity')).fixture, key);

function forward(request) {
  return new Promise((resolve, reject) => {
    const url = new URL(request.url), body = request.postData === undefined ? null : Buffer.from(request.postData);
    const headers = {...request.headers, host: new URL(ORIGIN).host};
    for (const h of Object.keys(headers)) if (h.toLowerCase() === 'content-length') delete headers[h];
    if (body) headers['Content-Length'] = String(body.length);
    const out = httpRequest({hostname: endpoint.hostname, port: endpoint.port, ca:certificate,servername:'example.com',path: url.pathname + url.search, method: request.method, headers}, response => {
      const chunks = []; response.on('data', c => chunks.push(c));
      response.on('end', () => resolve({responseCode: response.statusCode, responseHeaders: Object.entries(response.headers).flatMap(([name, values]) => [].concat(values).map(value => ({name, value: String(value)}))), body: Buffer.concat(chunks).toString('base64')}));
    });
    out.on('error', reject); out.setTimeout(10000, () => out.destroy(new Error('fixture timeout')));
    if (body) out.write(body); out.end();
  });
}
async function browser(name) {
  const profile = mkdtempSync(join(tmpdir(), PROFILE_PREFIX));
  checkFreshProfile(profile, tmpdir(), readdirSync(profile));
  const child = spawn(chromePath, ['--headless=new', '--remote-debugging-port=0', '--user-data-dir=' + profile, '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--no-proxy-server','--ignore-certificate-errors-spki-list='+pin,'--host-resolver-rules=MAP workshop.invalid 127.0.0.1, MAP * ~NOTFOUND, EXCLUDE 127.0.0.1', 'about:blank'], {stdio: 'ignore', windowsHide: true});
  const owner = {profile, child, connection: null, offline: false, proxyDown: false, ambiguous: false, dropNext: false, hangNext: false, swVersion: 1, requests: []}; owners.push(owner);
  let activePort;
  for (let attempt = 0; attempt < 150; attempt++) {
    try { activePort = readFileSync(join(profile, 'DevToolsActivePort'), 'utf8'); readActivePort(activePort); break; } catch { await sleep(100); }
  }
  assert.ok(activePort, 'Chrome did not publish its own debugging endpoint');
  const version = await (await fetch(`http://127.0.0.1:${readActivePort(activePort).port}/json/version`)).json();
  const connection = await proveBrowser({activePort, version, childPid: child.pid, alive: () => child.exitCode === null}); owner.connection = connection;
  console.log(`OWNED ${name}: PID ${child.pid}; profile ${profile}; debugger port ${readActivePort(activePort).port}`);
  connection.on(async msg => {
    if (msg.method !== 'Fetch.requestPaused') return;
    const {requestId, request} = msg.params;
    const reply = (method, params) => connection.send(method, {requestId, ...params}, msg.sessionId).catch(() => {});
    try {
      const url = new URL(request.url);
      if (url.origin !== ORIGIN || !url.pathname.startsWith('/workshop/')) return await reply('Fetch.failRequest', {errorReason: 'BlockedByClient'});
      if (owner.offline) return await reply('Fetch.failRequest', {errorReason: 'InternetDisconnected'});
      if (owner.proxyDown) return await reply('Fetch.fulfillRequest', {responseCode: 502, responseHeaders: [{name:'Content-Type',value:'text/plain'}], body:Buffer.from('Fixture proxy unavailable').toString('base64')});
      if (owner.ambiguous && url.pathname.startsWith('/workshop/api/')) return await reply('Fetch.fulfillRequest', {responseCode: 401, responseHeaders: [{name: 'Content-Type', value: 'application/json'}], body: Buffer.from('{}').toString('base64')});
      const response = await forward(request);
      if (url.pathname === '/workshop/sw.js') console.log('SW FETCH', owner.swVersion, response.responseCode, Buffer.from(response.body,'base64').length);
      if (url.pathname === '/workshop/api/operations') {
        owner.requests.push(JSON.parse(request.postData).op_id);
        if (owner.hangNext) { owner.hangNext = false; setTimeout(() => reply('Fetch.failRequest',{errorReason:'ConnectionReset'}),16000); return; }
        if (owner.dropNext) { owner.dropNext = false; return await reply('Fetch.failRequest', {errorReason: 'ConnectionReset'}); }
      }
      if (url.pathname === '/workshop/sw.js' && owner.swVersion === 2) response.body = Buffer.from(Buffer.from(response.body, 'base64').toString().replace('ix-workshop-shell-v1', 'ix-workshop-shell-v2')).toString('base64');
      await reply('Fetch.fulfillRequest', response);
    } catch (error) { console.error('intercept', error.message); await reply('Fetch.failRequest', {errorReason: 'Failed'}); }
  });
  const enabled = await connection.send('Fetch.enable', {patterns: [{urlPattern: '*'}]}); assert.ok(!enabled.error);
  const tab = await connection.openTab(); owner.tab = tab;
  await tab.send('Page.enable');
  owner.eval = async expression => {
    const result = await tab.send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (result.result?.exceptionDetails) throw new Error(result.result.exceptionDetails.exception?.description || 'page evaluation failed');
    if (result.error) throw new Error(result.error.message);
    return result.result?.result?.value;
  };
  owner.until = async (expression, explanation, timeout = 20000) => {
    const end = Date.now() + timeout;
    while (Date.now() < end) { if (connection.lost) throw connection.lost; try { if (await owner.eval(expression)) return; } catch {} await sleep(100); }
    throw new Error(explanation + ': ' + await owner.eval('document.body.innerText'));
  };
  owner.click = id => owner.eval(`document.getElementById(${JSON.stringify(id)}).click()`);
  owner.type = (id, value) => owner.eval(`(() => {const input=document.getElementById(${JSON.stringify(id)});input.value=${JSON.stringify(value)};input.dispatchEvent(new Event('input'));})()`);
  owner.state = () => owner.eval(`import('/workshop/sync.js').then(async m => (await m.Replica.open()).read())`);
  owner.navigate = async url => { await tab.send('Page.navigate', {url}); await owner.until(`document.querySelector('#interpret')?.textContent === 'Interpréter' || document.querySelector('#interpret')?.textContent === 'Interpret'`, 'app did not boot'); };
  owner.reload = async () => {
    await owner.eval('window.__beforeReload=true'); await tab.send('Page.reload');
    await owner.until(`window.__beforeReload!==true && document.querySelector('#interpret')?.textContent.length>0`, 'app did not reopen');
  };
  owner.shot = async file => { if (!artifacts) return; await owner.eval('scrollTo(0,0)'); const layout=await tab.send('Page.getLayoutMetrics');const size=layout.result.cssContentSize; const r = await tab.send('Page.captureScreenshot', {format: 'png',captureBeyondViewport:true,clip:{x:0,y:0,width:size.width,height:size.height,scale:1}}); writeFileSync(join(artifacts, file + '.png'), Buffer.from(r.result.data, 'base64')); };
  return owner;
}
async function connect(owner, name) {
  const invite = await fixture('invite');
  if (artifacts) writeFileSync(join(artifacts, 'fixture-pairing.svg'), invite.svg);
  await owner.navigate(ORIGIN + '/workshop/#pair=' + invite.code);
  assert.equal(await owner.eval('location.hash'), '', 'QR fragment cleared');
  await owner.type('device-name', name); await owner.click('pair');
  await owner.until(`document.getElementById('connection').textContent.includes('approbation')`, 'device pending');
  const devices = await fixture('devices'), d = devices.find(item => item.name === name);
  assert.equal(d.status, 'pending'); await fixture('approved', d.id); await owner.click('sync');
  await owner.until(`document.getElementById('connection').textContent==='Desktop connecté'`, 'device approval');
  return d.id;
}
async function rename(owner, title) {
  await owner.type('project-title', title);
  await owner.until(`import('/workshop/sync.js').then(async m=>{const s=await(await m.Replica.open()).read();return s.projects[s.selected]?.document.title===${JSON.stringify(title)}})`, 'local rename stored');
  await owner.click('save');
}
async function saved(owner, title) {
  await owner.until(`document.getElementById('save-status').textContent==='Enregistré sur Desktop' && document.getElementById('project-title').value===${JSON.stringify(title)}`, 'committed Desktop version');
}

try {
  const pc = await browser('unpaired-PC');
  await pc.navigate(ORIGIN + '/workshop/');
  await pc.click('new-project');
  await pc.until(`import('/workshop/sync.js').then(async m=>Object.keys((await(await m.Replica.open()).read()).projects).length===1)`, '+ Project did not create a local project', 2000);
  let local = await pc.state(), localID = local.selected;
  assert.equal(local.token, ''); assert.equal(local.projects[localID].document.title, 'Nouveau projet');
  assert.equal(await pc.eval(`document.getElementById('project-editor').hidden`), false);
  assert.equal(local.outbox.length, 1); assert.equal(local.projects[localID].revision, 0);
  await pc.reload(); assert.equal((await pc.state()).selected, localID);
  assert.equal(await pc.eval(`document.getElementById('save-status').textContent`), 'Gardé sur cet appareil');
  await pc.eval(`(()=>{const select=document.getElementById('language');select.value='en';select.dispatchEvent(new Event('change'))})()`);
  await pc.until(`document.getElementById('new-project').textContent==='+ Project'`, 'English project button');
  await pc.click('new-project');
  await pc.until(`import('/workshop/sync.js').then(async m=>Object.keys((await(await m.Replica.open()).read()).projects).length===2)`, 'second project missing');
  local=await pc.state(); assert.equal(local.projects[local.selected].document.title, 'New project');
  assert.equal(local.projects[localID].document.title, 'Nouveau projet'); assert.equal(local.outbox.length, 2);
  assert.equal(pc.requests.length, 0, 'unpaired local creation must not write to Desktop');
  assert.equal((await fixture('devices')).length, 0, 'no access created');
  const usefulID=local.selected;
  await pc.type('transcript', 'Create a workspace "QA room" with a counter "Checks"');
  await pc.until(`import('/workshop/sync.js').then(async m=>(await(await m.Replica.open()).read()).transcript.includes('Checks'))`, 'local transcript persisted');
  await pc.click('interpret'); await pc.until(`!document.getElementById('apply').hidden`, 'local intent visible');
  assert.equal((await pc.state()).outbox.length, 2, 'interpretation has no side effect');
  await pc.click('apply');
  await pc.until(`import('/workshop/sync.js').then(async m=>{const s=await(await m.Replica.open()).read();return s.projects[s.selected].document.workspaces.length===1})`, 'useful workspace missing');
  await pc.type('project-title', 'QA local project');
  await pc.until(`document.getElementById('project-title').value==='QA local project'`, 'renamed project visible');
  await pc.click('save'); await pc.reload(); local=await pc.state();
  assert.equal(local.selected,usefulID); assert.equal(local.projects[usefulID].document.title,'QA local project');
  assert.equal(local.projects[usefulID].document.workspaces[0].widgets[0].title,'Checks');
  assert.equal(local.projects[usefulID].document.workspaces[0].ixlet_definitions[0].status,'draft');
  assert.equal(local.outbox.length,4); assert.equal(pc.requests.length,0); assert.equal(local.token,'');
  await pc.shot('project-button-unpaired-en');
  log('+ Project -> transcript -> workspace/widget -> rename/save -> reload in FR/EN; previous projects and exact outbox retained, no pairing or backend write');
  if (process.env.WORKSHOP_PROJECT_ONLY !== '1') {
  const a = await browser('tablet-A'); await connect(a, 'Fixture A');
  await a.until(`navigator.serviceWorker.controller!==null`, 'scoped worker installed');
  await a.type('transcript', 'Créer un workspace "Maison" avec un compteur "Lectures"');
  await a.until(`import('/workshop/sync.js').then(async m=>(await(await m.Replica.open()).read()).transcript.includes('Lectures'))`, 'transcript stored');
  await a.click('interpret'); await a.until(`!document.getElementById('apply').hidden`, 'intent preview');
  assert.equal((await a.state()).outbox.length, 0, 'interpretation cannot execute');
  await a.click('apply'); await saved(a, 'Maison');
  let state = await a.state(); const projectID = state.selected;
  assert.equal(state.projects[projectID].revision, 1); assert.equal(state.projects[projectID].document.workspaces[0].ixlet_definitions[0].status, 'draft');
  await a.reload(); await saved(a, 'Maison'); log('transcript → intent → workspace + widget + IXLET draft → SQLite → reopen');

  const b = await browser('tablet-B'); const deviceB = await connect(b, 'Fixture B'); await saved(b, 'Maison');
  a.offline = true; b.offline = true;
  await rename(a, 'Version A'); await rename(b, 'Version B');
  await a.until(`document.getElementById('connection').textContent.startsWith('Non connecté')`, 'offline shown');
  const queued = (await a.state()).outbox[0].op_id;
  await a.reload();
  await a.until(`document.getElementById('project-title').value==='Version A'`, 'offline shell reopened');
  assert.equal((await a.state()).outbox[0].op_id, queued); log('offline shell reopen retains IndexedDB draft and exact outbox operation');
  a.offline = false; a.hangNext = true; const timeoutStart=Date.now(); await a.click('sync');
  await a.until(`document.getElementById('connection').textContent==='Synchronisation…'`, 'in-flight request shown');
  await a.until(`document.getElementById('connection').textContent.startsWith('Non connecté')`, 'lost response shown');
  assert.ok(Date.now()-timeoutStart>=14000,'real response timeout elapsed');
  assert.equal((await a.state()).outbox[0].op_id, queued);
  await a.click('sync'); await saved(a, 'Version A');
  assert.equal((await a.state()).projects[projectID].revision, 2);
  assert.ok(a.requests.filter(id => id === queued).length >= 2); log('timeout after commit retries same op_id without an extra revision');

  b.offline = false; await b.click('sync');
  await b.until(`!document.getElementById('conflict').hidden`, 'visible two-device conflict');
  state = await b.state(); assert.equal(state.projects[projectID].document.title, 'Version B'); assert.equal(state.projects[projectID].remote.project.title, 'Version A'); assert.equal(state.outbox.length, 1);
  await b.tab.send('Emulation.setDeviceMetricsOverride', {width: 1280, height: 800, deviceScaleFactor: 1, mobile: true});
  assert.equal(await b.eval('document.documentElement.scrollWidth<=innerWidth'), true); await b.shot('landscape-conflict-fr');
  await b.click('keep-local'); await saved(b, 'Version B'); assert.equal((await b.state()).projects[projectID].revision, 3);
  await a.click('sync'); await saved(a, 'Version B'); log('concurrent revisions retain both intentions; explicit resolution saves revision 3');

  await fixture('restart'); await a.reload(); await saved(a, 'Version B'); log('SQLite close/reopen preserves projects, tokens, revisions and operation receipts');
  a.ambiguous = true; await rename(a, '401 protected');
  state = await a.state(); const protectedToken = state.token, protectedOp = state.outbox[0].op_id;
  const sibling=await a.connection.openTab();
  const siblingEval=async expression=>{const r=await sibling.send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.result?.exceptionDetails)throw new Error(r.result.exceptionDetails.text);return r.result?.result?.value;};
  await sibling.send('Page.enable');
  await sibling.send('Page.addScriptToEvaluateOnNewDocument',{source:`class FixtureRecognition {static async available(){return 'available'} start(){window.__fixtureRecognition=this} stop(){this.onend?.()}};FixtureRecognition.prototype.processLocally=false;window.SpeechRecognition=FixtureRecognition;`});
  await sibling.send('Page.navigate',{url:ORIGIN+'/workshop/'});
  for(let n=0;n<200;n++){try{if(await siblingEval(`document.getElementById('interpret')?.textContent==='Interpréter'`))break;}catch{}await sleep(100);}
  await siblingEval(`document.getElementById('dictate').click()`);
  for(let n=0;n<100;n++){if(await siblingEval(`document.getElementById('voice-status').textContent==='Dictée locale en cours…'`))break;await sleep(100);}
  assert.equal(await siblingEval(`document.getElementById('voice-status').textContent`),'Dictée locale en cours…');
  await a.eval(`document.getElementById('transcript').focus()`);
  a.swVersion = 2; await fixture('upgrade');
  await a.eval(`navigator.serviceWorker.getRegistration('/workshop/').then(r=>r.update())`);
  await a.until(`navigator.serviceWorker.getRegistration('/workshop/').then(r=>!!r.waiting)`, 'worker update ready');
  await sleep(3000);assert.ok((await a.eval(`document.getElementById('client-version').textContent`)).includes('0.1.1'),'editing prevents reload');
  await a.eval(`document.activeElement.blur()`);await sleep(3000);
  assert.ok((await a.eval(`document.getElementById('client-version').textContent`)).includes('0.1.1'),'dictation in sibling tab prevents activation');
  await siblingEval(`document.getElementById('stop-dictation').click()`);
  await a.until(`document.getElementById('client-version').textContent.includes('0.2.0')`, 'automatic safe reload to new client');
  for(let n=0;n<200;n++){try{if(await siblingEval(`document.getElementById('client-version').textContent.includes('0.2.0')`))break;}catch{}await sleep(100);}
  assert.ok((await siblingEval(`document.getElementById('client-version').textContent`)).includes('0.2.0'),'sibling reloaded to new client');
  state = await a.state();
  assert.equal(state.token, protectedToken); assert.equal(state.outbox[0].op_id, protectedOp); assert.equal(state.projects[projectID].document.title, '401 protected');
  await fixture('proxy-down');await a.reload();
  await a.until(`document.getElementById('project-title').value==='401 protected'`, 'HTTP 502 shell fallback');
  assert.equal((await a.state()).outbox[0].op_id,protectedOp);await fixture('proxy-up');
  const cached = await a.eval(`(async()=>{const result=[];for(const key of await caches.keys())for(const request of await(await caches.open(key)).keys())result.push(new URL(request.url).pathname);return result})()`);
  assert.ok(cached.length > 0); assert.ok(cached.every(path => path.startsWith('/workshop/') && !path.includes('/api/')));
  a.ambiguous = false; await a.click('sync'); await saved(a, '401 protected'); log('automatic v0.1 → v0.2 update coordinates typing/dictation across tabs; 401, 502 and reload retain token/draft/outbox; no API cache');

  await a.tab.send('Emulation.setDeviceMetricsOverride', {width: 768, height: 1024, deviceScaleFactor: 1, mobile: true});
  await a.eval(`(()=>{const s=document.getElementById('language');s.value='en';s.dispatchEvent(new Event('change'));})()`);
  await a.until(`document.documentElement.lang==='en' && document.getElementById('save').textContent==='Save to Desktop'`, 'English interface');
  await a.eval(`(()=>{const s=document.getElementById('theme');s.value='light';s.dispatchEvent(new Event('change'));})()`);
  await a.until(`document.documentElement.dataset.theme==='light'`, 'light theme');
  assert.equal(await a.eval('document.documentElement.scrollWidth<=innerWidth'), true); await a.shot('portrait-en-light');
  await a.tab.send('Emulation.setDeviceMetricsOverride', {width: 360, height: 800, deviceScaleFactor: 1, mobile: true});
  assert.equal(await a.eval('document.documentElement.scrollWidth<=innerWidth'), true); await a.shot('phone-en-light');
  const manifest = JSON.parse(Buffer.from((await forward({url: ORIGIN + '/workshop/manifest.webmanifest', method: 'GET', headers: {}})).body, 'base64').toString());
  assert.equal(manifest.scope, '/workshop/'); assert.equal(manifest.display, 'standalone'); assert.equal(manifest.icons.length, 2);
  log('portrait/landscape/360px, FR/EN, light theme and scoped install manifest');

  b.offline = true; await rename(b, 'Revoked local draft'); await fixture('revoked', deviceB); b.offline = false; await b.click('sync');
  await b.until(`document.getElementById('connection').textContent.includes('révoqué')`, 'explicit revocation shown');
  state = await b.state(); assert.equal(state.projects[projectID].document.title, 'Revoked local draft'); assert.equal(state.outbox.length, 1); log('revoked device cannot save; local draft and outbox remain');
  const newInvite=await fixture('invite');await b.click('replace-access');await b.type('pair-code',newInvite.code);await b.type('device-name','Fixture B replacement');await b.click('pair');
  await b.until(`document.getElementById('connection').textContent.includes('approbation')`,'replacement awaits fresh approval');
  assert.equal((await b.state()).projects[projectID].document.title,'Revoked local draft');assert.equal((await b.state()).outbox.length,1);
  const replacement=(await fixture('devices')).find(d=>d.name==='Fixture B replacement');await fixture('refused',replacement.id);await b.click('sync');
  await b.until(`document.getElementById('connection').textContent.includes('refusé')`,'explicit refusal shown');assert.equal((await b.state()).outbox.length,1);log('explicit re-pair retains drafts/outbox and fresh refusal blocks access');
  a.offline=true;
  const atomic=await a.eval(`(async()=>{const m=await import('/workshop/sync.js');const one=await m.Replica.open(),two=await m.Replica.open();const stale=await one.read();await two.create({workspaceTitle:'Added concurrently',counterTitle:'Two'},stale.selected);await one.rename(stale.selected,'Atomic title');await one.create({workspaceTitle:'Third workspace',counterTitle:'Three'},stale.selected);const current=await one.read();return current.projects[stale.selected].document;})()`);
  assert.equal(atomic.workspaces.length,3);assert.equal(atomic.title,'Atomic title');assert.equal(atomic.workspaces[1].title,'Added concurrently');log('concurrent IndexedDB transforms preserve workspaces added after an earlier read');
  await pc.eval(`(()=>{const select=document.getElementById('language');select.value='fr';select.dispatchEvent(new Event('change'))})()`);
  await pc.until(`document.documentElement.lang==='fr'`, 'French fixture pairing labels');
  await pc.tab.send('Page.navigate',{url:'about:blank'});
  await pc.until(`location.href==='about:blank'`, 'fresh document before fixture pairing');
  await connect(pc,'Fixture PC');
  await pc.until(`import('/workshop/sync.js').then(async m=>{const s=await(await m.Replica.open()).read();return s.outbox.length===0 && s.projects[${JSON.stringify(usefulID)}].revision===3})`, 'queued PC project did not save after fixture approval');
  await fixture('restart'); await pc.reload(); local=await pc.state();
  assert.equal(local.projects[usefulID].document.title,'QA local project');
  assert.equal(local.projects[usefulID].document.workspaces[0].widgets[0].title,'Checks');
  assert.equal(local.projects[usefulID].revision,3); assert.equal(local.outbox.length,0);
  log('previously unpaired PC project saves once after synthetic approval, survives SQLite restart and reload');
  }
} finally {
  for (const owner of owners.reverse()) {
    if (owner.connection) { try { await owner.connection.send('Browser.close'); } catch {} owner.connection.close(); }
    for (let n = 0; n < 100 && owner.child.exitCode === null; n++) await sleep(100);
    if (owner.child.exitCode === null) { owner.child.kill(); await new Promise(resolve => owner.child.once('exit', resolve)); }
    assert.equal(resolve(dirname(owner.profile)), resolve(tmpdir())); assert.ok(owner.profile.startsWith(join(tmpdir(), PROFILE_PREFIX)));
    rmSync(owner.profile, {recursive: true, force: true, maxRetries: 10, retryDelay: 200});
    assert.equal(owner.child.exitCode === null, false); console.log('CLEANED owned Chrome PID ' + owner.child.pid + ' and its fresh profile');
  }
}
