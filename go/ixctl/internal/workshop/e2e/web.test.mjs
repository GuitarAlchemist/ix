import test from 'node:test';
import assert from 'node:assert/strict';
import {interpret, createDraft, SyncEngine, initialState} from '../web/sync.js';
import {startDictation, installLocalVoice} from '../web/voice.js';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';

test('French and English transcripts produce drafts, with no execution fields', () => {
  const fr = interpret('Créer un workspace "Maison" avec un compteur "Livres"', 'fr');
  const en = interpret('Create a workspace "Home" with a counter "Books"', 'en');
  assert.equal(fr.workspaceTitle, 'Maison'); assert.equal(en.counterTitle, 'Books');
  const p = createDraft(fr);
  assert.equal(p.workspaces[0].widgets[0].initial_value, 0);
  assert.equal(p.workspaces[0].ixlet_definitions[0].status, 'draft');
  const extended = createDraft(en, p);
  assert.equal(extended.workspaces.length, 2); assert.equal(p.workspaces.length, 1);
  assert.ok(!JSON.stringify(p).includes('shell'));
  assert.throws(() => interpret('run powershell Get-Content Scribe.db', 'en'), /command_unrecognized/);
});
test('unsupported voice never starts a recognizer', async () => {
  let state = ''; let starts = 0;
  class RemoteRecognition { start() { starts++; } }
  const result = await startDictation('en', () => assert.fail('audio'), s => { state = s; }, {SpeechRecognition: RemoteRecognition});
  assert.equal(result, null); assert.equal(starts, 0); assert.equal(state, 'voiceUnsupported');
});
test('local voice adapter uses requested language and exposes denial without audio', async () => {
  let state = '', text = '', instance, options;
  class Recognition {
    static async available(value) { options = value; return 'available'; }
    constructor() { instance = this; }
    start() {}
  }
  Recognition.prototype.processLocally = false;
  await startDictation('en', value => { text = value; }, value => { state = value; }, {SpeechRecognition: Recognition});
  assert.deepEqual(options, {langs: ['en-US'], processLocally: true});
  assert.equal(instance.lang, 'en-US'); assert.equal(instance.processLocally, true); assert.equal(state, 'voiceListening');
  instance.onresult({results: [[{transcript: 'Synthetic text only'}]]}); assert.equal(text, 'Synthetic text only');
  instance.onerror({error: 'not-allowed'}); assert.equal(state, 'voiceDenied');
});
// Fake recognizers only: these tests prove the adapter's decisions, never a real transcription.
function fakeRecognition(availability, calls = {}) {
  class Recognition {
    static async available(value) { calls.available = value; return availability; }
    static async install(value) { calls.install = value; return calls.installResult ?? true; }
    constructor() { calls.instance = this; }
    start() { calls.starts = (calls.starts || 0) + 1; }
  }
  Recognition.prototype.processLocally = false;
  return Recognition;
}
test('a missing on-device pack is reported as such, without download or audio', async () => {
  const calls = {}, states = [];
  const result = await startDictation('fr', () => assert.fail('audio'), s => states.push(s), {SpeechRecognition: fakeRecognition('downloadable', calls)});
  assert.equal(result, null); assert.equal(calls.starts, undefined); assert.equal(calls.install, undefined);
  assert.deepEqual(calls.available, {langs: ['fr-FR'], processLocally: true});
  assert.deepEqual(states, ['voicePackMissing']);
  for (const [availability, state] of [['downloading', 'voicePackDownloading'], ['unavailable', 'voiceUnsupported']]) {
    const seen = []; await startDictation('en', () => {}, s => seen.push(s), {SpeechRecognition: fakeRecognition(availability)});
    assert.deepEqual(seen, [state]);
  }
});
test('the prefixed Chrome constructor is accepted', async () => {
  const calls = {}, states = [];
  await startDictation('en', () => {}, s => states.push(s), {webkitSpeechRecognition: fakeRecognition('available', calls)});
  assert.equal(calls.starts, 1); assert.deepEqual(states, ['voiceListening']);
});
test('errors after listening started keep their real cause', async () => {
  for (const [error, state] of [['no-speech', 'voiceNoSpeech'], ['audio-capture', 'voiceNoMicrophone'], ['service-not-allowed', 'voiceDenied'], ['language-not-supported', 'voicePackMissing'], ['aborted', 'voiceReady'], ['network', 'voiceFailed'], ['bad-grammar', 'voiceFailed']]) {
    const calls = {}, states = [];
    await startDictation('fr', () => {}, (s, detail) => states.push([s, detail]), {SpeechRecognition: fakeRecognition('available', calls)});
    calls.instance.onerror({error}); calls.instance.onend();
    assert.deepEqual(states, [['voiceListening', undefined], [state, error]], error);
  }
});
test('the on-device pack is installed only on request, for the chosen language', async () => {
  const calls = {};
  assert.equal(await installLocalVoice('fr', {SpeechRecognition: fakeRecognition('downloadable', calls)}), 'voiceReady');
  assert.deepEqual(calls.install, {langs: ['fr-FR'], processLocally: true}); assert.equal(calls.starts, undefined);
  const refused = {installResult: false};
  assert.equal(await installLocalVoice('en', {SpeechRecognition: fakeRecognition('downloadable', refused)}), 'voicePackFailed');
  const Throws = fakeRecognition('downloadable'); Throws.install = async () => { throw new Error('no'); };
  assert.equal(await installLocalVoice('en', {SpeechRecognition: Throws}), 'voicePackFailed');
  assert.equal(await installLocalVoice('en', {}), 'voiceUnsupported');
});
test('a throwing recognizer reports its error name instead of a microphone denial', async () => {
  const Broken = fakeRecognition('available'); Broken.available = async () => { throw new TypeError('bad'); };
  const states = []; await startDictation('en', () => {}, (s, d) => states.push([s, d]), {SpeechRecognition: Broken});
  assert.deepEqual(states, [['voiceFailed', 'TypeError']]);
  const Denied = fakeRecognition('available'); Denied.prototype.start = () => { throw new DOMException('no', 'NotAllowedError'); };
  const denied = []; await startDictation('en', () => {}, s => denied.push(s), {SpeechRecognition: Denied});
  assert.deepEqual(denied, ['voiceDenied']);
});
test('simultaneous update checks share one negotiation and settle all waitUntil promises', async () => {
  const listeners={},prepared=[],waiting=[];let activated=0;
  const client={id:'fixture-client',url:'https://workshop.invalid/workshop/',postMessage:message=>prepared.push(message)};
  const self={addEventListener:(type,fn)=>{listeners[type]=fn;},clients:{matchAll:async()=>[client]},skipWaiting:async()=>{activated++;}};
  vm.runInNewContext(readFileSync(new URL('../web/sw.js',import.meta.url),'utf8'),{self,URL,crypto,setTimeout,clearTimeout,console});
  const event=data=>({data,source:client,waitUntil:promise=>waiting.push(promise)});
  listeners.message(event({type:'workshop-check-update'}));listeners.message(event({type:'workshop-check-update'}));
  await new Promise(resolve=>setImmediate(resolve));
  try {
    assert.equal(prepared.length,1);
    listeners.message(event({type:'workshop-update-readiness',id:prepared[0].id,ready:true}));
    await Promise.all(waiting);assert.equal(activated,1);
  } finally {
    for(const message of prepared)listeners.message(event({type:'workshop-update-readiness',id:message.id,ready:false}));
  }
});
test('failed success-response bodies and missing identity remain uncertain, not a different store', async () => {
  const previous=globalThis.fetch;
  const state=initialState();state.token='fixture-token';state.storeID='fixture-store';
  const replica={read:async()=>structuredClone(state)};
  try {
    for(const json of [async()=>{throw new DOMException('fixture aborted','AbortError')},async()=>({})]){
      globalThis.fetch=async()=>({ok:true,json});
      const engine=new SyncEngine(replica);await engine.run();assert.equal(engine.status,'notConnected');assert.equal(state.token,'fixture-token');
    }
  } finally {globalThis.fetch=previous;}
});
