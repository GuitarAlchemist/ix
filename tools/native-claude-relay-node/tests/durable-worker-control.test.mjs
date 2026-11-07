import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,rmSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve,sep} from 'node:path';
import {randomUUID} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {launchNativeWorker} from '../agent.mjs';
const workerScript=fileURLToPath(new URL('./fixtures/fake-native-worker.mjs',import.meta.url));
async function fixture(mode,body){const root=mkdtempSync(join(tmpdir(),'gaia-worker-'));try{writeFileSync(join(root,'fixture-mode.txt'),mode);await body(root);}finally{assert.ok(resolve(root).startsWith(resolve(tmpdir())+sep));rmSync(root,{recursive:true,force:true});}}
const claim=()=>({claimId:randomUUID(),sessionId:randomUUID(),request:{id:randomUUID(),nonce:randomUUID(),targetName:'IX',budgetUsd:0.05,expiresAt:Date.now()+60000}});
test('production fork adapter consumes a correlated fixture result after worker close',async()=>fixture('success',async root=>{const r=await launchNativeWorker(claim(),{root,workerScript,shouldStop:()=>false,progress:()=>{}});assert.equal(r.state,'DONE');assert.equal(r.proof.verified,true);}));
test('explicit pre-send refusal stays REJECTED; lost receipt stays UNKNOWN',async()=>{for(const mode of ['reject','crash'])await fixture(mode,async root=>{const r=await launchNativeWorker(claim(),{root,workerScript,shouldStop:()=>false,progress:()=>{}});assert.equal(r.state,mode==='reject'?'REJECTED':'UNKNOWN');});});
test('stop sends IPC only to the owned worker and preserves admitted uncertainty',async()=>fixture('stop',async root=>{let ready=false;const c=claim();const result=await launchNativeWorker(c,{root,workerScript,progress:()=>{ready=true;},shouldStop:()=>ready});assert.equal(result.state,'UNKNOWN');const recorded=JSON.parse(readFileSync(join(root,'jobs',c.request.id,c.claimId,'attempt.json')));assert.equal(recorded.state,'OPERATOR_STOP_NO_RETRY');assert.equal(recorded.closeObserved,true);}));
