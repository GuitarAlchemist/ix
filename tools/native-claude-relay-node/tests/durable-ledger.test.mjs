import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {randomUUID} from 'node:crypto';
import {RelayLedger,acquireLeader} from '../ledger.mjs';
const request=()=>({schema:'gaia-native-ack-request/1',id:randomUUID(),targetName:'IX',nonce:randomUUID(),issuedAt:100,expiresAt:100000,budgetUsd:0.05,authorizationRef:'explicit-test-only'});
function fixture(){const root=mkdtempSync(join(tmpdir(),'gaia-ledger-'));return{root,clean:()=>rmSync(root,{recursive:true,force:true})};}
test('SQLite leader excludes another connection and releases on close',()=>{const f=fixture();let release;try{release=acquireLeader(f.root);assert.throws(()=>acquireLeader(f.root),/ALREADY_RUNNING/);release();release=null;const next=acquireLeader(f.root);next();}finally{release?.();f.clean();}});
test('duplicate ID converges; changed content is rejected; one durable claim wins',()=>{const f=fixture();const a=new RelayLedger(f.root),b=new RelayLedger(f.root);try{const r=request();a.enqueue(r);b.enqueue({...r});assert.throws(()=>b.enqueue({...r,nonce:randomUUID()}),/REQUEST_ID_CONFLICT/);const claim=a.claim(200);assert.equal(claim.request.id,r.id);assert.equal(b.claim(200),null);assert.equal(a.get(r.id).state,'CLAIMED');}finally{a.close();b.close();f.clean();}});
test('crash after intent never replays and blocks more effects pending reconciliation',()=>{const f=fixture();const r=request();let a=new RelayLedger(f.root);try{a.enqueue(r);const claim=a.claim(200);a.close();a=new RelayLedger(f.root);a.recover();assert.equal(a.get(r.id).state,'UNKNOWN');a.enqueue(request());assert.equal(a.claim(300),null);assert.throws(()=>a.finish(r.id,'wrong-generation',{state:'DONE'}),/STALE_CLAIM/);a.finish(r.id,claim.claimId,{state:'DONE',proof:{verified:true}});assert.equal(a.get(r.id).state,'DONE');assert.ok(a.claim(400));}finally{a.close();f.clean();}});
test('expiry, malformed target and unapproved work cannot become an effect',()=>{const f=fixture(),a=new RelayLedger(f.root);try{const r=request();a.enqueue(r);assert.equal(a.claim(100001),null);assert.equal(a.get(r.id).state,'EXPIRED');assert.throws(()=>a.enqueue({...request(),targetName:'Jay'}),/REQUEST_INVALID/);assert.throws(()=>a.enqueue({...request(),text:'perform arbitrary work'}),/REQUEST_INVALID/);}finally{a.close();f.clean();}});
