import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fork} from 'node:child_process';
import {once} from 'node:events';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {acquireLeader} from '../ledger.mjs';
test('OS-backed SQLite exclusion crosses processes and releases after owned process crash',{timeout:10000},async()=>{
 const root=mkdtempSync(join(tmpdir(),'gaia-leader-'));const child=fork(fileURLToPath(new URL('./fixtures/leader-holder.mjs',import.meta.url)),[root],{stdio:['ignore','ignore','pipe','ipc'],windowsHide:true});
 try{const [message]=await Promise.race([once(child,'message'),once(child,'close').then(()=>{throw Error('LEADER_NOT_READY');})]);assert.equal(message.state,'LOCK_HELD');assert.throws(()=>acquireLeader(root),/ALREADY_RUNNING/);const closed=once(child,'close');child.kill('SIGTERM');await closed;const release=acquireLeader(root);release();}
 finally{if(child.exitCode===null&&child.signalCode===null){const closed=once(child,'close');child.kill('SIGTERM');await closed;}assert.ok(resolve(root).startsWith(resolve(tmpdir())+sep));rmSync(root,{recursive:true,force:true});}
});
