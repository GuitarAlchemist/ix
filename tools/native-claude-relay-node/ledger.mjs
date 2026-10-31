/** Local durable request ledger; native Claude inbox remains owned by Claude. */
import {DatabaseSync} from 'node:sqlite';
import {mkdirSync,lstatSync,existsSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {createHash,randomUUID} from 'node:crypto';
const uuid=v=>typeof v==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(v);
function path(root,name){root=resolve(root);mkdirSync(root,{recursive:true});const file=join(root,name);if(lstatSync(root).isSymbolicLink()||existsSync(file)&&lstatSync(file).isSymbolicLink())throw Error('STATE_PATH_REFUSED');return file;}
export function acquireLeader(root){
 const db=new DatabaseSync(path(root,'leader.sqlite'));db.exec('PRAGMA busy_timeout=0;CREATE TABLE IF NOT EXISTS leader(id INTEGER PRIMARY KEY);');
 try{db.exec('BEGIN IMMEDIATE');}catch{db.close();throw Error('ALREADY_RUNNING');}
 let closed=false;return()=>{if(!closed){closed=true;db.exec('ROLLBACK');db.close();}};
}
function validate(r){
 const fields='schema,id,targetName,nonce,issuedAt,expiresAt,budgetUsd,authorizationRef'.split(',').sort().join(',');
 if(!r||Object.keys(r).sort().join(',')!==fields||r.schema!=='gaia-native-ack-request/1'||!uuid(r.id)||!uuid(r.nonce)
  ||!['IX','Indigo IX'].includes(r.targetName)||!Number.isSafeInteger(r.issuedAt)||!Number.isSafeInteger(r.expiresAt)
  ||r.expiresAt<=r.issuedAt||r.expiresAt-r.issuedAt>3600000||!(r.budgetUsd>0&&r.budgetUsd<=0.05)
  ||typeof r.authorizationRef!=='string'||r.authorizationRef.length<1||r.authorizationRef.length>160)throw Error('REQUEST_INVALID');
 return Object.fromEntries(fields.split(',').map(key=>[key,r[key]]));
}
export class RelayLedger{
 constructor(root){this.db=new DatabaseSync(path(root,'requests.sqlite'));this.db.exec("PRAGMA journal_mode=WAL;PRAGMA synchronous=FULL;PRAGMA busy_timeout=5000;PRAGMA max_page_count=16384;CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,digest TEXT NOT NULL,body TEXT NOT NULL,state TEXT NOT NULL,claimId TEXT,sessionId TEXT,result TEXT,updatedAt INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS native_returns(requestId TEXT PRIMARY KEY,claimId TEXT NOT NULL,sessionId TEXT NOT NULL,body TEXT NOT NULL,receivedAtUtc TEXT NOT NULL);");}
 enqueue(input){const r=validate(input),body=JSON.stringify(r),digest=createHash('sha256').update(body).digest('hex');
  this.db.prepare("INSERT OR IGNORE INTO requests(id,digest,body,state,updatedAt) SELECT ?,?,?,'READY',? WHERE (SELECT COUNT(*) FROM requests)<256").run(r.id,digest,body,Date.now());
  const row=this.get(r.id);if(!row)throw Error('RETENTION_CAP_NO_NEW_EFFECT');if(row.digest!==digest)throw Error('REQUEST_ID_CONFLICT');return row;
 }
 get(id){return this.db.prepare('SELECT * FROM requests WHERE id=?').get(id)??null;}
 nativeReturn(id){const row=this.db.prepare('SELECT * FROM native_returns WHERE requestId=?').get(id);return row?{...row,proof:JSON.parse(row.body)}:null;}
 recordNativeReturn(id,claimId,proof){
  this.db.exec('BEGIN IMMEDIATE');try{
   const request=this.get(id),scope=request?JSON.parse(request.body):null;
   if(!request||request.claimId!==claimId||proof?.requestId!==id||proof.claimId!==claimId||proof.sessionId!==request.sessionId||proof.nonce!==scope.nonce||!proof.target?.startsWith(scope.targetName+' [')||proof.nativeProof?.text!=='ACK '+scope.nonce)throw Error('STALE_RETURN_SCOPE');
   const body=JSON.stringify(proof),prior=this.nativeReturn(id);let row=prior;
   if(prior){if(prior.body!==body)throw Error('RETURN_ID_CONFLICT');}
   else {if(!['CLAIMED','UNKNOWN'].includes(request.state))throw Error('RETURN_STATE_REFUSED');this.db.prepare('INSERT INTO native_returns(requestId,claimId,sessionId,body,receivedAtUtc) VALUES(?,?,?,?,?)').run(id,claimId,request.sessionId,body,new Date().toISOString());row=this.nativeReturn(id);}
   this.db.exec('COMMIT');return {type:'DURABLE_NATIVE_RETURN_RECEIPT',duplicate:!!prior,requestId:id,replyMessageId:proof.replyMessageId,nativeAckAtUtc:proof.ackAtUtc,receivedAtUtc:row.receivedAtUtc};
  }catch(e){this.db.exec('ROLLBACK');throw e;}
 }
 claim(now=Date.now()){
  this.db.exec('BEGIN IMMEDIATE');try{
   if(this.db.prepare("SELECT id FROM requests WHERE state IN ('CLAIMED','UNKNOWN') LIMIT 1").get()){this.db.exec('COMMIT');return null;}
   for(const row of this.db.prepare("SELECT * FROM requests WHERE state='READY' ORDER BY updatedAt,id").all()){
    const request=JSON.parse(row.body);if(now<request.issuedAt)continue;
    if(now>=request.expiresAt){this.db.prepare("UPDATE requests SET state='EXPIRED',updatedAt=? WHERE id=? AND state='READY'").run(now,row.id);continue;}
    const claimId=randomUUID(),sessionId=randomUUID();
    const changed=this.db.prepare("UPDATE requests SET state='CLAIMED',claimId=?,sessionId=?,updatedAt=? WHERE id=? AND state='READY'").run(claimId,sessionId,now,row.id);
    if(changed.changes!==1)throw Error('CLAIM_CONFLICT');this.db.exec('COMMIT');return {request,claimId,sessionId};
   }this.db.exec('COMMIT');return null;
  }catch(error){this.db.exec('ROLLBACK');throw error;}
 }
 recover(){return this.db.prepare("UPDATE requests SET state='UNKNOWN',result=?,updatedAt=? WHERE state='CLAIMED'").run(JSON.stringify({reason:'SUPERVISOR_RESTART_NO_RETRY'}),Date.now()).changes;}
 finish(id,claimId,result){
  if(!['DONE','UNKNOWN','REJECTED'].includes(result?.state))throw Error('RESULT_INVALID');
  const update=this.db.prepare("UPDATE requests SET state=?,result=?,updatedAt=? WHERE id=? AND claimId=? AND state IN ('CLAIMED','UNKNOWN')").run(result.state,JSON.stringify(result),Date.now(),id,claimId);
  if(update.changes!==1)throw Error('STALE_CLAIM');return this.get(id);
 }
 summary(){return this.db.prepare('SELECT state,COUNT(*) AS count FROM requests GROUP BY state').all();}
 close(){this.db.close();}
}
