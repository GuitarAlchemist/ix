/** On-demand local supervisor. No network server, credentials or idle model calls. */
import {fork} from 'node:child_process';
import {existsSync,mkdirSync,readFileSync,writeFileSync,renameSync,appendFileSync,statSync,unlinkSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomUUID} from 'node:crypto';
import {RelayLedger,acquireLeader} from './ledger.mjs';
import {verifyDurableNativeReturn} from './durable-native-return.mjs';
export const VERSION='0.1.1-capture1';
const codeRoot=fileURLToPath(new URL('.',import.meta.url));
const readJson=file=>{try{return JSON.parse(readFileSync(file,'utf8'));}catch{return null;}};
function writeJson(file,value){const temporary=file+'.'+randomUUID()+'.tmp';writeFileSync(temporary,JSON.stringify(value,null,2)+'\n');renameSync(temporary,file);}
function safeError(error){const code=String(error?.code??error?.message??'UNKNOWN');return /^[A-Z0-9_]{1,80}$/.test(code)?code:'REDACTED_RUNTIME_ERROR';}
function log(root,event){const file=join(root,'agent.log');if(existsSync(file)&&statSync(file).size>262144){const old=file+'.1';if(existsSync(old))unlinkSync(old);renameSync(file,old);}appendFileSync(file,JSON.stringify({at:new Date().toISOString(),...event})+'\n');}
const sleep=milliseconds=>new Promise(resolve=>setTimeout(resolve,milliseconds));
export function readNativeOutcome(claim,root){
 const artifactRoot=join(root,'jobs',claim.request.id,claim.claimId),own=readJson(join(artifactRoot,'attempt.json'));
 let capture=null;if(own?.captureRequired){const ledger=new RelayLedger(root);try{capture=ledger.nativeReturn(claim.request.id);}finally{ledger.close();}}
 const verified=own?.sessionId===claim.sessionId&&own.nonce===claim.request.nonce&&own.closeObserved===true&&!own.requestExpired
  &&own.sendAdmissions===1&&own.sendCalls?.length===1&&own.nativeAck?.senderName===claim.request.targetName
  &&own.nativeAck?.text==='ACK '+claim.request.nonce&&own.correlatedAckResult?.sessionId===claim.sessionId
  &&own.correlatedAckResult?.text?.trim()==='ACK '+claim.request.nonce&&own.correlatedAckResult?.isError!==true
  &&typeof own.admission?.targetAddress==='string'&&own.admission.targetAddress.startsWith(claim.request.targetName+' [')&&Array.isArray(own.results)
  &&(!own.captureRequired||capture?.claimId===claim.claimId&&capture?.sessionId===claim.sessionId&&capture?.proof.replyMessageId===own.nativeAck.messageId&&own.nativeReturnReceipt?.replyMessageId===own.nativeAck.messageId);
 if(verified)return {state:'DONE',proof:{verified:true,at:own.nativeAck.observedAt,sessionId:claim.sessionId,
  target:own.admission.targetAddress,nonce:claim.request.nonce,replyMessageId:own.nativeAck.messageId,lastCliTotalCostEstimateUsd:own.results.at(-1)?.costUsdEstimate,ixUsageMeasured:false,artifactRoot}};
 const noSend=own?.sessionId===claim.sessionId&&own.nonce===claim.request.nonce&&own.closeObserved===true&&own.sendAdmissions===0&&own.sendCalls?.length===0;
 return {state:noSend?'REJECTED':'UNKNOWN',reason:own?.state??'WORKER_CLOSED_WITHOUT_RECEIPT',sessionId:claim.sessionId,artifactRoot};
}
export async function launchNativeWorker(claim,{root,shouldStop,progress,commitNativeReturn,workerScript=join(codeRoot,'native-worker.mjs')}={}){
 const artifactRoot=join(root,'jobs',claim.request.id,claim.claimId);mkdirSync(artifactRoot,{recursive:true});
 const config=join(artifactRoot,'request.json');
 writeFileSync(config,JSON.stringify({requestId:claim.request.id,sessionId:claim.sessionId,nonce:claim.request.nonce,targetName:claim.request.targetName,
  budgetUsd:claim.request.budgetUsd,expiresAt:claim.request.expiresAt,artifactRoot:artifactRoot.replaceAll('\\','/')+'/'})+'\n',{flag:'wx'});
 return new Promise(resolveResult=>{
  const worker=fork(workerScript,['--approved-durable-request',config],{execPath:process.execPath,stdio:['ignore','pipe','pipe','ipc'],windowsHide:true,cwd:codeRoot});
  let outputBytes=0,stopSent=false;
  const watch=setInterval(()=>{
   const own=readJson(join(artifactRoot,'attempt.json'));
   if(own?.sessionId===claim.sessionId)progress({workerPid:worker.pid,claudePid:own.pid??null,phase:own.state,
    claudeVersion:own.observations?.events?.find(e=>e.type==='system'&&e.subtype==='init')?.init?.version??null});
   if(shouldStop()&&!stopSent&&worker.connected){stopSent=true;worker.send({type:'STOP'});}
  },500);
  const drain=chunk=>{outputBytes+=chunk.length;if(outputBytes>1048576&&!stopSent&&worker.connected){stopSent=true;worker.send({type:'STOP'});}};
  worker.stdout.on('data',drain);worker.stderr.on('data',drain);
  worker.on('message',message=>{if(message?.type!=='VERIFIED_NATIVE_RETURN')return;try{const proof=verifyDurableNativeReturn(claim,message.packet);if(typeof commitNativeReturn!=='function')throw Error('DURABLE_RECEIVER_REQUIRED');const receipt=commitNativeReturn(proof);if(worker.connected)worker.send({type:'NATIVE_RETURN_COMMITTED',receipt});}catch(error){progress({lastError:safeError(error)});if(worker.connected)worker.send({type:'STOP'});}});
  worker.on('error',()=>{});
  worker.on('close',()=>{
   clearInterval(watch);resolveResult(readNativeOutcome(claim,root));
  });
 });
}
export async function runAgent({root,wait=sleep,launch=launchNativeWorker}={}){
 root=resolve(root);mkdirSync(root,{recursive:true});const disabled=join(root,'disabled.flag');
 if(existsSync(disabled))return {state:'DISABLED'};
 const release=acquireLeader(root),ledger=new RelayLedger(root),previous=readJson(join(root,'status.json'));
 let stopping=false,failures=0;
 const status={schema:'gaia-native-agent-status/1',version:VERSION,pid:process.pid,nodeVersion:process.version,startedAt:new Date().toISOString(),
  updatedAt:null,finishedAt:null,state:'STARTING',activeRequest:null,workerPid:null,claudePid:null,claudeVersion:null,
  lastRoundTrip:previous?.lastRoundTrip??null,lastNativeReturn:previous?.lastNativeReturn??null,lastError:null,idleModelCalls:0,transport:'existing Codex-to-PC hop; local durable ACK requests; native Claude messaging'};
 const save=()=>{status.updatedAt=new Date().toISOString();status.queue=ledger.summary();writeJson(join(root,'status.json'),status);};
 const shouldStop=()=>stopping||existsSync(disabled);
 const pause=async milliseconds=>{if(wait!==sleep)return wait(milliseconds);const until=Date.now()+milliseconds;while(!shouldStop()&&Date.now()<until)await sleep(Math.min(250,until-Date.now()));};
 const signal=()=>{stopping=true;writeFileSync(disabled,'operator stop\n');};
 process.on('SIGINT',signal);process.on('SIGTERM',signal);
 try{
  const uncertain=ledger.recover();status.state=uncertain?'NEEDS_RECONCILIATION':'IDLE';save();log(root,{event:'started',pid:process.pid,recoveredUnknown:uncertain});
  while(!shouldStop()){
   const claim=ledger.claim();
   if(!claim){status.state=ledger.summary().some(row=>row.state==='UNKNOWN')?'NEEDS_RECONCILIATION':'IDLE';save();await pause(1000);continue;}
   status.state='RUNNING_REQUEST';status.activeRequest=claim.request.id;status.lastError=null;save();
   let result;
   try{result=await launch(claim,{root,shouldStop,progress:value=>{Object.assign(status,value);save();},commitNativeReturn:proof=>{const receipt=ledger.recordNativeReturn(claim.request.id,claim.claimId,proof);status.lastNativeReturn=receipt;save();log(root,{event:'native-return-stored',requestId:claim.request.id,replyMessageId:proof.replyMessageId,receivedAtUtc:receipt.receivedAtUtc});return receipt;}});}
   catch(error){result={state:'UNKNOWN',reason:safeError(error)};}
   ledger.finish(claim.request.id,claim.claimId,result);
   if(result.state==='DONE'){status.lastRoundTrip=result.proof;failures=0;}else{failures++;status.lastError=result.reason??result.state;}
   status.activeRequest=null;status.workerPid=null;status.claudePid=null;status.state=result.state==='UNKNOWN'?'NEEDS_RECONCILIATION':'IDLE';save();
   log(root,{event:'request-finished',requestId:claim.request.id,state:result.state,error:status.lastError});
   if(failures)await pause(Math.min(60000,5000*2**Math.min(failures-1,4)));
  }
  status.state='STOPPED';status.finishedAt=new Date().toISOString();save();log(root,{event:'stopped'});return status;
 }catch(error){status.state='FAULTED';status.lastError=safeError(error);status.finishedAt=new Date().toISOString();try{save();log(root,{event:'faulted',error:status.lastError});}catch{}throw error;}
 finally{process.off('SIGINT',signal);process.off('SIGTERM',signal);ledger.close();release();}
}
if(resolve(process.argv[1]??'')===fileURLToPath(import.meta.url)){
 const command=process.argv[2],rootIndex=process.argv.indexOf('--root'),root=rootIndex>=0?process.argv[rootIndex+1]:null;
 if(!root)throw Error('EXPLICIT_STATE_ROOT_REQUIRED');
 if(command==='run')await runAgent({root});
 else if(command==='status'){
  const status=readJson(join(root,'status.json'))??{state:'NOT_STARTED'},age=Date.now()-Date.parse(status.updatedAt);
  console.log(JSON.stringify({...status,observedAt:new Date().toISOString(),heartbeatAgeMs:Number.isFinite(age)?age:null,heartbeatFresh:Number.isFinite(age)&&age>=0&&age<10000}));
 }
 else if(command==='enqueue'){
  const i=process.argv.indexOf('--request');if(i<0)throw Error('REQUEST_FILE_REQUIRED');
  const ledger=new RelayLedger(root);try{console.log(JSON.stringify(ledger.enqueue(readJson(process.argv[i+1]))));}finally{ledger.close();}
 }else if(command==='reconcile'){
  const i=process.argv.indexOf('--request-id');if(i<0)throw Error('REQUEST_ID_REQUIRED');
  const ledger=new RelayLedger(root);try{const row=ledger.get(process.argv[i+1]);if(!row||row.state!=='UNKNOWN')throw Error('REQUEST_NOT_UNKNOWN');
   const result=readNativeOutcome({request:JSON.parse(row.body),claimId:row.claimId,sessionId:row.sessionId},root);
   if(result.state!=='UNKNOWN')ledger.finish(row.id,row.claimId,result);console.log(JSON.stringify(result));
  }finally{ledger.close();}
 }else if(command==='stop'){if(!existsSync(root))throw Error('NOT_INSTALLED');writeFileSync(join(root,'disabled.flag'),'operator stop\n');console.log(JSON.stringify({state:'STOP_REQUESTED'}));}
 else throw Error('COMMAND_INVALID_USE_MANAGE_PS1_FOR_START_OR_UNINSTALL');
}
