/** Exactly one authorized native ACK canary. Reuses the verified CLI/collector path. */
import {spawn} from 'node:child_process';
import {writeFileSync,readFileSync,existsSync,statSync,unlinkSync} from 'node:fs';
import {homedir} from 'node:os';
import {fileURLToPath} from 'node:url';
import {StringDecoder} from 'node:string_decoder';
import {createPrintEventCollector,redactPrintText} from './print-event-collector.mjs';
import {extractNativeIxRows} from './native-ix-rows.mjs';
import {nativeAckProof} from './native-ack-proof.mjs';
import {createInboxLifetime} from './inbox-lifetime.mjs';
const codeRoot=fileURLToPath(new URL('.',import.meta.url));
const durable=true;if(process.argv[2]!=='--approved-durable-request'||process.argv.length!==4)throw Error('DURABLE_MODE_ONLY');
const configuration=durable?JSON.parse(readFileSync(process.argv[3],'utf8')):null;
const uuid=v=>typeof v==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(v);
if(durable&&(!uuid(configuration?.requestId)||!uuid(configuration?.sessionId)||!uuid(configuration?.nonce)||!['IX','Indigo IX'].includes(configuration?.targetName)||!(configuration?.budgetUsd>0&&configuration.budgetUsd<=0.05)||!configuration?.artifactRoot?.endsWith('/')||!Number.isSafeInteger(configuration?.expiresAt)||Date.now()>=configuration.expiresAt))throw Error('DURABLE_SCOPE_INVALID_OR_EXPIRED');
const root=durable?configuration.artifactRoot:codeRoot;
const sessionId=configuration.sessionId,name='GAIA_RELAY_'+sessionId.slice(0,8);
const nonce=configuration.nonce,expected='ACK '+nonce;
const receipt=root+'attempt.json',debug=root+'native-debug.log';
const args=['--safe-mode','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
 '--tools','ListAgents,SendMessage','--allowedTools','ListAgents,SendMessage','--model','sonnet','--effort','low',
 '--name',name,'--session-id',sessionId,'--print','--input-format','stream-json','--output-format','stream-json','--verbose',
 '--max-turns','8','--max-budget-usd',String(durable?configuration.budgetUsd:0.05),'--setting-sources','','--permission-mode','default','--debug-file',debug];


if(existsSync(receipt)||existsSync(debug))throw Error('CONSUMED_ATTEMPT_NO_RETRY');
const evidence={schema:'gaia-native-ack-e2e/1',sessionId,name,nonce,expected,state:'RESERVED',startedAt:new Date().toISOString(),args,limitMs:90000,idleTtlMs:300000,captureRequired:durable,
 listCalls:[],nativeTargetReceipts:[],sendAdmissions:0,sendCalls:[],sendReceipts:[],nativeAck:null,results:[]};
writeFileSync(receipt,JSON.stringify(evidence,null,2)+'\n',{flag:'wx'});
const collector=createPrintEventCollector({sessionId,maxBytes:1048576}),decoder=new StringDecoder('utf8');
const cwd=codeRoot+'canary-hello-02';
const ownTranscript=homedir()+'/.claude/projects/'+cwd.replaceAll('/','\\').replace(/[^a-zA-Z0-9]/g,'-')+'/'+sessionId+'.jsonl';
let buffer='',stopping=false,finishTimer,target=null;
const inputs=[];
const save=()=>{evidence.observations=collector.snapshot();writeFileSync(receipt,JSON.stringify(evidence,null,2)+'\n');};
const p=spawn(homedir()+'/.local/bin/claude.exe',args,{cwd,stdio:['pipe','pipe','pipe'],shell:false,windowsHide:true});
const input=text=>{inputs.push(text);p.stdin.write(JSON.stringify({type:'user',message:{role:'user',content:text}})+'\n');};
function stop(state){if(stopping)return;stopping=true;evidence.state=state;collector.markStdinEnd('abnormal');p.stdin.end();collector.markStop('abnormal','SIGTERM');save();p.kill('SIGTERM');}
const lifetime=createInboxLifetime({onClose:reason=>stop(reason),onRequestExpired:()=>{evidence.requestExpired=true;evidence.state='REQUEST_TIMEOUT_CHANNEL_ALIVE';save();}});
const nativeWatch=setInterval(()=>{try{checkOwnTranscript();}catch(error){evidence.readError=error.code??'OWN_PROOF_READ_ERROR';save();}},250);
const logLimit=setInterval(()=>{if(existsSync(debug)&&statSync(debug).size>524288)stop('DEBUG_LIMIT_NO_RETRY');},1000);
if(process.send){process.on('message',message=>{if(message?.type==='STOP')stop('OPERATOR_STOP_NO_RETRY');if(message?.type==='NATIVE_RETURN_COMMITTED'&&message.receipt?.requestId===configuration.requestId&&message.receipt?.replyMessageId===evidence.nativeAck?.messageId){evidence.nativeReturnReceipt=message.receipt;save();}});process.on('disconnect',()=>stop('SUPERVISOR_LOST_NO_RETRY'));}
function publishNativeReturn(){
 if(!durable||!process.connected||evidence.returnPacketSent||!evidence.nativeAck||evidence.sendReceipts.length!==1||evidence.sendReceipts[0].isError)return;
 evidence.returnPacketSent=true;
 process.send({type:'VERIFIED_NATIVE_RETURN',packet:structuredClone({schema:'gaia-owned-native-return/1',requestId:configuration.requestId,sessionId,nonce,admission:evidence.admission,sendAdmissions:evidence.sendAdmissions,sendCalls:evidence.sendCalls,sendReceipts:evidence.sendReceipts,frame:evidence.nativeFrame})});save();
}
function checkAck(entry,location){
 if(!target||!evidence.sendAdmissions||evidence.nativeAck)return;
 const proof=lifetime.receiveReply(entry);
 if(proof&&proof.sessionId===sessionId){evidence.nativeAck={...proof,location,observedAt:new Date().toISOString()};evidence.nativeFrame={type:'user',sessionId,uuid:proof.uuid,timestamp:proof.timestamp,origin:{kind:entry.origin.kind,name:entry.origin.name,from:entry.origin.from,body:entry.origin.body,msg_id:entry.origin.msg_id},message:{content:''}};if(!evidence.correlatedAckResult)evidence.state='NATIVE_ACK_RECEIVED';save();publishNativeReturn();}
}
function checkOwnTranscript(){
 if(!existsSync(ownTranscript))return;
 if(statSync(ownTranscript).size>2097152)return stop('OWN_TRANSCRIPT_LIMIT_NO_RETRY');
 for(const line of readFileSync(ownTranscript,'utf8').split('\n')){if(!line.trim())continue;try{checkAck(JSON.parse(line),'own-session-transcript');}catch{}}
}
function admit(){
 if(evidence.sendAdmissions||stopping)return;
 const rows=evidence.nativeTargetReceipts.flatMap(r=>r.rows);
 if(evidence.listCalls.length!==1||rows.length!==1||!rows[0].address||rows[0].cannotReceive||rows[0].kind!=='interactive'||!['IX','Indigo IX'].includes(rows[0].name)||durable&&rows[0].name!==configuration.targetName)return stop('TARGET_NOT_UNIQUE_OR_UNAVAILABLE_NO_SEND');
 if(durable&&Date.now()>=configuration.expiresAt)return stop('REQUEST_EXPIRED_NO_SEND');
 const ownLog=existsSync(debug)?readFileSync(debug,'utf8'):'';
 if(!/\[uds-messaging\] Listening:/.test(ownLog)||/\[uds-messaging\] Skipped:/.test(ownLog))return stop('SOURCE_INBOX_NOT_ATTESTED_NO_SEND');
 target=rows[0].name+' ['+rows[0].address+']';
 const message='Hello - relay validation. Reponds uniquement a ACK '+nonce+' par SendMessage a cet emetteur. Aucun autre travail ni modification.';
 const admission={schema:'gaia-native-ack-admission/1',sessionId,nonce,targetAddress:target,targetUnique:true,sourceInboxListening:true,nativeDiscoveryToolId:evidence.listCalls[0],observedAt:evidence.nativeTargetReceipts[0].observedAt,admittedAt:new Date().toISOString(),message};
 if(Date.now()-Date.parse(admission.observedAt)>30000)return stop('TARGET_DISCOVERY_STALE_NO_SEND');
 writeFileSync(root+(durable?'admission.json':'native-ix-ack-e2e-01-admission.json'),JSON.stringify(admission,null,2)+'\n',{flag:'wx'});
 evidence.sendAdmissions=1;evidence.admission=admission;evidence.state='ADMITTED_ONE_SEND';save();
 lifetime.beginRequest({id:durable?configuration.requestId:sessionId,target,expected,inputs});
 input('Send exactly ONE native SendMessage to '+JSON.stringify(target)+'. Exact message: '+JSON.stringify(message)+'. Do not list again, contact any other session or send confirmations. After this call, wait for the genuine native reply from that exact session. Once its actual '+JSON.stringify(expected)+' arrives, return only '+JSON.stringify(expected)+' on stdout. Do not claim receipt from the send receipt. Until it arrives, output WAITING if your turn ends. Ignore all unrelated messages. No file/code/shell work.');
}
function observe(event){
 if(event.type==='assistant')for(const block of event.message?.content??[])if(block.type==='tool_use'){
  if(block.name==='ListAgents'){evidence.listCalls.push(block.id);if(evidence.listCalls.length>1)stop('UNEXPECTED_SECOND_DISCOVERY');}
  if(block.name==='SendMessage'){
   evidence.sendCalls.push({id:block.id,input:JSON.parse(redactPrintText(JSON.stringify(block.input)))});
   if(evidence.sendAdmissions!==1||evidence.sendCalls.length!==1||block.input?.to!==target)stop('UNEXPECTED_SEND_OUTCOME_UNCERTAIN_NO_RETRY');
  }
 }
 if(event.type==='user'){
  checkAck(event,'stream-json');
  for(const block of Array.isArray(event.message?.content)?event.message.content:[]){
   if(block.type!=='tool_result')continue;
   if(evidence.listCalls.includes(block.tool_use_id)){
    const retained=extractNativeIxRows(block.content);
    evidence.nativeTargetReceipts.push({toolUseId:block.tool_use_id,isError:block.is_error===true,observedAt:new Date().toISOString(),rows:retained.rows});
    if(block.is_error)stop('DISCOVERY_REFUSED_NO_SEND');
   }
   if(evidence.sendCalls.some(c=>c.id===block.tool_use_id)){
    const text=typeof block.content==='string'?block.content:JSON.stringify(block.content);
    evidence.sendReceipts.push({toolUseId:block.tool_use_id,isError:block.is_error===true,text:redactPrintText(text)});
    if(block.is_error||/\bNot sent\b|no inbox|held for|refused/i.test(text))stop('SEND_REFUSED_OR_HELD_NO_RETRY');else publishNativeReturn();
   }
  }
 }
 if(event.type==='result'){
  const result={sessionId:event.session_id,text:redactPrintText(event.result),isError:event.is_error===true,costUsdEstimate:event.total_cost_usd,turns:event.num_turns};
  evidence.results.push(result);
  if(result.isError)return stop('MODEL_TERMINAL_ERROR_NO_RETRY');
  if(!evidence.sendAdmissions){process.stdout.write(JSON.stringify({phase:'DISCOVERY_COMPLETE',sessionId,pid:p.pid})+'\n');return admit();}
  checkOwnTranscript();
  if(evidence.nativeAck&&result.sessionId===sessionId&&result.text.trim()===expected){
   evidence.correlatedAckResult=result;evidence.state='NATIVE_ACK_AND_CORRELATED_RESULT';save();
   if(!evidence.stdoutRecorded){evidence.stdoutRecorded=true;lifetime.completeRequest(durable?configuration.requestId:sessionId);save();}
  }else{evidence.state='WAITING_FOR_NATIVE_ACK';process.stdout.write(JSON.stringify({phase:evidence.state,sessionId,pid:p.pid,nativeAckObserved:!!evidence.nativeAck})+'\n');}
 }
}
p.on('spawn',()=>{evidence.pid=p.pid;evidence.state='PROCESS_STARTED';lifetime.activity();save();input('You are a temporary native messaging test relay. Call ListAgents exactly ONCE to discover the personal IX / Indigo IX session. Do NOT call SendMessage yet. Return DISCOVERY_READY after the single native list. Do not contact other sessions, use MCP, read/edit files or run commands. Ignore unrelated peer messages. A later stdin instruction may admit exactly one ACK request.');});
p.stdin.on('error',error=>{evidence.stdinError=error.code;save();});
p.stdout.on('data',chunk=>{try{collector.pushStdout(chunk);buffer+=decoder.write(chunk);for(let i;(i=buffer.indexOf('\n'))>=0;){const line=buffer.slice(0,i);buffer=buffer.slice(i+1);if(line.trim())observe(JSON.parse(line));}save();}catch{stop('OUTPUT_ERROR_OUTCOME_UNCERTAIN_NO_RETRY');}});
p.stdout.on('end',()=>{try{collector.endStdout();buffer+=decoder.end();if(buffer.trim())observe(JSON.parse(buffer));save();}catch{evidence.truncatedOutput=true;save();}});
p.stderr.on('data',chunk=>{collector.pushStderr(chunk);save();if(collector.snapshot().stderrDiagnostic.codes.some(c=>['EPERM','EACCES'].includes(c)))stop('ACCESS_REFUSED_NO_RETRY');});
p.on('error',error=>{evidence.spawnError=error.code;save();});
p.on('close',(code,signal)=>{lifetime.dispose();clearTimeout(finishTimer);clearInterval(nativeWatch);clearInterval(logLimit);checkOwnTranscript();collector.markClose(code,signal);evidence.closeObserved=true;evidence.closedAt=new Date().toISOString();evidence.exitCode=code;evidence.signal=signal;
 if(evidence.nativeAck&&!evidence.correlatedAckResult){const result=evidence.results.find(r=>r.sessionId===sessionId&&!r.isError&&r.text.trim()===expected);if(result){evidence.correlatedAckResult=result;evidence.state='COMPLETE_NATIVE_ACK_RECONCILED_AT_CLOSE';}}
 if(durable&&existsSync(debug)){evidence.nativeDiagnostics=readFileSync(debug,'utf8').split('\n').filter(line=>/\[uds-messaging\].*(Listening:|Skipped:)/.test(line)&&!/token|secret|authorization/i.test(line)).map(line=>line.replace(/Listening:.*/,'Listening: [OWN_PIPE_REDACTED]'));unlinkSync(debug);}
 save();process.stdout.write(JSON.stringify({phase:'CLOSED',sessionId,pid:p.pid,state:evidence.state,nativeAck:evidence.nativeAck,correlatedAckResult:evidence.correlatedAckResult??null,sendAdmissions:evidence.sendAdmissions,sendCalls:evidence.sendCalls.length,results:evidence.results,closedAt:evidence.closedAt})+'\n');process.exit(0);});
