// Hermetic IPC provider-shape fixture. No Claude, pipe API or native messages.
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {randomUUID} from 'node:crypto';
const c=JSON.parse(readFileSync(process.argv[3],'utf8')),target=c.targetName+' [123abc]',expected='ACK '+c.nonce,replyId=randomUUID(),now=new Date().toISOString();
const admission={sessionId:c.sessionId,nonce:c.nonce,targetAddress:target,targetUnique:true,sourceInboxListening:true};
const record={captureRequired:true,sessionId:c.sessionId,nonce:c.nonce,pid:process.pid,closeObserved:false,state:'FIXTURE_AWAITING_DURABLE_RECEIPT',sendAdmissions:1,admission,sendCalls:[{input:{to:target,message:'ACK-only fixture'}}],sendReceipts:[{isError:false,text:JSON.stringify([{type:'text',text:JSON.stringify({success:true,msg_id:randomUUID()})}])}],nativeAck:{senderName:c.targetName,text:expected,messageId:replyId,observedAt:now},results:[]};
const frame={type:'user',sessionId:c.sessionId,uuid:randomUUID(),timestamp:now,origin:{kind:'peer',name:c.targetName,from:'uds:fixture-only',body:expected,msg_id:replyId},message:{content:''}};
const save=()=>writeFileSync(join(c.artifactRoot,'attempt.json'),JSON.stringify(record));save();
process.on('message',m=>{if(m.type==='NATIVE_RETURN_COMMITTED'){record.nativeReturnReceipt=m.receipt;record.correlatedAckResult={sessionId:c.sessionId,text:expected};record.results=[{costUsdEstimate:0}];record.closeObserved=true;record.state='FIXTURE_COMPLETE';save();process.exit(0);}if(m.type==='STOP'){record.state='FIXTURE_STOPPED';record.closeObserved=true;save();process.exit(0);}});
process.send({type:'VERIFIED_NATIVE_RETURN',packet:{schema:'gaia-owned-native-return/1',requestId:c.requestId,sessionId:c.sessionId,nonce:c.nonce,admission,sendAdmissions:1,sendCalls:record.sendCalls,sendReceipts:record.sendReceipts,frame}});
