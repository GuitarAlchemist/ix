// Hermetic worker contract fixture. Never imports or launches Claude.
import {readFileSync,writeFileSync} from 'node:fs';
import {join,dirname} from 'node:path';
const c=JSON.parse(readFileSync(process.argv[3],'utf8'));
const root=dirname(dirname(dirname(dirname(process.argv[3])))),mode=readFileSync(join(root,'fixture-mode.txt'),'utf8').trim();
if(mode==='crash')process.exit(23);
const record={sessionId:c.sessionId,nonce:c.nonce,pid:process.pid,state:'FIXTURE_WAITING',closeObserved:false,sendAdmissions:0,sendCalls:[]};
const save=()=>writeFileSync(join(c.artifactRoot,'attempt.json'),JSON.stringify(record));save();
if(mode==='success'){
 Object.assign(record,{state:'COMPLETE_FIXTURE',closeObserved:true,sendAdmissions:1,sendCalls:[{}],admission:{targetAddress:c.targetName+' [123abc]'},
  nativeAck:{senderName:c.targetName,text:'ACK '+c.nonce,messageId:'fixture-only',observedAt:new Date().toISOString()},
  correlatedAckResult:{sessionId:c.sessionId,text:'ACK '+c.nonce},results:[{costUsdEstimate:0}]});save();process.exit(0);
}
if(mode==='reject'){Object.assign(record,{state:'TARGET_NOT_UNIQUE_OR_UNAVAILABLE_NO_SEND',closeObserved:true});save();process.exit(0);}
if(mode!=='stop')throw Error('FIXTURE_MODE_INVALID');
record.sendAdmissions=1;record.sendCalls=[{}];save();
const timeout=setTimeout(()=>process.exit(24),5000);
process.on('message',message=>{if(message?.type==='STOP'){clearTimeout(timeout);record.state='OPERATOR_STOP_NO_RETRY';record.closeObserved=true;save();process.exit(0);}});
