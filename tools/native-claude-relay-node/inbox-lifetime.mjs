/** Local timers and correlated reply retention. No process, model or transport calls. */
import {nativeAckProof} from './native-ack-proof.mjs';
export function createInboxLifetime({now=Date.now,setTimer=setTimeout,clearTimer=clearTimeout,requestMs=90000,idleMs=300000,onClose=()=>{},onRequestExpired=()=>{}}={}){
 let closed=false,idleTimer,requestTimer,idleExpiresAt=null,current=null,last=null;
 function close(reason='OPERATOR_STOP'){if(closed)return;closed=true;clearTimer(idleTimer);clearTimer(requestTimer);onClose(reason);}
 function activity(){if(closed)return;clearTimer(idleTimer);idleExpiresAt=now()+idleMs;idleTimer=setTimer(()=>close('IDLE_TTL_EXPIRED'),idleMs);}
 function beginRequest(scope){
  if(closed)throw Error('INBOX_CLOSED');if(current)throw Error('REQUEST_STILL_ACTIVE');
  current={...scope,state:'PENDING',startedAt:now(),deadlineAt:now()+requestMs,reply:null};activity();
  requestTimer=setTimer(()=>{if(!current||current.reply||closed)return;current.state='UNKNOWN';current.expiredAt=now();onRequestExpired(current.id);},requestMs);
 }
 function receiveReply(entry){
  if(closed||!current||current.reply)return null;
  const proof=nativeAckProof(entry,{target:current.target,expected:current.expected,inputs:current.inputs??[]});
  if(!proof||current.peerAddress&&proof.senderPeerAddress!==current.peerAddress)return null;
  clearTimer(requestTimer);current.reply=proof;current.receivedAt=now();
  current.late=!!current.expiredAt||now()>=current.deadlineAt;
  current.state=current.late?'LATE_ACK_RECORDED':'ACK_RECORDED';activity();return proof;
 }
 function completeRequest(id){
  if(!current||current.id!==id||!current.reply)throw Error('CORRELATED_REPLY_REQUIRED');
  current.stdoutAt=now();activity();if(current.late){current.state='LATE_ACK_STDOUT_RECORDED';return false;}
  current.state='DONE';last=current;current=null;return true;
 }
 function snapshot(){return structuredClone({closed,idleExpiresAt,request:current??last});}
 function dispose(){closed=true;clearTimer(idleTimer);clearTimer(requestTimer);}
 return Object.freeze({activity,beginRequest,receiveReply,completeRequest,snapshot,close,dispose});
}
