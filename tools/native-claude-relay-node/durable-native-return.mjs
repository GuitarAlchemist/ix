/** Validate the owned worker's scoped native peer frame before committing it. */
import {existsSync} from 'node:fs';
const local=new URL('./native-ack-proof.mjs',import.meta.url);
const {nativeAckProof}=await import(existsSync(local)?local:new URL('../native-ack-proof.mjs',import.meta.url));
export function verifyDurableNativeReturn(claim,packet){
 const expected='ACK '+claim.request.nonce,target=packet?.admission?.targetAddress;
 if(packet?.schema!=='gaia-owned-native-return/1'||packet.requestId!==claim.request.id||packet.sessionId!==claim.sessionId||packet.nonce!==claim.request.nonce||packet.sendAdmissions!==1||packet.sendCalls?.length!==1||packet.sendReceipts?.length!==1||packet.sendReceipts[0].isError||packet.admission?.sessionId!==claim.sessionId||packet.admission?.nonce!==claim.request.nonce||!packet.admission?.targetUnique||!packet.admission?.sourceInboxListening||typeof target!=='string'||!target.startsWith(claim.request.targetName+' [')||packet.sendCalls[0].input?.to!==target||typeof packet.sendCalls[0].input?.message!=='string')throw Error('NATIVE_RETURN_SCOPE_REFUSED');
 const proof=nativeAckProof(packet.frame,{target,expected});
 if(!proof||proof.sessionId!==claim.sessionId||!proof.messageId||!proof.timestamp||!proof.uuid)throw Error('NATIVE_RETURN_FRAME_REFUSED');
 let accepted;try{accepted=JSON.parse(JSON.parse(packet.sendReceipts[0].text)[0].text);}catch{throw Error('NATIVE_ADMISSION_REQUIRED');}
 if(accepted.success!==true||!accepted.msg_id)throw Error('NATIVE_ADMISSION_REQUIRED');
 return {schema:'gaia-durable-native-return/1',requestId:claim.request.id,claimId:claim.claimId,sessionId:claim.sessionId,target,nonce:claim.request.nonce,replyMessageId:proof.messageId,nativeAdmissionMessageId:accepted.msg_id,ackAtUtc:proof.timestamp,nativeProof:proof};
}
