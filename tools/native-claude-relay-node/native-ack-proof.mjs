/** Check a received user frame, never an assistant's claim or an echoed stdin request. */
export function nativeAckProof(entry,{target,expected,inputs=[]}={}){
 if(entry?.type!=='user'||entry.isReplay||entry.is_replay)return null;
 const content=entry.message?.content;
 const text=typeof content==='string'?content:Array.isArray(content)?content.filter(b=>b.type==='text').map(b=>b.text).join('\n'):'';
 if(inputs.includes(text))return null;
 // Observed native 2.1.312 transcript: trusted origin metadata identifies the
 // peer; the visible XML uses its pipe address and has explanatory text around it.
 const origin=entry.origin,name=typeof target==='string'?target.replace(/ \[[0-9a-f]{6,12}\]$/i,''):null;
 if(origin?.kind!=='peer'||origin.name!==name||typeof origin.from!=='string'||!origin.from.startsWith('uds:')||origin.body!==expected)return null;
 return {sender:target,senderName:origin.name,senderPeerAddress:origin.from,messageId:origin.msg_id??null,
  text:expected,uuid:entry.uuid??null,sessionId:entry.sessionId??entry.session_id??null,timestamp:entry.timestamp??null};
}
