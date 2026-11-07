import {test} from 'node:test';
import assert from 'node:assert/strict';
import {nativeAckProof} from '../native-ack-proof.mjs';
const target='IX [123abc]',expected='ACK unique-nonce';
const text='Another Claude session sent a message:\n<cross-session-message from="uds:test-own-peer" from-name="IX">'+expected+'</cross-session-message>\nNative explanation.';
const entry={type:'user',uuid:'received',origin:{kind:'peer',name:'IX',from:'uds:test-own-peer',body:expected,msg_id:'reply-id'},message:{content:[{type:'text',text}]}};
test('requires actual recipient and exact nonce inside received user wrapper',()=>{
 assert.equal(nativeAckProof(entry,{target,expected}).sender,target);
 assert.equal(nativeAckProof(entry,{target:'Other [456def]',expected}),null);
 assert.equal(nativeAckProof(entry,{target,expected:'ACK different'}),null);
});
test('rejects assistant claims, plain ACK, replay and echoed request',()=>{
 for(const altered of [{...entry,type:'assistant'},{...entry,origin:undefined},{...entry,isReplay:true},{...entry,origin:{...entry.origin,body:'ACK different'}},{...entry,origin:{...entry.origin,name:'Other'}}])assert.equal(nativeAckProof(altered,{target,expected}),null);
 assert.equal(nativeAckProof(entry,{target,expected,inputs:[text]}),null);
});
