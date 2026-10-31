import {acquireLeader} from '../../ledger.mjs';
const release=acquireLeader(process.argv[2]);
process.send({state:'LOCK_HELD'});
process.on('message',message=>{if(message.type==='CLOSE'){release();process.exit(0);}});
