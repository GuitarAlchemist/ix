/** Retain only IX metadata rows, never the surrounding peer inventory. */
import {redactPrintText} from './print-event-collector.mjs';
const addressPattern=/\b(?:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|[0-9a-f]{6,12})\b/gi;
const fields=new Set(['id','agent_id','session_id','short_id','address','name','session_name','status','state','kind','cwd','working_directory']);
export function extractNativeIxRows(content,{extraNames=[]}={}){
 const wantedNames=['IX','Indigo IX',...extraNames],escape=value=>value.replace(/[.*+?^$()|[\]\\]/g,'\\$&');
 const namePattern=new RegExp('\\b(?:'+wantedNames.map(escape).join('|')+')\\b(?=\\s*(?:[|()[\\],;:"\\x27-]|\\p{Pd}|$))','iu');
 const rows=[],seen=new Set();
 const keep=row=>{const key=JSON.stringify(row);if(!seen.has(key)){seen.add(key);rows.push(row);}};
 function structured(value){
  if(!value||typeof value!=='object')return;
  if(!Array.isArray(value)){
   const name=value.name??value.session_name,cwd=value.cwd??value.working_directory;
   if(typeof name==='string'&&wantedNames.some(n=>n.toLowerCase()===name.toLowerCase())||typeof cwd==='string'&&cwd.replaceAll('\\','/').toLowerCase()===(process.env.GAIA_IX_CWD??'').replaceAll('\\','/').toLowerCase()){
    const record={};for(const [key,v]of Object.entries(value))if(fields.has(key)&&['string','boolean','number'].includes(typeof v))record[key]=typeof v==='string'?redactPrintText(v).slice(0,500):v;
    keep({format:'structured',record});
   }
  }
  for(const v of Object.values(value))if(v&&typeof v==='object')structured(v);
 }
 function text(value){
  try{structured(JSON.parse(value));return;}catch{}
  for(const line of value.split(/\r?\n/)){
   const match=namePattern.exec(line);
   if(!match)continue;
   const prefix=line.slice(0,match.index).replace(addressPattern,'').replace(/\b(?:agent|session|id|name)\b/gi,'').replace(/[\s|()[\]{}:,*>"'\x60-]/g,'');
   if(prefix)continue;
   const addresses=[...new Set(line.match(addressPattern)??[])];
   if(addresses.length>1)continue;
   keep({format:'native-text',rawLine:redactPrintText(line.trim()).slice(0,1200),
    address:addresses[0]??null,name:match[0],
    state:line.match(/\b(waiting|idle|running|busy|offline|unavailable)\b/i)?.[0]??null,
    kind:line.match(/\b(interactive|background|sdk|print|cloud)\b/i)?.[0]??null,
    cannotReceive:/can't receive|cannot receive|off in that session|no inbox/i.test(line)});
  }
 }
 if(typeof content==='string')text(content);
 else if(Array.isArray(content))for(const block of content){if(block?.type==='text'&&typeof block.text==='string')text(block.text);}
 else structured(content);
 return {rows,rawInventoryRetained:false};
}
