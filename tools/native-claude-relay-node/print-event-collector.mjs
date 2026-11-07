/** Bounded observations only. No process, network, credentials or delivery logic. */
import {StringDecoder} from 'node:string_decoder';
const types=new Set(['system','assistant','user','result','stream_event','tool_progress','tool_use_summary','rate_limit_event','auth_status','prompt_suggestion','control_request','control_response','keep_alive']);
const subtypes=new Set(['init','api_retry','success','error_during_execution','error_max_turns','error_max_budget_usd','error_max_structured_output_retries','compact_boundary','status','permission_denied','task_started','task_progress','task_notification','hook_started','hook_progress','hook_response','stop_hook_summary','local_command_output']);
const retryErrors=new Set(['authentication_failed','oauth_org_not_allowed','account_on_hold','billing_error','rate_limit','overloaded','invalid_request','model_not_found','server_error','max_output_tokens','cloud_credential_error','unknown']);
const uuid=value=>typeof value==='string'&&/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(value)?value:null;
const number=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0?value:undefined;
const label=(value,allowed)=>allowed.has(value)?value:'unknown';
export function redactPrintText(value){
 return String(value??'').replace(/\b(?:authorization|x-api-key|cookie)\s*:[^\r\n]*/gi,'[CREDENTIAL_HEADER_REDACTED]')
  .replace(/\b(access_token|refresh_token|api_key|token|password|secret)\b(["']?\s*[:=]\s*)(["'])(.*?)\3/gi,'$1$2$3[REDACTED]$3')
  .replace(/sk-[A-Za-z0-9_-]+/g,'[REDACTED]')
  .replace(/Bearer\s+\S+/gi,'Bearer [REDACTED]')
  .replace(/\b(?:eyJ[A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g,'[REDACTED]')
  .replace(/\b(access_token|refresh_token|api_key|x-api-key|token|password|secret|cookie)\b(["']?\s*[:=]\s*["']?)[^\s"',;]+/gi,'$1$2[REDACTED]')
  .replace(/(https?:\/\/[^\s?#]+)[?#][^\s]*/gi,'$1?[REDACTED_QUERY]').slice(0,4000);
}
function diagnostic(value,knownPaths){
 let text;try{text=typeof value==='string'?value:JSON.stringify(value);}catch{text='';}
 const codes=[...new Set(text.match(/\b(?:EACCES|EPERM|ENOENT|EBUSY|EEXIST|ETIMEDOUT|ECONNREFUSED|ECONNRESET|ENOTFOUND|EPIPE|CERT_HAS_EXPIRED|DEPTH_ZERO_SELF_SIGNED_CERT)\b/g)??[])];
 const classes=[];
 for(const [name,pattern] of [
  ['config_lock',/failed to save config with lock/i],['network',/connection|network|fetch failed|ECONN|ENOTFOUND/i],
  ['authentication',/invalid.api.key|authentication|unauthorized/i],['permission',/permission.denied|access.denied|EACCES|EPERM/i],
  ['retry',/retry|rate.limit/i],['stdin_eof',/\bEOF\b|end.of.file|stdin/i],['timeout',/timeout|timed.out|ETIMEDOUT/i],
 ])if(pattern.test(text))classes.push(name);
 const paths=knownPaths.filter(path=>text.includes(path)||text.includes(JSON.stringify(path).slice(1,-1)));
 const status=value&&typeof value==='object'?[value.error_status,value.status,value.status_code,value.statusCode,value.error?.status,value.error?.statusCode].find(v=>Number.isInteger(v)&&v>=100&&v<=599):undefined;
 const messageValues=typeof value==='string'?[value]:Array.isArray(value)?value.filter(v=>typeof v==='string'):
  value&&typeof value==='object'?[value.message,value.error?.message,...(Array.isArray(value.errors)?value.errors:[])].filter(v=>typeof v==='string'):[];
 const messages=messageValues.slice(0,4).map(message=>redactPrintText(message)
  .replace(/https?:\/\/\S+/gi,'[URL_REDACTED]').replace(/[A-Za-z0-9_+/=-]{40,}/g,'[OPAQUE_VALUE_REDACTED]').slice(0,512));
 return {codes,classes,paths,httpStatus:status,messages,rawErrorTextRetained:false};
}
function usageOf(value){
 const out={};for(const key of ['input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens'])
  if(number(value?.[key])!==undefined)out[key]=value[key];return out;
}
export function createPrintEventCollector({sessionId,now=Date.now,maxBytes=262144,maxEvents=128,knownPaths=[]}={}){
 if(!uuid(sessionId)||!Number.isInteger(maxEvents)||maxEvents<1||maxEvents>512||!Number.isInteger(maxBytes)||maxBytes<1||maxBytes>1048576)throw Error('PRINT_COLLECTOR_CONFIGURATION');
 const origin=now(),decoder=new StringDecoder('utf8');let buffer='',ended=false,stderr='';
 const data={stdoutBytes:0,stderrBytes:0,events:[],eventCounts:{},eventsDropped:0,stderrTruncated:false,
  stderrDiagnostic:diagnostic('',knownPaths),stdinEnd:null,stdinFinish:null,stop:null,close:null};
 const time=()=>Math.max(0,now()-origin);
 function summarize(event){
  if(!event||typeof event!=='object'||Array.isArray(event))throw Error('PRINT_INVALID_STREAM_FRAME');
  const type=label(event.type,types),subtype=event.subtype===undefined?null:label(event.subtype,subtypes);
  const summary={type,subtype,sessionId:uuid(event.session_id),atMs:time(),diagnostic:diagnostic(event,knownPaths)};
  if(type==='system'&&subtype==='init')summary.init={sessionId:uuid(event.session_id),
   model:typeof event.model==='string'&&/^claude-[a-z0-9.-]{1,64}$/.test(event.model)?event.model:null,
   version:typeof event.claude_code_version==='string'&&/^\d+\.\d+\.\d+$/.test(event.claude_code_version)?event.claude_code_version:null,
   toolCount:Array.isArray(event.tools)?event.tools.length:null,mcpServerCount:Array.isArray(event.mcp_servers)?event.mcp_servers.length:null};
  if(type==='system'&&subtype==='api_retry')summary.retry={attempt:number(event.attempt),maxRetries:number(event.max_retries),delayMs:number(event.retry_delay_ms),
   errorStatus:event.error_status===null?null:Number.isInteger(event.error_status)&&event.error_status>=100&&event.error_status<=599?event.error_status:'not_reported',
   errorCategory:label(event.error,retryErrors),
   noResponse:event.no_response&&typeof event.no_response==='object'?{waitedMs:number(event.no_response.waited_ms),retryWaitMs:number(event.no_response.retry_wait_ms)}:undefined};
  if(type==='assistant'){
   const content=Array.isArray(event.message?.content)?event.message.content:[];
   summary.assistant={contentTypes:[...new Set(content.map(block=>label(block?.type,new Set(['text','thinking','redacted_thinking','tool_use']))))],
    textCharacters:content.reduce((sum,block)=>sum+(block?.type==='text'&&typeof block.text==='string'?block.text.length:0),0)};
  }
  if(type==='result')summary.result={sessionId:uuid(event.session_id),subtype,isError:event.is_error===true,
   text:event.is_error===false?redactPrintText(event.result):'',diagnostic:diagnostic(event.errors??event,knownPaths),
   usage:usageOf(event.usage),totalCostUsd:number(event.total_cost_usd),numTurns:number(event.num_turns),
   correlated:event.session_id===sessionId&&event.is_error===false&&typeof event.result==='string'&&event.result.trim().length>0};
  const key=type+(subtype?':'+subtype:'');data.eventCounts[key]=(data.eventCounts[key]??0)+1;
  if(data.events.length<maxEvents)data.events.push(summary);else data.eventsDropped++;
  return summary;
 }
 function parse(line){let event;try{event=JSON.parse(line);}catch{throw Error('PRINT_INVALID_STREAM_FRAME');}return summarize(event);}
 function drain(){const emitted=[];for(let index;(index=buffer.indexOf('\n'))>=0;){const line=buffer.slice(0,index);buffer=buffer.slice(index+1);if(line.trim())emitted.push(parse(line));}return emitted;}
 return {
  pushStdout(chunk){if(ended)throw Error('PRINT_STREAM_ALREADY_ENDED');data.stdoutBytes+=Buffer.byteLength(chunk);if(data.stdoutBytes>maxBytes)throw Error('PRINT_OUTPUT_BOUND_EXCEEDED');buffer+=decoder.write(Buffer.isBuffer(chunk)?chunk:Buffer.from(chunk));return drain();},
  endStdout(){if(ended)return [];ended=true;buffer+=decoder.end();const emitted=drain();if(buffer.trim())emitted.push(parse(buffer));buffer='';return emitted;},
  pushStderr(chunk){data.stderrBytes+=Buffer.byteLength(chunk);if(stderr.length<65536)stderr=(stderr+chunk.toString('utf8')).slice(0,65536);data.stderrTruncated=data.stderrBytes>65536;data.stderrDiagnostic=diagnostic(stderr,knownPaths);},
  markStdinEnd(reason){if(!data.stdinEnd)data.stdinEnd={reason:['result','timeout','access-denied','abnormal','one-shot-input'].includes(reason)?reason:'unknown',atMs:time()};},
  markStdinFinish(){if(!data.stdinFinish)data.stdinFinish={atMs:time()};},
  markStop(reason,signal){if(!data.stop)data.stop={reason:['timeout','access-denied','abnormal','result-grace'].includes(reason)?reason:'unknown',signal:signal==='SIGTERM'?'SIGTERM':'unknown',atMs:time()};},
  markClose(code,signal){data.close={code:Number.isInteger(code)?code:null,signal:signal==='SIGTERM'?'SIGTERM':signal===null?null:'unknown',atMs:time()};},
  snapshot(){return structuredClone(data);}
 };
}
