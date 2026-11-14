import {spawn} from 'node:child_process';
import {mkdtempSync,readFileSync,readdirSync,rmSync,mkdirSync,writeFileSync} from 'node:fs';
import {join,dirname,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import assert from 'node:assert/strict';
import {proveBrowser} from './cdp.mjs';
import {PROFILE_PREFIX,checkFreshProfile,readActivePort} from './browser-owner.mjs';
const local=new URL(process.env.WORKSHOP_OWNER_ORIGIN),app=new URL(process.env.WORKSHOP_TABLET_ORIGIN),key=process.env.WORKSHOP_OWNER_SESSION,pin=process.env.WORKSHOP_FIXTURE_PIN;
assert.equal(local.hostname,'127.0.0.1');assert.equal(local.protocol,'http:');assert.equal(app.hostname,'workshop.invalid');assert.equal(app.protocol,'https:');for(const p of [local.port,app.port])assert.ok(![8737,8738,8739,8740].includes(Number(p)));assert.match(key,/^[a-f0-9]{48}$/i);
const sleep=ms=>new Promise(r=>setTimeout(r,ms)),profile=mkdtempSync(join(tmpdir(),PROFILE_PREFIX));checkFreshProfile(profile,tmpdir(),readdirSync(profile));
const child=spawn(process.env.CHROME||'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',['--headless=new','--remote-debugging-port=0','--user-data-dir='+profile,'--no-first-run','--no-default-browser-check','--disable-background-networking','--no-proxy-server','--ignore-certificate-errors-spki-list='+pin,'--host-resolver-rules=MAP workshop.invalid 127.0.0.1, MAP * ~NOTFOUND, EXCLUDE 127.0.0.1','about:blank'],{stdio:'ignore',windowsHide:true});
let browser;
try{
 let active;for(let n=0;n<150;n++){try{active=readFileSync(join(profile,'DevToolsActivePort'),'utf8');readActivePort(active);break}catch{await sleep(100)}}
 const version=await(await fetch('http://127.0.0.1:'+readActivePort(active).port+'/json/version')).json();browser=await proveBrowser({activePort:active,version,childPid:child.pid,alive:()=>child.exitCode===null});console.log('OWNED owner/tablet browser PID '+child.pid+' profile '+profile);
 const open=async url=>{const tab=await browser.openTab();await tab.send('Page.enable');const evaluate=async expression=>{const r=await tab.send('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.result?.exceptionDetails)throw new Error(r.result.exceptionDetails.exception?.description||r.result.exceptionDetails.text);return r.result?.result?.value};const until=async(expression,why)=>{for(let n=0;n<200;n++){try{if(await evaluate(expression))return}catch{}await sleep(100)}throw new Error(why+': '+await evaluate('document.body.innerText'))};await tab.send('Page.navigate',{url});return{tab,eval:evaluate,until,click:id=>evaluate(`document.getElementById(${JSON.stringify(id)}).click()`)}};
 const owner=await open(local.origin+'/workshop-owner/#session='+key);
 await owner.until(`document.getElementById('status').textContent==='Desktop connecté'`,'owner boot');assert.equal(await owner.eval('location.hash'),'');await owner.click('invite');
 await owner.until(`!document.getElementById('invite-view').hidden && document.getElementById('qr').complete`,'QR display');const link=await owner.eval(`document.getElementById('pair-link').href`);assert.ok(link.startsWith(app.origin+'/workshop/#pair='));
 const tablet=await open(link);await tablet.until(`document.getElementById('interpret').textContent==='Interpréter'`,'tablet boot');await tablet.eval(`document.getElementById('device-name').value='Synthetic tablet UI'`);await tablet.click('pair');await tablet.until(`document.getElementById('connection').textContent.includes('approbation')`,'tablet pending');
 await owner.until(`!!document.querySelector('[data-decision=approved]')`,'pending shown on Desktop');
 const forbidden=await tablet.eval(`fetch('/workshop-owner/api/state').then(r=>r.status)`);assert.equal(forbidden,403);
 const noControl=await tablet.eval(`fetch('/v1/status').then(r=>r.status)`);assert.equal(noControl,404);
 const selfApprove=await tablet.eval(`(async()=>{const m=await import('/workshop/sync.js'),s=await(await m.Replica.open()).read();return(await fetch('/workshop-owner/api/invite',{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+s.token},body:'{}'})).status})()`);assert.equal(selfApprove,403);
 if(process.env.WORKSHOP_ARTIFACT_DIR){mkdirSync(process.env.WORKSHOP_ARTIFACT_DIR,{recursive:true});const r=await owner.tab.send('Page.captureScreenshot',{format:'png'});writeFileSync(join(process.env.WORKSHOP_ARTIFACT_DIR,'owner-qr-pending-fr.png'),Buffer.from(r.result.data,'base64'))}
 await owner.eval(`document.querySelector('[data-decision=approved]').click()`);await tablet.click('sync');await tablet.until(`document.getElementById('connection').textContent==='Desktop connecté'`,'owner approval effective');
 await owner.eval(`(()=>{const s=document.getElementById('language');s.value='en';s.dispatchEvent(new Event('change'))})()`);await owner.until(`document.documentElement.lang==='en' && document.querySelector('[data-decision=revoked]')?.textContent==='Revoke'`,'owner EN');
 browser.on(message=>{if(message.method==='Page.javascriptDialogOpening'&&message.sessionId===owner.tab.sessionId)owner.tab.send('Page.handleJavaScriptDialog',{accept:true})});
 await owner.eval(`document.querySelector('[data-decision=revoked]').click()`);await tablet.click('sync');await tablet.until(`document.getElementById('connection').textContent.includes('révoqué')`,'revoke effective');
 console.log('PASS real host owner UI: QR display → pending tablet → explicit approval → revocation; FR/EN; tablet admin and /v1 denied');
}finally{
 if(browser){try{await browser.send('Browser.close')}catch{}browser.close()}
 for(let n=0;n<100&&child.exitCode===null;n++)await sleep(100);if(child.exitCode===null){child.kill();await new Promise(r=>child.once('exit',r))}
 assert.equal(resolve(dirname(profile)),resolve(tmpdir()));assert.ok(profile.startsWith(join(tmpdir(),PROFILE_PREFIX)));rmSync(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});console.log('CLEANED Chrome '+child.pid+' and its fresh profile');
}

