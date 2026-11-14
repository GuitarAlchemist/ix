'use strict';
const text={fr:{translation:{title:'Administrer IxForge',intro:'Administration locale sur ce Desktop. Aucun contrôle d’exécution IXLET.',language:'Langue',pairTitle:'Relier la tablette',pairHelp:'Créez un QR temporaire, scannez-le sur la tablette, puis approuvez sa demande ci-dessous. Le code expire après deux minutes et ne sert qu’une fois.',newInvite:'Créer un QR de pairage',devicesTitle:'Appareils',connected:'Desktop connecté',offline:'Non connecté · état des appareils à vérifier',missing:'Ouvrez cette page par le raccourci Administrer IxForge sur le Desktop.',pending:'En attente',approved:'Approuvé',refused:'Refusé',revoked:'Révoqué',approve:'Approuver',refuse:'Refuser',revoke:'Révoquer',empty:'Aucun appareil pour le moment.',until:'Valable jusqu’à {{time}}',error:'Action impossible. Actualisez ou rouvrez le raccourci local.',confirm:'Révoquer l’accès de « {{name}} » ?'}},en:{translation:{title:'Manage IxForge',intro:'Local administration on this Desktop. No IXLET execution controls.',language:'Language',pairTitle:'Connect your tablet',pairHelp:'Create a temporary QR, scan it on the tablet, then approve its request below. The code expires after two minutes and can be used once.',newInvite:'Create pairing QR',devicesTitle:'Devices',connected:'Desktop connected',offline:'Not connected · device state needs verification',missing:'Open this page using the Manage IxForge Desktop shortcut.',pending:'Pending',approved:'Approved',refused:'Refused',revoked:'Revoked',approve:'Approve',refuse:'Refuse',revoke:'Revoke',empty:'No devices yet.',until:'Valid until {{time}}',error:'Action unavailable. Refresh or reopen the local shortcut.',confirm:'Revoke access for “{{name}}” ?'}}};
const $=id=>document.getElementById(id),t=(key,values)=>i18next.t(key,values);
let language='fr',key='',qrObject=null,invite=null,devices=[];
const fragment=new URLSearchParams(location.hash.slice(1));
if(fragment.has('session')){key=fragment.get('session');history.replaceState(null,'',location.pathname);try{sessionStorage.setItem('ix-workshop-owner-session',key)}catch{}}
else{try{key=sessionStorage.getItem('ix-workshop-owner-session')||''}catch{}}
async function api(path,body){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);try{
 const response=await fetch('/workshop-owner/api/'+path,{method:body===undefined?'GET':'POST',cache:'no-store',signal:controller.signal,headers:{'Content-Type':'application/json',Authorization:'Bearer '+key},body:body===undefined?undefined:JSON.stringify(body)});
 if(!response.ok)throw new Error('owner_request_refused');return await response.json();
}finally{clearTimeout(timer)}}
function render(){
 document.documentElement.lang=language;document.title=t('title');for(const node of document.querySelectorAll('[data-i18n]'))node.textContent=t(node.dataset.i18n);
 $('devices').replaceChildren();if(!devices.length){const p=document.createElement('p');p.textContent=t('empty');$('devices').append(p)}
 for(const device of devices){const row=document.createElement('article');row.className='device';row.dataset.deviceId=device.id;
  const title=document.createElement('h3');title.textContent=device.name;const status=document.createElement('p');status.textContent=t(device.status);row.append(title,status);
  const decisions=device.status==='pending'?['approved','refused']:device.status==='approved'?['revoked']:[];
  for(const decision of decisions){const button=document.createElement('button');button.dataset.decision=decision;button.textContent=t({approved:'approve',refused:'refuse',revoked:'revoke'}[decision]);button.onclick=async()=>{if(decision==='revoked'&&!confirm(t('confirm',{name:device.name})))return;button.disabled=true;try{await api('devices/'+encodeURIComponent(device.id)+'/'+decision,{});await refresh()}catch{$('error').textContent=t('error');button.disabled=false}};row.append(button)}$('devices').append(row);
 }
 if(invite)$('expires').textContent=t('until',{time:new Date(invite.expires).toLocaleTimeString(language)});
}
async function refresh(){if(!/^[0-9a-f]{48}$/i.test(key)){$('error').textContent=t('missing');$('invite').disabled=true;return}try{const state=await api('state');devices=state.devices;$('status').textContent=t('connected');$('error').textContent='';render()}catch{$('status').textContent=t('offline')}}
async function boot(){await i18next.init({lng:language,fallbackLng:'en',resources:text,interpolation:{escapeValue:false}});render();$('language').onchange=async()=>{language=$('language').value;await i18next.changeLanguage(language);render();await refresh()};$('invite').onclick=async()=>{$('invite').disabled=true;try{invite=await api('invite',{});if(qrObject)URL.revokeObjectURL(qrObject);qrObject=URL.createObjectURL(new Blob([invite.qr_svg],{type:'image/svg+xml'}));$('qr').src=qrObject;$('pair-link').href=invite.url;$('pair-link').textContent=invite.url;$('invite-view').hidden=false;render()}catch{$('error').textContent=t('error')}finally{$('invite').disabled=false}};await refresh();setInterval(refresh,2000)}
boot();



