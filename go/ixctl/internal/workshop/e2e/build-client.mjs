import {createHash} from 'node:crypto';
import {readdirSync,readFileSync,writeFileSync} from 'node:fs';
import {join,relative,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=join(dirname(fileURLToPath(import.meta.url)),'..','web');
const check=process.argv.includes('--check');
const brand=JSON.parse(readFileSync(join(root,'branding.json'),'utf8'));
function generated(path,text){const file=join(root,path),old=readFileSync(file,'utf8');if(old===text)return;if(check)throw new Error(path+' is stale; run npm run build');writeFileSync(file,text);}
const manifest=JSON.parse(readFileSync(join(root,'manifest.webmanifest'),'utf8'));manifest.name=brand.name;manifest.short_name=brand.short_name;
generated('manifest.webmanifest',JSON.stringify(manifest)+'\n');
generated('index.html',readFileSync(join(root,'index.html'),'utf8').replace(/<title>[^<]*<\/title>/,'<title>'+brand.name+'</title>').replace(/<h1 id="app-name">[^<]*<\/h1>/,'<h1 id="app-name">'+brand.name+'</h1>'));
function files(dir){return readdirSync(dir,{withFileTypes:true}).flatMap(entry=>entry.isDirectory()?files(join(dir,entry.name)):[join(dir,entry.name)]);}
const hash=createHash('sha256');
for(const path of files(root).filter(path=>relative(root,path)!=='sw.js').sort()){
  hash.update(relative(root,path).replaceAll('\\','/')+'\0');hash.update(readFileSync(path));hash.update('\0');
}
const cache='ix-workshop-shell-'+hash.digest('hex').slice(0,16);
const source=readFileSync(join(root,'sw.js'),'utf8');
generated('sw.js',source.replace(/const CACHE = '[^']+';/,"const CACHE = '"+cache+"';"));
console.log((check?'Verified':'Built')+' client asset digest '+cache);
