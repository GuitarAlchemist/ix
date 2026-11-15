import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const jsQR=require(process.env.WORKSHOP_QR_DECODER || 'jsqr');
const svg=readFileSync(0,'utf8');
const modules=Number(svg.match(/viewBox="0 0 (\d+) \d+"/)[1]);
const scale=8,width=modules*scale,data=new Uint8ClampedArray(width*width*4).fill(255);
for(const match of svg.matchAll(/<rect x="(\d+)" y="(\d+)" width="1" height="1"\/>/g)){
  for(let y=Number(match[2])*scale;y<(Number(match[2])+1)*scale;y++)for(let x=Number(match[1])*scale;x<(Number(match[1])+1)*scale;x++){
    const offset=(y*width+x)*4;data[offset]=data[offset+1]=data[offset+2]=0;
  }
}
const decoded=jsQR(data,width,width);
assert.equal(decoded?.data,process.argv[2]);
console.log('QR fixture roundtrip verified');
