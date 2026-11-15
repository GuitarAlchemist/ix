// IxForge widgets: custom HTML elements built with Lit, composed like blocks. They render into the light DOM (no
// shadow root), so the page's stylesheet and its strict CSP (no inline style attributes) apply to them as to the rest.
//
//   <ix-widget>     picks the block for a tool's kind
//   <ix-checklist>  a list with check boxes, an add field and a progress bar
//   <ix-counter>    a big number with −1, +1 and reset
//   <ix-queries>    queries to read and copy, by file (read-only; the queries are in the widget, .entries)
//
// A block never talks to the network. It is given the IXLET state (.x), whether it can act (.usable), how to act
// (.run, which returns true once the action is kept) and the translation function (.t, with .lang to re-render).
import {LitElement, html, svg, repeat, live} from './vendor/lit-all.min.js';

const PATHS = {
  devices: 'M3 5h13v9H3zM7 18h5M9.5 14v4M18 9h3v10h-3z',
  surface: 'M3 3h8v8H3zM13 3h8v5h-8zM13 10h8v11h-8zM3 13h8v8H3z',
  sun: 'M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
  moon: 'M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5',
  system: 'M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M12 3v18',
  mic: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3M5 11a7 7 0 0 0 14 0M12 18v3',
  detach: 'M14 4h6v6M20 4l-8 8M18 14v6H4V6h6',
  reattach: 'M10 14l-6 6M4 14v6h6M14 4h6v16H8',
  pin: 'M9 4h6l-1 6l3 3H7l3-3zM12 13v8',
  export: 'M12 4v11M7 10l5 5l5-5M5 20h14',
  import: 'M12 15V4M7 9l5-5l5 5M5 20h14',
  plus: 'M12 5v14M5 12h14',
  x: 'M6 6l12 12M18 6L6 18',
  list: 'M9 6h11M9 12h11M9 18h11M4 6l1 1l2-2M4 12l1 1l2-2M4 18l1 1l2-2',
  counter: 'M4 5h16v14H4zM9 12h6M12 9v6',
  update: 'M20 12a8 8 0 1 1-2.3-5.6M20 4v5h-5',
  install: 'M12 3v10M8 9l4 4l4-4M4 15v5h16v-5',
  settings: 'M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6M10.3 3h3.4l.5 2.4l1.9 1.1l2.3-.8l1.7 2.9l-1.8 1.6v2.2l1.8 1.6l-1.7 2.9l-2.3-.8l-1.9 1.1l-.5 2.4h-3.4l-.5-2.4l-1.9-1.1l-2.3.8l-1.7-2.9l1.8-1.6v-2.2L3.9 8.6l1.7-2.9l2.3.8l1.9-1.1z',
  terminal: 'M3 5h18v14H3zM7 9l3 3l-3 3M12 15h5',
  spark: 'M12 3l2 6l6 2l-6 2l-2 6l-2-6l-6-2l6-2zM19 3v4M17 5h4',
  database: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3s-3.6 3-8 3s-8-1.3-8-3M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
  copy: 'M9 9h11v11H9zM5 15H4V4h11v1',
};
export const icon = name => html`<svg class="i" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${svg`<path d=${PATHS[name]}></path>`}</svg>`;

class Block extends LitElement {
  static properties = {x: {attribute: false}, definition: {}, usable: {type: Boolean}, run: {attribute: false}, t: {attribute: false}, lang: {}, compact: {type: Boolean, reflect: true}};
  createRenderRoot() { return this; }
}

export class IxChecklist extends Block {
  static properties = {...Block.properties, draft: {state: true}};
  constructor() { super(); this.draft = ''; this.usable = false; }
  async add() {
    const text = this.draft.trim();
    if (!text || !this.usable) return;
    if (await this.run('add', {text})) this.draft = '';
  }
  render() {
    const t = this.t, items = this.x?.items || [], done = items.filter(i => i.done).length, def = this.definition;
    return html`
      <div class="progress-row">
        <progress max=${items.length || 1} .value=${done} aria-hidden="true"></progress>
        <span data-xlet-progress=${def}>${items.length ? t('checklistProgress', {done, total: items.length}) : t('checklistEmpty')}</span>
      </div>
      <ul class="checklist">${repeat(items, i => i.id, i => html`
        <li data-item=${i.id} class=${i.done ? 'done' : ''}>
          <label><input type="checkbox" data-item-check=${i.id} .checked=${live(i.done)} ?disabled=${!this.usable}
            @change=${e => this.run(e.target.checked ? 'check' : 'uncheck', {item_id: i.id})}><span class="item-text">${i.text}</span></label>
          <button class="icon ghost" data-item-remove=${i.id} ?disabled=${!this.usable} aria-label=${t('removeItem') + ' · ' + i.text} title=${t('removeItem')}
            @click=${() => this.run('remove', {item_id: i.id})}>${icon('x')}</button>
        </li>`)}
      </ul>
      <form class="add-item" @submit=${e => { e.preventDefault(); this.add(); }}>
        <input data-item-text=${def} maxlength="240" autocomplete="off" placeholder=${t('itemPlaceholder')} aria-label=${t('itemPlaceholder')}
          .value=${live(this.draft)} @input=${e => { this.draft = e.target.value; }}>
        <button data-item-add=${def} ?disabled=${!this.usable}>${icon('plus')}<span>${t('addItem')}</span></button>
      </form>`;
  }
}

export class IxCounter extends Block {
  constructor() { super(); this.usable = false; }
  render() {
    const t = this.t, def = this.definition;
    const button = (action, label, extra = '') => html`<button class=${extra} data-xlet-action=${action} data-xlet=${def} ?disabled=${!this.usable} @click=${() => this.run(action)}>${label}</button>`;
    return html`
      <p class="number" data-xlet-value=${def}>${this.x?.value ?? 0}</p>
      <div class="counter-actions">${button('decrement', t('decrement'))}${button('increment', t('increment'))}${button('reset', t('reset'), 'ghost')}</div>`;
  }
}

export class IxQueries extends Block {
  static properties = {...Block.properties, entries: {attribute: false}, copied: {state: true}};
  constructor() { super(); this.entries = []; this.copied = null; }
  async copy(index, sql) {
    try { await navigator.clipboard.writeText(sql); this.copied = {index, ok: true}; }
    catch { this.copied = {index, ok: false}; }
  }
  render() {
    const t = this.t, groups = [];
    (this.entries || []).forEach((entry, index) => {
      let group = groups.find(g => g.source === entry.source);
      if (!group) groups.push(group = {source: entry.source, queries: []});
      group.queries.push({entry, index});
    });
    return html`<div class="queries">${groups.map(g => html`
      <section data-query-source=${g.source}><h3>${g.source}</h3>${g.queries.map(({entry, index}) => html`
        <details data-query=${index}><summary>${entry.title}</summary><pre>${entry.sql}</pre>
          <div class="row"><button class="ghost" data-query-copy=${index} @click=${() => this.copy(index, entry.sql)}>${icon('copy')}<span>${t('copySQL')}</span></button>
          <span data-query-copied role="status">${this.copied?.index === index ? t(this.copied.ok ? 'copied' : 'copyFailed') : ''}</span></div>
        </details>`)}
      </section>`)}</div>`;
  }
}

export class IxWidget extends Block {
  static properties = {...Block.properties, kind: {}, entries: {attribute: false}};
  render() {
    const props = {x: this.x, definition: this.definition, usable: this.usable, run: this.run, t: this.t, lang: this.lang, compact: this.compact};
    if (this.kind === 'queries') return html`<ix-queries .entries=${this.entries} .t=${props.t} lang=${props.lang} ?compact=${props.compact}></ix-queries>`;
    return this.kind === 'checklist'
      ? html`<ix-checklist .x=${props.x} definition=${props.definition} ?usable=${props.usable} .run=${props.run} .t=${props.t} lang=${props.lang} ?compact=${props.compact}></ix-checklist>`
      : html`<ix-counter .x=${props.x} definition=${props.definition} ?usable=${props.usable} .run=${props.run} .t=${props.t} lang=${props.lang} ?compact=${props.compact}></ix-counter>`;
  }
}

customElements.define('ix-checklist', IxChecklist);
customElements.define('ix-counter', IxCounter);
customElements.define('ix-queries', IxQueries);
customElements.define('ix-widget', IxWidget);

// --- Moving a widget between IxForge and elsewhere (a Claude artifact, a file) -----------------------------------
// The exchange format is a JSON block: {"format":"ixforge-widget/1","kind":"checklist","title":"…","items":[{text,done}]}
// or {"format":"ixforge-widget/1","kind":"counter","title":"…","value":3}
// or {"format":"ixforge-widget/1","kind":"queries","title":"…","entries":[{title,source,sql}]}. An exported page carries it in
// <script type="application/json" id="ixforge-widget">, works by itself, and copies its current state back.
export const FORMAT = 'ixforge-widget/1';
const bytes = s => new TextEncoder().encode(s).length;
const titleOK = s => typeof s === 'string' && s.trim() !== '' && bytes(s) <= 240;

export function widgetData(kind, title, x, entries) {
  if (kind === 'queries') return {format: FORMAT, kind, title, entries: (entries || []).map(e => ({title: e.title, source: e.source, sql: e.sql}))};
  return kind === 'counter' ? {format: FORMAT, kind, title, value: x?.value ?? 0}
    : {format: FORMAT, kind, title, items: (x?.items || []).map(i => ({text: i.text, done: Boolean(i.done)}))};
}

// parseWidget accepts the JSON itself, or a whole exported page (HTML), and returns validated widget data.
export function parseWidget(text) {
  let source = String(text || '').trim();
  if (source.startsWith('<')) {
    const block = new DOMParser().parseFromString(source, 'text/html').getElementById('ixforge-widget');
    if (!block) throw new Error('import_invalid');
    source = block.textContent;
  }
  let data;
  try { data = JSON.parse(source); } catch { throw new Error('import_invalid'); }
  if (!data || data.format !== FORMAT || !titleOK(data.title)) throw new Error('import_invalid');
  if (data.kind === 'counter') {
    if (data.value !== undefined && !Number.isSafeInteger(data.value)) throw new Error('import_invalid');
    return {format: FORMAT, kind: 'counter', title: data.title.trim(), value: data.value ?? 0};
  }
  if (data.kind === 'queries') {
    // The Desktop's rules (queries.go): 1 to 100 queries, each titled, from a named file, SQL of at most 8 KiB.
    const entries = Array.isArray(data.entries) ? data.entries : [];
    const ok = e => e && titleOK(e.title) && titleOK(e.source) && typeof e.sql === 'string' && e.sql.trim() !== '' && bytes(e.sql) <= 8192;
    if (!entries.length || entries.length > 100 || !entries.every(ok)) throw new Error('import_invalid');
    return {format: FORMAT, kind: 'queries', title: data.title.trim(), entries: entries.map(e => ({title: e.title, source: e.source, sql: e.sql}))};
  }
  if (data.kind !== 'checklist' || !Array.isArray(data.items ?? []) || (data.items || []).length > 200) throw new Error('import_invalid');
  const items = (data.items || []).map(i => ({text: String(i?.text ?? ''), done: i?.done === true}));
  if (!items.every(i => titleOK(i.text))) throw new Error('import_invalid');
  return {format: FORMAT, kind: 'checklist', title: data.title.trim(), items};
}

const escapeHTML = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[c]);

// exportPage is a standalone HTML page: no network, no IxForge needed. It can be opened from disk, attached to a
// conversation with Claude, or published as an artifact; "Copier pour IxForge" copies the JSON back.
export function exportPage(data, labels) {
  const json = JSON.stringify(data).replace(/</g, '\\u003c');
  return `<!doctype html>
<html lang="${escapeHTML(labels.lang)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHTML(data.title)}</title>
<style>
:root{color-scheme:light dark;--bg:#f4f7fa;--panel:#fff;--line:#d4dde7;--text:#192637;--muted:#55677a;--accent:#12694c;--ink:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#10151e;--panel:#19212d;--line:#303d4f;--text:#edf3fb;--muted:#a6b7cc;--accent:#98edc6;--ink:#102c22}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,sans-serif;display:grid;place-items:start center;padding:24px 16px}
main{width:min(640px,100%);background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:24px}
h1{margin:0 0 12px;font-size:22px}ul{list-style:none;margin:12px 0;padding:0;display:grid;gap:6px}li{display:flex;gap:10px;align-items:center;min-height:40px}
li.done span{text-decoration:line-through;color:var(--muted)}li span{flex:1}input[type=checkbox]{width:22px;height:22px;accent-color:var(--accent)}
form{display:flex;gap:8px}input[type=text]{flex:1;min-width:0;font:inherit;padding:10px;border-radius:9px;border:1px solid var(--line);background:var(--bg);color:var(--text)}
button{font:inherit;padding:10px 14px;border-radius:9px;border:1px solid transparent;background:var(--accent);color:var(--ink);cursor:pointer}
button.ghost{background:transparent;color:var(--text);border-color:var(--line)}.number{font-size:96px;font-weight:650;text-align:center;margin:8px 0}
.row{display:flex;gap:8px;justify-content:center;flex-wrap:wrap}details{border-top:1px solid var(--line);padding:8px 0}summary{cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word;font:13px/1.45 ui-monospace,Consolas,monospace;background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:10px;margin:8px 0 0}footer{margin-top:20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:13px}
</style></head><body><main><h1 id="title"></h1><div id="widget"></div>
<footer><button id="copy" class="ghost">${escapeHTML(labels.copy)}</button><span id="copied" role="status"></span><span>${escapeHTML(labels.made)}</span></footer></main>
<script type="application/json" id="ixforge-widget">${json}</script>
<script>
(()=>{const block=document.getElementById('ixforge-widget');const data=JSON.parse(block.textContent);const root=document.getElementById('widget');
document.getElementById('title').textContent=data.title;
const save=()=>{block.textContent=JSON.stringify(data).replace(/</g,'\\\\u003c');draw();};
const el=(tag,props,...kids)=>{const e=Object.assign(document.createElement(tag),props);e.append(...kids);return e;};
function draw(){root.replaceChildren();
if(data.kind==='queries'){data.entries.forEach(q=>root.append(el('details',{},el('summary',{textContent:q.source+' · '+q.title}),el('pre',{textContent:q.sql}))));return;}
if(data.kind==='counter'){root.append(el('p',{className:'number',textContent:String(data.value)}),el('div',{className:'row'},
el('button',{textContent:'−1',onclick:()=>{data.value--;save();}}),el('button',{textContent:'+1',onclick:()=>{data.value++;save();}}),
el('button',{className:'ghost',textContent:${JSON.stringify(labels.reset)},onclick:()=>{data.value=0;save();}})));return;}
const ul=el('ul',{});data.items.forEach((item,i)=>{const box=el('input',{type:'checkbox',checked:item.done,onchange:()=>{item.done=box.checked;save();}});
ul.append(el('li',{className:item.done?'done':''},box,el('span',{textContent:item.text}),el('button',{className:'ghost',textContent:'×',title:${JSON.stringify(labels.remove)},onclick:()=>{data.items.splice(i,1);save();}})));});
const input=el('input',{type:'text',maxLength:240,placeholder:${JSON.stringify(labels.placeholder)}});
const form=el('form',{onsubmit:e=>{e.preventDefault();const text=input.value.trim();if(!text||data.items.length>=200)return;data.items.push({text,done:false});save();root.querySelector('input[type=text]').focus();}},input,el('button',{textContent:${JSON.stringify(labels.add)}}));
root.append(ul,form);}
document.getElementById('copy').onclick=async()=>{const text=JSON.stringify(data);try{await navigator.clipboard.writeText(text);document.getElementById('copied').textContent=${JSON.stringify(labels.copied)};}
catch{prompt(${JSON.stringify(labels.copyHere)},text);}};
draw();})();
</script></body></html>
`;
}
