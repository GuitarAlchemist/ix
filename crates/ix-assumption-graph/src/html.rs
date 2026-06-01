//! Self-contained 2D viewer for the assumption graph.
//!
//! [`render`] inlines the Prime-Radiant graph JSON into a single static HTML
//! page — double-click to open, no server, no build, no external library.
//!
//! The viewer is a **two-panel master/detail browser**, not a node-link graph
//! (a node-link layout is illegible past a few dozen nodes; this stays readable
//! at thousands):
//! - **left** — the assumptions, grouped by *functional category* (runtime
//!   asserts / safety obligations / invariants / assumptions), each with its
//!   hexavalent verdict;
//! - **right** — the *code* the assumptions live in, as a collapsible tree
//!   (repo → crate → file → line) built purely from path segments, so it works
//!   for any language;
//! - the two are cross-linked: selecting an assumption reveals it in the tree;
//!   clicking a tree node filters the assumptions to that subtree; a search box
//!   and verdict chips filter both. Selection opens an evidence-first detail
//!   strip (claim, verdict, source, certainty, `path:line`).
//!
//! Emitted by `ix-assumption-graph-report --html`.

/// Build the viewer HTML with `graph_json` (the output of
/// [`crate::AssumptionGraph::prime_radiant_graph`], serialized) inlined.
pub fn render(graph_json: &str) -> String {
    // Escape `<`/`>` so a claim containing `</script>` (or `<!--`) can't break
    // out of the inline <script> block. `<`/`>` parse back to the same
    // characters in JS, so the data is unchanged. (Codex P2 on #76.)
    let safe = graph_json.replace('<', "\\u003c").replace('>', "\\u003e");
    TEMPLATE.replace("/*__GRAPH__*/null", &safe)
}

const TEMPLATE: &str = r####"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Assumption Browser</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --panel2:#0d1117; --fg:#e6edf3; --muted:#8b949e;
    --border:#30363d; --accent:#58a6ff; --sel:#1f6feb33; --hover:#21262d;
    --T:#2ea043; --P:#3fb950; --U:#8b949e; --D:#d29922; --F:#f85149; --C:#bc8cff;
  }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
    font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  #app { display:flex; flex-direction:column; height:100%; }

  header { padding:8px 14px; border-bottom:1px solid var(--border); background:var(--panel);
    display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:14px; margin:0; font-weight:600; white-space:nowrap; }
  header .stat { color:var(--muted); font-size:12px; white-space:nowrap; }
  header .stat b { color:var(--fg); font-weight:600; }
  #search { flex:1; min-width:160px; max-width:380px; background:var(--bg); color:var(--fg);
    border:1px solid var(--border); border-radius:6px; padding:5px 9px; font-size:13px; }
  #search:focus { outline:none; border-color:var(--accent); }
  .chips { display:flex; gap:5px; flex-wrap:wrap; align-items:center; }
  .chip { cursor:pointer; user-select:none; border:1px solid var(--border); border-radius:11px;
    padding:1px 9px; font-size:11px; font-weight:600; color:var(--fg); background:var(--bg);
    display:flex; align-items:center; gap:5px; }
  .chip .dot { width:8px; height:8px; border-radius:50%; }
  .chip.off { opacity:.32; }
  .btn { cursor:pointer; background:var(--bg); color:var(--fg); border:1px solid var(--border);
    border-radius:6px; padding:4px 10px; font-size:12px; }
  .btn:hover { background:var(--hover); }

  #cols { flex:1; display:grid; grid-template-columns:1fr 1fr; min-height:0; }
  .pane { display:flex; flex-direction:column; min-height:0; border-right:1px solid var(--border); }
  .pane:last-child { border-right:none; }
  .pane-head { padding:6px 12px; border-bottom:1px solid var(--border); background:var(--panel);
    display:flex; align-items:center; gap:10px; font-size:12px; color:var(--muted); }
  .pane-head b { color:var(--fg); }
  .pane-head select { background:var(--bg); color:var(--fg); border:1px solid var(--border);
    border-radius:6px; padding:3px 6px; font-size:12px; }
  .pane-body { overflow:auto; flex:1; padding:4px 0; }
  #activeFilter { margin-left:auto; color:var(--accent); cursor:pointer; font-size:11px; display:none; }
  #activeFilter.on { display:inline; }

  /* ---- left: assumptions grouped by category ---- */
  .group { margin:0; }
  .group > summary { cursor:pointer; padding:5px 12px; font-weight:600; list-style:none;
    display:flex; align-items:center; gap:8px; background:var(--panel);
    border-bottom:1px solid var(--border); }
  .group > summary:hover { background:var(--hover); }
  .group > summary::-webkit-details-marker { display:none; }
  .group > summary .caret { color:var(--muted); transition:transform .12s; flex:none; }
  .group[open] > summary .caret { transform:rotate(90deg); }
  .group > summary .glabel { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .group > summary .gcount { color:var(--muted); font-weight:400; font-size:11px; flex:none; }
  /* rollup: child verdicts summed into a stacked bar on the parent */
  .vbar { display:inline-flex; width:78px; height:9px; border-radius:3px; overflow:hidden;
    margin-left:auto; flex:none; border:1px solid var(--border); }
  .vbar > span { display:block; min-width:2px; }
  .row { padding:4px 12px 4px 28px; cursor:pointer; display:flex; align-items:flex-start; gap:8px;
    border-left:3px solid transparent; }
  .row:hover { background:var(--hover); }
  .row.sel { background:var(--sel); border-left-color:var(--accent); }
  .row .v { flex:none; width:16px; text-align:center; font-weight:700; font-size:11px; margin-top:1px; }
  .row .claim { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; }
  .row .loc { flex:none; color:var(--muted); font-size:11px; max-width:38%; overflow:hidden;
    text-overflow:ellipsis; white-space:nowrap; }

  /* ---- right: code tree ---- */
  details.tree { margin:0; }
  details.tree > summary { cursor:pointer; padding:3px 12px; list-style:none; display:flex;
    align-items:center; gap:6px; }
  details.tree > summary::-webkit-details-marker { display:none; }
  details.tree > summary:hover { background:var(--hover); }
  details.tree > summary.sel { background:var(--sel); }
  details.tree > summary .caret { color:var(--muted); width:10px; transition:transform .12s; }
  details.tree[open] > summary .caret { transform:rotate(90deg); }
  .tnode { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tnode.crate { font-weight:600; }
  .tnode.file { color:var(--accent); }
  .tcount { color:var(--muted); font-size:11px; margin-left:auto; padding-left:8px; }
  .leaf { padding:3px 12px; cursor:pointer; display:flex; gap:8px; align-items:baseline; }
  .leaf:hover { background:var(--hover); }
  .leaf.sel { background:var(--sel); }
  .leaf .ln { color:var(--muted); font-size:11px; flex:none; font-family:ui-monospace,monospace; }
  .leaf .claim { overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    font-family:ui-monospace,monospace; font-size:12px; }

  /* ---- detail strip ---- */
  #detail { border-top:1px solid var(--border); background:var(--panel); padding:10px 16px;
    max-height:32%; overflow:auto; display:none; }
  #detail.on { display:block; }
  #detail .dclaim { font-family:ui-monospace,monospace; font-size:14px; color:var(--fg);
    white-space:pre-wrap; word-break:break-word; }
  #detail .dmeta { margin-top:8px; display:flex; flex-wrap:wrap; gap:7px 18px; color:var(--muted);
    font-size:12px; }
  #detail .dmeta b { color:var(--fg); font-weight:600; }
  #detail .badge { display:inline-block; padding:1px 8px; border-radius:4px; font-weight:700;
    color:#0d1117; font-size:11px; }
  #detail .evid { margin-top:8px; font-family:ui-monospace,monospace; font-size:12px;
    color:var(--accent); }
  .empty { color:var(--muted); padding:24px 16px; text-align:center; font-size:13px; }
  ::-webkit-scrollbar { width:10px; height:10px; }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:5px; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>Assumption Browser</h1>
    <span class="stat"><b id="shown">0</b> / <span id="total">0</span> assumptions</span>
    <input id="search" placeholder="Search claims and paths…" autocomplete="off">
    <div class="chips" id="verdicts"></div>
    <button class="btn" id="reset">Reset</button>
  </header>

  <div id="cols">
    <section class="pane">
      <div class="pane-head">
        <span>Assumptions — group by</span>
        <select id="groupby">
          <option value="type">Crate &rarr; file</option>
          <option value="category">Category</option>
          <option value="truth_value">Verdict</option>
          <option value="kind">Kind</option>
        </select>
        <b id="leftcount"></b>
      </div>
      <div class="pane-body" id="left"></div>
    </section>

    <section class="pane">
      <div class="pane-head">
        <span><b>Code</b> — repo / crate / file / line</span>
        <span id="activeFilter">✕ clear path filter</span>
      </div>
      <div class="pane-body" id="right"></div>
    </section>
  </div>

  <div id="detail"></div>
</div>

<script>
const RAW = /*__GRAPH__*/null;
const VC = { T:'var(--T)', P:'var(--P)', U:'var(--U)', D:'var(--D)', F:'var(--F)', C:'var(--C)' };
const VERDICTS = ['T','P','U','D','F','C'];

// ---- normalize nodes ----
function lineOf(n){
  const e = n.evidence || '';
  const m = /:(\d+)\s*$/.exec(e);
  return m ? parseInt(m[1],10) : 0;
}
function category(n){
  if (n.source === 'code:assert') return 'Runtime asserts';
  if (n.truth_value === 'U')       return 'Safety obligations (unverified)';
  if (n.kind === 'invariant')      return 'Documented invariants';
  return 'Assumptions';
}
const NODES = (RAW && RAW.nodes ? RAW.nodes : []).map(n => ({
  id:n.id, claim:n.name||'', path:(n.path||'').replace(/\\/g,'/'),
  verdict:n.truth_value||'U', kind:n.kind||'', cert:n.certainty||'', conf:n.confidence,
  source:n.source||'', evidence:n.evidence||'', crate:n.type||'(unknown)', line:lineOf(n),
  category:'' ,
}));
NODES.forEach(n => n.category = category(n));
const BY_ID = new Map(NODES.map(n => [n.id, n]));

// ---- state ----
const state = { q:'', verdicts:new Set(VERDICTS), prefix:null, groupby:'type', sel:null };

function visible(){
  const q = state.q.toLowerCase();
  return NODES.filter(n =>
    state.verdicts.has(n.verdict) &&
    (!q || n.claim.toLowerCase().includes(q) || n.path.toLowerCase().includes(q))
  );
}
function leftSet(){
  const base = visible();
  if (!state.prefix) return base;
  return base.filter(n => n.path === state.prefix || n.path.startsWith(state.prefix + '/') ||
    n.path.startsWith(state.prefix.endsWith('/')?state.prefix:state.prefix+'/'));
}

// ---- header chips ----
function renderChips(){
  const box = document.getElementById('verdicts'); box.textContent='';
  VERDICTS.forEach(v => {
    const c = document.createElement('span');
    c.className = 'chip' + (state.verdicts.has(v)?'':' off');
    const dot = document.createElement('span'); dot.className='dot'; dot.style.background=VC[v];
    c.appendChild(dot); c.appendChild(document.createTextNode(v));
    c.onclick = () => { state.verdicts.has(v)?state.verdicts.delete(v):state.verdicts.add(v); render(); };
    box.appendChild(c);
  });
}

// ---- left: rollup tree (child verdicts summed into each parent) ----
function fileOf(p){ const a=p.split('/'); return a[a.length-1]||p; }
// Each entry: [label, key-fn]. Grouping by crate nests crate → file → claim,
// so counts/verdicts roll up at every level; the others are a single level.
function levelsFor(){
  if (state.groupby==='type')        return [n=>n.crate, n=>fileOf(n.path)];
  if (state.groupby==='truth_value') return [n=>'Verdict '+n.verdict];
  if (state.groupby==='kind')        return [n=>n.kind || '(none)'];
  return [n=>n.category];
}
function nest(rows, levels){
  if (!levels.length)
    return { leaf: rows.slice().sort((a,b)=>a.path.localeCompare(b.path)||a.line-b.line) };
  const fn=levels[0], m=new Map();
  rows.forEach(r => { const k=fn(r); if(!m.has(k)) m.set(k,[]); m.get(k).push(r); });
  const children=new Map();
  [...m.keys()].sort().forEach(k => children.set(k, nest(m.get(k), levels.slice(1))));
  return { children };
}
function leavesOf(node){
  if (node.leaf) return node.leaf;
  let out=[]; node.children.forEach(c => out=out.concat(leavesOf(c))); return out;
}
function verdictBar(claims){
  const counts={}; VERDICTS.forEach(v=>counts[v]=0);
  claims.forEach(c => counts[c.verdict]=(counts[c.verdict]||0)+1);
  const bar=document.createElement('span'); bar.className='vbar';
  bar.title=VERDICTS.filter(v=>counts[v]).map(v=>v+':'+counts[v]).join('   ') || 'no verdicts';
  VERDICTS.forEach(v => { if(!counts[v]) return;
    const s=document.createElement('span'); s.style.flexGrow=counts[v]; s.style.background=VC[v];
    bar.appendChild(s); });
  return bar;
}
function renderLeft(){
  const rows = leftSet();
  document.getElementById('shown').textContent = rows.length;
  document.getElementById('leftcount').textContent = state.prefix ? ('in '+state.prefix) : '';
  const host = document.getElementById('left'); host.textContent='';
  if (!rows.length){ const e=document.createElement('div'); e.className='empty';
    e.textContent='No assumptions match.'; host.appendChild(e); return; }
  const top = nest(rows, levelsFor()).children;
  top.forEach((node,name) => host.appendChild(renderGroup(name, node, 0, top.size<=8)));
}
function renderGroup(name, node, depth, openDefault){
  const claims = leavesOf(node);
  const d=document.createElement('details'); d.className='group'; d.open=openDefault;
  const s=document.createElement('summary'); s.style.paddingLeft=(8+depth*14)+'px';
  const car=document.createElement('span'); car.className='caret'; car.textContent='▶';
  const t=document.createElement('span'); t.className='glabel'; t.textContent=name;
  const c=document.createElement('span'); c.className='gcount'; c.textContent=claims.length;
  s.append(car,t,c,verdictBar(claims)); d.appendChild(s);
  if (node.leaf){
    node.leaf.forEach(n => d.appendChild(renderRow(n, depth+1)));
  } else {
    node.children.forEach((child,cname) =>
      d.appendChild(renderGroup(cname, child, depth+1, node.children.size<=12)));
  }
  return d;
}
function renderRow(n, depth){
  const r=document.createElement('div'); r.className='row'+(state.sel===n.id?' sel':''); r.dataset.id=n.id;
  r.style.paddingLeft=(16+depth*14)+'px';
  const v=document.createElement('span'); v.className='v'; v.textContent=n.verdict; v.style.color=VC[n.verdict];
  const cl=document.createElement('span'); cl.className='claim'; cl.textContent=n.claim; cl.title=n.claim;
  const lo=document.createElement('span'); lo.className='loc'; lo.textContent=n.crate+(n.line?(':'+n.line):'');
  r.append(v,cl,lo);
  r.onclick=()=>select(n.id,'left');
  return r;
}

// ---- right: code tree (path segments) ----
function buildTree(nodes){
  const root={children:new Map(),files:new Map(),count:0};
  nodes.forEach(n=>{
    const parts=n.path.split('/').filter(Boolean);
    if(!parts.length) return;
    const file=parts.pop();
    let cur=root; cur.count++;
    let acc=[];
    parts.forEach(p=>{ acc.push(p);
      if(!cur.children.has(p)) cur.children.set(p,{name:p,path:acc.join('/'),children:new Map(),files:new Map(),count:0});
      cur=cur.children.get(p); cur.count++;
    });
    const fp=(parts.length?parts.join('/')+'/':'')+file;
    if(!cur.files.has(file)) cur.files.set(file,{name:file,path:fp,nodes:[]});
    cur.files.get(file).nodes.push(n);
  });
  return root;
}
function crateOf(path){ const p=path.split('/'); const i=p.indexOf('crates'); return i>=0?p[i+1]:p[0]; }
function renderRight(){
  const host=document.getElementById('right'); host.textContent='';
  const nodes=visible();
  if(!nodes.length){ const e=document.createElement('div'); e.className='empty';
    e.textContent='No code matches.'; host.appendChild(e); return; }
  const root=buildTree(nodes);
  // collapse the long shared prefix (e.g. crates/) by starting one level down at crate granularity
  renderTreeLevel(root, host, 0);
}
function renderTreeLevel(node, host, depth){
  // folders
  [...node.children.values()].sort((a,b)=>a.name.localeCompare(b.name)).forEach(ch=>{
    const d=document.createElement('details'); d.className='tree'; d.dataset.path=ch.path;
    d.open = depth>0 || node.children.size<=2;
    const s=document.createElement('summary'); s.dataset.path=ch.path;
    if(state.prefix===ch.path) s.classList.add('sel');
    const car=document.createElement('span'); car.className='caret'; car.textContent='▶';
    const nm=document.createElement('span'); nm.className='tnode'+(ch.name && depth===0?' crate':''); nm.textContent=ch.name;
    const cnt=document.createElement('span'); cnt.className='tcount'; cnt.textContent=ch.count;
    nm.style.paddingLeft=(depth*12)+'px';
    s.append(car,nm,cnt); d.appendChild(s);
    s.onclick=(ev)=>{ ev.preventDefault(); d.open=!d.open; setPrefix(ch.path); };
    renderTreeLevel(ch, d, depth+1);
    host.appendChild(d);
  });
  // files at this level
  [...node.files.values()].sort((a,b)=>a.name.localeCompare(b.name)).forEach(f=>{
    const d=document.createElement('details'); d.className='tree'; d.dataset.path=f.path;
    const s=document.createElement('summary'); s.dataset.path=f.path;
    if(state.prefix===f.path) s.classList.add('sel');
    const car=document.createElement('span'); car.className='caret'; car.textContent='▶';
    const nm=document.createElement('span'); nm.className='tnode file'; nm.textContent=f.name;
    nm.style.paddingLeft=(depth*12)+'px';
    const cnt=document.createElement('span'); cnt.className='tcount'; cnt.textContent=f.nodes.length;
    s.append(car,nm,cnt); d.appendChild(s);
    s.onclick=(ev)=>{ ev.preventDefault(); d.open=!d.open; setPrefix(f.path); };
    f.nodes.sort((a,b)=>a.line-b.line).forEach(n=>{
      const leaf=document.createElement('div'); leaf.className='leaf'+(state.sel===n.id?' sel':''); leaf.dataset.id=n.id;
      const ln=document.createElement('span'); ln.className='ln'; ln.textContent=n.line?(':'+n.line):'';
      const v=document.createElement('span'); v.className='v'; v.textContent=n.verdict; v.style.color=VC[n.verdict];
      const cl=document.createElement('span'); cl.className='claim'; cl.textContent=n.claim; cl.title=n.claim;
      leaf.append(ln,v,cl);
      leaf.onclick=()=>select(n.id,'right');
      d.appendChild(leaf);
    });
    host.appendChild(d);
  });
}

// ---- cross-linking ----
function setPrefix(p){ state.prefix = (state.prefix===p)?null:p; syncActiveFilter(); renderLeft(); renderRight(); }
function syncActiveFilter(){ document.getElementById('activeFilter').classList.toggle('on', !!state.prefix); }

function select(id, from){
  state.sel = id; renderLeft(); renderRight(); showDetail(id);
  // reveal in the opposite tree
  requestAnimationFrame(()=>{
    if(from==='left'){ revealInTree(id); }
    const sel = document.querySelector(from==='left' ? '.leaf.sel' : '.row.sel');
    const cur = document.querySelector('.row.sel') || document.querySelector('.leaf.sel');
    if(cur) cur.scrollIntoView({block:'nearest'});
  });
}
function revealInTree(id){
  const n=BY_ID.get(id); if(!n) return;
  // open every ancestor <details> whose data-path prefixes the node path
  document.querySelectorAll('#right details.tree').forEach(d=>{
    const p=d.dataset.path||'';
    if(n.path===p || n.path.startsWith(p+'/')) d.open=true;
  });
  const leaf=document.querySelector('#right .leaf[data-id="'+CSS.escape(id)+'"]');
  if(leaf) leaf.scrollIntoView({block:'center'});
}

function showDetail(id){
  const n=BY_ID.get(id); const box=document.getElementById('detail');
  if(!n){ box.className=''; return; }
  box.className='on'; box.textContent='';
  const claim=document.createElement('div'); claim.className='dclaim'; claim.textContent=n.claim;
  const meta=document.createElement('div'); meta.className='dmeta';
  const badge=document.createElement('span'); badge.className='badge'; badge.style.background=VC[n.verdict];
  badge.textContent='verdict '+n.verdict;
  const mk=(label,val)=>{ const s=document.createElement('span'); const b=document.createElement('b');
    b.textContent=label+': '; s.append(b,document.createTextNode(val)); return s; };
  meta.append(badge, mk('category',n.category), mk('kind',n.kind||'—'),
    mk('source',n.source||'—'), mk('certainty',n.cert||'—'),
    mk('confidence', n.conf!=null?n.conf.toFixed(2):'—'), mk('crate',n.crate));
  const ev=document.createElement('div'); ev.className='evid';
  ev.textContent='📄 '+n.path+(n.line?(':'+n.line):'');
  box.append(claim, meta, ev);
}

// ---- wiring ----
function render(){ renderChips(); renderLeft(); renderRight(); syncActiveFilter(); }
document.getElementById('search').oninput = e => { state.q=e.target.value; renderLeft(); renderRight(); };
document.getElementById('groupby').onchange = e => { state.groupby=e.target.value; renderLeft(); };
document.getElementById('activeFilter').onclick = () => setPrefix(state.prefix);
document.getElementById('reset').onclick = () => {
  state.q=''; state.verdicts=new Set(VERDICTS); state.prefix=null; state.sel=null;
  document.getElementById('search').value=''; document.getElementById('detail').className='';
  render();
};
document.getElementById('total').textContent = NODES.length;
render();
</script>
</body>
</html>
"####;
