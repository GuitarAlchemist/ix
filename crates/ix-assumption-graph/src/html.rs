//! Self-contained 2D viewer for the assumption graph.
//!
//! [`render`] inlines the Prime-Radiant graph JSON into a single static HTML
//! page (Cytoscape.js from CDN) — double-click to open, no server, no build.
//! Readable node-link view: claims colored by hexavalent verdict, grouped into
//! per-namespace columns; clicking a node focuses its neighborhood and opens a
//! detail panel led by *evidence* and a clickable list of what it contradicts.
//! Hover for full claim text; filter by verdict / contradictions / isolation.
//! Emitted by `ix-assumption-graph-report --html`.

/// Build the viewer HTML with `graph_json` (the output of
/// [`crate::AssumptionGraph::prime_radiant_graph`], serialized) inlined.
pub fn render(graph_json: &str) -> String {
    TEMPLATE.replace("/*__GRAPH__*/null", graph_json)
}

const TEMPLATE: &str = r####"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Assumption Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --fg:#e6edf3; --muted:#8b949e; --border:#30363d; --accent:#58a6ff; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
    font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  #app { display:flex; flex-direction:column; height:100%; }
  header { padding:9px 16px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; white-space:nowrap; }
  .stats { color:var(--muted); white-space:nowrap; }
  .stats b { color:var(--fg); }
  #search { background:#0d1117; color:var(--fg); border:1px solid var(--border);
    border-radius:6px; padding:5px 10px; width:180px; outline:none; }
  #search:focus { border-color:var(--accent); }
  .toggle { background:#0d1117; color:var(--muted); border:1px solid var(--border);
    border-radius:6px; padding:5px 10px; cursor:pointer; white-space:nowrap; font-size:12px; }
  .toggle.on { background:var(--accent); color:#0d1117; border-color:var(--accent); font-weight:600; }
  .legend { display:flex; gap:9px; flex-wrap:wrap; margin-left:auto; }
  .chip { display:flex; align-items:center; gap:5px; color:var(--muted); font-size:12px;
    cursor:pointer; user-select:none; }
  .chip.off { opacity:.32; text-decoration:line-through; }
  .dot { width:11px; height:11px; border-radius:50%; display:inline-block; flex:0 0 auto; }
  #main { flex:1; display:flex; min-height:0; position:relative; }
  #cy { flex:1; min-width:0; }
  #ctrl { position:absolute; left:12px; bottom:12px; display:flex; gap:6px; z-index:5; }
  #ctrl button { background:var(--panel); color:var(--fg); border:1px solid var(--border);
    border-radius:6px; padding:5px 11px; cursor:pointer; font-size:12px; }
  #ctrl button:hover { border-color:var(--accent); }
  #tip { position:absolute; z-index:9; display:none; pointer-events:none; max-width:280px;
    background:#000; color:#fff; border:1px solid var(--border); border-radius:5px;
    padding:5px 9px; font-size:12px; box-shadow:0 3px 12px rgba(0,0,0,.6); }
  #side { width:340px; border-left:1px solid var(--border); background:var(--panel);
    padding:16px; overflow:auto; display:none; }
  #side.open { display:block; }
  #side h2 { font-size:15px; margin:0 0 10px; line-height:1.35; }
  #side .row { margin:11px 0; }
  #side .k { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.6px; margin-bottom:3px; }
  #side .v { word-break:break-word; }
  #side .sub { color:var(--muted); }
  .badge { display:inline-block; padding:2px 9px; border-radius:11px; font-weight:700; color:#0d1117; }
  .conf { color:var(--muted); margin-left:8px; font-size:12px; }
  .ev { background:#0d1117; border-left:3px solid var(--accent); padding:8px 10px; border-radius:0 6px 6px 0; }
  .rels { display:flex; flex-direction:column; gap:4px; }
  .rel { display:flex; align-items:center; gap:7px; padding:5px 8px; background:#0d1117;
    border:1px solid var(--border); border-radius:6px; cursor:pointer; font-size:12px; }
  .rel:hover { border-color:var(--accent); }
  .idfoot { margin-top:16px; padding-top:10px; border-top:1px solid var(--border);
    font-family:monospace; font-size:10px; color:#586069; word-break:break-all; }
  #empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    color:var(--muted); text-align:center; padding:40px; font-size:14px; }
  code { background:#0d1117; padding:1px 5px; border-radius:4px; font-size:12px; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>Assumption Graph</h1>
    <span class="stats" id="stats"></span>
    <input id="search" placeholder="filter claims…" autocomplete="off">
    <button class="toggle" id="f-contra">⚠ contradictions only</button>
    <button class="toggle" id="f-iso">hide isolated</button>
    <span class="legend" id="legend"></span>
  </header>
  <div id="main">
    <div id="cy"></div>
    <div id="tip"></div>
    <div id="ctrl"><button id="b-fit">Fit</button><button id="b-reset">Reset</button></div>
    <aside id="side"></aside>
  </div>
</div>
<script>
const GRAPH = /*__GRAPH__*/null;
const VERDICT = {
  T:{c:'#2ea043',n:'True'}, P:{c:'#3fb950',n:'Probable'}, U:{c:'#8b949e',n:'Unknown'},
  D:{c:'#d29922',n:'Doubtful'}, F:{c:'#f85149',n:'False'}, C:{c:'#bc8cff',n:'Contradictory'}
};
function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
(function(){
  const g = GRAPH || {nodes:[],edges:[]};
  const nodes = g.nodes||[], edges = g.edges||[];
  const byId = {}; nodes.forEach(n=>byId[n.id]=n);
  const nContra = edges.filter(e=>e.relation==='contradicts').length;
  document.getElementById('stats').innerHTML =
    `<b>${nodes.length}</b> claims &middot; <b>${edges.length}</b> links &middot; ` +
    `<b style="color:#f85149">${nContra}</b> contradictions`;

  // clickable verdict legend (toggles visibility)
  const hiddenV = new Set();
  document.getElementById('legend').innerHTML = Object.entries(VERDICT)
    .map(([k,v])=>`<span class="chip" data-v="${k}"><i class="dot" style="background:${v.c}"></i>${v.n}</span>`).join('');

  if(!nodes.length){
    document.getElementById('main').insertAdjacentHTML('beforeend',
      '<div id="empty">No assumptions yet.<br>Add <code>@ai:</code> annotations (or research claims), then re-run ' +
      '<code>ix-assumption-graph-report . --html out.html</code>.</div>');
    return;
  }

  // degree + contradiction membership (for filters)
  const degree={}, inContra=new Set();
  edges.forEach(e=>{ degree[e.source]=(degree[e.source]||0)+1; degree[e.target]=(degree[e.target]||0)+1;
    if(e.relation==='contradicts'){ inContra.add(e.source); inContra.add(e.target); }});

  // namespace columns
  const nsList=[], nsCount={}, byNs={};
  nodes.forEach(n=>{ const k=n.type||'(none)'; if(!(k in nsCount)){nsList.push(k);nsCount[k]=0;byNs[k]=[];}
    nsCount[k]++; byNs[k].push(n.id); });
  const COLW=240, ROWH=96, X0=120, Y0=140, pos={};
  nsList.forEach((k,ci)=> byNs[k].forEach((id,ri)=>{ pos[id]={x:X0+ci*COLW, y:Y0+ri*ROWH}; }));

  const els=[];
  nsList.forEach(ns => els.push({data:{id:'ns:'+ns, label:`${ns}  (${nsCount[ns]})`, isGroup:1}}));
  nodes.forEach(n => els.push({data:{
    id:n.id, parent:'ns:'+(n.type||'(none)'),
    label:(n.name||'').length>38 ? (n.name||'').slice(0,36)+'…' : (n.name||''),
    tv:n.truth_value||'U', full:n }}));
  edges.forEach((e,i) => els.push({data:{
    id:'e'+i, source:e.source, target:e.target, rel:e.relation||'',
    contradicts: e.relation==='contradicts' ? 1 : 0 }}));

  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: els, wheelSensitivity: 0.2,
    style: [
      {selector:'node[isGroup]', style:{
        'background-color':'#161b22','background-opacity':0.45,'border-color':'#30363d','border-width':1,
        'label':'data(label)','color':'#8b949e','font-size':11,'font-weight':'bold','text-valign':'top',
        'text-halign':'center','padding':'16px','shape':'round-rectangle','text-margin-y':-3 }},
      {selector:'node[tv]', style:{
        'background-color': n => (VERDICT[n.data('tv')]||VERDICT.U).c,
        'label':'data(label)','color':'#0d1117','font-size':9,'font-weight':'bold',
        'text-valign':'center','text-halign':'center','text-wrap':'wrap','text-max-width':'72px',
        'width':58,'height':58,'border-width':2,'border-color':'#0d1117' }},
      {selector:'node[tv]:selected', style:{'border-color':'#58a6ff','border-width':5}},
      {selector:'edge', style:{
        'width':1.5,'line-color':'#484f58','curve-style':'bezier',
        'target-arrow-color':'#484f58','target-arrow-shape':'triangle','arrow-scale':0.8 }},
      {selector:'edge[?contradicts]', style:{
        'width':3,'line-color':'#f85149','line-style':'dashed',
        'target-arrow-color':'#f85149','target-arrow-shape':'triangle','arrow-scale':1 }},
      {selector:'.faded', style:{'opacity':0.12}},
      {selector:'.hidden', style:{'display':'none'}}
    ],
    layout:{ name:'preset', positions: pos, fit:true, padding:60 }
  });

  // relations of a node, split into contradicts vs related
  function relsOf(id){ const out={contra:[],rel:[]};
    edges.forEach(e=>{ if(e.source!==id && e.target!==id) return;
      const other=byId[e.source===id?e.target:e.source]; if(!other) return;
      (e.relation==='contradicts'?out.contra:out.rel).push(other); });
    return out; }

  const side=document.getElementById('side');
  function relList(arr){ return '<div class="rels">'+arr.map(o=>{ const v=VERDICT[o.truth_value]||VERDICT.U;
    return `<div class="rel" data-goto="${esc(o.id)}"><i class="dot" style="background:${v.c}"></i>${esc(o.name)}</div>`;
    }).join('')+'</div>'; }
  function showNode(n){
    const v=VERDICT[n.truth_value]||VERDICT.U, r=relsOf(n.id);
    const conf=(typeof n.confidence==='number')?`<span class="conf">conf ${Math.round(n.confidence*100)}%</span>`:'';
    let h=`<h2>${esc(n.name)}</h2>`+
      `<div class="row"><span class="badge" style="background:${v.c}">${v.n} (${esc(n.truth_value)})</span>${conf}</div>`+
      `<div class="row sub">${esc(n.source||'?')} &middot; ${esc(n.certainty||'')} &middot; ${esc(n.type||'')}</div>`;
    if(n.evidence) h+=`<div class="row"><div class="k">Evidence</div><div class="v ev">${esc(n.evidence)}</div></div>`;
    if(r.contra.length) h+=`<div class="row"><div class="k" style="color:#f85149">⚠ Contradicts (${r.contra.length})</div>${relList(r.contra)}</div>`;
    if(r.rel.length) h+=`<div class="row"><div class="k">Related (${r.rel.length})</div>${relList(r.rel)}</div>`;
    if(n.path) h+=`<div class="row"><div class="k">Source path</div><div class="v">${esc(n.path)}</div></div>`;
    h+=`<div class="idfoot">${esc(n.id)}</div>`;
    side.innerHTML=h; side.classList.add('open');
    side.querySelectorAll('[data-goto]').forEach(el=> el.onclick=()=>selectNode(el.getAttribute('data-goto')));
  }
  function focusOn(ele){ cy.elements().addClass('faded'); ele.closedNeighborhood().removeClass('faded'); }
  function selectNode(id){ const ele=cy.$id(id); if(ele.empty())return;
    cy.elements().unselect(); ele.select(); focusOn(ele);
    cy.animate({center:{eles:ele}, zoom:Math.max(cy.zoom(),0.85)},{duration:300}); showNode(byId[id]); }

  cy.on('tap','node[tv]', e=>{ cy.elements().unselect(); e.target.select(); focusOn(e.target); showNode(e.target.data('full')); });
  cy.on('tap', e=>{ if(e.target===cy){ side.classList.remove('open'); cy.elements().removeClass('faded').unselect(); }});

  // hover tooltip with full claim
  const tip=document.getElementById('tip');
  cy.on('mouseover','node[tv]', e=>{ const p=e.target.renderedPosition();
    tip.textContent=e.target.data('full').name||''; tip.style.display='block';
    tip.style.left=(p.x+16)+'px'; tip.style.top=(p.y-12)+'px'; });
  cy.on('mouseout','node[tv]', ()=> tip.style.display='none');
  cy.on('pan zoom drag', ()=> tip.style.display='none');

  // filters
  const search=document.getElementById('search');
  const fContra=document.getElementById('f-contra'), fIso=document.getElementById('f-iso');
  function applyFilters(){
    const q=search.value.toLowerCase().trim(), cOnly=fContra.classList.contains('on'), hideIso=fIso.classList.contains('on');
    cy.nodes('[tv]').forEach(n=>{ const d=n.data('full'); let hide=false;
      if(hiddenV.has(d.truth_value)) hide=true;
      if(q && !((d.name||'').toLowerCase().includes(q))) hide=true;
      if(cOnly && !inContra.has(d.id)) hide=true;
      if(hideIso && (degree[d.id]||0)===0) hide=true;
      n.toggleClass('hidden', hide); });
    cy.edges().forEach(e=> e.toggleClass('hidden', e.source().hasClass('hidden')||e.target().hasClass('hidden')));
  }
  search.addEventListener('input', applyFilters);
  fContra.onclick=()=>{ fContra.classList.toggle('on'); applyFilters(); };
  fIso.onclick=()=>{ fIso.classList.toggle('on'); applyFilters(); };
  document.querySelectorAll('.chip').forEach(chip=> chip.onclick=()=>{
    const k=chip.getAttribute('data-v'); if(hiddenV.has(k)){hiddenV.delete(k);chip.classList.remove('off');}
    else{hiddenV.add(k);chip.classList.add('off');} applyFilters(); });

  // controls
  document.getElementById('b-fit').onclick=()=> cy.fit(null,55);
  document.getElementById('b-reset').onclick=()=>{
    search.value=''; hiddenV.clear(); fContra.classList.remove('on'); fIso.classList.remove('on');
    document.querySelectorAll('.chip').forEach(c=>c.classList.remove('off'));
    cy.elements().removeClass('hidden faded').unselect(); side.classList.remove('open'); cy.fit(null,55); };
})();
</script>
</body>
</html>
"####;
