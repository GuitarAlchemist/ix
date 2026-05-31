//! Self-contained 2D viewer for the assumption graph.
//!
//! [`render`] inlines the Prime-Radiant graph JSON into a single static HTML
//! page (Cytoscape.js from CDN) — double-click to open, no server, no build.
//! Nodes are claims colored by hexavalent verdict, grouped by namespace via
//! Cytoscape compound nodes; `contradicts` edges are red + dashed; clicking a
//! node opens a detail panel. Emitted by `ix-assumption-graph-report --html`.

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
  :root { --bg:#0d1117; --panel:#161b22; --fg:#e6edf3; --muted:#8b949e; --border:#30363d; }
  * { box-sizing:border-box; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
    font:13px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  #app { display:flex; flex-direction:column; height:100%; }
  header { padding:10px 16px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .stats { color:var(--muted); }
  .stats b { color:var(--fg); }
  .legend { display:flex; gap:11px; flex-wrap:wrap; margin-left:auto; }
  .legend span { display:flex; align-items:center; gap:5px; color:var(--muted); font-size:12px; }
  .dot { width:11px; height:11px; border-radius:50%; display:inline-block; }
  #main { flex:1; display:flex; min-height:0; position:relative; }
  #cy { flex:1; min-width:0; }
  #side { width:330px; border-left:1px solid var(--border); background:var(--panel);
    padding:16px; overflow:auto; display:none; }
  #side.open { display:block; }
  #side h2 { font-size:14px; margin:0 0 12px; line-height:1.35; }
  #side .row { margin:9px 0; }
  #side .k { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.5px; }
  #side .v { word-break:break-word; }
  .badge { display:inline-block; padding:2px 9px; border-radius:11px; font-weight:700; color:#0d1117; }
  #search { background:var(--panel); color:var(--fg); border:1px solid var(--border);
    border-radius:6px; padding:5px 10px; width:220px; outline:none; }
  #search:focus { border-color:#58a6ff; }
  #empty { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    color:var(--muted); text-align:center; padding:40px; font-size:14px; }
  code { background:#161b22; padding:1px 5px; border-radius:4px; font-size:12px; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>Assumption Graph</h1>
    <span class="stats" id="stats"></span>
    <input id="search" placeholder="filter claims…" autocomplete="off">
    <span class="legend" id="legend"></span>
  </header>
  <div id="main">
    <div id="cy"></div>
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
  const contradicts = edges.filter(e=>e.relation==='contradicts').length;
  document.getElementById('stats').innerHTML =
    `<b>${nodes.length}</b> claims &middot; <b>${edges.length}</b> links &middot; ` +
    `<b style="color:#f85149">${contradicts}</b> contradictions`;
  document.getElementById('legend').innerHTML = Object.values(VERDICT)
    .map(v=>`<span><i class="dot" style="background:${v.c}"></i>${v.n}</span>`).join('');

  if(!nodes.length){
    document.getElementById('main').insertAdjacentHTML('beforeend',
      '<div id="empty">No assumptions yet.<br>Add <code>@ai:</code> annotations (or research claims), then re-run ' +
      '<code>ix-assumption-graph-report . --html out.html</code>.</div>');
    return;
  }

  const namespaces = [...new Set(nodes.map(n=>n.type||'(none)'))];
  const els = [];
  namespaces.forEach(ns => els.push({data:{id:'ns:'+ns, label:ns, isGroup:1}}));
  nodes.forEach(n => els.push({data:{
    id:n.id, parent:'ns:'+(n.type||'(none)'),
    label:(n.name||'').length>40 ? (n.name||'').slice(0,38)+'…' : (n.name||''),
    tv:n.truth_value||'U', full:n }}));
  edges.forEach((e,i) => els.push({data:{
    id:'e'+i, source:e.source, target:e.target, rel:e.relation||'',
    contradicts: e.relation==='contradicts' ? 1 : 0 }}));

  // Deterministic namespace-column layout: one column per namespace, claims
  // stacked within it. Readable and stable (no force-layout squish); cross-
  // column edges surface cross-namespace links (e.g. a research claim
  // contradicting a code assumption).
  const byNs = {};
  nodes.forEach(n => { const k='ns:'+(n.type||'(none)'); (byNs[k]=byNs[k]||[]).push(n.id); });
  const COLW=240, ROWH=96, X0=120, Y0=140, pos={};
  Object.keys(byNs).forEach((k,ci)=> byNs[k].forEach((id,ri)=> { pos[id]={x:X0+ci*COLW, y:Y0+ri*ROWH}; }));

  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: els,
    wheelSensitivity: 0.2,
    style: [
      {selector:'$node > node', style:{}},
      {selector:'node[isGroup]', style:{
        'background-color':'#161b22','background-opacity':0.45,'border-color':'#30363d','border-width':1,
        'label':'data(label)','color':'#8b949e','font-size':11,'text-valign':'top','text-halign':'center',
        'padding':'16px','shape':'round-rectangle','text-margin-y':-3 }},
      {selector:'node[tv]', style:{
        'background-color': n => (VERDICT[n.data('tv')]||VERDICT.U).c,
        'label':'data(label)','color':'#0d1117','font-size':9,'font-weight':'bold',
        'text-valign':'center','text-halign':'center','text-wrap':'wrap','text-max-width':'72px',
        'width':56,'height':56,'border-width':2,'border-color':'#0d1117' }},
      {selector:'node[tv]:selected', style:{'border-color':'#58a6ff','border-width':4}},
      {selector:'edge', style:{
        'width':1.5,'line-color':'#484f58','curve-style':'bezier',
        'target-arrow-color':'#484f58','target-arrow-shape':'triangle','arrow-scale':0.8 }},
      {selector:'edge[?contradicts]', style:{
        'width':3,'line-color':'#f85149','line-style':'dashed',
        'target-arrow-color':'#f85149','target-arrow-shape':'triangle','arrow-scale':1 }},
      {selector:'.dim', style:{'opacity':0.1}}
    ],
    layout:{ name:'preset', positions: pos, fit:true, padding:60 }
  });

  const side = document.getElementById('side');
  cy.on('tap','node[tv]', evt => {
    const n = evt.target.data('full'), v = VERDICT[n.truth_value]||VERDICT.U;
    side.classList.add('open');
    side.innerHTML =
      `<h2>${esc(n.name)}</h2>` +
      `<div class="row"><span class="badge" style="background:${v.c}">${v.n} &nbsp;(${esc(n.truth_value)})</span></div>` +
      `<div class="row"><div class="k">kind</div><div class="v">${esc(n.kind)}</div></div>` +
      `<div class="row"><div class="k">namespace</div><div class="v">${esc(n.type)}</div></div>` +
      `<div class="row"><div class="k">domain</div><div class="v">${esc(n.domain)}</div></div>` +
      `<div class="row"><div class="k">source</div><div class="v">${esc(n.path)||'&mdash;'}</div></div>` +
      `<div class="row"><div class="k">id</div><div class="v" style="font-family:monospace;font-size:10px;color:#8b949e">${esc(n.id)}</div></div>`;
  });
  cy.on('tap', evt => { if(evt.target===cy){ side.classList.remove('open'); cy.elements().removeClass('dim').unselect(); }});

  document.getElementById('search').addEventListener('input', e => {
    const q = e.target.value.toLowerCase().trim();
    if(!q){ cy.elements().removeClass('dim'); return; }
    cy.nodes('[tv]').forEach(n => n.toggleClass('dim', !(n.data('full').name||'').toLowerCase().includes(q)));
  });
})();
</script>
</body>
</html>
"####;
