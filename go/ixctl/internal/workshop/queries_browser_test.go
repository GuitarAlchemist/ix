package workshop

import (
	"encoding/json"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestTheDuckDBQueriesWidgetListsOurQueries: the queries widget, made by ixforge-queries from the ix repo, is imported
// on the linked PC. It lists the 15 queries by file; a query opens to its SQL, which one button copies. Nothing can be
// typed into it. The Desktop keeps it, so another device shows the same list, and it exports as a page that works by
// itself.
func TestTheDuckDBQueriesWidgetListsOurQueries(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the browser fixture runs on Windows")
	}
	f := newBrowserFixture(t)
	widget, err := QueriesWidget("Requêtes DuckDB", readTestQueries(t), testMade)
	if err != nil {
		t.Fatal(err)
	}
	link, err := f.store.NewLinkInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	pc, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	pc.navigate(link.URL)
	pc.until("linked", `document.getElementById('connection').textContent==='Desktop connecté'`)
	settle(t, pc)

	pc.typeInto("new-name", string(widget))
	pc.press("#new-list")
	const groups = `[...document.querySelectorAll('#widget-host [data-query-source]')].map(s=>s.dataset.querySource+':'+s.querySelectorAll('[data-query]').length).join(', ')`
	pc.until("the queries, by file", `document.getElementById('tool-title').value==='Requêtes DuckDB' && `+groups+`==='docs/streeling/queries.sql:5, docs/value/queries.sql:3, docs/duck/pipelines.sql:7'`)
	pc.until("counted in the rail", `[...document.querySelectorAll('#tools [data-tool]')].find(b=>b.title==='Requêtes DuckDB').querySelector('.badge').textContent==='15'`)
	var inputs int
	pc.evalInto(`document.querySelectorAll('#widget-host input, #widget-host textarea').length`, &inputs)
	if inputs != 0 {
		t.Errorf("the queries widget is read-only: %d fields", inputs)
	}

	// A query opens to its SQL; the button copies it.
	pc.evalInto(`(()=>{window.__copied=null;navigator.clipboard.writeText=async s=>{window.__copied=s};return true})()`, nil)
	const first = `document.querySelector('#widget-host [data-query]')`
	pc.evalInto(`(()=>{`+first+`.querySelector('summary').click();return true})()`, nil)
	pc.until("the SQL shown", first+`.open && `+first+`.querySelector('pre').textContent.startsWith('SELECT repo, kind, count(*) AS n')`)
	pc.evalInto(`(()=>{`+first+`.querySelector('[data-query-copy]').click();return true})()`, nil)
	pc.until("the SQL copied", `window.__copied!==null && window.__copied.startsWith('SELECT repo, kind, count(*) AS n') && window.__copied.endsWith('ORDER BY n DESC;') && `+first+`.querySelector('[data-query-copied]').textContent==='Copié'`)
	pc.until("kept by the Desktop", `document.getElementById('tool-status').textContent==='15 requêtes · partagé par le Desktop'`)
	// How fresh it is: when it was generated (date and time), from which ix commit (a link to it, with its subject), and
	// when this device last received it from the Desktop.
	const made = `document.querySelector('#widget-host [data-query-made]')`
	pc.until("says when and from what it was made", `/^Générée le .*2026 à \d\d:\d\d depuis ix 8c97be7 \(docs: number our DuckDB queries\)$/.test(`+made+`?.textContent.replace(/\s+/g,' ').trim())`)
	var link string
	pc.evalInto(made+`.querySelector('a[data-query-commit]')?.href||''`, &link)
	if link != "https://github.com/GuitarAlchemist/ix/commit/8c97be79" {
		t.Errorf("the commit must link to it on GitHub: %q", link)
	}
	pc.until("says when this device last synced", `/^Synchronisée le .*2026.* \d\d:\d\d$/.test(document.querySelector('#widget-host [data-query-synced]')?.textContent.replace(/\s+/g,' ').trim())`)
	pc.screenshot("requetes-duckdb.png")

	// The export: a page that lists the same queries, offline.
	pc.evalInto(`(()=>{window.__exported=null;URL.createObjectURL=b=>{window.__exported=b;return 'about:blank'};HTMLAnchorElement.prototype.click=function(){};return true})()`, nil)
	pc.click("export")
	var exported string
	pc.evalInto(`(async()=>{for(let i=0;i<50&&!window.__exported;i++)await new Promise(r=>setTimeout(r,100));return window.__exported?await window.__exported.text():''})()`, &exported)
	if !strings.Contains(exported, `"kind":"queries"`) || !strings.Contains(exported, "ix_hll_merge") {
		t.Fatalf("the export must carry the queries")
	}

	// Another device shows the list the Desktop kept.
	tablet := f.join("Tablette")
	tablet.until("the tool reaches the tablet", `[...document.querySelectorAll('#tools [data-tool]')].some(b=>b.title==='Requêtes DuckDB')`)
	tablet.evalInto(`(()=>{[...document.querySelectorAll('#tools [data-tool]')].find(b=>b.title==='Requêtes DuckDB').click();return true})()`, nil)
	tablet.until("the same queries on the tablet", groups+`==='docs/streeling/queries.sql:5, docs/value/queries.sql:3, docs/duck/pipelines.sql:7'`)
}

// TestReimportingOurQueriesReplacesThem: when the queries change in ix, the owner generates the widget again and
// imports it. The queries widget of the same name takes the new queries and says it was made from the new commit; no
// second tool appears, on this device or on the Desktop.
func TestReimportingOurQueriesReplacesThem(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the browser fixture runs on Windows")
	}
	f := newBrowserFixture(t)
	entries := readTestQueries(t)
	before, err := QueriesWidget("Requêtes DuckDB", entries[:2], Generation{At: testMadeAt.Add(-24 * time.Hour), Commit: "1111111"})
	if err != nil {
		t.Fatal(err)
	}
	after, err := QueriesWidget("Requêtes DuckDB", entries, testMade)
	if err != nil {
		t.Fatal(err)
	}
	link, err := f.store.NewLinkInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	pc, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	pc.navigate(link.URL)
	pc.until("linked", `document.getElementById('connection').textContent==='Desktop connecté'`)
	settle(t, pc)

	const ours = `[...document.querySelectorAll('#tools [data-tool]')].filter(b=>b.title==='Requêtes DuckDB')`
	const status = `document.getElementById('tool-status').textContent`
	pc.typeInto("new-name", string(before))
	pc.press("#new-list")
	pc.until("the first list, kept by the Desktop", ours+`.length===1 && `+ours+`[0].querySelector('.badge').textContent==='2' && `+status+`==='2 requêtes · partagé par le Desktop'`)

	pc.typeInto("new-name", string(after))
	pc.press("#new-list")
	pc.until("the same tool, with the new queries, kept by the Desktop", ours+`.length===1 && `+ours+`[0].querySelector('.badge').textContent==='15' && `+status+`==='15 requêtes · partagé par le Desktop'`)
	pc.until("made from the new commit", `/depuis ix 8c97be7 \(docs: number our DuckDB queries\)$/.test(document.querySelector('#widget-host [data-query-made]')?.textContent.replace(/\s+/g,' ').trim())`)

	rows, err := f.store.db.Query(`SELECT document FROM workshop_projects`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var widgets []Widget
	for rows.Next() {
		var raw string
		var p Project
		if err := rows.Scan(&raw); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatal(err)
		}
		for _, ws := range p.Workspaces {
			for _, w := range ws.Widgets {
				if w.Kind == "queries" {
					widgets = append(widgets, w)
				}
			}
		}
	}
	if len(widgets) != 1 || len(widgets[0].Entries) != 15 || widgets[0].SourceCommit != testSourceCommit || widgets[0].GeneratedAt != "2026-11-15T01:02:03Z" {
		t.Fatalf("the Desktop must keep one queries widget, the new one: %d widgets %+v", len(widgets), widgets)
	}
}
