package workshop

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

// TestATabletRunsAQueryOnTheDesktop: on the tablet (an approved device, not the PC), a query of the list opens to its
// SQL and a button runs it on the Desktop. Its rows come back as a table; a query DuckDB refuses shows DuckDB's own
// message, here a pipeline that needs the ix extension.
func TestATabletRunsAQueryOnTheDesktop(t *testing.T) {
	repo := filepath.Join("testdata", "ix")
	abs, err := filepath.Abs(repo)
	if err != nil {
		t.Fatal(err)
	}
	runner := &QueryRunner{Repo: abs, Files: DefaultQueryFiles,
		Run: func(ctx context.Context, dir, sql string) ([]byte, []byte, error) {
			if strings.HasPrefix(sql, "SELECT repo, kind, count(*) AS n") {
				return []byte(`[{"repo":"ix","kind":"solution","n":42},{"repo":"ga","kind":"plan","n":7}]`), nil, nil
			}
			return nil, []byte("Catalog Error: Scalar Function with name ix_ndcg does not exist!"), errors.New("exit status 1")
		}}
	f := newBrowserFixture(t, WithQueries(runner))
	widget, err := QueriesWidget("Requêtes DuckDB", readTestQueries(t), testMade)
	if err != nil {
		t.Fatal(err)
	}
	tablet := f.join("Tablette")
	tablet.typeInto("new-name", string(widget))
	tablet.press("#new-list")
	tablet.until("the list", `document.querySelectorAll('#widget-host [data-query]').length===15`)

	const first = `document.querySelector('#widget-host [data-query]')`
	tablet.evalInto(`(()=>{`+first+`.querySelector('summary').click();return true})()`, nil)
	tablet.until("a run button", first+`.open && `+first+`.querySelector('[data-query-run]')?.textContent.trim()==='Exécuter sur le Desktop'`)
	tablet.evalInto(`(()=>{`+first+`.querySelector('[data-query-run]').click();return true})()`, nil)
	const table = first + `.querySelector('[data-query-result] table')`
	tablet.until("the rows as a table", table+` && [...`+table+`.querySelectorAll('th')].map(c=>c.textContent).join(' ')==='repo kind n' && [...`+table+`.querySelectorAll('tbody tr')].map(r=>[...r.cells].map(c=>c.textContent).join(' ')).join(', ')==='ix solution 42, ga plan 7'`)
	tablet.until("how many rows", first+`.querySelector('[data-query-status]').textContent.startsWith('2 lignes')`)
	tablet.screenshot("requete-executee.png")

	const pipeline = `[...document.querySelectorAll('#widget-host [data-query]')].find(d=>d.querySelector('summary').textContent.startsWith('Pipeline 1:'))`
	tablet.evalInto(`(()=>{const d=`+pipeline+`;d.querySelector('summary').click();d.querySelector('[data-query-run]').click();return true})()`, nil)
	tablet.until("DuckDB's own error", pipeline+`.querySelector('[data-query-error]')?.textContent==='Catalog Error: Scalar Function with name ix_ndcg does not exist!'`)
	tablet.screenshot("requete-refusee.png")
}

// TestNoRunButtonWhereTheDesktopRunsNoQueries: a Desktop without the repo offers no run button; the list stays one to
// read and copy.
func TestNoRunButtonWhereTheDesktopRunsNoQueries(t *testing.T) {
	f := newBrowserFixture(t)
	widget, err := QueriesWidget("Requêtes DuckDB", readTestQueries(t), testMade)
	if err != nil {
		t.Fatal(err)
	}
	tablet := f.join("Tablette")
	tablet.typeInto("new-name", string(widget))
	tablet.press("#new-list")
	tablet.until("the list, shared", `document.querySelectorAll('#widget-host [data-query]').length===15 && document.getElementById('tool-status').textContent.endsWith('partagé par le Desktop')`)
	var buttons int
	tablet.evalInto(`document.querySelectorAll('#widget-host [data-query-run]').length`, &buttons)
	if buttons != 0 {
		t.Fatalf("no run button without queries on the Desktop: %d", buttons)
	}
}
