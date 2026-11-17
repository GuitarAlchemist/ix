package workshop

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/repograph"
)

const graphFile = "// Preamble, no query.\n\n// 1) Files.\nMATCH (f:File) RETURN f.path AS file;\n\n// 2) Broken: no such table.\nMATCH (n:Nope) RETURN n;\n"

// graphRepo is an ix repo with one Cypher query file and the given graph builds in state/graph: each is finished or
// not, oldest first. It answers the repo and the database of each build.
func graphRepo(t *testing.T, finished ...bool) (string, []string) {
	t.Helper()
	repo := t.TempDir()
	path := filepath.Join(repo, "docs", "streeling", "graph.cypher")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(graphFile), 0o644); err != nil {
		t.Fatal(err)
	}
	var dbs []string
	at := time.Date(2026, 11, 16, 20, 0, 0, 0, time.UTC)
	for i, done := range finished {
		dir, err := repograph.NewBuild(filepath.Join(repo, "state", "graph"), at.Add(time.Duration(i)*time.Hour))
		if err != nil {
			t.Fatal(err)
		}
		if done {
			if err := repograph.Finish(dir); err != nil {
				t.Fatal(err)
			}
		}
		dbs = append(dbs, filepath.Join(dir, "db"))
	}
	return repo, dbs
}

// TestTheDesktopRunsAListedCypherQueryOnTheNewestGraph: a listed query of a .cypher file runs with the LadybugDB CLI
// on the newest finished build of the repo's graph (state/graph), never on one still being built. LadybugDB reports an
// error on its output and still exits 0: that line comes back as the query's error, not as rows. Without the CLI, or
// without a finished build, the Desktop says so by code.
func TestTheDesktopRunsAListedCypherQueryOnTheNewestGraph(t *testing.T) {
	if got := ParseQueries("docs/streeling/graph.cypher", graphFile); len(got) != 2 || got[0].Title != "1) Files." || got[0].SQL != "MATCH (f:File) RETURN f.path AS file;" {
		t.Fatalf("a .cypher file is read like a .sql one, its headings being // comments: %+v", got)
	}
	s, tablet, _ := fixture(t)
	repo, dbs := graphRepo(t, true, true, false)
	var ran []string
	runner := &QueryRunner{Repo: repo, Files: []string{"docs/streeling/graph.cypher"}, Lbug: "lbug.exe", Graphs: filepath.Join(repo, "state", "graph"),
		RunGraph: func(ctx context.Context, db, query string) ([]byte, error) {
			ran = append(ran, db)
			if strings.Contains(query, "Nope") {
				return []byte("Error: Binder exception: Table Nope does not exist.\n"), nil
			}
			return []byte("[\n{\"file\":\"go/a.go\"}\n]\n"), nil
		}}
	handler, err := Handler(s, "https://workshop.invalid", WithQueries(runner))
	if err != nil {
		t.Fatal(err)
	}
	run := func(title string) map[string]json.RawMessage {
		t.Helper()
		w := request(t, handler, "POST", "/workshop/api/queries/run", tablet, `{"source":"docs/streeling/graph.cypher","title":"`+title+`"}`)
		var reply map[string]json.RawMessage
		if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &reply) != nil {
			t.Fatalf("%s: %d %s", title, w.Code, w.Body)
		}
		return reply
	}
	if reply := run("1) Files."); strings.Join(strings.Fields(string(reply["rows"])), "") != `[{"file":"go/a.go"}]` || reply["error"] != nil {
		t.Fatalf("rows of the graph: %v", reply)
	}
	if len(ran) != 1 || ran[0] != dbs[1] {
		t.Fatalf("ran on %q, want the newest finished build %q", ran, dbs[1])
	}
	var lbugErr string
	if reply := run("2) Broken: no such table."); json.Unmarshal(reply["error"], &lbugErr) != nil || lbugErr != "Error: Binder exception: Table Nope does not exist." || reply["rows"] != nil {
		t.Fatalf("LadybugDB's error comes back as an error: %v", reply)
	}

	runner.Lbug = ""
	if reply := run("1) Files."); string(reply["error"]) != `"graph_engine_missing"` {
		t.Fatalf("no LadybugDB CLI named: %v", reply)
	}
	empty, _ := graphRepo(t, false)
	runner.Lbug, runner.Repo, runner.Graphs = "lbug.exe", empty, filepath.Join(empty, "state", "graph")
	if reply := run("1) Files."); string(reply["error"]) != `"graph_missing"` {
		t.Fatalf("no finished build: %v", reply)
	}
	if len(ran) != 2 {
		t.Fatalf("a refused run runs nothing: %q", ran)
	}
}

// TestLadybugRunsAQuery drives the real LadybugDB CLI named by IXFORGE_LBUG on a graph built in the repo.
func TestLadybugRunsAQuery(t *testing.T) {
	lbug := os.Getenv("IXFORGE_LBUG")
	if lbug == "" {
		t.Skip("IXFORGE_LBUG names no lbug.exe")
	}
	repo, dbs := graphRepo(t, true)
	cmd := exec.Command(lbug, dbs[0], "-b", "-s")
	cmd.Stdin = strings.NewReader("CREATE NODE TABLE File(id STRING PRIMARY KEY, path STRING);\nCREATE (:File {id: 'ix:go/a.go', path: 'go/a.go'});\n")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("lbug: %v %s", err, out)
	}
	r := &QueryRunner{Repo: repo, Files: []string{"docs/streeling/graph.cypher"}, Lbug: lbug, Graphs: filepath.Join(repo, "state", "graph")}
	got, err := r.Query(context.Background(), "docs/streeling/graph.cypher", "1) Files.")
	if err != nil || got.Error != "" || strings.Join(strings.Fields(string(got.Rows)), "") != `[{"file":"go/a.go"}]` {
		t.Fatalf("lbug rows: %+v %v", got, err)
	}
	got, err = r.Query(context.Background(), "docs/streeling/graph.cypher", "2) Broken: no such table.")
	if err != nil || !strings.Contains(got.Error, "Nope") || got.Rows != nil {
		t.Fatalf("lbug error: %+v %v", got, err)
	}
}
