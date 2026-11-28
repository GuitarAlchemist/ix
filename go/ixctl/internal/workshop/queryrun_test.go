package workshop

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const queriesFile = "-- Preamble, no query.\n\n-- 1) One row: a number and a label.\nSELECT 1 AS n, 'é' AS label;\n\n-- 2) Broken: no such table.\nSELECT * FROM nope;\n"

// queryRepo is an ix repo with one query file.
func queryRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	path := filepath.Join(repo, "docs", "streeling", "queries.sql")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(queriesFile), 0o644); err != nil {
		t.Fatal(err)
	}
	return repo
}

// TestAnApprovedDeviceRunsAListedQueryOnTheDesktop: any approved device asks the Desktop to run one of the listed
// queries, named by its file and title (owner's choice, 2026-11-16: the listed queries only, every approved device).
// The Desktop reads that query's SQL from the repo itself, never from the device, runs it with DuckDB in the repo's
// folder, and answers its rows, or DuckDB's own error.
func TestAnApprovedDeviceRunsAListedQueryOnTheDesktop(t *testing.T) {
	s, tablet, _ := fixture(t)
	repo := queryRepo(t)
	var ran []string
	runner := &QueryRunner{Repo: repo, Files: []string{"docs/streeling/queries.sql"},
		Run: func(ctx context.Context, dir, sql string) ([]byte, []byte, error) {
			if dir != repo {
				t.Errorf("queries run in the repo's folder, not %q", dir)
			}
			ran = append(ran, sql)
			if strings.Contains(sql, "nope") {
				return nil, []byte("Catalog Error: Table with name nope does not exist!\n"), errors.New("exit status 1")
			}
			return []byte(`[{"n":1,"label":"é"}]` + "\n"), nil, nil
		}}
	handler, err := Handler(s, "https://workshop.invalid", WithQueries(runner))
	if err != nil {
		t.Fatal(err)
	}
	run := func(token, body string) (int, map[string]json.RawMessage) {
		t.Helper()
		w := request(t, handler, "POST", "/workshop/api/queries/run", token, body)
		var reply map[string]json.RawMessage
		_ = json.Unmarshal(w.Body.Bytes(), &reply)
		return w.Code, reply
	}

	var me struct{ Queries bool }
	if w := request(t, handler, "GET", "/workshop/api/me", tablet, ""); json.Unmarshal(w.Body.Bytes(), &me) != nil || !me.Queries {
		t.Fatalf("an approved device must learn that the Desktop runs the listed queries: %s", w.Body)
	}
	code, reply := run(tablet, `{"source":"docs/streeling/queries.sql","title":"1) One row: a number and a label."}`)
	if code != 200 || string(reply["rows"]) != `[{"n":1,"label":"é"}]` || reply["error"] != nil {
		t.Fatalf("the tablet runs the first query: %d %v", code, reply)
	}
	var sql string
	_ = json.Unmarshal(reply["sql"], &sql)
	if sql != "SELECT 1 AS n, 'é' AS label;" || len(ran) != 1 || ran[0] != sql {
		t.Fatalf("the Desktop runs and answers the repo's SQL: answered %q, ran %q", sql, ran)
	}
	code, reply = run(tablet, `{"source":"docs/streeling/queries.sql","title":"2) Broken: no such table."}`)
	var duckErr string
	_ = json.Unmarshal(reply["error"], &duckErr)
	if code != 200 || duckErr != "Catalog Error: Table with name nope does not exist!" || reply["rows"] != nil {
		t.Fatalf("DuckDB's error comes back as it said it: %d %v", code, reply)
	}

	// The device names a query; it never sends SQL.
	if code, _ := run(tablet, `{"source":"docs/streeling/queries.sql","title":"1) One row: a number and a label.","sql":"COPY (SELECT 1) TO 'x.csv'"}`); code != 400 {
		t.Fatalf("SQL from a device must be refused: %d", code)
	}
	for _, body := range []string{
		`{"source":"docs/streeling/queries.sql","title":"3) Not there."}`,
		`{"source":"../secret.sql","title":"1) One row: a number and a label."}`,
		`{"source":"docs/other.sql","title":"1) One row: a number and a label."}`,
	} {
		if code, reply := run(tablet, body); code != 400 || string(reply["code"]) != `"query_unknown"` {
			t.Fatalf("a query that is not listed is refused (%s): %d %v", body, code, reply)
		}
	}
	if len(ran) != 2 {
		t.Fatalf("refused requests run nothing: ran %q", ran)
	}
	if code, _ := run("", `{"source":"docs/streeling/queries.sql","title":"1) One row: a number and a label."}`); code != 401 {
		t.Fatalf("an anonymous request is refused: %d", code)
	}
	invite, err := s.NewInvite("https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	_, pending, err := s.Claim(invite.Code, "Phone")
	if err != nil {
		t.Fatal(err)
	}
	if code, _ := run(pending, `{"source":"docs/streeling/queries.sql","title":"1) One row: a number and a label."}`); code != 403 {
		t.Fatalf("a device not yet approved is refused: %d", code)
	}
}

// TestAQueryIsBoundedInTimeAndSize: a query that runs too long is stopped, and a result too large to send is refused
// whole rather than cut (a cut JSON array is not a result).
func TestAQueryIsBoundedInTimeAndSize(t *testing.T) {
	s, tablet, _ := fixture(t)
	big := "[" + strings.Repeat(`{"n":1},`, maxQueryOutput/8) + `{"n":1}]`
	runner := &QueryRunner{Repo: queryRepo(t), Files: []string{"docs/streeling/queries.sql"}, Timeout: 50 * time.Millisecond,
		Run: func(ctx context.Context, dir, sql string) ([]byte, []byte, error) {
			if strings.Contains(sql, "nope") {
				<-ctx.Done()
				return nil, nil, ctx.Err()
			}
			return []byte(big), nil, nil
		}}
	handler, err := Handler(s, "https://workshop.invalid", WithQueries(runner))
	if err != nil {
		t.Fatal(err)
	}
	for title, want := range map[string]string{"2) Broken: no such table.": "query_timed_out", "1) One row: a number and a label.": "query_result_too_large"} {
		w := request(t, handler, "POST", "/workshop/api/queries/run", tablet, `{"source":"docs/streeling/queries.sql","title":"`+title+`"}`)
		var reply map[string]any
		_ = json.Unmarshal(w.Body.Bytes(), &reply)
		if w.Code != 200 || reply["error"] != want || reply["rows"] != nil {
			t.Fatalf("%s: %d %s", title, w.Code, w.Body)
		}
	}
}

// TestWithoutQueriesTheDesktopOffersNone: a Desktop not given the repo has no route and says so in /me.
func TestWithoutQueriesTheDesktopOffersNone(t *testing.T) {
	s, tablet, _ := fixture(t)
	handler, err := Handler(s, "https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	var me map[string]any
	_ = json.Unmarshal(request(t, handler, "GET", "/workshop/api/me", tablet, "").Body.Bytes(), &me)
	if _, ok := me["queries"]; ok {
		t.Fatalf("no queries offered: %v", me)
	}
	if w := request(t, handler, "POST", "/workshop/api/queries/run", tablet, `{"source":"a","title":"b"}`); w.Code != 404 {
		t.Fatalf("no route without the repo: %d", w.Code)
	}
}

// TestDuckDBRunsAQuery drives the real duckdb CLI when this machine has one (DUCKDB, or on PATH).
func TestDuckDBRunsAQuery(t *testing.T) {
	exe := os.Getenv("DUCKDB")
	if exe == "" {
		var err error
		if exe, err = exec.LookPath("duckdb"); err != nil {
			t.Skip("no duckdb CLI here")
		}
	}
	r := &QueryRunner{DuckDB: exe}
	out, errOut, err := r.run(context.Background(), t.TempDir(), "SELECT 1 AS n, 'é' AS label;")
	if err != nil || strings.TrimSpace(string(out)) != `[{"n":1,"label":"é"}]` {
		t.Fatalf("duckdb: %v %q %q", err, out, errOut)
	}
	_, errOut, err = r.run(context.Background(), t.TempDir(), "SELECT * FROM nope;")
	if err == nil || !strings.Contains(string(errOut), "nope") {
		t.Fatalf("duckdb must fail on a missing table: %v %q", err, errOut)
	}
}
