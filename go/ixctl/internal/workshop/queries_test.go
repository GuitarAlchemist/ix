package workshop

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The DuckDB queries of the ix repo, copied verbatim into testdata/ix (ix 8c97be79).
func readTestQueries(t *testing.T) []QueryEntry {
	t.Helper()
	entries, err := ReadQueries(filepath.Join("testdata", "ix"), DefaultQueryFiles)
	if err != nil {
		t.Fatal(err)
	}
	return entries
}

// TestTheDuckDBQueriesAreReadFromTheirFiles: each numbered query of the three .sql files of the ix repo is one entry,
// with its heading as title, its file as source, and the SQL under the heading (its comments included) as it is
// written, ready to paste into DuckDB.
func TestTheDuckDBQueriesAreReadFromTheirFiles(t *testing.T) {
	entries := readTestQueries(t)
	count := map[string]int{}
	for _, e := range entries {
		count[e.Source]++
	}
	if len(entries) != 15 || count["docs/streeling/queries.sql"] != 5 || count["docs/value/queries.sql"] != 3 || count["docs/duck/pipelines.sql"] != 7 {
		t.Fatalf("want 5 + 3 + 7 queries, got %d: %v", len(entries), count)
	}
	first, value, pipeline6 := entries[0], entries[5], entries[13]
	if first.Title != "1) Enrollment: how many learnings per repo and kind (cross-repo overview)." || !strings.HasPrefix(first.SQL, "SELECT repo, kind, count(*) AS n\n") || !strings.HasSuffix(first.SQL, "ORDER BY n DESC;") {
		t.Errorf("first query: %+v", first)
	}
	if value.Title != "1. Top demos across the ecosystem by continuous score (excludes repo rollups)." || value.Source != "docs/value/queries.sql" {
		t.Errorf("first value query: %+v", value)
	}
	if pipeline6.Title != "Pipeline 6: probabilistic sketches (build → blob → probe / merge)" || !strings.Contains(pipeline6.SQL, "-- 6a. Bloom") || !strings.Contains(pipeline6.SQL, "ix_hll_merge") {
		t.Errorf("pipeline 6 keeps both its parts (6a, 6b): %+v", pipeline6)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.SQL, "\n") || strings.HasSuffix(e.SQL, "\n") || strings.Contains(e.SQL, "── Pipeline") {
			t.Errorf("%q: the SQL must hold its own lines only, trimmed: %q", e.Title, e.SQL)
		}
	}
}

// The generation the tests stamp on a queries widget: when, and from which ix commit (the one testdata/ix copies).
var testMadeAt = time.Date(2026, 11, 15, 1, 2, 3, 0, time.UTC)

const (
	testSourceCommit = "8c97be79"
	testSubject      = "docs: number our DuckDB queries"
)

var testMade = Generation{At: testMadeAt, Commit: testSourceCommit, Subject: testSubject}

// TestTheQueriesWidgetIsAnIxForgeWidget: the generator writes an ixforge-widget/1 block of kind "queries", which
// IxForge imports like any other widget. It says when it was made and from which ix commit (with its subject), so a
// reader knows how fresh the list is.
func TestTheQueriesWidgetIsAnIxForgeWidget(t *testing.T) {
	raw, err := QueriesWidget("Requêtes DuckDB", readTestQueries(t), Generation{At: testMadeAt.In(time.FixedZone("EST", -5*3600)), Commit: testSourceCommit, Subject: testSubject})
	if err != nil {
		t.Fatal(err)
	}
	var data struct {
		Format, Kind, Title string
		Entries             []QueryEntry
		GeneratedAt         string `json:"generated_at"`
		SourceCommit        string `json:"source_commit"`
		SourceSubject       string `json:"source_subject"`
	}
	if err := json.Unmarshal(raw, &data); err != nil || data.Format != "ixforge-widget/1" || data.Kind != "queries" || data.Title != "Requêtes DuckDB" || len(data.Entries) != 15 {
		t.Fatalf("%s %v", raw, err)
	}
	if data.GeneratedAt != "2026-11-15T01:02:03Z" || data.SourceCommit != testSourceCommit || data.SourceSubject != testSubject {
		t.Errorf("made at %q (want UTC 2026-11-15T01:02:03Z) from %q (%q)", data.GeneratedAt, data.SourceCommit, data.SourceSubject)
	}
	if _, err := QueriesWidget("Vide", nil, testMade); err == nil {
		t.Error("a queries widget with no query must be refused")
	}
	if _, err := QueriesWidget("Requêtes DuckDB", readTestQueries(t), Generation{At: testMadeAt, Commit: "main"}); err == nil {
		t.Error("a source that is not a commit must be refused")
	}
}

// TestTheDesktopKeepsAQueriesWidget: the Desktop saves a project holding a queries widget (read-only, no IXLET) and
// refuses one it could not show or that would carry more than its share.
func TestTheDesktopKeepsAQueriesWidget(t *testing.T) {
	entries := readTestQueries(t)
	op := func(change func(*Widget, *Workspace)) Operation {
		w := Widget{ID: "w-q", Title: "Requêtes DuckDB", Kind: "queries", Entries: append([]QueryEntry(nil), entries...)}
		ws := Workspace{ID: "ws", Title: "Requêtes DuckDB", Widgets: []Widget{w}, IXLETDefinitions: []IXLETDefinition{}}
		change(&ws.Widgets[0], &ws)
		return Operation{OpID: "op-1", Project: Project{ID: "p-q", Title: "Requêtes DuckDB", Workspaces: []Workspace{ws}}}
	}
	if err := validate(op(func(*Widget, *Workspace) {})); err != nil {
		t.Fatalf("a queries widget must be accepted: %v", err)
	}
	if err := validate(op(func(w *Widget, _ *Workspace) {
		w.GeneratedAt, w.SourceCommit, w.SourceSubject = "2026-11-15T01:02:03Z", testSourceCommit, testSubject
	})); err != nil {
		t.Fatalf("a queries widget that says when and from what it was made must be accepted: %v", err)
	}
	long := strings.Repeat("x", 8193)
	refused := map[string]func(*Widget, *Workspace){
		"no query":          func(w *Widget, _ *Workspace) { w.Entries = nil },
		"empty SQL":         func(w *Widget, _ *Workspace) { w.Entries[0].SQL = "  " },
		"SQL over 8 KiB":    func(w *Widget, _ *Workspace) { w.Entries[0].SQL = long },
		"untitled query":    func(w *Widget, _ *Workspace) { w.Entries[0].Title = "" },
		"over 100 queries":  func(w *Widget, _ *Workspace) { w.Entries = make([]QueryEntry, 101) },
		"initial value":     func(w *Widget, _ *Workspace) { w.InitialValue = 1 },
		"queries on a list": func(w *Widget, _ *Workspace) { w.Kind = "checklist" },
		"an IXLET for queries": func(_ *Widget, ws *Workspace) {
			ws.IXLETDefinitions = []IXLETDefinition{{ID: "d", Title: "Requêtes DuckDB", Kind: "queries", Status: "draft"}}
		},
		"made at no time":      func(w *Widget, _ *Workspace) { w.GeneratedAt = "hier" },
		"made at a local time": func(w *Widget, _ *Workspace) { w.GeneratedAt = "2026-11-14T20:02:03-05:00" },
		"made from a branch":   func(w *Widget, _ *Workspace) { w.SourceCommit = "main" },
		"a blank subject":      func(w *Widget, _ *Workspace) { w.SourceSubject = "  " },
		"a subject over 240 B": func(w *Widget, _ *Workspace) { w.SourceSubject = strings.Repeat("x", 241) },
		"a list's subject": func(w *Widget, _ *Workspace) {
			w.Kind, w.Entries, w.SourceSubject = "checklist", nil, testSubject
		},
		"a list's generation": func(w *Widget, _ *Workspace) {
			w.Kind, w.Entries, w.SourceCommit = "checklist", nil, testSourceCommit
		},
	}
	for name, change := range refused {
		if err := validate(op(change)); !errors.Is(err, ErrInvalid) {
			t.Errorf("%s: want invalid_document, got %v", name, err)
		}
	}
}
