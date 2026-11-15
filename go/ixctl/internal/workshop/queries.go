package workshop

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"
)

// A queries widget lists SQL queries for reading and copying: our DuckDB queries, read from the ix repo by
// cmd/ixforge-queries. It runs nothing and has no IXLET; its queries live in the widget itself, so the Desktop keeps
// them with the project and every device shows them.

// QueryEntry is one query: its heading, the file it comes from, and its SQL as written there.
type QueryEntry struct {
	Title  string `json:"title"`
	Source string `json:"source"`
	SQL    string `json:"sql"`
}

const (
	maxQueries  = 100
	maxQuerySQL = 8 << 10
)

// DefaultQueryFiles are the ix repo's numbered DuckDB query files, relative to its root.
var DefaultQueryFiles = []string{"docs/streeling/queries.sql", "docs/value/queries.sql", "docs/duck/pipelines.sql"}

// A heading is a comment that numbers a query ("-- 1) …", "-- 1. …") or a pipeline ("-- ── Pipeline 1: … ──").
var queryHeading = regexp.MustCompile(`^--\s*(?:(\d+[.)]\s+.+?)|── (Pipeline \d+:.+?)\s*─*)\s*$`)

// ParseQueries splits one .sql file into its numbered queries. What precedes the first heading (the file's preamble)
// belongs to no query.
func ParseQueries(source, text string) []QueryEntry {
	var entries []QueryEntry
	var body []string
	flush := func() {
		if len(entries) > 0 {
			entries[len(entries)-1].SQL = strings.Trim(strings.Join(body, "\n"), "\n")
		}
		body = nil
	}
	for _, line := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		if m := queryHeading.FindStringSubmatch(line); m != nil {
			flush()
			entries = append(entries, QueryEntry{Title: m[1] + m[2], Source: source})
			continue
		}
		body = append(body, strings.TrimRight(line, " \t"))
	}
	flush()
	return entries
}

// ReadQueries reads the given files of a repo, in order.
func ReadQueries(root string, files []string) ([]QueryEntry, error) {
	var entries []QueryEntry
	for _, name := range files {
		raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
		if err != nil {
			return nil, err
		}
		entries = append(entries, ParseQueries(name, string(raw))...)
	}
	return entries, nil
}

// QueriesWidget is the ixforge-widget/1 block IxForge imports.
func QueriesWidget(title string, entries []QueryEntry) ([]byte, error) {
	w := Widget{ID: "w", Title: title, Kind: "queries", Entries: entries}
	if !titleOK(title) || !queriesOK(w) {
		return nil, errors.New("queries_invalid")
	}
	return json.Marshal(struct {
		Format  string       `json:"format"`
		Kind    string       `json:"kind"`
		Title   string       `json:"title"`
		Entries []QueryEntry `json:"entries"`
	}{"ixforge-widget/1", "queries", title, entries})
}

// queriesOK says whether a widget's queries are fit to keep: only a queries widget has any, at least one and at most
// 100, each titled, from a named file, with SQL of at most 8 KiB.
func queriesOK(w Widget) bool {
	if w.Kind != "queries" {
		return len(w.Entries) == 0
	}
	if w.InitialValue != 0 || len(w.Entries) == 0 || len(w.Entries) > maxQueries {
		return false
	}
	for _, e := range w.Entries {
		if !titleOK(e.Title) || !titleOK(e.Source) || strings.TrimSpace(e.SQL) == "" || len(e.SQL) > maxQuerySQL || !utf8.ValidString(e.SQL) {
			return false
		}
	}
	return true
}
