package workshop

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
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

// QueriesWidget is the ixforge-widget/1 block IxForge imports. It says when it was generated (stored in UTC) and from
// which commit of the repo the queries were read.
func QueriesWidget(title string, entries []QueryEntry, generatedAt time.Time, sourceCommit string) ([]byte, error) {
	w := Widget{ID: "w", Title: title, Kind: "queries", Entries: entries, GeneratedAt: generatedAt.UTC().Format(time.RFC3339), SourceCommit: sourceCommit}
	if !titleOK(title) || !queriesOK(w) {
		return nil, errors.New("queries_invalid")
	}
	return json.Marshal(struct {
		Format       string       `json:"format"`
		Kind         string       `json:"kind"`
		Title        string       `json:"title"`
		Entries      []QueryEntry `json:"entries"`
		GeneratedAt  string       `json:"generated_at"`
		SourceCommit string       `json:"source_commit"`
	}{"ixforge-widget/1", "queries", title, entries, w.GeneratedAt, sourceCommit})
}

// A source commit is a git object name, abbreviated or full.
var commitName = regexp.MustCompile(`^[0-9a-f]{7,40}$`)

// generationOK: a queries widget may say when it was generated, as a UTC RFC 3339 time, and from which commit.
func generationOK(w Widget) bool {
	if w.GeneratedAt != "" {
		at, err := time.Parse(time.RFC3339, w.GeneratedAt)
		if err != nil || at.Location() != time.UTC || !strings.HasSuffix(w.GeneratedAt, "Z") {
			return false
		}
	}
	return w.SourceCommit == "" || commitName.MatchString(w.SourceCommit)
}

// queriesOK says whether a widget's queries are fit to keep: only a queries widget has any (and a generation), at
// least one and at most 100, each titled, from a named file, with SQL of at most 8 KiB.
func queriesOK(w Widget) bool {
	if w.Kind != "queries" {
		return len(w.Entries) == 0 && w.GeneratedAt == "" && w.SourceCommit == ""
	}
	if w.InitialValue != 0 || len(w.Entries) == 0 || len(w.Entries) > maxQueries || !generationOK(w) {
		return false
	}
	for _, e := range w.Entries {
		if !titleOK(e.Title) || !titleOK(e.Source) || strings.TrimSpace(e.SQL) == "" || len(e.SQL) > maxQuerySQL || !utf8.ValidString(e.SQL) {
			return false
		}
	}
	return true
}
