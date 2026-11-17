// Command ixforge-graph builds the LadybugDB graph of the ix repo (see internal/repograph): the Streeling learnings
// with their repo and tags, the repo's recent commits with the files they changed, and the files each learning names.
// Then, with -queries, it runs the numbered queries of a .cypher file (// N) title, then the query) and prints each
// one's rows with its time.
//
//	go run ./cmd/ixforge-graph -ix C:\Users\<you>\source\repos\ix -lbug <path to lbug.exe> -out <empty or absent dir>
//	go run ./cmd/ixforge-graph ... -queries C:\Users\<you>\source\repos\ix\docs\streeling\graph.cypher
//
// The output folder holds the CSV files, build.cypher and the database (db). Nothing is fetched; the repo is only read.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/repograph"
)

func main() {
	ix := flag.String("ix", "", "the ix repo (absolute)")
	lbug := flag.String("lbug", "", "the LadybugDB CLI, lbug.exe")
	out := flag.String("out", "", "the folder for the CSV files, build.cypher and the database; must be empty or absent")
	commits := flag.Int("commits", 2000, "how many recent non-merge commits to read")
	maxFiles := flag.Int("max-files", 40, "a commit changing more files than this is left out")
	queries := flag.String("queries", "", "a .cypher file of numbered queries to run once the graph is built")
	flag.Parse()
	if err := build(*ix, *lbug, *out, *commits, *maxFiles, *queries); err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-graph:", err)
		os.Exit(1)
	}
}

func build(ix, lbug, out string, commits, maxFiles int, queries string) error {
	if !filepath.IsAbs(ix) || !filepath.IsAbs(out) || lbug == "" {
		return fmt.Errorf("-ix and -out must be absolute and -lbug given")
	}
	if entries, err := os.ReadDir(out); err == nil && len(entries) > 0 {
		return fmt.Errorf("%s is not empty", out)
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	start := time.Now()
	learnings, err := repograph.ReadCatalog(filepath.Join(ix, "state", "streeling", "catalog.jsonl"))
	if err != nil {
		return err
	}
	history, err := repograph.ReadCommits(ix, commits, maxFiles)
	if err != nil {
		return err
	}
	mentions, err := repograph.Mentions(map[string]string{"ix": ix}, learnings, history)
	if err != nil {
		return err
	}
	if err := repograph.Export(out, learnings, history, mentions); err != nil {
		return err
	}
	named := 0
	for _, files := range mentions {
		named += len(files)
	}
	read := time.Since(start)
	start = time.Now()
	db := filepath.Join(out, "db")
	if err := repograph.Build(lbug, db, filepath.Join(out, "build.cypher")); err != nil {
		return err
	}
	fmt.Printf("graph: %d learnings, %d commits, %d learning→file mentions; read %v, built %v → %s\n",
		len(learnings), len(history), named, read.Round(time.Millisecond), time.Since(start).Round(time.Millisecond), db)
	if queries == "" {
		return nil
	}
	text, err := os.ReadFile(queries)
	if err != nil {
		return err
	}
	for _, q := range numbered(string(text)) {
		start := time.Now()
		rows, err := repograph.Query(lbug, db, q.query)
		fmt.Printf("\n// %s  (%v)\n", q.title, time.Since(start).Round(time.Millisecond))
		if err != nil {
			fmt.Println("ERROR:", err)
			continue
		}
		fmt.Println(string(rows))
	}
	return nil
}

type entry struct{ title, query string }

var heading = regexp.MustCompile(`^//\s*(\d+\).*)$`)

// numbered splits a .cypher file into its numbered queries: a "// N) title" line starts one; other comments are kept
// out of the query.
func numbered(text string) []entry {
	var out []entry
	for _, line := range strings.Split(strings.ReplaceAll(text, "\r", ""), "\n") {
		if m := heading.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			out = append(out, entry{title: strings.TrimSpace(m[1])})
			continue
		}
		if len(out) == 0 || strings.HasPrefix(strings.TrimSpace(line), "//") {
			continue
		}
		out[len(out)-1].query += line + "\n"
	}
	return out
}
