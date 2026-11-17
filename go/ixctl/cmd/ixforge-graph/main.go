// Command ixforge-graph builds the LadybugDB graph of the ix and ga repos (see internal/repograph): the Streeling
// learnings with their repo and tags, each repo's recent commits with the files they changed, the files each learning
// names, and with -github the repos' pull requests and issues, read with gh and its own sign-in. Each build goes into a
// new stamped folder of -into; once its database is built the folder is marked finished, and older builds beyond -keep
// are removed. With -queries it then runs the numbered queries of a .cypher file (// N) title, then the query) and
// prints each one's rows with its time.
//
//	go run ./cmd/ixforge-graph -ix C:\Users\<you>\source\repos\ix -ga C:\Users\<you>\source\repos\ga -github `
//	    -lbug <path to lbug.exe> -into C:\Users\<you>\source\repos\ix\state\graph
//	go run ./cmd/ixforge-graph ... -queries C:\Users\<you>\source\repos\ix\docs\streeling\graph.cypher
//
// The Desktop runs the queries of docs/streeling/graph.cypher on the newest finished build of <ix>\state\graph.
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
	var o options
	flag.StringVar(&o.ix, "ix", "", "the ix repo (absolute); its state/streeling/catalog.jsonl lists the learnings")
	flag.StringVar(&o.ga, "ga", "", "the ga repo (absolute), optional")
	flag.BoolVar(&o.github, "github", false, "also list the repos' pull requests and issues with gh (GuitarAlchemist/<repo>)")
	flag.IntVar(&o.prs, "prs", 300, "with -github, how many recent pull requests of each repo")
	flag.IntVar(&o.issues, "issues", 300, "with -github, how many recent issues of each repo")
	flag.StringVar(&o.lbug, "lbug", "", "the LadybugDB CLI, lbug.exe")
	flag.StringVar(&o.into, "into", "", "the folder of the builds (absolute); each build is a new stamped folder in it")
	flag.IntVar(&o.keep, "keep", 3, "how many finished builds to keep")
	flag.IntVar(&o.commits, "commits", 2000, "how many recent non-merge commits of each repo to read")
	flag.IntVar(&o.maxFiles, "max-files", 40, "a commit changing more files than this is left out")
	flag.StringVar(&o.queries, "queries", "", "a .cypher file of numbered queries to run once the graph is built")
	flag.Parse()
	if err := build(o); err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-graph:", err)
		os.Exit(1)
	}
}

type options struct {
	ix, ga, lbug, into, queries          string
	github                               bool
	prs, issues, keep, commits, maxFiles int
}

func build(o options) error {
	if !filepath.IsAbs(o.ix) || !filepath.IsAbs(o.into) || (o.ga != "" && !filepath.IsAbs(o.ga)) || o.lbug == "" || o.keep < 1 {
		return fmt.Errorf("-ix, -into (and -ga) must be absolute, -lbug given and -keep at least 1")
	}
	start := time.Now()
	learnings, err := repograph.ReadCatalog(filepath.Join(o.ix, "state", "streeling", "catalog.jsonl"))
	if err != nil {
		return err
	}
	d := repograph.Data{Roots: map[string]string{"ix": o.ix}, Learnings: learnings}
	if o.ga != "" {
		d.Roots["ga"] = o.ga
	}
	for _, name := range []string{"ix", "ga"} {
		dir, ok := d.Roots[name]
		if !ok {
			continue
		}
		commits, err := repograph.ReadCommits(dir, name, o.commits, o.maxFiles)
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		d.Commits = append(d.Commits, commits...)
		if o.github {
			prs, issues, err := repograph.ReadGitHub(name, "GuitarAlchemist/"+name, o.prs, o.issues)
			if err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			d.PRs, d.Issues = append(d.PRs, prs...), append(d.Issues, issues...)
		}
	}
	dir, err := repograph.NewBuild(o.into, time.Now())
	if err != nil {
		return err
	}
	stats, err := repograph.ExportStats(dir, d)
	if err != nil {
		return err
	}
	read := time.Since(start)
	start = time.Now()
	db := filepath.Join(dir, "db")
	if err := repograph.Build(o.lbug, db, filepath.Join(dir, "build.cypher")); err != nil {
		return err
	}
	if err := repograph.Finish(dir); err != nil {
		return err
	}
	if err := repograph.Prune(o.into, o.keep); err != nil {
		return err
	}
	fmt.Printf("graph: %d learnings, %d commits, %d files, %d pull requests, %d issues; %d learning→file mentions, %d PR→file touches, %d issue→file names, %d PR→issue closes; read %v, built %v → %s\n",
		len(d.Learnings), len(d.Commits), stats.Files, len(d.PRs), len(d.Issues), stats.Mentions, stats.Touches, stats.Names, stats.Closes,
		read.Round(time.Millisecond), time.Since(start).Round(time.Millisecond), db)
	if o.queries == "" {
		return nil
	}
	text, err := os.ReadFile(o.queries)
	if err != nil {
		return err
	}
	for _, q := range numbered(string(text)) {
		start := time.Now()
		rows, err := repograph.Query(o.lbug, db, q.query)
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
