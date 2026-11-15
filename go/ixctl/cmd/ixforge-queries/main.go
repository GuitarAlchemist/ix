// Command ixforge-queries reads our DuckDB queries from the ix repo (the numbered queries of its .sql query files)
// and writes them as an IxForge widget, a list to read and copy. Import the file in IxForge with the import button,
// or paste its content in the new-tool field. Run it again and re-import when the queries change: the import updates
// the widget of the same name. The widget says when it was made and from which ix commit (git rev-parse HEAD).
//
//	go run ./cmd/ixforge-queries -ix C:\Users\<you>\source\repos\ix -out requetes-duckdb.json
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/workshop"
)

func main() {
	ix := flag.String("ix", "", "the ix repo's root folder")
	out := flag.String("out", "", "the widget file to write")
	title := flag.String("title", "Requêtes DuckDB", "the widget's title")
	files := flag.String("files", strings.Join(workshop.DefaultQueryFiles, ","), "the .sql files to read, relative to the repo, comma-separated")
	flag.Parse()
	if *ix == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "ixforge-queries: -ix and -out are required")
		os.Exit(2)
	}
	entries, err := workshop.ReadQueries(*ix, strings.Split(*files, ","))
	if err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-queries:", err)
		os.Exit(1)
	}
	commit, err := exec.Command("git", "-C", *ix, "rev-parse", "HEAD").Output()
	if err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-queries: cannot read the ix commit (git rev-parse HEAD):", err)
		os.Exit(1)
	}
	// The widget names the commit its queries come from: that is only true if the files are as committed.
	changed, err := exec.Command("git", append([]string{"-C", *ix, "status", "--porcelain", "--"}, strings.Split(*files, ",")...)...).Output()
	if err != nil || len(strings.TrimSpace(string(changed))) > 0 {
		fmt.Fprintf(os.Stderr, "ixforge-queries: the query files differ from ix %.7s; commit them or set them aside first:\n%s", commit, changed)
		os.Exit(1)
	}
	widget, err := workshop.QueriesWidget(*title, entries, time.Now(), strings.TrimSpace(string(commit)))
	if err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-queries: these queries do not fit a widget (1 to 100, SQL up to 8 KiB each):", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, widget, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-queries:", err)
		os.Exit(1)
	}
	fmt.Printf("%d queries from ix %.7s written to %s\n", len(entries), commit, *out)
}
