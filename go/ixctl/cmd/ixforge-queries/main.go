// Command ixforge-queries reads our DuckDB queries from the ix repo (the numbered queries of its .sql query files)
// and writes them as an IxForge widget, a list to read and copy. Import the file in IxForge with the import button,
// or paste its content in the new-tool field. Run it again and re-import when the queries change.
//
//	go run ./cmd/ixforge-queries -ix C:\Users\<you>\source\repos\ix -out requetes-duckdb.json
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

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
	widget, err := workshop.QueriesWidget(*title, entries)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-queries: these queries do not fit a widget (1 to 100, SQL up to 8 KiB each):", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, widget, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-queries:", err)
		os.Exit(1)
	}
	fmt.Printf("%d queries written to %s\n", len(entries), *out)
}
