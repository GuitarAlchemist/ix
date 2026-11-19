// Command ixforge-site writes the standalone IxForge (the client alone, installable, data in the browser, optional AI
// on the person's own Claude key) as static files for a static host. It writes into an empty or new folder only.
//
//	go run ./cmd/ixforge-site -out <folder>         # site with _headers and _redirects (Cloudflare Pages, Netlify)
//	go run ./cmd/ixforge-site -app -out <folder>    # the app alone, for any folder (GitHub Pages)
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/workshop"
)

func main() {
	out := flag.String("out", "", "folder to write the site into (new or empty)")
	app := flag.Bool("app", false, "write the app alone, to serve from any folder of a header-less host such as GitHub Pages")
	flag.Parse()
	write := workshop.WriteSite
	if *app {
		write = workshop.WriteApp
	}
	if *out == "" {
		fmt.Fprintln(os.Stderr, "ixforge-site: -out is required")
		os.Exit(2)
	}
	if entries, err := os.ReadDir(*out); err == nil && len(entries) > 0 {
		fmt.Fprintf(os.Stderr, "ixforge-site: %s is not empty; nothing written\n", *out)
		os.Exit(1)
	}
	if err := write(*out); err != nil {
		fmt.Fprintln(os.Stderr, "ixforge-site:", err)
		os.Exit(1)
	}
	fmt.Println("site written to", *out)
}
