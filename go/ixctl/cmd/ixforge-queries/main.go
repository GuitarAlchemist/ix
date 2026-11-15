// Command ixforge-queries puts our DuckDB queries (the numbered queries of the ix repo's .sql query files) in IxForge,
// as a widget to read and copy, which says when it was made and from which ix commit.
//
// Kept up to date by itself: pair once (an invite code from the owner page), approve the device "Veilleur requêtes" on
// the owner page, then leave the watcher running. Each time the query files are committed with changes, it sends the
// widget to the Desktop, and every device shows it at its next sync, without a reload. While the files differ from
// their commit, it waits.
//
//	go run ./cmd/ixforge-queries -pair <code> -origin https://<desktop host>
//	go run ./cmd/ixforge-queries -watch -ix C:\Users\<you>\source\repos\ix
//
// Or once, as a file to import with the import button:
//
//	go run ./cmd/ixforge-queries -ix C:\Users\<you>\source\repos\ix -out requetes-duckdb.json
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/workshop"
)

// access is the watcher's device access, kept in the user's profile (readable by this account only).
type access struct {
	Origin  string `json:"origin"`
	Token   string `json:"token"`
	StoreID string `json:"store_id"`
}

func accessPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "IxForge", "queries-watcher.json"), nil
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "ixforge-queries:", err)
	os.Exit(1)
}

func main() {
	ix := flag.String("ix", "", "the ix repo's root folder")
	out := flag.String("out", "", "write the widget to this file, once, to import it")
	title := flag.String("title", "Requêtes DuckDB", "the widget's title")
	files := flag.String("files", strings.Join(workshop.DefaultQueryFiles, ","), "the .sql files to read, relative to the repo, comma-separated")
	pair := flag.String("pair", "", "pair this watcher with the Desktop using an invite code")
	origin := flag.String("origin", "", "the Desktop's origin, with -pair (https://…)")
	name := flag.String("name", "Veilleur requêtes", "the watcher's device name, with -pair")
	watch := flag.Bool("watch", false, "keep the Desktop's widget up to date with the repo")
	every := flag.Duration("every", 10*time.Second, "how often the watcher reads the repo")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	switch {
	case *pair != "":
		if !strings.HasPrefix(*origin, "https://") {
			fail(errors.New("-pair needs -origin https://<the Desktop's host>"))
		}
		desktop := &workshop.DesktopClient{Origin: strings.TrimRight(*origin, "/")}
		if err := desktop.Pair(ctx, *pair, *name); err != nil {
			fail(err)
		}
		path, err := accessPath()
		if err != nil {
			fail(err)
		}
		raw, _ := json.Marshal(access{desktop.Origin, desktop.Token, desktop.StoreID})
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			fail(err)
		}
		if err := os.WriteFile(path, raw, 0o600); err != nil {
			fail(err)
		}
		fmt.Printf("Paired as %q. Approve it on the owner page, then run with -watch.\n", *name)

	case *watch:
		if *ix == "" {
			fail(errors.New("-watch needs -ix <the ix repo>"))
		}
		path, err := accessPath()
		if err != nil {
			fail(err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			fail(fmt.Errorf("no access yet (%v): pair first with -pair <code> -origin https://…", err))
		}
		var a access
		if err := json.Unmarshal(raw, &a); err != nil || a.Token == "" {
			fail(errors.New("the saved access is unreadable: pair again"))
		}
		w := &workshop.QueriesWatch{Desktop: &workshop.DesktopClient{Origin: a.Origin, Token: a.Token, StoreID: a.StoreID},
			Repo: *ix, Files: strings.Split(*files, ","), Title: *title, Now: time.Now}
		fmt.Printf("Watching %s every %s for %q (Ctrl+C to stop).\n", *ix, *every, *title)
		last := ""
		for {
			step, err := w.Step(ctx)
			// Say what happened only when it changes: one line per sent version, per wait, per error.
			line := ""
			switch {
			case err != nil:
				line = "not sent: " + err.Error() + " (next try in " + every.String() + ")"
			case step.Pushed:
				line = fmt.Sprintf("sent %d queries from ix %.7s", step.Queries, step.Commit)
			case step.Waiting == "uncommitted":
				line = "waiting: the query files differ from their commit; commit them to send them"
			default:
				line = fmt.Sprintf("up to date: %d queries from ix %.7s", step.Queries, step.Commit)
			}
			if line != last || step.Pushed {
				fmt.Println(time.Now().Format("15:04:05"), line)
				last = line
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(*every):
			}
		}

	default:
		if *ix == "" || *out == "" {
			fmt.Fprintln(os.Stderr, "ixforge-queries: give -ix and -out (a file to import), or -pair, or -watch")
			os.Exit(2)
		}
		widget, commit, err := workshop.QueriesWidgetFromRepo(*ix, strings.Split(*files, ","), *title, time.Now())
		if err != nil {
			fail(err)
		}
		if err := os.WriteFile(*out, widget, 0o644); err != nil {
			fail(err)
		}
		fmt.Printf("queries from ix %.7s written to %s\n", commit, *out)
	}
}
