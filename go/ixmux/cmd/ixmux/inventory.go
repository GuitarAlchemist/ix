package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/inventory"
	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

type dirs []string

func (d *dirs) String() string     { return strings.Join(*d, ",") }
func (d *dirs) Set(v string) error { *d = append(*d, v); return nil }

// defaultClaudeHome is Claude Code's configuration directory: CLAUDE_CONFIG_DIR, else ~/.claude.
func defaultClaudeHome() string {
	if d := os.Getenv("CLAUDE_CONFIG_DIR"); d != "" {
		return d
	}
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, ".claude")
	}
	return ""
}

// runInventory prints, read-only, the local Claude Code sessions and the ixmux state directories
// given with their mandates: ixmux inventory [--state <dir> ...] [--claude-home <dir>] [--no-claude] [--json]
func runInventory(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("inventory", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var states dirs
	fs.Var(&states, "state", "an ixmux state directory (repeatable)")
	home := fs.String("claude-home", defaultClaudeHome(), "Claude Code's configuration directory, whose sessions/ is read")
	noClaude := fs.Bool("no-claude", false, "do not read the Claude Code sessions")
	asJSON := fs.Bool("json", false, "print the report as JSON ("+inventory.Format+")")
	if err := fs.Parse(args); err != nil {
		return server.ExitUsage
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: ixmux inventory [--state <dir> ...] [--claude-home <dir>] [--no-claude] [--json]")
		return server.ExitUsage
	}
	o := inventory.Options{IxmuxStates: states, ClaudeHome: *home}
	if *noClaude {
		o.ClaudeHome = ""
	}
	r := inventory.Collect(o)
	if *asJSON {
		e := json.NewEncoder(stdout)
		e.SetIndent("", "  ")
		e.Encode(r)
		return server.ExitOK
	}
	printInventory(stdout, r)
	return server.ExitOK
}

// printInventory writes one line per entry, then what may be done, the duplicates, what a bound
// left out and what this inventory does not see.
func printInventory(w io.Writer, r inventory.Report) {
	fmt.Fprintf(w, "inventory read at %s (%s): liveness is proven only by a live pid with its recorded creation time; the rest is what files declare\n",
		r.ReadAt.Format(time.RFC3339), r.Format)
	for _, e := range r.Entries {
		age := "-"
		if e.AgeSeconds != nil {
			age = (time.Duration(*e.AgeSeconds) * time.Second).String()
		}
		live := string(e.Liveness)
		if live == "" {
			live = "-"
		}
		fmt.Fprintf(w, "%-11s %-14s %-38s %-12s last %-10s %s\n", e.Status, e.Kind, e.ID, live, age, e.Name)
		if e.Problem != "" {
			fmt.Fprintf(w, "    problem: %s\n", e.Problem)
		}
		if e.Status != inventory.Done {
			fmt.Fprintf(w, "    resume:  %s\n", e.Resume)
		}
	}
	for _, d := range r.Duplicates {
		fmt.Fprintf(w, "duplicate %s %s: %d incarnations, %d live: %s\n", d.Kind, d.ID, len(d.Incarnations), d.Live, strings.Join(d.Incarnations, ", "))
	}
	for why, n := range r.Omitted {
		fmt.Fprintf(w, "omitted: %d %s\n", n, why)
	}
	for _, g := range r.Gaps {
		fmt.Fprintf(w, "not read: %s\n", g)
	}
}

// inventory is the MCP tool ixmux_inventory: the inventory of this adapter's state directory
// alone, read from its files, so it answers while ixmux is down. It shows no Claude Code session
// of the machine (their directories are private to their user) and no text that came from a
// target (a mandate's reason): what the tunnel may carry stays what ixmux_status already shows.
func (m *mcp) inventory(raw json.RawMessage) (any, *toolErr) {
	var a struct{}
	if terr := args(raw, &a); terr != nil {
		return nil, terr
	}
	r := inventory.Collect(inventory.Options{IxmuxStates: []string{m.state}})
	for i := range r.Entries {
		delete(r.Entries[i].Declared, "reason")
	}
	return r, nil
}
