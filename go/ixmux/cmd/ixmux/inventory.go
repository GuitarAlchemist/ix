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

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
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

// mcpInventory is what ixmux_inventory carries: the reduced view of the inventory of the adapter's
// own state directory. A tool answer may later cross a tunnel, so it carries what ixmux_status and
// ixmux_result already give and no more: no path, pid, address, exit code or creation time; no
// Claude Code session of the machine; no text from a target (a reason, an error); a cause only from
// the closed set; and the finished mandates as a count, their ids being what ixmux_result answers.
type mcpInventory struct {
	Format   string     `json:"format"`
	ReadAt   time.Time  `json:"read_at"`
	Entries  []mcpEntry `json:"entries"`
	Finished int        `json:"finished"` // finished mandates (delivered, refused), not listed
	Gaps     []string   `json:"gaps"`
}

type mcpEntry struct {
	Kind         string             `json:"kind"` // ixmux, mandate, mandate-log
	MandateID    string             `json:"mandate_id,omitempty"`
	Target       string             `json:"target,omitempty"`
	Incarnation  string             `json:"incarnation,omitempty"` // a mandate's, or ixmux's epoch
	Status       inventory.Status   `json:"status"`
	Liveness     inventory.Liveness `json:"liveness,omitempty"`
	State        string             `json:"state,omitempty"` // as the command log records it
	Cause        string             `json:"cause,omitempty"`
	Problem      string             `json:"problem,omitempty"` // a mandate's, written by the inventory
	LastActivity *time.Time         `json:"last_activity,omitempty"`
	AgeSeconds   *int64             `json:"age_s,omitempty"`
	Resume       string             `json:"resume"`
}

// mcpInventoryFormat names the reduced report of ixmux_inventory.
const mcpInventoryFormat = "ixmux.inventory.mcp.v1"

// inventory is the MCP tool ixmux_inventory, read from the files of the adapter's state directory,
// so it answers while ixmux is down.
// @ai:invariant ixmux_inventory carries no path, pid, address, Claude session, reason or unknown cause [T:test conf:0.85 src:go/ixmux/cmd/ixmux/inventory_test.go:TestInventoryAcrossAKillDuringATurn]
func (m *mcp) inventory(raw json.RawMessage) (any, *toolErr) {
	var a struct{}
	if terr := args(raw, &a); terr != nil {
		return nil, terr
	}
	r := inventory.Collect(inventory.Options{IxmuxStates: []string{m.state}})
	out := mcpInventory{Format: mcpInventoryFormat, ReadAt: r.ReadAt, Entries: []mcpEntry{}, Gaps: r.Gaps}
	for _, e := range r.Entries {
		v := mcpEntry{Kind: e.Kind, Status: e.Status, LastActivity: e.LastActivity, AgeSeconds: e.AgeSeconds, Resume: e.Resume}
		switch e.Kind {
		case "ixmux":
			v.Liveness = e.Liveness
			if strings.HasPrefix(e.Incarnation, "epoch:") {
				v.Incarnation = e.Incarnation
			}
		case "mandate":
			if s := e.Declared["state"]; s == string(deliver.Delivered) || s == string(deliver.Refused) {
				out.Finished++
				continue
			}
			v.MandateID, v.Target, v.Incarnation, v.State, v.Problem = e.ID, e.Name, e.Incarnation, e.Declared["state"], e.Problem
			if uncertainCauses[e.Declared["cause"]] {
				v.Cause = e.Declared["cause"]
			}
		case "mandate-log":
			v.Resume = "the command log could not be read here; never delete it"
		default:
			continue
		}
		out.Entries = append(out.Entries, v)
	}
	for why, n := range r.Omitted { // only finished mandates are ever left out of an ixmux's
		if strings.HasPrefix(why, "finished mandates") {
			out.Finished += n
		}
	}
	return out, nil
}
