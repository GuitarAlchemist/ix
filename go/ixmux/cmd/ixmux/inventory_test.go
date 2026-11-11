package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ixmux/inventory"
)

func runInv(t *testing.T, args ...string) inventory.Report {
	t.Helper()
	var out, errb bytes.Buffer
	if code := runInventory(append(args, "--no-claude", "--json"), &out, &errb); code != 0 {
		t.Fatalf("inventory exit %d: %s", code, errb.String())
	}
	var r inventory.Report
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("inventory JSON: %v\n%s", err, out.String())
	}
	return r
}

func entryOf(r inventory.Report, kind, id string) *inventory.Entry {
	for i := range r.Entries {
		if r.Entries[i].Kind == kind && r.Entries[i].ID == id {
			return &r.Entries[i]
		}
	}
	return nil
}

// The restart story read by the inventory, on a real ixmux and fake claude: a turn under way is
// running; ixmux killed in it, the inventory, from files alone, says ixmux is unreachable and the
// mandate uncertain, for a person; the MCP tool says the same while ixmux is down; ixmux started
// again finds it uncertain and sends nothing again. The inventory changes no file.
func TestInventoryAcrossAKillDuringATurn(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	p := startClaude(t, state, eff, "FAKE_HOLD="+hold)
	m := startMCP(t, state, p.token)
	m.initialize()
	inc := claudeOf(p).Incarnation
	if rc := m.tool("ixmux_submit", submit("k1", inc, "A slow one.")); rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("submit k1: %v", rc.Structured)
	}
	waitTurn(t, p, eff)

	logPath := filepath.Join(state, "commands.jsonl")
	before, _ := os.ReadFile(logPath)
	r := runInv(t, "--state", state)
	if e := entryOf(r, "ixmux", state); e == nil || e.Status != inventory.Running || e.Liveness != inventory.Proven {
		t.Errorf("a live ixmux: %+v", e)
	}
	if e := entryOf(r, "mandate", "k1"); e == nil || e.Status != inventory.Running {
		t.Errorf("a turn under way: %+v", e)
	}
	if after, _ := os.ReadFile(logPath); !bytes.Equal(before, after) {
		t.Error("the inventory changed the command log")
	}

	p.cmd.Process.Kill()
	<-p.exited
	r = runInv(t, "--state", state)
	if e := entryOf(r, "ixmux", state); e == nil || e.Status != inventory.Unreachable || e.Liveness != inventory.Absent {
		t.Errorf("a killed ixmux: %+v", e)
	}
	if e := entryOf(r, "mandate", "k1"); e == nil || e.Status != inventory.Uncertain {
		t.Errorf("the turn of a killed ixmux: %+v", e)
	}
	inv := m.tool("ixmux_inventory", map[string]any{})
	if inv.IsError {
		t.Fatalf("ixmux_inventory with ixmux down: %v", inv.Structured)
	}
	var mr inventory.Report
	b, _ := json.Marshal(inv.Structured)
	json.Unmarshal(b, &mr)
	if e := entryOf(mr, "mandate", "k1"); e == nil || e.Status != inventory.Uncertain || e.Declared["reason"] != "" {
		t.Errorf("ixmux_inventory's k1: %+v", e)
	}
	for _, e := range mr.Entries {
		if e.Kind == "claude-session" {
			t.Errorf("ixmux_inventory shows a Claude session of the machine: %+v", e)
		}
	}

	os.Remove(hold)
	q := startClaude(t, state, eff)
	m.secrets = append(m.secrets, q.token)
	m.resultOf("k1", "uncertain")
	if again := m.tool("ixmux_submit", submit("k1", inc, "A slow one.")); !again.IsError {
		t.Errorf("resubmitting the uncertain k1 was accepted: %v", again.Structured)
	}
	r = runInv(t, "--state", state)
	if e := entryOf(r, "ixmux", state); e == nil || e.Status != inventory.Running {
		t.Errorf("ixmux started again: %+v", e)
	}
	if e := entryOf(r, "mandate", "k1"); e == nil || e.Status != inventory.Uncertain || e.Declared["cause"] != "restart" || e.Declared["reason"] == "" {
		t.Errorf("k1 after the restart: %+v", e)
	}
	inv = m.tool("ixmux_inventory", map[string]any{})
	b, _ = json.Marshal(inv.Structured)
	mr = inventory.Report{}
	json.Unmarshal(b, &mr)
	if e := entryOf(mr, "mandate", "k1"); e == nil || e.Status != inventory.Uncertain || e.Declared["reason"] != "" {
		t.Errorf("ixmux_inventory's k1 after the restart, its reason must not pass: %+v", e)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}
