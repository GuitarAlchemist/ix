package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
	state, eff, hold := filepath.Join(dir, "ixstatedir"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	// A Claude session record where the adapter would find one, were it to read the machine's.
	home := filepath.Join(dir, "claude-home")
	os.MkdirAll(filepath.Join(home, "sessions"), 0o700)
	os.WriteFile(filepath.Join(home, "sessions", "1.json"), []byte(`{"pid":1,"sessionId":"plantedsession","startedAt":1,"status":"busy"}`), 0o600)
	t.Setenv("CLAUDE_CONFIG_DIR", home)
	p := startClaude(t, state, eff, "FAKE_HOLD="+hold)
	m := startMCP(t, state, p.token)
	m.initialize()
	inc := claudeOf(p).Incarnation
	if rc := m.tool("ixmux_submit", submit("k1", inc, "A slow one.")); rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("submit k1: %v", rc.Structured)
	}
	waitTurn(t, p, eff)
	var out, errb bytes.Buffer // the CLI does read it: the planted record is there to be leaked
	runInventory([]string{"--json"}, &out, &errb)
	if !strings.Contains(out.String(), "plantedsession") {
		t.Fatalf("the planted session record is not read by the CLI: %s", out.String())
	}

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
	if mr := mcpInv(t, m, state, p.cmd.Process.Pid); mr.k1 == nil || mr.k1.Status != inventory.Uncertain || mr.k1.State != "delivering" {
		t.Errorf("ixmux_inventory's k1 with ixmux down: %+v", mr.k1)
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
	if mr := mcpInv(t, m, state, q.cmd.Process.Pid); mr.k1 == nil || mr.k1.Status != inventory.Uncertain || mr.k1.Cause != "restart" {
		t.Errorf("ixmux_inventory's k1 after the restart: %+v", mr.k1)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}

type mcpInvSeen struct {
	mcpInventory
	k1 *mcpEntry
}

// mcpInv calls ixmux_inventory and checks, on its raw text, that nothing passes beyond what
// ixmux_status and ixmux_result give: no path of the state directory, no pid, no address, no
// reason, no Claude session (one is planted in the adapter's CLAUDE_CONFIG_DIR, see
// TestInventoryAcrossAKillDuringATurn).
func mcpInv(t *testing.T, m *mcpProc, state string, pid int) mcpInvSeen {
	t.Helper()
	inv := m.tool("ixmux_inventory", map[string]any{})
	if inv.IsError {
		t.Fatalf("ixmux_inventory: %v", inv.Structured)
	}
	text := inv.Content[0].Text
	esc, _ := json.Marshal(state)
	for _, leak := range []string{string(esc[1 : len(esc)-1]), filepath.Base(state), "127.0.0.1", `"pid"`, strconv.Itoa(pid), "reason", "claude-session", "plantedsession", "addr", "exit_code"} {
		if strings.Contains(text, leak) {
			t.Errorf("ixmux_inventory carries %q: %s", leak, text)
		}
	}
	var out mcpInvSeen
	if err := json.Unmarshal([]byte(text), &out.mcpInventory); err != nil {
		t.Fatal(err)
	}
	for i := range out.Entries {
		if out.Entries[i].MandateID == "k1" {
			out.k1 = &out.Entries[i]
		}
	}
	return out
}
