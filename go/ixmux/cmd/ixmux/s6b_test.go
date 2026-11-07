package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// S6B P1-1 over the real binary: a turn longer than --timeout is uncertain with still_running,
// then delivered with its whole answer when it ends before TurnMax, read by GET
// /v1/mandates/{id} and by ixmux_result. A second mandate during the turn is refused target-busy,
// with nothing written for it in the command log. A stop with a turn held is bounded.
func TestLateAnswerOverHTTPAndMCP(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	p := start(t, state, []string{"FAKE_EFFECTS=" + eff, "FAKE_HOLD=" + hold}, "--claude-target", "c", "--claude-bin", fakeClaude,
		"--claude-env", "FAKE_EFFECTS", "--claude-env", "FAKE_HOLD", "--timeout", "1s")
	m := startMCP(t, state, p.token)
	m.initialize()
	inc := claudeOf(p).Incarnation

	if rc := m.tool("ixmux_submit", submit("late", inc, "A slow one.")); rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("submit late: %v", rc.Structured)
	}
	m.resultOf("late", "uncertain")
	var e struct {
		State        string `json:"state"`
		StillRunning bool   `json:"still_running"`
		Ack          *struct {
			Text string `json:"text"`
		} `json:"ack"`
	}
	if p.do("GET", "/v1/mandates/late", nil, &e); e.State != "uncertain" || !e.StillRunning {
		t.Errorf("past --timeout: %+v, want uncertain with still_running", e)
	}

	rc := m.tool("ixmux_submit", submit("second", inc, "Another."))
	if !rc.IsError || rc.Structured["error"] != "target-busy" {
		t.Errorf("a second mandate during the turn: %v, want isError target-busy", rc.Structured)
	}
	if log, _ := os.ReadFile(filepath.Join(state, "commands.jsonl")); bytes.Contains(log, []byte(`"mandate_id":"second"`)) {
		t.Errorf("the refused second mandate is in the command log:\n%s", log)
	}

	os.Remove(hold)
	res := m.resultOf("late", "delivered")
	if !strings.Contains(fmt.Sprint(res["text"]), "BONJOUR") || res["ack_seen"] != true || res["still_running"] != false {
		t.Errorf("the late answer through ixmux_result: %v", res)
	}
	if p.do("GET", "/v1/mandates/late", nil, &e); e.State != "delivered" || e.Ack == nil || !strings.Contains(e.Ack.Text, "BONJOUR") {
		t.Errorf("the late answer through GET: %+v", e)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}

// S6B P1-1: a stop with a turn held ends within StopWait plus the engine's deadline, not TurnMax,
// and exits 0: the child is killed, the call returns, the engine closes.
func TestStopWithATurnHeld(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	p := start(t, state, []string{"FAKE_EFFECTS=" + eff, "FAKE_HOLD=" + hold}, "--claude-target", "c", "--claude-bin", fakeClaude,
		"--claude-env", "FAKE_EFFECTS", "--claude-env", "FAKE_HOLD", "--timeout", "1s")
	m := startMCP(t, state, p.token)
	m.initialize()
	if rc := m.tool("ixmux_submit", submit("held", claudeOf(p).Incarnation, "A slow one.")); rc.IsError {
		t.Fatalf("submit held: %v", rc.Structured)
	}
	m.resultOf("held", "uncertain")
	p.do("POST", "/v1/shutdown", nil, nil)
	select {
	case <-p.exited:
	case <-time.After(60 * time.Second): // TurnMax is 10 minutes: a guard against waiting for it
		t.Fatal("ixmux still waits for the held turn")
	}
	if code := p.cmd.ProcessState.ExitCode(); code != 0 {
		t.Errorf("exit code %d, want 0\n%s", code, p.out)
	}
}
