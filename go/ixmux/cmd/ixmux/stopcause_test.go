package main

// Adopted from the throwaway probe TestProbeGracefulStopDuringADelivery of session 1b129d7c
// (GAIA-IXMUX-CLOUD-QUALIFICATION-R1), with the assertion on the cause it lacked.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func waitTurn(t *testing.T, p *proc, eff string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for lines(t, eff) == 0 {
		if time.Now().After(deadline) {
			t.Fatalf("the held turn never reached the fake claude\n%s", p.out)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// A clean stop (POST /v1/shutdown) during a turn leaves the mandate uncertain with cause restart:
// ixmux stopped with the delivery under way. adapter-error would say the target answered an error.
func TestACleanStopDuringATurnIsCauseRestart(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	p := startClaude(t, state, eff, "FAKE_HOLD="+hold)
	m := startMCP(t, state, p.token)
	m.initialize()
	inc := claudeOf(p).Incarnation
	if rc := m.tool("ixmux_submit", submit("g1", inc, "A slow one.")); rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("submit g1: %v", rc.Structured)
	}
	waitTurn(t, p, eff)
	before := m.tool("ixmux_result", map[string]any{"mandate_id": "g1"})
	t.Logf("before the stop: %v", before.Structured)
	began := time.Now()
	p.do("POST", "/v1/shutdown", nil, nil)
	select {
	case <-p.exited:
	case <-time.After(60 * time.Second):
		t.Fatal("ixmux did not stop")
	}
	t.Logf("stopped in %s, exit code %d", time.Since(began), p.cmd.ProcessState.ExitCode())
	if out := p.out.String(); out != "" {
		t.Logf("first ixmux output:\n%s", out)
	}
	os.Remove(hold)
	q := startClaude(t, state, eff)
	m.secrets = append(m.secrets, q.token)
	t.Logf("restart output: %q", strings.TrimSpace(q.out.String()))
	res := m.tool("ixmux_result", map[string]any{"mandate_id": "g1"})
	t.Logf("after the restart: isError=%v %v", res.IsError, res.Structured)
	if res.Structured["state"] != "uncertain" {
		t.Errorf("state after a graceful stop mid-turn: %v, want uncertain", res.Structured["state"])
	}
	if res.Structured["cause"] != "restart" {
		t.Errorf("cause after a clean stop mid-turn: %v, want restart", res.Structured["cause"])
	}
	if again := m.tool("ixmux_submit", submit("g1", inc, "A slow one.")); !again.IsError {
		t.Errorf("resubmitting an uncertain mandate was accepted: %v", again.Structured)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}
