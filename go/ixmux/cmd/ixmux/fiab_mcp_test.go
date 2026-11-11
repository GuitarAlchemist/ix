package main

import (
	"path/filepath"
	"testing"
)

// GAIA-20261110 slice 2: through the MCP adapter, a client tells why a mandate is uncertain without
// any free text: an adapter error, a delivery deadline that passed, or (see
// TestMCPIxmuxKilledDuringATurn) a restart, each a typed cause; saturation is a typed refusal,
// and none of them makes a second turn.
func TestMCPSaysWhyAMandateIsUncertain(t *testing.T) {
	dir := t.TempDir()
	state, eff := filepath.Join(dir, "state"), filepath.Join(dir, "effects")
	env := []string{"FAKE_EFFECTS=" + eff, "FAKE_ISERROR=1", "FAKE_MUTE=1"}
	args := []string{"--claude-target", "c", "--claude-bin", fakeClaude, "--timeout", "1s"}
	for _, kv := range []string{"FAKE_EFFECTS", "FAKE_ISERROR", "FAKE_MUTE"} {
		args = append(args, "--claude-env", kv)
	}
	p := start(t, state, env, args...)
	m := startMCP(t, state, p.token)
	m.initialize()
	inc := claudeOf(p).Incarnation

	// An error: the session's result has is_error.
	if rc := m.tool("ixmux_submit", submit("e1", inc, "fail, please")); rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("submit e1: %v", rc.Structured)
	}
	if res := m.resultOf("e1", "uncertain"); res["cause"] != "adapter-error" {
		t.Errorf("e1, whose result had is_error: %v, want cause adapter-error", res)
	}

	// A timeout: the session never answers within the 1 s delivery deadline.
	if rc := m.tool("ixmux_submit", submit("d1", inc, "mute, please")); rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("submit d1: %v", rc.Structured)
	}
	if res := m.resultOf("d1", "uncertain"); res["cause"] != "deadline" {
		t.Errorf("d1, never answered: %v, want cause deadline", res)
	}

	// Saturation: d1's call has not returned, the target takes nothing else.
	if rc := m.tool("ixmux_submit", submit("b1", inc, "Bonjour.")); !rc.IsError || rc.Structured["error"] != "target-busy" {
		t.Errorf("a mandate while d1's call runs: %v, want isError target-busy", rc.Structured)
	}
	if n := lines(t, eff); n != 2 {
		t.Errorf("%d turns, want 2 (e1 and d1)", n)
	}
}
