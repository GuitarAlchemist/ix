package main

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

const stillActive = 259

// alive: the process pid exists and has not exited.
func alive(pid int) bool {
	h, err := syscall.OpenProcess(0x1000 /* PROCESS_QUERY_LIMITED_INFORMATION */, false, uint32(pid))
	if err != nil {
		return false
	}
	defer syscall.CloseHandle(h)
	var code uint32
	return syscall.GetExitCodeProcess(h, &code) == nil && code == stillActive
}

// Oracles S4-7: ixmux is killed mid-mandate; its Job Object kills the child; the restart has a
// new child, so a new incarnation: the mandate is uncertain, its reconcile stale, its resend
// refused, and the old child took it once.
func TestClaudeKillIxmuxKillsTheChild(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	p := startClaude(t, state, eff, "FAKE_HOLD="+hold)
	c := claudeOf(p)
	go func() {
		req, _ := http.NewRequest("POST", "http://"+p.addr+"/v1/mandates", strings.NewReader(`{"mandate_id":"slow-1","target":"c","incarnation":"`+c.Incarnation+`","text":"slow"}`))
		req.Header.Set("Authorization", "Bearer "+p.token)
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
		}
	}()
	deadline := time.Now().Add(10 * time.Second)
	for lines(t, eff) == 0 {
		if time.Now().After(deadline) {
			t.Fatalf("slow-1 never reached the child\n%s", p.out)
		}
		time.Sleep(20 * time.Millisecond)
	}
	p.cmd.Process.Kill()
	<-p.exited
	for alive(c.PID) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if alive(c.PID) {
		t.Fatalf("claude child %d survived the kill of ixmux", c.PID)
	}
	os.Remove(hold)

	q := startClaude(t, state, eff)
	c2 := claudeOf(q)
	if c2.Incarnation == c.Incarnation || c2.PID == c.PID {
		t.Errorf("after the restart: %+v, want a new child and incarnation (was %+v)", c2, c)
	}
	var e result
	if q.do("GET", "/v1/mandates/slow-1", nil, &e); e.State != "uncertain" {
		t.Errorf("slow-1 after the kill: %+v, want uncertain", e)
	}
	var res claudeResult
	if code := q.do("POST", "/v1/mandates/slow-1/reconcile", nil, &res); code != http.StatusConflict || res.Error != "stale-incarnation" {
		t.Errorf("reconcile: %d %+v, want 409 stale-incarnation", code, res)
	}
	if code := q.do("POST", "/v1/mandates", claudeMandate("slow-1", c2.Incarnation, "slow"), &res); code != http.StatusConflict {
		t.Errorf("resend of slow-1: %d %+v, want 409", code, res)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1 (the killed child's)", n)
	}
}
