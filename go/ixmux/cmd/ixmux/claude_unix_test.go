//go:build unix

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

// GAIA-20261110 slice 2, the unix side of TestClaudeKillIxmuxKillsTheChild. No Job Object ties the
// child to ixmux here (claudestream/job_unix.go): a killed ixmux leaves its claude child running
// while its turn runs, and the child ends only once that turn ends. The test measures this weaker
// guarantee instead of assuming it, and checks what holds on every system: the restart neither
// forgets the mandate nor sends it again.
func TestClaudeKillIxmuxOrphansTheChildUntilItsTurnEnds(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	p := startClaude(t, state, eff, "FAKE_HOLD="+hold)
	c := claudeOf(p)
	t.Cleanup(func() { // an orphan left by a failure here must not outlive the test
		os.Remove(hold)
		syscall.Kill(c.PID, syscall.SIGKILL)
	})
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

	// While its turn is held, the child outlives ixmux.
	for held := time.Now().Add(time.Second); time.Now().Before(held); time.Sleep(50 * time.Millisecond) {
		if !alive(c.PID) {
			t.Fatalf("claude child %d ended with ixmux while its turn ran: unix holds more than job_unix.go says; update it", c.PID)
		}
	}
	// Once the turn ends, its answer meets a closed pipe and the child ends.
	os.Remove(hold)
	for deadline = time.Now().Add(10 * time.Second); alive(c.PID) && time.Now().Before(deadline); {
		time.Sleep(20 * time.Millisecond)
	}
	if alive(c.PID) {
		t.Fatalf("claude child %d still runs 10 s after its turn ended without ixmux", c.PID)
	}

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
	if code := q.do("POST", "/v1/mandates", claudeMandate("slow-1", c2.Incarnation, "slow"), &res); code != http.StatusConflict {
		t.Errorf("resend of slow-1: %d %+v, want 409", code, res)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1 (the orphaned child's)", n)
	}
}
