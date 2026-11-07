package claudestream

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// failKills makes every kill of a job fail until the test ends; then the child c is killed for
// real, so no process is left.
func failKills(t *testing.T, c *child) {
	old := killJob
	killJob = func(*job) error { return errors.New("test: the kill fails") }
	t.Cleanup(func() {
		killJob = old
		select {
		case <-c.exited:
		default:
			old(c.job)
		}
	})
}

func (a *Adapter) isReserved() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.reserved
}

// S6C P1-2 (Codex 9): the kill at TurnMax fails. The wait after it is bounded by StopWait; past
// it the incarnation is lost (kill-failed), the turn resolves in error, so the mandate stays
// uncertain with no call running, and the reservation is free.
func TestKillThatFailsAtTurnMax(t *testing.T) {
	a, e, _ := setupCfg(t, 200*time.Millisecond, func(c *Config) {
		c.TurnMax, c.StopWait = 500*time.Millisecond, 300*time.Millisecond
	}, "FAKE_MUTE=1")
	failKills(t, a.live())
	if res, _ := send(t, a, e, "m1", "mute"); res.State != deliver.Uncertain {
		t.Fatalf("m1 never answered: %+v, want uncertain", res)
	}
	deadline := time.Now().Add(5 * time.Second) // TurnMax plus StopWait is 0.8 s
	for {
		en, _ := entryOf(e, "m1")
		if !en.StillRunning && !a.isReserved() {
			if en.State != deliver.Uncertain || !strings.Contains(en.Reason, ReasonKillFailed) {
				t.Errorf("after a failed kill: %+v, want uncertain with reason %s", en, ReasonKillFailed)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("a failed kill holds the turn: %+v, reserved %v", en, a.isReserved())
		}
		time.Sleep(20 * time.Millisecond)
	}
	if notes(t, a, ReasonKillFailed) != 1 {
		t.Errorf("no %s line in the stream journal", ReasonKillFailed)
	}
	if _, err := a.Incarnation(); err == nil {
		t.Error("the lost incarnation is still offered")
	}
}

// S6C P1-2: a stop whose kill fails is bounded too: StopWait for the exit, then StopWait after
// the kill, and the step is traced kill-failed.
func TestKillThatFailsAtStop(t *testing.T) {
	const stopWait = 300 * time.Millisecond
	a, _, _ := setupCfg(t, time.Second, func(c *Config) { c.StopWait = stopWait }, "FAKE_LINGER=5s")
	failKills(t, a.live())
	began := time.Now()
	var err error
	within(t, 10*time.Second, "Stop with a failing kill", func() { err = a.Stop() })
	if d := time.Since(began); d > 2*stopWait+time.Second {
		t.Errorf("Stop took %s with a failing kill, want twice StopWait (%s) plus a margin", d, stopWait)
	}
	if err == nil || !strings.Contains(err.Error(), ReasonKillFailed) {
		t.Errorf("Stop: %v, want a %s error", err, ReasonKillFailed)
	}
	// The stop trace names it: its last step is kill-failed.
	var steps []string // the stop lines only: the child's exit may follow the job's close
	for _, m := range stopSteps(t, a.StreamPath()) {
		if s, ok := m["step"].(string); ok && m["ixmux"] == "stop" {
			steps = append(steps, s)
		}
	}
	if len(steps) == 0 || steps[len(steps)-1] != ReasonKillFailed {
		t.Errorf("stop trace %v, want %s last", steps, ReasonKillFailed)
	}
}
