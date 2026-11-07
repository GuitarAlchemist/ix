package claudestream

import (
	"bytes"
	"os"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// notes counts the ixmux lines of kind in the stream journal.
func notes(t *testing.T, a *Adapter, kind string) int {
	t.Helper()
	b, _ := os.ReadFile(a.StreamPath())
	return bytes.Count(b, []byte(`"ixmux":"`+kind+`"`))
}

func (a *Adapter) isStopped() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.stopped
}

// S6C P1-1 (Codex 9): a Stop that lands between watch's read of stopped and the relaunch at
// TurnMax. No relaunch starts once stopped is set: no new child, no opening turn, and the stop
// keeps its bound.
func TestStopDuringTheRelaunch(t *testing.T) {
	const stopWait = 500 * time.Millisecond
	a, e, eff := setupCfg(t, 200*time.Millisecond, func(c *Config) {
		c.TurnMax, c.StopWait, c.PrimeAfter = 500*time.Millisecond, stopWait, 100*time.Millisecond
	}, "FAKE_INIT=first", "FAKE_MUTE=1")
	stopped := make(chan time.Duration, 1)
	testHookBeforeRelaunch = func() {
		began := time.Now()
		go func() { a.Stop(); stopped <- time.Since(began) }()
		for !a.isStopped() {
			time.Sleep(time.Millisecond)
		}
	}
	t.Cleanup(func() { testHookBeforeRelaunch = nil })
	if res, _ := send(t, a, e, "m1", "mute"); res.State != deliver.Uncertain {
		t.Fatalf("m1 never answered: %+v, want uncertain", res)
	}
	select {
	case d := <-stopped:
		if d > stopWait+time.Second {
			t.Errorf("Stop took %s during a relaunch, want StopWait (%s) plus a margin", d, stopWait)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Stop never returned during a relaunch")
	}
	if n := notes(t, a, "start"); n != 1 {
		t.Errorf("%d children started, want 1: a relaunch began after Stop", n)
	}
	if n := effects(t, eff); n != 2 { // the first child's opening turn, then m1
		t.Errorf("%d turns, want 2: an opening turn was sent during the stop", n)
	}
}

// S6C P1-1: a Stop while the relaunched child waits for its opening turn kills that child at
// once: no opening turn, and the stop does not wait InitWait.
func TestStopKillsTheChildBeingStarted(t *testing.T) {
	const stopWait = 500 * time.Millisecond
	a, e, eff := setupCfg(t, 200*time.Millisecond, func(c *Config) {
		c.TurnMax, c.StopWait, c.PrimeAfter = 500*time.Millisecond, stopWait, 3*time.Second
	}, "FAKE_INIT=first", "FAKE_MUTE=1")
	// Start waited PrimeAfter for the first child's init, then primed it: one turn so far.
	if res, _ := send(t, a, e, "m1", "mute"); res.State != deliver.Uncertain {
		t.Fatalf("m1 never answered: %+v, want uncertain", res)
	}
	for deadline := time.Now().Add(10 * time.Second); notes(t, a, "start") < 2; time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("no relaunch at TurnMax")
		}
	}
	began := time.Now()
	a.Stop()
	if d := time.Since(began); d > stopWait+time.Second {
		t.Errorf("Stop took %s with a child being started, want StopWait (%s) plus a margin", d, stopWait)
	}
	if n := effects(t, eff); n != 2 {
		t.Errorf("%d turns, want 2: the child being started was primed during the stop", n)
	}
}
