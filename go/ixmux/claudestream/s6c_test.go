package claudestream

import (
	"bytes"
	"context"
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

// S6C, the end-of-turn window (Branche: closed): the adapter frees its reservation before it
// hands the turn back. The watcher is held between the end of the turn and its release; a mandate
// sent as soon as Deliver has returned, as the engine would once its slot is free, is delivered.
func TestReservationIsFreeWhenDeliverReturns(t *testing.T) {
	a, _, _ := setup(t, 5*time.Second)
	testHookBeforeRelease = func() { time.Sleep(300 * time.Millisecond) }
	t.Cleanup(func() { testHookBeforeRelease = nil })
	inc, _ := a.Incarnation()
	for _, id := range []string{"w1", "w2"} {
		var ack deliver.Ack
		var err error
		within(t, 10*time.Second, "Deliver "+id, func() {
			ack, err = a.Deliver(context.Background(), deliver.Mandate{MandateID: id, Target: "c", Incarnation: inc, Text: "x"})
		})
		if err != nil || !ack.AckSeen {
			t.Fatalf("%s, sent once the previous Deliver returned: %+v %v, want delivered", id, ack, err)
		}
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
	within(t, 10*time.Second, "Stop with a child being started", func() { a.Stop() })
	if d := time.Since(began); d > stopWait+time.Second {
		t.Errorf("Stop took %s with a child being started, want StopWait (%s) plus a margin", d, stopWait)
	}
	if n := effects(t, eff); n != 2 {
		t.Errorf("%d turns, want 2: the child being started was primed during the stop", n)
	}
}
