package deliver

import (
	"context"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// lateTarget blocks before its effect until release is closed, ignoring ctx; its Status answers
// from the effects it has made, as a target that remembers nothing else would.
type lateTarget struct {
	inc      string
	release  chan struct{}
	returned chan struct{}
	effects  atomic.Int64
}

func (l *lateTarget) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	<-l.release
	if l.effects.Add(1) == 1 {
		defer close(l.returned)
	}
	return Ack{Receipt: "late " + m.MandateID}, nil
}

func (l *lateTarget) Status(ctx context.Context, mandateID string) (Outcome, error) {
	if l.effects.Load() > 0 {
		return Received, nil
	}
	return NotReceived, nil
}

func waitState(e *Engine, target, id string, want State, within time.Duration) State {
	deadline := time.Now().Add(within)
	for {
		got := stateOf(e, target, id)
		if got == want || time.Now().After(deadline) {
			return got
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// S2D-1: Reconcile while the original adapter call is still running would refuse a delivery
// that can still take effect; a resend would then make a second effect.
func TestReconcileWaitsForTheRunningDelivery(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{Timeout: 200 * time.Millisecond})
	tg := &lateTarget{inc: "i", release: make(chan struct{}), returned: make(chan struct{})}
	e.Register("a", Target{Adapter: tg, Incarnation: func() (string, error) { return "i", nil }})
	m := Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "go"}

	res, err := e.Deliver(context.Background(), m)
	if err != nil || res.State != Uncertain {
		t.Fatalf("deliver past its deadline: %+v %v", res, err)
	}
	rres, rerr := e.Reconcile(context.Background(), "a", "m1")
	if refusalCode(rerr) != DeliveryStillRunning {
		t.Errorf("reconcile while the adapter still runs: %+v %v, want %s", rres, rerr, DeliveryStillRunning)
	}

	close(tg.release) // the adapter takes effect late
	<-tg.returned
	got := waitState(e, "a", "m1", Delivered, 2*time.Second)
	if got != Delivered {
		t.Errorf("after the late answer: %s, want delivered (the late answer is an observation)", got)
	}
	e.Deliver(context.Background(), m) // a resend
	if n := tg.effects.Load(); n != 1 {
		t.Errorf("%d effects, want 1", n)
	}
}
