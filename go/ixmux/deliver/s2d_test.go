package deliver

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

// S2D-3: a Register that replaces the target between the incarnation check and the call.
func TestReplacedTargetGetsNoStaleMandate(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	old := &fakeTarget{inc: "i"}
	repl := &fakeTarget{inc: "i2"}
	e.Register("a", old.target())
	testHookAfterIncarnation = func() { e.Register("a", repl.target()) }
	defer func() { testHookAfterIncarnation = nil }()
	res, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "go"})
	if refusalCode(err) != StaleIncarnation {
		t.Errorf("target replaced after the check: %+v %v, want %s", res, err, StaleIncarnation)
	}
	if old.calls.Load() != 0 || repl.calls.Load() != 0 {
		t.Errorf("old adapter %d calls, new adapter %d calls; want 0 and 0", old.calls.Load(), repl.calls.Load())
	}
}

// hangingStatus answers Status only once hold is closed, ignoring ctx.
type hangingStatus struct{ hold chan struct{} }

func (h hangingStatus) Deliver(ctx context.Context, m Mandate) (Ack, error) { return Ack{}, nil }
func (h hangingStatus) Status(ctx context.Context, id string) (Outcome, error) {
	<-h.hold
	return Unknown, nil
}

// S2D-4: reconciliations whose Status ignores its context do not pile up goroutines: at most
// one in flight per (target, mandate id) and MaxInFlight in all.
func TestReconcileGoroutinesAreBounded(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	var log strings.Builder
	log.WriteString(logHeader)
	for i := range 10 {
		fmt.Fprintf(&log, `{"at":"2026-11-05T00:00:00Z","target":"a","mandate_id":"m%d","incarnation":"i","state":"delivering"}`+"\n", i)
	}
	if err := os.WriteFile(path, []byte(log.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	const slots = 4
	e, rec := openEngine(t, path, Options{Timeout: 30 * time.Millisecond, MaxInFlight: slots})
	if rec.MadeUncertain != 10 {
		t.Fatalf("recovery %+v", rec)
	}
	h := hangingStatus{hold: make(chan struct{})}
	e.Register("a", Target{Adapter: h, Incarnation: func() (string, error) { return "i", nil }})

	base := runtime.NumGoroutine()
	codes := map[string]int{}
	for range 5 {
		for i := range 10 {
			res, err := e.Reconcile(context.Background(), "a", fmt.Sprintf("m%d", i))
			if err == nil && res.State != Uncertain {
				t.Fatalf("m%d: %+v", i, res)
			}
			codes[refusalCode(err)]++
		}
	}
	grown := runtime.NumGoroutine() - base
	t.Logf("goroutines grew by %d after 50 reconciles; answers by refusal code: %v", grown, codes)
	if grown > slots {
		t.Errorf("goroutines grew by %d, bound %d (MaxInFlight)", grown, slots)
	}
	close(h.hold) // the hung Status calls return
	time.Sleep(100 * time.Millisecond)
	if grown := runtime.NumGoroutine() - base; grown > 0 {
		t.Errorf("%d goroutines left once Status answered", grown)
	}
}

// halfWriter writes half of its first write, then fails it; later writes pass.
type halfWriter struct {
	logFile
	failed bool
}

func (h *halfWriter) Write(b []byte) (int, error) {
	if h.failed {
		return h.logFile.Write(b)
	}
	h.failed = true
	n, _ := h.logFile.Write(b[:len(b)/2])
	return n, errors.New("disk full")
}

// S2D-5: after a partial write, the log stops: no fragment is glued to a later record, and a
// restart cuts the fragment as a torn tail instead of refusing the log as corrupt.
func TestPartialWriteBreaksTheLog(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	tg := &fakeTarget{inc: "i"}
	e.Register("a", tg.target())
	e.log.f = &halfWriter{logFile: e.log.f}
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i"}); err == nil {
		t.Fatal("a delivery whose accepted record was half written succeeded")
	}
	_, err = e.Deliver(context.Background(), Mandate{MandateID: "m2", Target: "a", Incarnation: "i"})
	if refusalCode(err) != LogBroken {
		t.Errorf("after a partial write: %v, want %s", err, LogBroken)
	}
	if tg.calls.Load() != 0 {
		t.Errorf("%d adapter calls, want 0", tg.calls.Load())
	}
	e.Close()
	e2, rec, err := Open(path, Options{})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer e2.Close()
	if rec.Torn == nil || rec.Records != 0 {
		t.Errorf("reopen: %+v, want the fragment cut as a torn tail and no record", rec)
	}
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
