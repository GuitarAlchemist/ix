package deliver

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// closeRace starts Close once the delivery or reconcile under test has reached its injection
// point, and reports whether Close waited for it (closeWaits fired) or returned first.
func closeRace(t *testing.T, e *Engine) (waited <-chan struct{}, closed <-chan error) {
	t.Helper()
	w, c := make(chan struct{}), make(chan error, 1)
	var once sync.Once
	testHookCloseWaits = func() { once.Do(func() { close(w) }) }
	t.Cleanup(func() { testHookCloseWaits = nil })
	go func() { c <- e.Close() }()
	return w, c
}

// S3B-1 (Codex round 6, P1): an adapter that succeeds while Close runs. Close must wait for the
// final state to be written; a restart then reads delivered, never uncertain.
func TestCloseWaitsForTheFinalWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	tg := &fakeTarget{inc: "i"}
	e.Register("a", tg.target())
	reached, resume := make(chan struct{}), make(chan struct{})
	testHookBeforeFinish = func() { close(reached); <-resume }
	t.Cleanup(func() { testHookBeforeFinish = nil })

	done := make(chan error, 1)
	go func() {
		_, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "go"})
		done <- err
	}()
	<-reached // the adapter answered; its final state is not written yet
	waited, closed := closeRace(t, e)
	var cerr error
	select {
	case <-waited:
		close(resume)
		cerr = <-closed
	case cerr = <-closed: // Close did not wait
		close(resume)
	}
	derr := <-done
	if cerr != nil {
		t.Errorf("Close: %v, want nil once the final state is written", cerr)
	}
	e2, _ := openEngine(t, path, Options{})
	if got := stateOf(e2, "a", "m1"); got != Delivered {
		t.Errorf("after the restart: %s (Deliver %v), want delivered: the success was lost by Close", got, derr)
	}
}

// uncertainLog writes a log whose mandate m1 for target a, incarnation i, was left delivering:
// Open makes it uncertain.
func uncertainLog(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	log := logHeader + `{"at":"2026-11-06T00:00:00Z","target":"a","mandate_id":"m1","incarnation":"i","state":"delivering","fp":"x"}` + "\n"
	if err := os.WriteFile(path, []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// holdStatus answers Received, once hold is closed; entered is closed when Status starts.
type holdStatus struct {
	entered, hold chan struct{}
	calls         int
}

func (h *holdStatus) Deliver(ctx context.Context, m Mandate) (Ack, error) { return Ack{}, nil }
func (h *holdStatus) Status(ctx context.Context, mandateID, incarnation string) (Outcome, error) {
	h.calls++
	if h.entered != nil {
		close(h.entered)
	}
	<-h.hold
	return Received, nil
}

// S3B-1 (Codex round 6, P1): a Status running when Close begins. Close waits for its answer to
// be written: a restart reads delivered, and Close returns nil.
func TestCloseWaitsForAReconcile(t *testing.T) {
	path := uncertainLog(t)
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	h := &holdStatus{entered: make(chan struct{}), hold: make(chan struct{})}
	e.Register("a", Target{Adapter: h, Incarnation: func() (string, error) { return "i", nil }})
	done := make(chan error, 1)
	go func() { _, err := e.Reconcile(context.Background(), "a", "m1"); done <- err }()
	<-h.entered
	waited, closed := closeRace(t, e)
	var cerr error
	select {
	case <-waited:
		close(h.hold)
		cerr = <-closed
	case cerr = <-closed:
		close(h.hold)
	}
	rerr := <-done
	if cerr != nil {
		t.Errorf("Close: %v, want nil once the reconcile is written", cerr)
	}
	e2, _ := openEngine(t, path, Options{})
	if got := stateOf(e2, "a", "m1"); got != Delivered {
		t.Errorf("after the restart: %s (Reconcile %v), want delivered", got, rerr)
	}
}

// S3B-1 (Codex round 6, P1): once Close has begun, Reconcile and Deliver refuse engine-closed
// without calling the target.
func TestCloseBegunRefusesNewCalls(t *testing.T) {
	path := uncertainLog(t)
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	h := &holdStatus{hold: make(chan struct{})}
	close(h.hold)
	e.Register("a", Target{Adapter: h, Incarnation: func() (string, error) { return "i", nil }})
	busy := &fakeTarget{inc: "i", block: make(chan struct{}), entered: make(chan struct{})}
	e.Register("b", busy.target())
	go e.Deliver(context.Background(), Mandate{MandateID: "m2", Target: "b", Incarnation: "i"})
	<-busy.entered // Close will wait for this call
	waited, closed := closeRace(t, e)
	<-waited
	_, rerr := e.Reconcile(context.Background(), "a", "m1")
	_, derr := e.Deliver(context.Background(), Mandate{MandateID: "m3", Target: "a", Incarnation: "i"})
	close(busy.block)
	if cerr := <-closed; cerr != nil {
		t.Errorf("Close: %v", cerr)
	}
	if refusalCode(rerr) != EngineClosed || h.calls != 0 {
		t.Errorf("Reconcile after Close began: %v, %d Status calls; want %s and none", rerr, h.calls, EngineClosed)
	}
	if refusalCode(derr) != EngineClosed {
		t.Errorf("Deliver after Close began: %v, want %s", derr, EngineClosed)
	}
}

// S3B-1: a transition that could not be written makes Close fail, so a stop does not say 0.
func TestCloseReportsALostTransition(t *testing.T) {
	e, _, err := Open(filepath.Join(t.TempDir(), "cmd.jsonl"), Options{Timeout: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	e.Register("a", (&fakeTarget{inc: "i"}).target())
	e.log.f = failingSync{e.log.f}
	e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i"})
	if err := e.Close(); refusalCode(err) != LogBroken {
		t.Errorf("Close after a failed write: %v, want %s", err, LogBroken)
	}
}
