package deliver

import (
	"context"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// firstLate blocks its first Deliver before the effect until release is closed, ignoring ctx;
// later calls take effect at once. Status answers from the effects made so far.
type firstLate struct {
	release  chan struct{}
	returned chan struct{}
	calls    atomic.Int64
	effects  atomic.Int64
}

func (f *firstLate) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	if f.calls.Add(1) == 1 {
		<-f.release
		defer close(f.returned)
	}
	f.effects.Add(1)
	return Ack{Receipt: "took " + m.MandateID}, nil
}

func (f *firstLate) Status(ctx context.Context, mandateID string) (Outcome, error) {
	if f.effects.Load() > 0 {
		return Received, nil
	}
	return NotReceived, nil
}

func (f *firstLate) target() Target {
	return Target{Adapter: f, Incarnation: func() (string, error) { return "i", nil }}
}

// S2F-1 (Codex round 4): Close, then a reopen in the same process, would forget that an adapter
// call is still running; reconciled not-received and sent again, the mandate would take effect
// twice once the first call resumes. Close refuses while a call runs and keeps the lock.
func TestCloseKeepsTheGuardOfARunningCall(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	tg := &firstLate{release: make(chan struct{}), returned: make(chan struct{})}
	m := Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "go"}
	e1, _, err := Open(path, Options{Timeout: 50 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	e1.Register("a", tg.target())
	if res, err := e1.Deliver(context.Background(), m); err != nil || res.State != Uncertain {
		t.Fatalf("deliver past its deadline: %+v %v", res, err)
	}

	cerr := e1.Close()
	e2, _, oerr := Open(path, Options{Timeout: 50 * time.Millisecond})
	if oerr == nil {
		// The guard is lost: reconcile, resend, then the first call resumes.
		e2.Register("a", tg.target())
		rres, rerr := e2.Reconcile(context.Background(), "a", "m1")
		dres, derr := e2.Deliver(context.Background(), m)
		close(tg.release)
		<-tg.returned
		e2.Close()
		t.Fatalf("Close %v, then reopen accepted while the call ran: reconcile %+v %v, resend %+v %v; %d effects, want at most 1",
			cerr, rres, rerr, dres, derr, tg.effects.Load())
	}
	if refusalCode(cerr) != CallsInFlight {
		t.Errorf("Close with a running call: %v, want %s", cerr, CallsInFlight)
	}
	if refusalCode(oerr) != JournalLocked {
		t.Errorf("reopen while the call runs: %v, want %s", oerr, JournalLocked)
	}

	close(tg.release) // the call returns; its late answer is recorded
	<-tg.returned
	deadline := time.Now().Add(2 * time.Second)
	for {
		if cerr = e1.Close(); cerr == nil || time.Now().After(deadline) {
			break
		}
	}
	if cerr != nil {
		t.Fatalf("Close once the call returned: %v", cerr)
	}
	e3, _ := openEngine(t, path, Options{})
	e3.Register("a", tg.target())
	if res, err := e3.Deliver(context.Background(), m); err != nil || !res.Replayed {
		t.Errorf("resend after the late answer: %+v %v, want a replay", res, err)
	}
	if n := tg.effects.Load(); n != 1 {
		t.Errorf("%d effects, want 1", n)
	}
}
