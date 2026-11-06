package deliver

import (
	"context"
	"errors"
	"os"
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

func (f *firstLate) Status(ctx context.Context, mandateID, incarnation string) (Outcome, error) {
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

// statusFn is a target whose Status runs fn, then answers NotReceived.
type statusFn struct {
	inc string
	fn  func()
}

func (s *statusFn) Deliver(ctx context.Context, m Mandate) (Ack, error) { return Ack{}, nil }
func (s *statusFn) Status(ctx context.Context, mandateID, incarnation string) (Outcome, error) {
	if s.fn != nil {
		s.fn()
	}
	return NotReceived, nil
}
func (s *statusFn) target() Target {
	return Target{Adapter: s, Incarnation: func() (string, error) { return s.inc, nil }}
}

// S2F-2 (Codex round 4): a delivery left uncertain for incarnation i1 is not reconciled by the
// target's next incarnation, which cannot know what i1 received. Nothing is written.
func TestReconcileAsksOnlyTheMandatesIncarnation(t *testing.T) {
	for _, tc := range []struct {
		name string
		reg  func(e *Engine) // registers target a
	}{
		{"restarted before", func(e *Engine) { e.Register("a", (&statusFn{inc: "i2"}).target()) }},
		{"restarted during Status", func(e *Engine) {
			e.Register("a", (&statusFn{inc: "i1", fn: func() { e.Register("a", (&statusFn{inc: "i2"}).target()) }}).target())
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "cmd.jsonl")
			log := logHeader + `{"at":"2026-11-05T00:00:00Z","target":"a","mandate_id":"m1","incarnation":"i1","state":"accepted","fp":"x"}` + "\n" +
				`{"at":"2026-11-05T00:00:01Z","target":"a","mandate_id":"m1","incarnation":"i1","state":"delivering"}` + "\n"
			if err := os.WriteFile(path, []byte(log), 0o600); err != nil {
				t.Fatal(err)
			}
			e, rec := openEngine(t, path, Options{})
			if rec.MadeUncertain != 1 {
				t.Fatalf("recovery %+v", rec)
			}
			before, _ := os.ReadFile(path)
			tc.reg(e)
			res, err := e.Reconcile(context.Background(), "a", "m1")
			if refusalCode(err) != StaleIncarnation {
				t.Errorf("reconcile of an i1 delivery: %+v %v, want %s", res, err, StaleIncarnation)
			}
			if got := stateOf(e, "a", "m1"); got != Uncertain {
				t.Errorf("state %s, want uncertain", got)
			}
			if after, _ := os.ReadFile(path); string(after) != string(before) {
				t.Errorf("reconcile wrote to the log:\n%s", after[len(before):])
			}
		})
	}
}

// S2F-3 (Codex round 4): an Open whose directory sync failed leaves the new log behind; the next
// Open syncs the directory again before its first transition, instead of trusting a log whose
// directory entry may not be durable.
func TestFailedDirectorySyncIsRetried(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	syncs := 0
	dirSync = func(dir string) error { syncs++; return errors.New("injected directory sync failure") }
	t.Cleanup(func() { dirSync = syncDir })
	if e, _, err := Open(path, Options{}); err == nil {
		e.Close()
		t.Fatal("Open succeeded although the directory sync failed")
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the failed Open left no log behind (%v): the scenario needs one", err)
	}
	syncs = 0
	dirSync = func(dir string) error { syncs++; return nil }
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if syncs == 0 {
		t.Error("the second Open trusted the log left by the failed one without syncing its directory")
	}
}

// S2F-4 (P2): Status shows a delivery whose adapter call has not returned.
func TestStatusShowsStillRunning(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{Timeout: 50 * time.Millisecond})
	tg := &firstLate{release: make(chan struct{}), returned: make(chan struct{})}
	e.Register("a", tg.target())
	if res, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i"}); err != nil || res.State != Uncertain {
		t.Fatalf("deliver past its deadline: %+v %v", res, err)
	}
	if st := e.Status(); len(st) != 1 || !st[0].StillRunning {
		t.Errorf("while the adapter call runs: %+v, want still_running", st)
	}
	close(tg.release)
	<-tg.returned
	waitState(e, "a", "m1", Delivered, 2*time.Second)
	deadline := time.Now().Add(2 * time.Second)
	for e.Status()[0].StillRunning && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if st := e.Status(); st[0].StillRunning {
		t.Errorf("once the call returned: %+v, want not still_running", st)
	}
}

// S2F-4 (P2): one Status call per (target, mandate id) at a time, even with slots to spare.
func TestReconcileIsOnePerKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	log := logHeader + `{"at":"2026-11-05T00:00:00Z","target":"a","mandate_id":"m1","incarnation":"i","state":"delivering"}` + "\n"
	if err := os.WriteFile(path, []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	e, _ := openEngine(t, path, Options{Timeout: 30 * time.Millisecond, MaxInFlight: 16})
	h := hangingStatus{hold: make(chan struct{}), calls: new(atomic.Int64)}
	defer close(h.hold)
	e.Register("a", Target{Adapter: h, Incarnation: func() (string, error) { return "i", nil }})
	codes := map[string]int{}
	for range 5 {
		_, err := e.Reconcile(context.Background(), "a", "m1")
		codes[refusalCode(err)]++
	}
	if n := h.calls.Load(); n != 1 || codes[InFlight] != 4 {
		t.Errorf("5 reconciles of one key with 16 slots: %d Status calls, answers %v; want 1 call and 4 %s", n, codes, InFlight)
	}
}

// failingSync writes, then fails every Sync.
type failingSync struct{ logFile }

func (f failingSync) Sync() error { return errors.New("injected sync failure") }

// S2F-4 (P2): a failed Sync breaks the log as a partial write does: the transition does not
// count, the adapter is not called, and later appends are refused.
func TestFailedSyncBreaksTheLog(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	tg := &fakeTarget{inc: "i"}
	e.Register("a", tg.target())
	e.log.f = failingSync{e.log.f}
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i"}); err == nil {
		t.Fatal("a delivery whose accepted record failed its sync succeeded")
	}
	if got := stateOf(e, "a", "m1"); got != "" {
		t.Errorf("the unsynced transition counted: state %s", got)
	}
	_, err := e.Deliver(context.Background(), Mandate{MandateID: "m2", Target: "a", Incarnation: "i"})
	if refusalCode(err) != LogBroken {
		t.Errorf("after a failed sync: %v, want %s", err, LogBroken)
	}
	if tg.calls.Load() != 0 {
		t.Errorf("%d adapter calls, want 0", tg.calls.Load())
	}
}
