package deliver

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"
)

// gatedTarget answers once released, with an Ack of n bytes; Status answers outcome.
type gatedTarget struct {
	n       int
	entered chan struct{} // closed once Deliver is entered
	release chan struct{} // Deliver answers once it is closed
	err     error         // when set, Deliver answers it instead of an Ack
	outcome Outcome
	mu      sync.Mutex
	calls   int
}

func newGated(n int) *gatedTarget {
	return &gatedTarget{n: n, entered: make(chan struct{}), release: make(chan struct{}), outcome: Received}
}

func (g *gatedTarget) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	g.mu.Lock()
	g.calls++
	g.mu.Unlock()
	close(g.entered)
	<-g.release
	if g.err != nil {
		return Ack{}, g.err
	}
	return Ack{Receipt: "took " + m.MandateID, Text: strings.Repeat("y", g.n)}, nil
}

func (g *gatedTarget) Status(ctx context.Context, mandateID, incarnation string) (Outcome, error) {
	return g.outcome, nil
}

func (g *gatedTarget) callCount() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.calls
}

func (g *gatedTarget) target() Target {
	return Target{Adapter: g, Incarnation: func() (string, error) { return "i", nil }}
}

// fillTo delivers small mandates to a until the log holds at least size bytes.
func fillTo(t *testing.T, e *Engine, a *fakeTarget, size int64) {
	t.Helper()
	for i := 0; ; i++ {
		e.log.mu.Lock()
		s := e.log.size
		e.log.mu.Unlock()
		if s >= size {
			return
		}
		m := Mandate{MandateID: fmt.Sprintf("f%05d", i), Target: "a", Incarnation: "i", Text: "t"}
		if res, err := e.Deliver(context.Background(), m); err != nil || res.State != Delivered {
			t.Fatalf("fill at %d bytes, %s: %+v %v", s, m.MandateID, res, err)
		}
	}
}

// R2-1: deliveries accepted together share the log's room. Once each is accepted, its final record
// (with the target's answer) must find room: otherwise a target that took the mandate leaves it
// uncertain, its answer lost. A delivery that cannot be given that room is refused log-full before
// its adapter is called.
func TestConcurrentDeliveriesKeepTheirAcks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	opt := Options{MaxLogBytes: 256 << 10}
	e, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	fillTo(t, e, a, 200<<10)

	const n, ackBytes = 8, 12 << 10
	gs := make([]*gatedTarget, n)
	type out struct {
		res Result
		err error
	}
	outs := make([]chan out, n)
	for i := range gs {
		gs[i] = newGated(ackBytes)
		e.Register(fmt.Sprintf("g%d", i), gs[i].target())
		outs[i] = make(chan out, 1)
	}
	for i := range gs {
		go func(i int) {
			res, err := e.Deliver(context.Background(), Mandate{MandateID: fmt.Sprintf("c%d", i), Target: fmt.Sprintf("g%d", i), Incarnation: "i", Text: "t"})
			outs[i] <- out{res, err}
		}(i)
	}
	// Every delivery is either inside its adapter or already answered (refused) before any answers.
	got := make([]*out, n)
	for i := range gs {
		select {
		case <-gs[i].entered:
		case o := <-outs[i]:
			got[i] = &o
		case <-time.After(10 * time.Second):
			t.Fatalf("c%d neither entered its adapter nor answered", i)
		}
	}
	// While they run, new mandates take whatever room is left.
	for j := 0; ; j++ {
		_, err := e.Deliver(context.Background(), Mandate{MandateID: fmt.Sprintf("x%03d", j), Target: "a", Incarnation: "i", Text: "t"})
		if refusalCode(err) == LogFull {
			break
		}
		if err != nil || j > 1000 {
			t.Fatalf("x%03d: %v", j, err)
		}
	}
	for i := range gs {
		close(gs[i].release)
	}
	delivered := 0
	for i := range gs {
		if got[i] == nil {
			o := <-outs[i]
			got[i] = &o
		}
		o := got[i]
		switch {
		case refusalCode(o.err) == LogFull:
			if gs[i].callCount() != 0 {
				t.Errorf("c%d refused %s after its adapter was called", i, LogFull)
			}
		case o.err != nil || o.res.State != Delivered || o.res.Ack == nil || len(o.res.Ack.Text) != ackBytes:
			t.Errorf("c%d, its adapter called and answered: %+v %v; want delivered with its %d-byte answer, or refused %s before the call",
				i, o.res, o.err, ackBytes, LogFull)
		default:
			delivered++
		}
	}
	if delivered == 0 {
		t.Errorf("no delivery went through: want at least one")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	e2, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	defer e2.Close()
	for i := range gs {
		if got[i].err != nil {
			continue
		}
		if en, ack, ok := e2.Mandate(fmt.Sprintf("c%d", i)); !ok || en.State != Delivered || ack == nil || len(ack.Text) != ackBytes {
			t.Errorf("c%d after a restart: %+v %v; want delivered with its answer", i, en, ack)
		}
	}
}

// failNext fails its next write whole (nothing written); later writes pass.
type failNext struct {
	logFile
	failed bool
}

func (f *failNext) Write(b []byte) (int, error) {
	if !f.failed {
		f.failed = true
		return 0, errors.New("disk error")
	}
	return f.logFile.Write(b)
}

// R2-1, log-broken: a write that fails breaks the log for every delivery. One whose adapter was
// already called when it finds the log broken may have been taken: that is no refusal (a refusal
// says nothing was sent). It is uncertain, cause log-write-failed, and never sent again.
func TestALogBrokenUnderAnotherDeliveryIsNoRefusalAfterTheCall(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	ga, gb := newGated(10), newGated(10)
	e.Register("a", ga.target())
	e.Register("b", gb.target())
	type out struct {
		res Result
		err error
	}
	oa, ob := make(chan out, 1), make(chan out, 1)
	mb := Mandate{MandateID: "b1", Target: "b", Incarnation: "i", Text: "t"}
	go func() {
		res, err := e.Deliver(context.Background(), Mandate{MandateID: "a1", Target: "a", Incarnation: "i", Text: "t"})
		oa <- out{res, err}
	}()
	go func() { res, err := e.Deliver(context.Background(), mb); ob <- out{res, err} }()
	<-ga.entered
	<-gb.entered
	e.log.mu.Lock()
	e.log.f = &failNext{logFile: e.log.f}
	e.log.mu.Unlock()
	close(ga.release)
	if o := <-oa; o.err == nil || refusalCode(o.err) != "" || o.res.State != Uncertain || o.res.Cause != CauseLogWrite {
		t.Errorf("a1, whose final write failed: %+v %v; want an error that is no refusal, uncertain, cause %s", o.res, o.err, CauseLogWrite)
	}
	close(gb.release)
	if o := <-ob; o.err == nil || refusalCode(o.err) != "" || o.res.State != Uncertain || o.res.Cause != CauseLogWrite {
		t.Errorf("b1, answered once the log was broken: %+v %v (refusal code %q); want an error that is no refusal, uncertain, cause %s",
			o.res, o.err, refusalCode(o.err), CauseLogWrite)
	}
	if _, err := e.Deliver(context.Background(), mb); refusalCode(err) != UncertainNeedsReconcile || gb.callCount() != 1 {
		t.Errorf("b1 sent again: %v, %d calls; want %s and 1 call", err, gb.callCount(), UncertainNeedsReconcile)
	}
	e.Close()
}

// receivedTarget answers every Deliver with an error that is not a refusal (uncertain), and every
// Status with received.
type receivedTarget struct{}

func (receivedTarget) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	return Ack{}, errors.New("the pipe broke after the write")
}
func (receivedTarget) Status(ctx context.Context, mandateID, incarnation string) (Outcome, error) {
	return Received, nil
}

// R2-1, reconcile: an uncertain delivery keeps room in the log for the record that settles it. Other
// records filling the log must not leave a reconcile unable to write its answer: it would answer an
// error and lose what the target said, and after a restart it would stay unreconcilable too.
func TestAnUncertainMandateKeepsRoomToBeReconciled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	opt := Options{MaxLogBytes: 16 << 10}
	e, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	e.Register("u", Target{Adapter: receivedTarget{}, Incarnation: func() (string, error) { return "i", nil }})
	for _, id := range []string{"u1", "u2"} {
		if res, err := e.Deliver(context.Background(), Mandate{MandateID: id, Target: "u", Incarnation: "i", Text: "t"}); err != nil || res.State != Uncertain {
			t.Fatalf("%s: %+v %v, want uncertain", id, res, err)
		}
	}
	// Records that need no room after them (refused deliveries) until the log takes no more.
	for i := 0; ; i++ {
		if i > 1000 {
			t.Fatal("the log never filled")
		}
		if err := e.transition(key{"z", fmt.Sprintf("z%04d", i)}, "i", Refused, "filler", "", nil); err != nil {
			break
		}
	}
	if res, err := e.Reconcile(context.Background(), "u", "u1"); err != nil || res.State != Delivered {
		t.Errorf("reconcile u1 on a full log: %+v %v; want delivered", res, err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	e2, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	defer e2.Close()
	e2.Register("u", Target{Adapter: receivedTarget{}, Incarnation: func() (string, error) { return "i", nil }})
	if res, err := e2.Reconcile(context.Background(), "u", "u2"); err != nil || res.State != Delivered {
		t.Errorf("reconcile u2 after a restart on a full log: %+v %v; want delivered", res, err)
	}
	if st := stateOf(e2, "u", "u1"); st != Delivered {
		t.Errorf("u1 after a restart: %s, want delivered", st)
	}
}

var atField = regexp.MustCompile(`"at":"[^"]*"`)

// ageLog moves every record of the log at path back by d, as if it had been written d earlier.
func ageLog(t *testing.T, path string, d time.Duration) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	b = atField.ReplaceAllFunc(b, func(m []byte) []byte {
		at, err := time.Parse(time.RFC3339Nano, string(m[6:len(m)-1]))
		if err != nil {
			t.Fatal(err)
		}
		return []byte(`"at":"` + at.Add(-d).Format(time.RFC3339Nano) + `"`)
	})
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
}

// R2-2: the log keeps one record per mandate for ever, so distinct mandates fill it for good; a
// restart's compaction then frees nothing, and every new mandate is refused log-full. A finished
// mandate (delivered or refused) older than the deduplication window is forgotten at compaction;
// an uncertain one never is, and one inside the window keeps its deduplication.
func TestOldFinishedMandatesAreForgottenAtRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	opt := Options{MaxLogBytes: 16 << 10}
	e, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	e.Register("u", Target{Adapter: receivedTarget{}, Incarnation: func() (string, error) { return "i", nil }})
	u1 := Mandate{MandateID: "u1", Target: "u", Incarnation: "i", Text: "t"}
	if res, err := e.Deliver(context.Background(), u1); err != nil || res.State != Uncertain {
		t.Fatalf("u1: %+v %v", res, err)
	}
	old := fillLog(t, e, a, "old", 1000)
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	// Fill and restart until a restart's compaction frees nothing: the log is full of mandates.
	for round := 1; ; round++ {
		if round > 30 {
			t.Fatal("the log never filled with mandates alone")
		}
		e, _, err := Open(path, opt)
		if err != nil {
			t.Fatal(err)
		}
		e.Register("a", a.target())
		d := fillLog(t, e, a, fmt.Sprintf("old%02d-", round), 1000)
		old = append(old, d...)
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		if len(d) == 0 {
			break
		}
	}
	ageLog(t, path, 40*24*time.Hour) // past the default window of 30 days

	e2, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	e2.Register("a", a.target())
	e2.Register("u", Target{Adapter: receivedTarget{}, Incarnation: func() (string, error) { return "i", nil }})
	calls := a.calls.Load()
	if res, err := e2.Deliver(context.Background(), Mandate{MandateID: "new1", Target: "a", Incarnation: "i", Text: "t"}); err != nil || res.State != Delivered {
		t.Errorf("a new mandate after a restart, the log full of mandates 40 days old: %+v %v; want delivered", res, err)
	}
	if _, err := e2.Deliver(context.Background(), u1); refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("u1, uncertain and 40 days old, sent again: %v; want %s (an uncertain mandate is never forgotten)", err, UncertainNeedsReconcile)
	}
	// A forgotten id is a new mandate: sent again, it is delivered again (the window is the guarantee).
	if res, err := e2.Deliver(context.Background(), old[0]); err != nil || res.State != Delivered || res.Replayed || a.calls.Load() != calls+2 {
		t.Errorf("%s, forgotten, sent again: %+v %v, %d calls; want delivered anew", old[0].MandateID, res, err, a.calls.Load()-calls)
	}
	// One still inside the window keeps its deduplication.
	if res, err := e2.Deliver(context.Background(), Mandate{MandateID: "new1", Target: "a", Incarnation: "i", Text: "t"}); err != nil || !res.Replayed {
		t.Errorf("new1 sent again: %+v %v; want replayed", res, err)
	}
	e2.Close()
}

// R2-2, without a restart: a running engine whose log has no room for a new mandate compacts it
// then and there, forgetting the finished mandates past the window; an uncertain one stays.
func TestARunningEngineForgetsOldMandatesWhenTheLogIsFull(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{MaxLogBytes: 16 << 10})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	e.Register("u", Target{Adapter: receivedTarget{}, Incarnation: func() (string, error) { return "i", nil }})
	u1 := Mandate{MandateID: "u1", Target: "u", Incarnation: "i", Text: "t"}
	if res, err := e.Deliver(context.Background(), u1); err != nil || res.State != Uncertain {
		t.Fatalf("u1: %+v %v", res, err)
	}
	old := fillLog(t, e, a, "old", 1000)
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "new0", Target: "a", Incarnation: "i", Text: "t"}); refusalCode(err) != LogFull {
		t.Fatalf("a new mandate on a log full of mandates inside the window: %v, want %s", err, LogFull)
	}
	timeNow = func() time.Time { return time.Now().Add(40 * 24 * time.Hour) }
	t.Cleanup(func() { timeNow = time.Now })
	calls := a.calls.Load()
	if res, err := e.Deliver(context.Background(), Mandate{MandateID: "new1", Target: "a", Incarnation: "i", Text: "t"}); err != nil || res.State != Delivered {
		t.Errorf("a new mandate 40 days later, no restart: %+v %v; want delivered", res, err)
	}
	if _, err := e.Deliver(context.Background(), u1); refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("u1, uncertain, sent again 40 days later: %v; want %s", err, UncertainNeedsReconcile)
	}
	if _, _, ok := e.Mandate(old[0].MandateID); ok {
		t.Errorf("%s, finished 40 days ago, is still remembered after the compaction", old[0].MandateID)
	}
	if a.calls.Load() != calls+1 {
		t.Errorf("%d adapter calls, want 1", a.calls.Load()-calls)
	}
}

// R2-1, late answer: a delivery uncertain at its deadline, its call still running, keeps room for
// the call's late answer and then for a reconcile, however full the log gets meanwhile.
func TestALateAnswerThenAReconcileFitOnAFullLog(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{MaxLogBytes: 16 << 10, Timeout: 200 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	g := newGated(0)
	g.err = errors.New("the pipe broke after the write")
	e.Register("g", g.target())
	if res, err := e.Deliver(context.Background(), Mandate{MandateID: "g1", Target: "g", Incarnation: "i", Text: "t"}); err != nil || res.State != Uncertain || res.Cause != CauseDeadline {
		t.Fatalf("g1 at its deadline: %+v %v; want uncertain, cause %s", res, err, CauseDeadline)
	}
	for i := 0; ; i++ {
		if i > 1000 {
			t.Fatal("the log never filled")
		}
		if err := e.transition(key{"z", fmt.Sprintf("z%04d", i)}, "i", Refused, "filler", "", nil); err != nil {
			break
		}
	}
	close(g.release)
	deadline := time.Now().Add(5 * time.Second)
	en := entryOf(e, "g", "g1")
	for (en.StillRunning || en.Cause != CauseAdapterError) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
		en = entryOf(e, "g", "g1")
	}
	if en.State != Uncertain || en.Cause != CauseAdapterError || en.StillRunning {
		t.Errorf("g1 after its late error answer on a full log: %+v; want uncertain, cause %s, its call returned", en, CauseAdapterError)
	}
	if res, err := e.Reconcile(context.Background(), "g", "g1"); err != nil || res.State != Delivered {
		t.Errorf("reconcile g1 on a full log: %+v %v; want delivered", res, err)
	}
}

// Review R2 P3-1 (its test, from the independent review): an online compaction that breaks the log
// (its directory sync fails after the rename) answers the mandate that triggered it log-broken,
// not log-full (which invites a retry), and Close then says log-broken too.
func TestAFatalOnlineCompactionSaysLogBroken(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{MaxLogBytes: 64 << 10})
	if err != nil {
		t.Fatal(err)
	}
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	fillLog(t, e, a, "f", 10000)
	timeNow = func() time.Time { return time.Now().Add(40 * 24 * time.Hour) }
	t.Cleanup(func() { timeNow = time.Now })
	dirSync = func(string) error { return errors.New("injected directory sync failure") }
	t.Cleanup(func() { dirSync = syncDir })
	calls := a.calls.Load()
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "new", Target: "a", Incarnation: "i", Text: "t"}); refusalCode(err) != LogBroken || a.calls.Load() != calls {
		t.Errorf("the mandate whose compaction broke the log: %v, %d calls; want %s and no call", err, a.calls.Load()-calls, LogBroken)
	}
	if err := e.Close(); refusalCode(err) != LogBroken {
		t.Errorf("Close after a compaction broke the log: %v, want %s", err, LogBroken)
	}
}

// Review R2 P3-4: a mandate with long keys delivered before the bound on keys existed is still
// replayed; only a new delivery is refused bad-mandate.
func TestAKnownMandateWithLongKeysIsReplayed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	m := Mandate{MandateID: strings.Repeat("L", 600), Target: "a", Incarnation: "i", Text: "t"}
	k, fp := key{"a", m.MandateID}, Fingerprint(m)
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transitionFP(k, "i", s, "", nil, fp); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.transition(k, "i", Delivered, "", "", &Ack{Receipt: "old"}); err != nil {
		t.Fatal(err)
	}
	if res, err := e.Deliver(context.Background(), m); err != nil || !res.Replayed {
		t.Errorf("a known mandate with long keys: %+v %v; want replayed", res, err)
	}
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: strings.Repeat("N", 600), Target: "a", Incarnation: "i", Text: "t"}); refusalCode(err) != BadMandate || a.calls.Load() != 0 {
		t.Errorf("a new mandate with long keys: %v, %d calls; want %s, no call", err, a.calls.Load(), BadMandate)
	}
}
