package deliver

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeTarget remembers nothing: every Deliver call is one more effect, counted in memory and,
// when file is set, appended to that file.
type fakeTarget struct {
	inc     string
	calls   atomic.Int64
	effects atomic.Int64
	file    string
	block   chan struct{} // when set, Deliver blocks on it after the effect, ignoring ctx
	entered chan struct{} // when set, closed once Deliver is entered
	outcome Outcome       // what Status answers
}

func (f *fakeTarget) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	f.calls.Add(1)
	f.effects.Add(1)
	if f.file != "" {
		if err := appendEffect(f.file, m.MandateID); err != nil {
			return Ack{}, err
		}
	}
	if f.entered != nil {
		close(f.entered)
	}
	if f.block != nil {
		<-f.block
	}
	return Ack{Receipt: "took " + m.MandateID}, nil
}

func (f *fakeTarget) Status(ctx context.Context, mandateID string) (Outcome, error) {
	return f.outcome, nil
}

func (f *fakeTarget) target() Target {
	return Target{Adapter: f, Incarnation: func() (string, error) { return f.inc, nil }}
}

func appendEffect(path, id string) error {
	fh, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer fh.Close()
	if _, err := fh.WriteString(id + "\n"); err != nil {
		return err
	}
	return fh.Sync()
}

func countEffects(t *testing.T, path string) int {
	t.Helper()
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Count(string(b), "\n")
}

func openEngine(t *testing.T, path string, opt Options) (*Engine, Recovery) {
	t.Helper()
	e, rec, err := Open(path, opt)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { e.Close() })
	return e, rec
}

func refusalCode(err error) string {
	var r *Refusal
	if errors.As(err, &r) {
		return r.Code
	}
	return ""
}

func stateOf(e *Engine, target, id string) State {
	for _, en := range e.Status() {
		if en.Target == target && en.MandateID == id {
			return en.State
		}
	}
	return ""
}

// Oracle 1: the adapter acts, the caller loses the answer and sends the same mandate again.
func TestLostAckThenRetryIsReplayed(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	tg := &fakeTarget{inc: "inc-1"}
	e.Register("a", tg.target())
	m := Mandate{MandateID: "m1", Target: "a", Incarnation: "inc-1", Text: "hello"}
	if _, err := e.Deliver(context.Background(), m); err != nil { // the answer is lost
		t.Fatal(err)
	}
	res, err := e.Deliver(context.Background(), m)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Replayed || res.State != Delivered || res.Ack == nil || res.Ack.Receipt != "took m1" {
		t.Errorf("retry: %+v", res)
	}
	if tg.effects.Load() != 1 || tg.calls.Load() != 1 {
		t.Errorf("%d effects, %d adapter calls; want 1 and 1", tg.effects.Load(), tg.calls.Load())
	}
}

// Oracle 3: target A never returns; B completes meanwhile; A ends uncertain at its timeout.
func TestBlockedTargetDoesNotHoldOthers(t *testing.T) {
	const timeout = time.Second
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{Timeout: timeout})
	a := &fakeTarget{inc: "ia", block: make(chan struct{}), entered: make(chan struct{})}
	b := &fakeTarget{inc: "ib"}
	e.Register("a", a.target())
	e.Register("b", b.target())

	var resA Result
	var errA error
	var elapsedA time.Duration
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		start := time.Now()
		resA, errA = e.Deliver(context.Background(), Mandate{MandateID: "ma", Target: "a", Incarnation: "ia"})
		elapsedA = time.Since(start)
	}()
	<-a.entered

	start := time.Now()
	resB, err := e.Deliver(context.Background(), Mandate{MandateID: "mb", Target: "b", Incarnation: "ib"})
	elapsedB := time.Since(start)
	if err != nil || resB.State != Delivered {
		t.Fatalf("B: %+v %v", resB, err)
	}
	t.Logf("B delivered in %v while A was blocked", elapsedB)
	if elapsedB > timeout/2 {
		t.Errorf("B took %v while A was blocked", elapsedB)
	}

	wg.Wait()
	t.Logf("A ended %s after %v", resA.State, elapsedA)
	if errA != nil || resA.State != Uncertain {
		t.Fatalf("A: %+v %v", resA, errA)
	}
	if elapsedA < timeout || elapsedA > timeout+timeout/2 {
		t.Errorf("A ended after %v, timeout %v", elapsedA, timeout)
	}
	_, err = e.Deliver(context.Background(), Mandate{MandateID: "ma", Target: "a", Incarnation: "ia"})
	if refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("resending A: %v, want %s", err, UncertainNeedsReconcile)
	}
	// A's adapter answers at last: its late answer settles the uncertain delivery, once.
	close(a.block)
	if got := waitState(e, "a", "ma", Delivered, 2*time.Second); got != Delivered {
		t.Errorf("A after its late answer: %s, want delivered", got)
	}
	if res, err := e.Deliver(context.Background(), Mandate{MandateID: "ma", Target: "a", Incarnation: "ia"}); err != nil || !res.Replayed {
		t.Errorf("resending A after its late answer: %+v %v, want replayed", res, err)
	}
	if a.effects.Load() != 1 || a.calls.Load() != 1 {
		t.Errorf("A: %d effects, %d calls; want 1 and 1", a.effects.Load(), a.calls.Load())
	}
}

// Oracle 4: a stale incarnation is refused before any adapter call.
func TestStaleIncarnationNeverCallsTheAdapter(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	tg := &fakeTarget{inc: "inc-2"}
	e.Register("a", tg.target())
	_, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "inc-1"})
	if refusalCode(err) != StaleIncarnation {
		t.Errorf("%v, want %s", err, StaleIncarnation)
	}
	if tg.calls.Load() != 0 || len(e.Status()) != 0 {
		t.Errorf("%d adapter calls, %d deliveries; want 0 and 0", tg.calls.Load(), len(e.Status()))
	}
}

// Oracle 5: a torn last record is ignored with a typed indication; earlier states are kept.
func TestTornLastRecordIsIgnored(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	tg := &fakeTarget{inc: "i"}
	e.Register("a", tg.target())
	for _, id := range []string{"m1", "m2"} {
		if _, err := e.Deliver(context.Background(), Mandate{MandateID: id, Target: "a", Incarnation: "i"}); err != nil {
			t.Fatal(err)
		}
	}
	e.Close()
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	good := st.Size()
	torn := `{"at":"2026-11-05T00:00:00Z","target":"a","mandate_id":"m3","sta`
	fh, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	fh.WriteString(torn)
	fh.Close()

	e2, rec := openEngine(t, path, Options{})
	if rec.Torn == nil || *rec.Torn != (Torn{At: good, Bytes: int64(len(torn))}) {
		t.Fatalf("recovery %+v, want torn tail at %d of %d bytes", rec, good, len(torn))
	}
	if stateOf(e2, "a", "m1") != Delivered || stateOf(e2, "a", "m2") != Delivered || len(e2.Status()) != 2 {
		t.Errorf("status after a torn tail: %+v", e2.Status())
	}
	// The log goes on cleanly after the cut.
	e2.Register("a", tg.target())
	if _, err := e2.Deliver(context.Background(), Mandate{MandateID: "m3", Target: "a", Incarnation: "i"}); err != nil {
		t.Fatal(err)
	}
	e2.Close()
	e3, rec3 := openEngine(t, path, Options{})
	if rec3.Torn != nil || stateOf(e3, "a", "m3") != Delivered {
		t.Errorf("reopen after the cut: %+v, m3 %s", rec3, stateOf(e3, "a", "m3"))
	}
}

func TestCorruptRecordInsideTheLogIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	if err := os.WriteFile(path, []byte(logHeader+"not json\n{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, _, err := Open(path, Options{})
	if refusalCode(err) != LogCorrupt {
		t.Errorf("%v, want %s", err, LogCorrupt)
	}
}

func TestBounds(t *testing.T) {
	dir := t.TempDir()
	e, _ := openEngine(t, filepath.Join(dir, "cmd.jsonl"), Options{MaxInFlight: 1, Timeout: time.Second})
	a := &fakeTarget{inc: "i", block: make(chan struct{}), entered: make(chan struct{})}
	b := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	e.Register("b", b.target())

	_, err := e.Deliver(context.Background(), Mandate{MandateID: "big", Target: "b", Incarnation: "i", Text: strings.Repeat("x", MaxText+1)})
	if refusalCode(err) != TextTooLong || b.calls.Load() != 0 {
		t.Errorf("text of %d bytes: %v, %d calls", MaxText+1, err, b.calls.Load())
	}

	done := make(chan struct{})
	var resA Result
	var errA error
	go func() {
		defer close(done)
		resA, errA = e.Deliver(context.Background(), Mandate{MandateID: "ma", Target: "a", Incarnation: "i"})
	}()
	<-a.entered
	start := time.Now()
	_, err = e.Deliver(context.Background(), Mandate{MandateID: "mb", Target: "b", Incarnation: "i"})
	if refusalCode(err) != Busy || time.Since(start) > 500*time.Millisecond || b.calls.Load() != 0 {
		t.Errorf("past MaxInFlight: %v after %v, %d calls; want an immediate %s", err, time.Since(start), b.calls.Load(), Busy)
	}
	close(a.block)
	<-done
	if errA != nil || resA.State != Delivered || a.calls.Load() != 1 {
		t.Errorf("the background delivery: %+v %v, %d calls; want delivered once", resA, errA, a.calls.Load())
	}

	big := filepath.Join(dir, "big.jsonl")
	if err := os.WriteFile(big, []byte(strings.Repeat("{}\n", 100)), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Open(big, Options{MaxLogBytes: 100}); refusalCode(err) != LogTooLarge {
		t.Errorf("log of 300 bytes, bound 100: %v, want %s", err, LogTooLarge)
	}
}

func TestStatusGivesStateAgeAndLastTransition(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	tg := &fakeTarget{inc: "i"}
	e.Register("a", tg.target())
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i"}); err != nil {
		t.Fatal(err)
	}
	st := e.Status()
	if len(st) != 1 || st[0].State != Delivered || st[0].Last != "delivering->delivered" || st[0].Age < 0 || st[0].Since.IsZero() {
		t.Errorf("status %+v", st)
	}
}
