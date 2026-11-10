package deliver

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// bigAckTarget takes every mandate and answers an Ack of n bytes.
type bigAckTarget struct {
	n     int
	calls int
}

func (b *bigAckTarget) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	b.calls++
	return Ack{Receipt: "took " + m.MandateID, Text: strings.Repeat("x", b.n)}, nil
}

func (b *bigAckTarget) target() Target {
	return Target{Adapter: b, Incarnation: func() (string, error) { return "i", nil }}
}

// Review P1-a: once the adapter was called, the target may have taken the mandate. A final record
// the log cannot hold is then no refusal (a refusal says nothing was sent, and a client would send
// the work again under another id): the delivery is uncertain, cause log-write-failed, and stays
// so after a restart, never sent again. The room an accepted delivery holds makes this impossible
// in a log this engine wrote; it stays possible when the bound is lowered under the log, which the
// test does during the call.
func TestAFinalRecordThatDoesNotFitIsNoRefusal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	opt := Options{MaxLogBytes: 16 << 10}
	e, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	g := newGated(100)
	e.Register("b", g.target())
	m := Mandate{MandateID: "m1", Target: "b", Incarnation: "i", Text: "t"}
	type out struct {
		res Result
		err error
	}
	o := make(chan out, 1)
	go func() { res, err := e.Deliver(context.Background(), m); o <- out{res, err} }()
	<-g.entered
	e.log.mu.Lock()
	e.log.max = e.log.size // not one byte left
	e.log.mu.Unlock()
	close(g.release)
	got := <-o
	if res, err := got.res, got.err; err == nil || refusalCode(err) != "" || res.State != Uncertain || res.Cause != CauseLogWrite || g.callCount() != 1 {
		t.Errorf("a delivery whose final record does not fit: %+v, %v (refusal code %q), %d calls; want an error that is no refusal, uncertain, cause %s, 1 call",
			res, err, refusalCode(err), g.callCount(), CauseLogWrite)
	}
	if _, err := e.Deliver(context.Background(), m); refusalCode(err) != UncertainNeedsReconcile || g.callCount() != 1 {
		t.Errorf("m1 sent again: %v, %d calls; want %s and 1 call", err, g.callCount(), UncertainNeedsReconcile)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	e2, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	defer e2.Close()
	if en := entryOf(e2, "b", "m1"); en.State != Uncertain || en.Cause != CauseRestart {
		t.Errorf("m1 after a restart: %+v, want uncertain, cause %s", en, CauseRestart)
	}
}

// R2-1: the room a delivery holds is bounded, so its final record is too: an Ack whose text does
// not fit in MaxRecordBytes is cut, said Truncated, and read back the same after a restart.
func TestAnAckTooLargeForARecordIsCutAndSaidTruncated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	b := &bigAckTarget{n: 2 * MaxRecordBytes}
	e.Register("b", b.target())
	res, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "b", Incarnation: "i", Text: "t"})
	if err != nil || res.State != Delivered || res.Ack == nil || !res.Ack.Truncated || len(res.Ack.Text) >= MaxRecordBytes || len(res.Ack.Text) < MaxRecordBytes/2 {
		t.Fatalf("a %d-byte answer: %+v %v; want delivered, its text cut under %d bytes and said truncated", b.n, res, err, MaxRecordBytes)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	e2, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e2.Close()
	if _, ack, ok := e2.Mandate("m1"); !ok || ack == nil || *ack != *res.Ack {
		t.Errorf("m1's Ack after a restart differs from the one answered")
	}
}

// Review P1-b: a compaction that fails (here its rename, as when another process holds the log
// open on Windows) leaves the old log, full. The restart's own transitions then do not fit: they
// are applied in memory, the log already says what makes them (a delivering record is uncertain at
// every Open), and Open still starts and says why the compaction was not done.
func TestARestartWhoseTransitionsDoNotFitStillStarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{MaxLogBytes: 8 << 10})
	if err != nil {
		t.Fatal(err)
	}
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	for _, id := range []string{"m1", "m2", "m3"} {
		if _, err := e.Deliver(context.Background(), Mandate{MandateID: id, Target: "a", Incarnation: "i", Text: "t"}); err != nil {
			t.Fatal(err)
		}
	}
	d1 := Mandate{MandateID: "d1", Target: "a", Incarnation: "i", Text: "t"}
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transitionFP(key{"a", "d1"}, "i", s, "", nil, fingerprint(d1)); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(path)
	full := Options{MaxLogBytes: st.Size()} // not one byte left

	renameLog = func(oldpath, newpath string) error { return errors.New("the log is held open by another process") }
	e2, rec, err := Open(path, full)
	renameLog = renameDurable
	if err != nil {
		t.Fatalf("Open of a full log whose compaction failed, with a delivery left delivering: %v", err)
	}
	if rec.CompactFailed == "" || rec.Compacted != nil || rec.MadeUncertain != 1 {
		t.Errorf("Recovery %+v, want CompactFailed said, no compaction, 1 made uncertain", rec)
	}
	if en := entryOf(e2, "a", "d1"); en.State != Uncertain || en.Cause != CauseRestart {
		t.Errorf("d1: %+v, want uncertain, cause %s", en, CauseRestart)
	}
	e2.Register("a", a.target())
	if _, err := e2.Deliver(context.Background(), d1); refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("d1 sent again: %v, want %s", err, UncertainNeedsReconcile)
	}
	if err := e2.Close(); err != nil {
		t.Fatal(err)
	}

	e3, rec, err := Open(path, full) // the rename works again: the log is compacted
	if err != nil {
		t.Fatal(err)
	}
	defer e3.Close()
	if rec.Compacted == nil || stateOf(e3, "a", "d1") != Uncertain {
		t.Errorf("the next start: %+v, d1 %q; want it compacted and d1 uncertain", rec, stateOf(e3, "a", "d1"))
	}
}

// Review P2-4: a compaction that would not make the log smaller is not done.
func TestACompactionThatWouldNotShrinkIsSkipped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{MaxLogBytes: 64 << 10})
	if err != nil {
		t.Fatal(err)
	}
	// Mandates with one record each (accepted, the adapter never called): compacted, each record
	// would gain its creation time. One mandate with two records makes the log a candidate.
	for i := 0; i < 20; i++ {
		m := Mandate{MandateID: "a" + strings.Repeat("0", i), Target: "a", Incarnation: "i", Text: "t"}
		if err := e.transitionFP(key{"a", m.MandateID}, "i", Accepted, "", nil, fingerprint(m)); err != nil {
			t.Fatal(err)
		}
	}
	k := Mandate{MandateID: "k1", Target: "a", Incarnation: "i", Text: "t"}
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transitionFP(key{"a", "k1"}, "i", s, "", nil, fingerprint(k)); err != nil {
			t.Fatal(err)
		}
	}
	e.Close()
	st, _ := os.Stat(path)
	before, _ := os.ReadFile(path)

	e2, rec, err := Open(path, Options{MaxLogBytes: st.Size() + 1})
	if err != nil {
		t.Fatal(err)
	}
	defer e2.Close()
	after, _ := os.ReadFile(path)
	if rec.Compacted != nil || rec.CompactFailed != "" || !strings.HasPrefix(string(after), string(before)) {
		t.Errorf("Recovery %+v; the log was rewritten: %v; want no compaction, the log appended to only", rec, !strings.HasPrefix(string(after), string(before)))
	}
}

// Review P2-6: a .compact file left by a crash during a compaction is removed at the next Open.
func TestALeftoverCompactFileIsRemoved(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	if err := os.WriteFile(path+".compact", []byte(strings.Repeat("x", 1000)), 0o600); err != nil {
		t.Fatal(err)
	}
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if _, err := os.Stat(path + ".compact"); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("the leftover %s.compact after Open: %v, want it removed", filepath.Base(path), err)
	}
}
