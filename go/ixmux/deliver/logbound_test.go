package deliver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// failingTarget answers every Deliver with an error that is not a refusal: the outcome is unknown,
// so the delivery ends uncertain.
type failingTarget struct{ inc string }

func (f *failingTarget) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	return Ack{}, errors.New("the target's pipe broke")
}

func (f *failingTarget) Status(ctx context.Context, mandateID, incarnation string) (Outcome, error) {
	return Unknown, nil
}

func (f *failingTarget) target() Target {
	return Target{Adapter: f, Incarnation: func() (string, error) { return f.inc, nil }}
}

// fillLog delivers mandates to a until the engine refuses one log-full (at most n), and returns
// those delivered. It fails the test on any other refusal, on a log-full given after a's adapter
// was called (a refusal says nothing was sent), and reports a log that never fills.
func fillLog(t *testing.T, e *Engine, a *fakeTarget, prefix string, n int) []Mandate {
	t.Helper()
	var delivered []Mandate
	for i := 0; i < n; i++ {
		m := Mandate{MandateID: fmt.Sprintf("%s%03d", prefix, i), Target: "a", Incarnation: "i", Text: "t"}
		calls := a.calls.Load()
		res, err := e.Deliver(context.Background(), m)
		if refusalCode(err) == LogFull {
			if a.calls.Load() != calls {
				t.Errorf("%s refused %s after its adapter was called", m.MandateID, LogFull)
			}
			return delivered
		}
		if err != nil || res.State != Delivered {
			t.Fatalf("Deliver %s: %+v %v", m.MandateID, res, err)
		}
		delivered = append(delivered, m)
	}
	t.Errorf("%d mandates went into a log bounded at Open, and none was refused %s", n, LogFull)
	return delivered
}

// The log is read back at Open up to MaxLogBytes. Its appends must hold the same bound: a log grown
// past it by its own engine is refused log-too-large at the next Open, and ixmux cannot start
// without the log being deleted, which would lose every mandate id it remembers.
func TestALogFilledByItsOwnAppendsStillOpens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	opt := Options{MaxLogBytes: 4 << 10}
	e, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	a, u := &fakeTarget{inc: "i"}, &failingTarget{inc: "i"}
	e.Register("a", a.target())
	e.Register("u", u.target())
	if res, err := e.Deliver(context.Background(), Mandate{MandateID: "u1", Target: "u", Incarnation: "i", Text: "t"}); err != nil || res.State != Uncertain {
		t.Fatalf("a delivery whose adapter fails: %+v %v, want uncertain", res, err)
	}
	delivered := fillLog(t, e, a, "m", 200)
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(path)

	e2, _, err := Open(path, opt)
	if err != nil {
		t.Fatalf("reopen the %d-byte log its own engine wrote, bound %d: %v", st.Size(), opt.MaxLogBytes, err)
	}
	defer e2.Close()
	e2.Register("a", a.target())
	e2.Register("u", u.target())

	calls := a.calls.Load()
	for _, m := range delivered {
		res, err := e2.Deliver(context.Background(), m)
		if err != nil || !res.Replayed || res.State != Delivered || res.Ack == nil || res.Ack.Receipt != "took "+m.MandateID {
			t.Errorf("%s sent again after the restart: %+v %v, want its first result replayed", m.MandateID, res, err)
		}
	}
	if a.calls.Load() != calls {
		t.Errorf("the adapter was called %d more times after the restart, want 0", a.calls.Load()-calls)
	}
	if _, err := e2.Deliver(context.Background(), Mandate{MandateID: delivered[0].MandateID, Target: "a", Incarnation: "i", Text: "other"}); refusalCode(err) != MandateIDReuse {
		t.Errorf("a delivered id with another text after the restart: %v, want %s", err, MandateIDReuse)
	}
	if s := stateOf(e2, "u", "u1"); s != Uncertain {
		t.Errorf("u1 after the restart is %q, want uncertain", s)
	}
	if _, err := e2.Deliver(context.Background(), Mandate{MandateID: "u1", Target: "u", Incarnation: "i", Text: "t"}); refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("u1 sent again after the restart: %v, want %s", err, UncertainNeedsReconcile)
	}
}

// records parses the log's records, after its header.
func records(t *testing.T, log []byte) []logRecord {
	t.Helper()
	lines := strings.Split(strings.TrimSuffix(string(log), "\n"), "\n")
	out := make([]logRecord, 0, len(lines))
	for _, ln := range lines[1:] {
		var r logRecord
		if err := json.Unmarshal([]byte(ln), &r); err != nil {
			t.Fatalf("record %q: %v", ln, err)
		}
		out = append(out, r)
	}
	return out
}

func firstRecord(t *testing.T, log []byte, id string) logRecord {
	t.Helper()
	for _, r := range records(t, log) {
		if r.MandateID == id {
			return r
		}
	}
	t.Fatalf("no record of %s", id)
	return logRecord{}
}

// compacted is id's only record in the compacted log at path.
func compacted(t *testing.T, path, id string) logRecord {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return firstRecord(t, b, id)
}

func entryOf(e *Engine, target, id string) Entry {
	for _, en := range e.Status() {
		if en.Target == target && en.MandateID == id {
			return en
		}
	}
	return Entry{}
}

// A full log stays full unless something frees it, and every new mandate is then refused for good.
// A restart compacts it: one record per mandate, its last state with what makes its id unique
// (fingerprint, incarnation) and what proves its outcome (Ack, reason). New mandates fit again,
// nothing remembered is lost, and a delivery left delivering by a crash is still made uncertain,
// never sent again.
func TestARestartCompactsAFullLogAndForgetsNothing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	opt := Options{MaxLogBytes: 4 << 10}
	e, _, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	a, u := &fakeTarget{inc: "i"}, &failingTarget{inc: "i"}
	e.Register("a", a.target())
	e.Register("u", u.target())
	if res, err := e.Deliver(context.Background(), Mandate{MandateID: "u1", Target: "u", Incarnation: "i", Text: "t"}); err != nil || res.State != Uncertain {
		t.Fatalf("a delivery whose adapter fails: %+v %v, want uncertain", res, err)
	}
	// d1 is left delivering, as by a crash during its adapter call.
	d1 := Mandate{MandateID: "d1", Target: "a", Incarnation: "i", Text: "t"}
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transitionFP(key{"a", "d1"}, "i", s, "", nil, fingerprint(d1)); err != nil {
			t.Fatal(err)
		}
	}
	delivered := fillLog(t, e, a, "m", 200)
	before, beforeU1 := entryOf(e, "a", delivered[0].MandateID), entryOf(e, "u", "u1")
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	full, _ := os.Stat(path)
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	compacting, rec, err := Open(path, opt)
	if err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(path)
	if rec.Compacted == nil || rec.Compacted.From != full.Size() || rec.Compacted.To >= full.Size() {
		t.Errorf("Recovery.Compacted %+v, log of %d bytes now %d; want it compacted", rec.Compacted, full.Size(), st.Size())
	}
	if rec.MadeUncertain != 1 {
		t.Errorf("d1, left delivering: %d made uncertain, want 1", rec.MadeUncertain)
	}
	if err := compacting.Close(); err != nil {
		t.Fatal(err)
	}
	// The engine that compacted answers from what it read before; e2 reads the compacted log.
	e2, _, err := Open(path, opt)
	if err != nil {
		t.Fatalf("reopen the compacted log: %v", err)
	}
	t.Cleanup(func() { e2.Close() })
	e2.Register("a", a.target())
	e2.Register("u", u.target())
	if s := stateOf(e2, "a", "d1"); s != Uncertain {
		t.Errorf("d1 after the compaction is %q, want uncertain", s)
	}
	calls := a.calls.Load()
	for _, m := range delivered {
		res, err := e2.Deliver(context.Background(), m)
		if err != nil || !res.Replayed || res.Ack == nil || res.Ack.Receipt != "took "+m.MandateID {
			t.Errorf("%s sent again after the compaction: %+v %v, want its first result replayed", m.MandateID, res, err)
		}
	}
	if _, err := e2.Deliver(context.Background(), d1); refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("d1 sent again after the compaction: %v, want %s", err, UncertainNeedsReconcile)
	}
	if a.calls.Load() != calls {
		t.Errorf("the adapter was called %d more times after the compaction, want 0", a.calls.Load()-calls)
	}
	if _, err := e2.Deliver(context.Background(), Mandate{MandateID: delivered[0].MandateID, Target: "a", Incarnation: "i", Text: "other"}); refusalCode(err) != MandateIDReuse {
		t.Errorf("a delivered id with another text after the compaction: %v, want %s", err, MandateIDReuse)
	}
	if u1 := entryOf(e2, "u", "u1"); u1.State != Uncertain || u1.Reason != beforeU1.Reason || u1.Reason == "" || u1.Cause != CauseAdapterError {
		t.Errorf("u1 after the compaction: %+v, want uncertain with its reason %q and cause %s", u1, beforeU1.Reason, CauseAdapterError)
	}
	// u1's last record is the compacted one: Reconcile asks the incarnation read back from it.
	if res, err := e2.Reconcile(context.Background(), "u", "u1"); err != nil || res.State != Uncertain {
		t.Errorf("Reconcile u1 after the compaction: %+v %v, want still uncertain (its target answers unknown)", res, err)
	}
	first, kept := firstRecord(t, original, delivered[0].MandateID), compacted(t, path, delivered[0].MandateID)
	if kept.Created == nil || !kept.Created.Equal(first.At) {
		t.Errorf("%s compacted with created %v, want %v (the time of its first record)", delivered[0].MandateID, kept.Created, first.At)
	}
	// d1 is reconciled with the incarnation it was sent to.
	a.outcome = Received
	if res, err := e2.Reconcile(context.Background(), "a", "d1"); err != nil || res.State != Delivered {
		t.Errorf("Reconcile d1 after the compaction: %+v %v, want delivered", res, err)
	}
	after := entryOf(e2, "a", delivered[0].MandateID)
	if after.Last != before.Last || !after.Since.Equal(before.Since) || after.Age < before.Age {
		t.Errorf("%s in Status: %+v before the compaction, %+v after; want the same last transition, since and creation", delivered[0].MandateID, before, after)
	}

	more := fillLog(t, e2, a, "n", 200)
	if len(more) == 0 {
		t.Fatal("no new mandate fits after the compaction")
	}
	if err := e2.Close(); err != nil {
		t.Fatal(err)
	}
	e3, rec3, err := Open(path, opt) // a second compaction keeps what the first kept
	if err != nil {
		t.Fatal(err)
	}
	if rec3.Compacted == nil {
		t.Errorf("the second restart, on a log filled again: %+v, want it compacted", rec3)
	}
	defer e3.Close()
	e3.Register("a", a.target())
	for _, m := range append(delivered[:1], more[0]) {
		if res, err := e3.Deliver(context.Background(), m); err != nil || !res.Replayed || res.Ack == nil || res.Ack.Receipt != "took "+m.MandateID {
			t.Errorf("%s after a second restart: %+v %v, want replayed", m.MandateID, res, err)
		}
	}
}
