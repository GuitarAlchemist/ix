package deliver

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
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
// those delivered. It fails the test on any other refusal, and reports a log that never fills.
func fillLog(t *testing.T, e *Engine, prefix string, n int) []Mandate {
	t.Helper()
	var delivered []Mandate
	for i := 0; i < n; i++ {
		m := Mandate{MandateID: fmt.Sprintf("%s%03d", prefix, i), Target: "a", Incarnation: "i", Text: "t"}
		res, err := e.Deliver(context.Background(), m)
		if refusalCode(err) == LogFull {
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
	delivered := fillLog(t, e, "m", 200)
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
