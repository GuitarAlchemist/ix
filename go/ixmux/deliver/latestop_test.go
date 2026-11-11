package deliver

// A mandate made uncertain by its deadline, whose adapter call then ends in an unknown error while
// ixmux stops, gets a second uncertain record from observeLate. That record must keep the
// transition that made it uncertain and, the error being the stop's, its cause: else both are
// lost from the log once compacted. Found by ix-ixmux-38 (GAIA-IXMUX-RECEIPTS-PROBE-R4), red on
// 03d916b0; the late-error path that Stopping (682e446f) did not cover.

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func lateRecords(t *testing.T, path, id string) []logRecord {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []logRecord
	for _, line := range strings.Split(string(b), "\n") {
		var lr logRecord
		if json.Unmarshal([]byte(line), &lr) == nil && lr.MandateID == id {
			out = append(out, lr)
		}
	}
	return out
}

func lateEntry(e *Engine, id string) Entry {
	en, _, _ := e.Mandate(id)
	return en
}

func TestALateErrorDuringAStopKeepsTheDeadlineAndItsTransition(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{Timeout: 100 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	g := newGated(1)
	g.err = errors.New("claude exited: exit status 1")
	e.Register("c", g.target())

	res, _ := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "c", Incarnation: "i", Text: "t"})
	if res.State != Uncertain || res.Cause != CauseDeadline {
		t.Fatalf("at the deadline: %s cause %q, want uncertain cause %s", res.State, res.Cause, CauseDeadline)
	}
	first := lateEntry(e, "m1")
	t.Logf("at the deadline: last %q cause %q", first.Last, first.Cause)

	e.Stopping()     // ixmux stops: it closes the children...
	close(g.release) // ...so the abandoned call ends in an unknown error
	for end := time.Now().Add(5 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		if !lateEntry(e, "m1").StillRunning {
			break
		}
		if time.Now().After(end) {
			t.Fatal("the late answer was never recorded")
		}
	}
	shown := lateEntry(e, "m1")
	t.Logf("after the late unknown answer: last %q cause %q reason %q", shown.Last, shown.Cause, shown.Reason)
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	inLog := lateRecords(t, path, "m1")
	t.Logf("log before compaction: %d records for m1", len(inLog))

	// Presentation: what Mandate/Status (so ixmux_result) say.
	if shown.Last != "delivering->uncertain" {
		t.Errorf("PRESENTATION: last %q, want delivering->uncertain (the transition that made it uncertain)", shown.Last)
	}
	if shown.Cause != CauseDeadline {
		t.Errorf("PRESENTATION: cause %q, want %s: nothing but the deadline made it uncertain; the late error is the stop's", shown.Cause, CauseDeadline)
	}

	// History, uncompacted: the deadline record is still in the log.
	kept := false
	for _, lr := range inLog {
		kept = kept || (lr.State == Uncertain && lr.Cause == CauseDeadline)
	}
	if !kept {
		t.Errorf("HISTORY (uncompacted): no uncertain record with cause deadline in the log")
	}

	// History, compacted: reopen with a bound the log is past three quarters of.
	st, _ := os.Stat(path)
	e2, rec, err := Open(path, Options{Timeout: 100 * time.Millisecond, MaxLogBytes: st.Size() * 5 / 4})
	if err != nil {
		t.Fatal(err)
	}
	if rec.Compacted == nil {
		t.Fatalf("the reopen did not compact (size %d); recovery %+v", st.Size(), rec)
	}
	after := lateEntry(e2, "m1")
	e2.Close()
	comp := lateRecords(t, path, "m1")
	t.Logf("after compaction: %d record(s) for m1: %+v; shown last %q cause %q", len(comp), comp, after.Last, after.Cause)
	if after.Cause != CauseDeadline || after.Last != "delivering->uncertain" {
		t.Errorf("HISTORY (compacted): the only record left says last %q cause %q; the deadline transition is gone from the log", after.Last, after.Cause)
	}
	if g.callCount() != 1 {
		t.Errorf("%d adapter calls, want 1", g.callCount())
	}
}

// Outside a stop a late error is the target's: the cause becomes adapter-error, but the reason
// keeps what made the mandate uncertain first (the deadline's text), then the error, so that a
// compacted log still says both (P3-7, case c of ix-ixmux-38's characterization).
func TestALateErrorKeepsTheDeadlinesReasonThroughACompaction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{Timeout: 100 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	g := newGated(1)
	g.err = errors.New("target said no: exit status 3")
	e.Register("c", g.target())
	if res, _ := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "c", Incarnation: "i", Text: "t"}); res.State != Uncertain || res.Cause != CauseDeadline {
		t.Fatalf("at the deadline: %s cause %q", res.State, res.Cause)
	}
	deadlineReason := lateEntry(e, "m1").Reason
	close(g.release)
	for end := time.Now().Add(5 * time.Second); lateEntry(e, "m1").StillRunning; time.Sleep(5 * time.Millisecond) {
		if time.Now().After(end) {
			t.Fatal("the late answer was never recorded")
		}
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(path)
	e2, rec, err := Open(path, Options{Timeout: 100 * time.Millisecond, MaxLogBytes: st.Size() * 5 / 4})
	if err != nil {
		t.Fatal(err)
	}
	after := lateEntry(e2, "m1")
	e2.Close()
	if rec.Compacted == nil {
		t.Fatalf("the reopen did not compact (size %d)", st.Size())
	}
	if after.Cause != CauseAdapterError || after.Last != "delivering->uncertain" {
		t.Errorf("compacted: last %q cause %q, want delivering->uncertain, adapter-error", after.Last, after.Cause)
	}
	if deadlineReason == "" || !strings.HasPrefix(after.Reason, deadlineReason) || !strings.Contains(after.Reason, "target said no") {
		t.Errorf("compacted reason %q: want the deadline's %q, then the late error", after.Reason, deadlineReason)
	}
}
