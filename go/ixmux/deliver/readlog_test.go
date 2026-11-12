package deliver

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// ReadLog gives each mandate its last recorded state, leaves the log byte for byte as it was,
// skips a torn last record, and refuses a log over its bound.
func TestReadLogGivesEachMandateItsLastState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _ := openEngine(t, path, Options{})
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "t"}); err != nil {
		t.Fatal(err)
	}
	g := newGated(1)
	e.Register("g", g.target())
	go e.Deliver(context.Background(), Mandate{MandateID: "m2", Target: "g", Incarnation: "i", Text: "t"})
	<-g.entered
	defer close(g.release) // registered after openEngine's cleanup Close, so it runs before it
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	torn := filepath.Join(t.TempDir(), "torn.jsonl")
	if err := os.WriteFile(torn, append(append([]byte{}, before...), `{"at":"2026`...), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{path, torn} {
		ms, tr, err := ReadLog(p, 0)
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		if len(ms) != 2 || ms[0].MandateID != "m1" || ms[0].State != Delivered ||
			ms[1].MandateID != "m2" || ms[1].State != Delivering || ms[1].Incarnation != "i" || ms[1].Created.IsZero() {
			t.Errorf("%s: %+v", p, ms)
		}
		if (tr != nil) != (p == torn) {
			t.Errorf("%s: torn %+v", p, tr)
		}
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(before, after) {
		t.Error("ReadLog changed the log")
	}
	if _, _, err := ReadLog(path, int64(len(before)-1)); refusalCode(err) != LogTooLarge {
		t.Errorf("a log over the bound: %v, want %s", err, LogTooLarge)
	}
}

// A reader of the log never breaks a running engine. On Windows its compaction fails while the
// file is open (MoveFileEx replaces no open file): that failure keeps the log as it was, is not
// fatal, and the next compaction, once the read is over, succeeds. Elsewhere it succeeds at once.
func TestReadLogNeverBreaksARunningEngine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _ := openEngine(t, path, Options{})
	a := &fakeTarget{inc: "i"}
	e.Register("a", a.target())
	for i := 0; i < 20; i++ {
		if _, err := e.Deliver(context.Background(), Mandate{MandateID: fmt.Sprintf("m%02d", i), Target: "a", Incarnation: "i", Text: "t"}); err != nil {
			t.Fatal(err)
		}
	}
	open, done := make(chan struct{}), make(chan struct{})
	testHookReadLogOpen = func() { close(open); <-done }
	t.Cleanup(func() { testHookReadLogOpen = nil })
	read := make(chan error, 1)
	go func() { _, _, err := ReadLog(path, 0); read <- err }()
	<-open
	e.log.mu.Lock()
	_, fatal, err := e.compactLocked()
	e.log.mu.Unlock()
	close(done)
	if fatal || (err != nil) != (runtime.GOOS == "windows") {
		t.Fatalf("compaction with a reader open on %s: %v (fatal %v)", runtime.GOOS, err, fatal)
	}
	if err := <-read; err != nil {
		t.Fatalf("the read during the compaction: %v", err)
	}
	testHookReadLogOpen = nil
	if runtime.GOOS == "windows" {
		e.log.mu.Lock()
		_, fatal, err = e.compactLocked()
		e.log.mu.Unlock()
		if err != nil || fatal {
			t.Fatalf("the compaction after the read: %v (fatal %v)", err, fatal)
		}
	}
	ms, _, err := ReadLog(path, 0)
	if err != nil || len(ms) != 20 {
		t.Fatalf("after the compaction: %d mandates, %v", len(ms), err)
	}
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "after", Target: "a", Incarnation: "i", Text: "t"}); err != nil {
		t.Errorf("a delivery after the compaction: %v", err)
	}
}
