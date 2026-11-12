package deliver

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

// An adapter call that ends in an error once the process is stopping (Stopping) is the stop's
// doing: uncertain, cause restart. Before Stopping, the same error is the target's: adapter-error.
// Found by session 1b129d7c (GAIA-IXMUX-CLOUD-QUALIFICATION-R1): a clean stop mid-turn said
// adapter-error, which tells the operator the target answered an error.
func TestAnAdapterErrorDuringAStopIsCauseRestart(t *testing.T) {
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	before, during := newGated(1), newGated(1)
	before.err = errors.New("stdin closed")
	during.err = errors.New("stdin closed")
	e.Register("before", before.target())
	e.Register("during", during.target())

	close(before.release)
	res, _ := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "before", Incarnation: "i", Text: "t"})
	if res.State != Uncertain || res.Cause != CauseAdapterError {
		t.Errorf("an adapter error with no stop: %s cause %q; want uncertain cause %s", res.State, res.Cause, CauseAdapterError)
	}

	out := make(chan Result, 1)
	go func() {
		res, _ := e.Deliver(context.Background(), Mandate{MandateID: "m2", Target: "during", Incarnation: "i", Text: "t"})
		out <- res
	}()
	<-during.entered
	e.Stopping()
	close(during.release)
	if res := <-out; res.State != Uncertain || res.Cause != CauseRestart {
		t.Errorf("an adapter error during a stop: %s cause %q; want uncertain cause %s", res.State, res.Cause, CauseRestart)
	}
	if during.callCount() != 1 {
		t.Errorf("%d calls, want 1", during.callCount())
	}
}
