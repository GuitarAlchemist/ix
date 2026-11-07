package claudestream

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// entryOf is the engine's view of one mandate.
func entryOf(e *deliver.Engine, id string) (deliver.Entry, *deliver.Ack) {
	en, ack, _ := e.Mandate(id)
	return en, ack
}

// S6B P1-1 (Argon): a turn that outlives the engine's deadline is uncertain, its adapter call
// still running; when it answers before TurnMax, the late answer settles it delivered, with the
// whole Ack. The adapter's call must not return at the engine's deadline.
func TestLateAnswerIsDelivered(t *testing.T) {
	hold := filepath.Join(t.TempDir(), "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	a, e, eff := setup(t, 300*time.Millisecond, "FAKE_HOLD="+hold)
	if res, _ := send(t, a, e, "s1", "slow"); res.State != deliver.Uncertain {
		t.Fatalf("held past the deadline: %+v, want uncertain", res)
	}
	if en, _ := entryOf(e, "s1"); en.State != deliver.Uncertain || !en.StillRunning {
		t.Errorf("during the turn: %+v, want uncertain with still_running", en)
	}

	// A second mandate to the same target during the turn: refused before anything is written.
	inc, _ := a.Incarnation()
	_, err := e.Deliver(context.Background(), deliver.Mandate{MandateID: "s2", Target: "c", Incarnation: inc, Text: "x"})
	if r := (*deliver.Refusal)(nil); !asRefusal(err, &r) || r.Code != deliver.TargetBusy {
		t.Errorf("a second mandate during the turn: %v, want target-busy", err)
	}
	if _, _, ok := e.Mandate("s2"); ok {
		t.Error("the refused second mandate has an entry: something was written for it")
	}

	os.Remove(hold)
	deadline := time.Now().Add(10 * time.Second) // a guard, not a timing bound
	for {
		en, ack := entryOf(e, "s1")
		if en.State == deliver.Delivered {
			if ack == nil || !ack.AckSeen || !strings.Contains(ack.Text, "BONJOUR") || en.StillRunning {
				t.Errorf("the late answer: %+v ack %+v, want delivered with its text", en, ack)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the late answer never settled s1: %+v", en)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if n := effects(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}

// S6B P1-1: with a turn held, Stop ends the child (it is killed past StopWait), so the adapter's
// call returns and Stop returns, far before TurnMax.
func TestStopWithATurnHeldIsBounded(t *testing.T) {
	hold := filepath.Join(t.TempDir(), "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	a, e, _ := setupCfg(t, 200*time.Millisecond, func(c *Config) { c.StopWait = 500 * time.Millisecond }, "FAKE_HOLD="+hold)
	if res, _ := send(t, a, e, "s1", "slow"); res.State != deliver.Uncertain {
		t.Fatalf("held past the deadline: %+v, want uncertain", res)
	}
	done := make(chan struct{})
	go func() { a.Stop(); e.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(30 * time.Second): // TurnMax is 10 minutes: a guard against waiting for it
		t.Fatal("Stop and Close still wait for a held turn")
	}
	if en, _ := entryOf(e, "s1"); en.StillRunning {
		t.Errorf("after Stop: %+v, want no call still running", en)
	}
}

// S6B P1-2 (Argon): a result with is_error is not an answer. The delivery is uncertain, and
// Reconcile does not make it delivered: it stays uncertain, with the reason written.
func TestErrorResultIsNotReceived(t *testing.T) {
	a, e, _ := setup(t, 5*time.Second, "FAKE_ISERROR=1")
	res, _ := send(t, a, e, "f1", "fail")
	if res.State != deliver.Uncertain {
		t.Fatalf("an is_error result: %+v, want uncertain", res)
	}
	if o, err := a.Status(context.Background(), "f1", a.sessionID()); o == deliver.Received {
		t.Errorf("Status of an is_error result: %s %v, want not received", o, err)
	}
	rec, err := e.Reconcile(context.Background(), "c", "f1")
	if err != nil || rec.State != deliver.Uncertain || !strings.Contains(rec.Reason, "is_error") {
		t.Errorf("Reconcile of an is_error result: %+v %v, want uncertain with an is_error reason", rec, err)
	}
	// The same flow without the error is received: the index still sees answers.
	if res, err := send(t, a, e, "ok1", "x"); err != nil || res.State != deliver.Delivered {
		t.Fatalf("a plain answer: %+v %v", res, err)
	}
	if o, _ := a.Status(context.Background(), "ok1", a.sessionID()); o != deliver.Received {
		t.Errorf("Status of a plain answer: %s, want received", o)
	}
}

func asRefusal(err error, r **deliver.Refusal) bool {
	if err == nil {
		return false
	}
	ref, ok := err.(*deliver.Refusal)
	*r = ref
	return ok
}
