package claudestream

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// S4B-4 (Codex round 7, P1): a child that sends its init on the opening message, then never
// answers it, fails the start within InitWait.
func TestSilentPrimeFailsTheStart(t *testing.T) {
	dir := t.TempDir()
	done := make(chan error, 1)
	go func() {
		a, err := Start(Config{Name: "c", Dir: filepath.Join(dir, "claude-c"), Bin: fake, Env: []string{"FAKE_INIT=first", "FAKE_MUTE=all"},
			InitWait: time.Second, PrimeAfter: 100 * time.Millisecond, StopWait: time.Second})
		if err == nil {
			a.Stop()
		}
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Error("a child that never answers its opening message was started")
		}
	case <-time.After(6 * time.Second):
		t.Fatal("Start still waits for an opening turn that never ends, past InitWait (1s)")
	}
}

// S4B-4 (Codex round 7, P1): a turn without answer is uncertain at its deadline, but stays
// reserved (no second turn) until TurnMax; then the child is killed, a new incarnation starts and
// the target is free. The first mandate cannot be reconciled under the new incarnation.
func TestTurnMaxFreesTheTarget(t *testing.T) {
	a, e, eff := setupCfg(t, 300*time.Millisecond, func(c *Config) { c.TurnMax = 1500 * time.Millisecond }, "FAKE_MUTE=1")
	inc1 := a.sessionID()
	if res, _ := send(t, a, e, "m1", "mute"); res.State != deliver.Uncertain {
		t.Fatalf("m1 never answered: %+v, want uncertain", res)
	}
	if res, _ := send(t, a, e, "m2", "x"); res.State == deliver.Delivered {
		t.Errorf("m2 sent while m1 may still answer: %+v", res)
	}
	if n := effects(t, eff); n != 1 {
		t.Errorf("%d turns while m1 is reserved, want 1", n)
	}
	deadline := time.Now().Add(8 * time.Second)
	for {
		if inc, err := a.Incarnation(); err == nil && inc != inc1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("no new incarnation after TurnMax")
		}
		time.Sleep(50 * time.Millisecond)
	}
	if res, err := send(t, a, e, "m3", "x"); err != nil || res.State != deliver.Delivered {
		t.Errorf("a mandate under the new incarnation: %+v %v, want delivered", res, err)
	}
	if _, err := e.Reconcile(context.Background(), "c", "m1"); err == nil {
		t.Error("m1 reconciled under the new incarnation, want stale-incarnation")
	}
}
