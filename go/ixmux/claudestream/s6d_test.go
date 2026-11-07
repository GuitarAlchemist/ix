package claudestream

import (
	"context"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// within runs f, an operation that may block, under a watchdog: past guard, the test fails at once
// instead of hanging the suite.
func within(t *testing.T, guard time.Duration, what string, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() { defer close(done); f() }()
	select {
	case <-done:
	case <-time.After(guard):
		t.Fatalf("%s still blocks after %s", what, guard)
	}
}

// S6D P1-4 (Codex 10): after kill-failed the target says lost (its process may still run, its PID
// is known), and a mandate for the lost incarnation is refused typed (stale-incarnation), not an
// untyped error that ixmux serve answers 500 internal.
func TestLostTargetIsSaidAndRefusedTyped(t *testing.T) {
	a, e, _ := setupCfg(t, 200*time.Millisecond, func(c *Config) {
		c.TurnMax, c.StopWait = 500*time.Millisecond, 300*time.Millisecond
	}, "FAKE_MUTE=1")
	failKills(t, a.live())
	inc, _ := a.Incarnation()
	if res, _ := send(t, a, e, "m1", "mute"); res.State != deliver.Uncertain {
		t.Fatalf("m1 never answered: %+v, want uncertain", res)
	}
	for deadline := time.Now().Add(5 * time.Second); a.Alive(); time.Sleep(20 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the kill at TurnMax never ended the incarnation")
		}
	}
	if !a.Lost() || a.PID() == 0 {
		t.Errorf("after kill-failed: lost %v, pid %d; want lost with its pid", a.Lost(), a.PID())
	}
	_, err := e.Deliver(context.Background(), deliver.Mandate{MandateID: "m2", Target: "c", Incarnation: inc, Text: "x"})
	if r := (*deliver.Refusal)(nil); !asRefusal(err, &r) || r.Code != deliver.StaleIncarnation {
		t.Errorf("a mandate for the lost incarnation: %v, want the refusal stale-incarnation", err)
	}
}
