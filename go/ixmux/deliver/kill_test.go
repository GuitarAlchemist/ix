package deliver

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// The helper process is this test binary run again with IXMUX_DELIVER_HELPER set: it opens the
// engine on IXMUX_LOG and delivers one mandate through an adapter that hangs, after
// (after-effect) or before (before-effect) writing its effect to IXMUX_EFFECT. It creates
// IXMUX_READY once it hangs, and waits to be killed.
const helperEnv = "IXMUX_DELIVER_HELPER"

type hangingAdapter struct{ mode, effect, ready string }

func (h hangingAdapter) Deliver(ctx context.Context, m Mandate) (Ack, error) {
	if h.mode == "after-effect" {
		if err := appendEffect(h.effect, m.MandateID); err != nil {
			return Ack{}, err
		}
		if err := os.WriteFile(h.ready, nil, 0o600); err != nil {
			return Ack{}, err
		}
		time.Sleep(time.Hour) // killed here, after the effect, before the answer
		return Ack{}, errors.New("not killed")
	}
	// before-effect: the kill lands on the step right before a real effect operation.
	if err := os.WriteFile(h.ready, nil, 0o600); err != nil {
		return Ack{}, err
	}
	time.Sleep(time.Hour) // killed here
	if err := appendEffect(h.effect, m.MandateID); err != nil {
		return Ack{}, err
	}
	return Ack{Receipt: "took " + m.MandateID}, nil
}

func TestHelperProcess(t *testing.T) {
	mode := os.Getenv(helperEnv)
	if mode == "" {
		t.Skip("helper process only")
	}
	if mode == "open" { // only try to open the log, and say how it went
		e, _, err := Open(os.Getenv("IXMUX_LOG"), Options{})
		if err != nil {
			fmt.Printf("OPEN-REFUSED %s\n", refusalCode(err))
			return
		}
		e.Close()
		fmt.Println("OPEN-OK")
		return
	}
	e, _, err := Open(os.Getenv("IXMUX_LOG"), Options{Timeout: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	a := hangingAdapter{mode: mode, effect: os.Getenv("IXMUX_EFFECT"), ready: os.Getenv("IXMUX_READY")}
	e.Register("a", Target{Adapter: a, Incarnation: func() (string, error) { return "i", nil }})
	e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "go"})
	t.Fatal("the delivery returned; it should hang until killed")
}

// killMidDelivery runs the helper in mode, waits until its adapter hangs, and kills it.
func killMidDelivery(t *testing.T, mode, log, effect string) {
	t.Helper()
	ready := filepath.Join(t.TempDir(), "ready")
	cmd := exec.Command(os.Args[0], "-test.run=^TestHelperProcess$", "-test.count=1")
	cmd.Env = append(os.Environ(), helperEnv+"="+mode, "IXMUX_LOG="+log, "IXMUX_EFFECT="+effect, "IXMUX_READY="+ready)
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()
	t.Cleanup(func() {
		cmd.Process.Kill()
		<-exited
	})
	deadline := time.After(20 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		select {
		case err := <-exited:
			exited <- err
			t.Fatalf("the helper exited before its adapter hung: %v\n%s", err, out.String())
		case <-deadline:
			t.Fatal("the helper never reached its adapter")
		case <-time.After(20 * time.Millisecond):
		}
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	t.Logf("helper killed mid-delivery (%s): %v", mode, <-exited)
	exited <- nil // for the cleanup
}

// Oracle 2: a real kill after the effect, before the final state.
func TestKillAfterEffectThenReconcileReceived(t *testing.T) {
	dir := t.TempDir()
	log, effect := filepath.Join(dir, "cmd.jsonl"), filepath.Join(dir, "effects")
	killMidDelivery(t, "after-effect", log, effect)

	e, rec := openEngine(t, log, Options{})
	tg := &fakeTarget{inc: "i", file: effect, outcome: Received}
	e.Register("a", tg.target())
	if rec.MadeUncertain != 1 || stateOf(e, "a", "m1") != Uncertain {
		t.Fatalf("after restart: recovery %+v, state %q", rec, stateOf(e, "a", "m1"))
	}
	m := Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "go"}
	if _, err := e.Deliver(context.Background(), m); refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("resend while uncertain: %v, want %s", err, UncertainNeedsReconcile)
	}
	if n := countEffects(t, effect); n != 1 {
		t.Fatalf("%d effects before reconcile, want 1", n)
	}
	res, err := e.Reconcile(context.Background(), "a", "m1")
	if err != nil || res.State != Delivered {
		t.Fatalf("reconcile: %+v %v", res, err)
	}
	res, err = e.Deliver(context.Background(), m)
	if err != nil || !res.Replayed || res.State != Delivered {
		t.Errorf("resend after reconcile: %+v %v", res, err)
	}
	if n := countEffects(t, effect); n != 1 || tg.calls.Load() != 0 {
		t.Errorf("%d effects, %d adapter calls after reconcile; want 1 and 0", n, tg.calls.Load())
	}
}

// Oracle 2, variant: a real kill before the effect; the target did not receive it.
func TestKillBeforeEffectThenReconcileNotReceived(t *testing.T) {
	dir := t.TempDir()
	log, effect := filepath.Join(dir, "cmd.jsonl"), filepath.Join(dir, "effects")
	killMidDelivery(t, "before-effect", log, effect)

	e, rec := openEngine(t, log, Options{})
	tg := &fakeTarget{inc: "i", file: effect, outcome: NotReceived}
	e.Register("a", tg.target())
	if rec.MadeUncertain != 1 || stateOf(e, "a", "m1") != Uncertain {
		t.Fatalf("after restart: recovery %+v, state %q", rec, stateOf(e, "a", "m1"))
	}
	m := Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "go"}
	if _, err := e.Deliver(context.Background(), m); refusalCode(err) != UncertainNeedsReconcile {
		t.Errorf("resend while uncertain: %v, want %s", err, UncertainNeedsReconcile)
	}
	res, err := e.Reconcile(context.Background(), "a", "m1")
	if err != nil || res.State != Refused {
		t.Fatalf("reconcile: %+v %v", res, err)
	}
	res, err = e.Deliver(context.Background(), m)
	if err != nil || res.Replayed || res.State != Delivered {
		t.Errorf("new attempt after not-received: %+v %v", res, err)
	}
	if n := countEffects(t, effect); n != 1 {
		t.Errorf("%d effects, want 1", n)
	}
	if _, err := os.Stat(effect); errors.Is(err, os.ErrNotExist) {
		t.Error("no effect file")
	}
}
