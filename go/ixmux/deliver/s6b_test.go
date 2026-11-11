package deliver

import (
	"path/filepath"
	"testing"
	"time"
)

// S6B Codex 4, redone in S6C: Mandate reads the entry and its Ack in one snapshot. A transition
// started between the two reads (delivering -> delivered with its Ack) must not land in between:
// never a delivering entry with the Ack of delivered. The hook gives the transition 200 ms to land;
// under one lock it cannot, and lands once Mandate is done.
func TestMandateReadsOneSnapshot(t *testing.T) {
	e, _, err := Open(filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	k := key{"t", "m1"}
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transitionFP(k, "inc", s, "", nil, "fp"); err != nil {
			t.Fatal(err)
		}
	}
	landed := make(chan error, 1)
	testHookMandateRead = func() {
		testHookMandateRead = nil
		go func() { landed <- e.transition(k, "inc", Delivered, "", "", &Ack{Receipt: "r", Text: "done"}) }()
		select {
		case err := <-landed: // it landed between the reads: the snapshot is broken
			landed <- err
		case <-time.After(200 * time.Millisecond):
		}
	}
	t.Cleanup(func() { testHookMandateRead = nil })
	en, ack, ok := e.Mandate("m1")
	if !ok || (en.State == Delivered) != (ack != nil) {
		t.Errorf("Mandate across a transition: state %s with ack %+v, want both from one snapshot", en.State, ack)
	}
	if err := <-landed; err != nil {
		t.Error(err)
	}
}
