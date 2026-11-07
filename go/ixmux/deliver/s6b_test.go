package deliver

import (
	"path/filepath"
	"testing"
)

// S6B Codex 4: Mandate reads the entry and its Ack in one snapshot. A transition written while it
// reads (delivering -> delivered with its Ack) is seen whole or not at all: never a delivering
// entry with the Ack of delivered.
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
	testHookMandateRead = func() {
		testHookMandateRead = nil
		if err := e.transition(k, "inc", Delivered, "", &Ack{Receipt: "r", Text: "done"}); err != nil {
			t.Error(err)
		}
	}
	t.Cleanup(func() { testHookMandateRead = nil })
	en, ack, ok := e.Mandate("m1")
	if !ok || (en.State == Delivered) != (ack != nil) {
		t.Errorf("Mandate across a transition: state %s with ack %+v, want both from one snapshot", en.State, ack)
	}
}
