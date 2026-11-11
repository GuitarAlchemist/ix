package deliver

import (
	"context"
	"path/filepath"
	"testing"
)

func TestEntryFingerprintSurvivesRecovery(t *testing.T) {
	path := filepath.Join(t.TempDir(), "commands.jsonl")
	e, _ := openEngine(t, path, Options{})
	f := &fakeTarget{inc: "inc"}
	e.Register("c", f.target())
	m := Mandate{MandateID: "m1", Target: "c", Incarnation: "inc", Text: "exact text"}
	if _, err := e.Deliver(context.Background(), m); err != nil {
		t.Fatal(err)
	}
	check := func(e *Engine) {
		t.Helper()
		en, _, ok := e.Mandate(m.MandateID)
		if !ok || en.Fingerprint != Fingerprint(m) {
			t.Fatalf("entry lost submission identity: %+v", en)
		}
	}
	check(e)
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, _ := openEngine(t, path, Options{})
	check(reopened)
}
