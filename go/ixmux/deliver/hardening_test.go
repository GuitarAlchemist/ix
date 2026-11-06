package deliver

import (
	"context"
	"path/filepath"
	"testing"
)

// S2C-5: the same mandate id with another text, target or incarnation is refused, not replayed.
func TestMandateIDReuseIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _ := openEngine(t, path, Options{})
	tg := &fakeTarget{inc: "i"}
	e.Register("a", tg.target())
	m := Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "first"}
	if _, err := e.Deliver(context.Background(), m); err != nil {
		t.Fatal(err)
	}
	other := m
	other.Text = "second"
	res, err := e.Deliver(context.Background(), other)
	if refusalCode(err) != MandateIDReuse {
		t.Errorf("same id, other text: %+v %v, want %s", res, err, MandateIDReuse)
	}
	if res, err := e.Deliver(context.Background(), m); err != nil || !res.Replayed {
		t.Errorf("same mandate again: %+v %v, want replayed", res, err)
	}
	if tg.calls.Load() != 1 {
		t.Errorf("%d adapter calls, want 1", tg.calls.Load())
	}
	// The fingerprint survives a restart.
	e.Close()
	e2, _ := openEngine(t, path, Options{})
	e2.Register("a", tg.target())
	if _, err := e2.Deliver(context.Background(), other); refusalCode(err) != MandateIDReuse {
		t.Errorf("after restart, same id, other text: %v, want %s", err, MandateIDReuse)
	}
	if tg.calls.Load() != 1 {
		t.Errorf("%d adapter calls after restart, want 1", tg.calls.Load())
	}
}
