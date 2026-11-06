package deliver

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// S2C-6: a second Open of a command log held by an engine, from another process, is refused.
func TestSecondWriterIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _ := openEngine(t, path, Options{})
	tg := &fakeTarget{inc: "i"}
	e.Register("a", tg.target())
	if _, err := e.Deliver(context.Background(), Mandate{MandateID: "m1", Target: "a", Incarnation: "i"}); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestHelperProcess$", "-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(), helperEnv+"=open", "IXMUX_LOG="+path)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("helper: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "OPEN-REFUSED "+JournalLocked) {
		t.Errorf("a second process opening the log: %s", strings.TrimSpace(string(out)))
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(before) {
		t.Error("the refused Open changed the log")
	}
	// In the same process too.
	if _, _, err := Open(path, Options{}); refusalCode(err) != JournalLocked {
		t.Errorf("a second Open in the same process: %v, want %s", err, JournalLocked)
	}
	// Once closed, the log opens again.
	e.Close()
	e2, _, err := Open(path, Options{})
	if err != nil {
		t.Fatalf("Open after Close: %v", err)
	}
	e2.Close()
}

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
