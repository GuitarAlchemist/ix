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
// A mandate id is unique in the whole log: reusing it with another target, incarnation or text
// is refused before any adapter call, also after a restart.
func TestMandateIDReuseIsRefused(t *testing.T) {
	for _, field := range []string{"target", "incarnation", "text"} {
		t.Run(field, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "cmd.jsonl")
			e, _ := openEngine(t, path, Options{})
			a, b := &fakeTarget{inc: "i"}, &fakeTarget{inc: "i"}
			e.Register("a", a.target())
			e.Register("b", b.target())
			m := Mandate{MandateID: "m1", Target: "a", Incarnation: "i", Text: "first"}
			if _, err := e.Deliver(context.Background(), m); err != nil {
				t.Fatal(err)
			}
			other := m
			switch field {
			case "target":
				other.Target = "b"
			case "incarnation":
				a.inc, other.Incarnation = "i2", "i2" // the session restarted
			case "text":
				other.Text = "second"
			}
			res, err := e.Deliver(context.Background(), other)
			if refusalCode(err) != MandateIDReuse {
				t.Errorf("same id, other %s: %+v %v, want %s", field, res, err, MandateIDReuse)
			}
			if a.calls.Load() != 1 || b.calls.Load() != 0 {
				t.Errorf("adapter calls a=%d b=%d, want 1 and 0", a.calls.Load(), b.calls.Load())
			}
			e.Close()
			e2, _ := openEngine(t, path, Options{})
			e2.Register("a", a.target())
			e2.Register("b", b.target())
			if _, err := e2.Deliver(context.Background(), other); refusalCode(err) != MandateIDReuse {
				t.Errorf("after restart, same id, other %s: %v, want %s", field, err, MandateIDReuse)
			}
			if a.calls.Load() != 1 || b.calls.Load() != 0 {
				t.Errorf("after restart, adapter calls a=%d b=%d, want 1 and 0", a.calls.Load(), b.calls.Load())
			}
		})
	}
}
