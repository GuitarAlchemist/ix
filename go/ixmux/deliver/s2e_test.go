package deliver

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// S2E-5 (P2): NUL bytes cannot make two different mandates share a fingerprint.
func TestFingerprintHasNoNULAmbiguity(t *testing.T) {
	a := Mandate{MandateID: "m", Target: "t", Incarnation: "b\x00c", Text: "d"}
	b := Mandate{MandateID: "m", Target: "t", Incarnation: "b", Text: "c\x00d"}
	if fingerprint(a) == fingerprint(b) {
		t.Errorf("incarnation %q + text %q and incarnation %q + text %q share a fingerprint", a.Incarnation, a.Text, b.Incarnation, b.Text)
	}
}

type failingRelease struct{ io.Closer }

func (f failingRelease) Close() error {
	f.Closer.Close()
	return errors.New("release failed")
}

// S2E-5 (P2): Close reports a failed lock release.
func TestCloseReportsTheLockRelease(t *testing.T) {
	e, _, err := Open(filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	e.log.lock = failingRelease{e.log.lock}
	if err := e.Close(); err == nil {
		t.Error("Close hid a failed lock release")
	}
}

// S2E-3 (Codex round 3): a log without a format version (records without fp) is refused at
// Open, with no transition written.
func TestLogWithoutFormatVersionIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	old := `{"at":"2026-11-05T00:00:00Z","target":"a","mandate_id":"m1","incarnation":"i","state":"accepted"}` + "\n" +
		`{"at":"2026-11-05T00:00:01Z","target":"a","mandate_id":"m1","incarnation":"i","state":"delivering"}` + "\n"
	if err := os.WriteFile(path, []byte(old), 0o600); err != nil {
		t.Fatal(err)
	}
	e, rec, err := Open(path, Options{})
	if err == nil {
		e.Close()
	}
	if refusalCode(err) != LogFormatUnsupported {
		t.Errorf("a log without a format version: recovery %+v, %v; want %s", rec, err, LogFormatUnsupported)
	}
	if b, _ := os.ReadFile(path); string(b) != old {
		t.Errorf("the refused log was changed:\n%s", b)
	}
	// An unknown version too.
	other := `{"format":"ixmux.cmdlog.v99"}` + "\n"
	if err := os.WriteFile(path, []byte(other), 0o600); err != nil {
		t.Fatal(err)
	}
	if e, _, err := Open(path, Options{}); refusalCode(err) != LogFormatUnsupported {
		if err == nil {
			e.Close()
		}
		t.Errorf("an unknown format version: %v, want %s", err, LogFormatUnsupported)
	}
}
