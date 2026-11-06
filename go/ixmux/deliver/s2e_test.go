package deliver

import (
	"os"
	"path/filepath"
	"testing"
)

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
