package server

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// replace puts a new file, of another session, at path: another file identity under the same path.
func replace(t *testing.T, path, content string) {
	t.Helper()
	tmp := path + ".new"
	if err := os.WriteFile(tmp, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, path); err != nil {
		t.Fatal(err)
	}
}

// S3B-2 (Codex round 6, P1): session.jsonl is replaced after the engine checked the incarnation
// and before the adapter writes. The adapter compares the incarnation on the file it opened, under
// the same lock as the write: it refuses, and the new session is left untouched.
func TestFakeSessionChecksTheFileItWrites(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	f, err := NewFakeSession("s", path)
	if err != nil {
		t.Fatal(err)
	}
	tg := f.Target()
	inc, err := tg.Incarnation()
	if err != nil {
		t.Fatal(err)
	}
	const other = `{"mandate_id":"x","incarnation":"another","text":"another session"}` + "\n"
	replace(t, path, other)
	_, err = f.Deliver(context.Background(), deliver.Mandate{MandateID: "m1", Target: "s", Incarnation: inc, Text: "hello"})
	if !errors.Is(err, deliver.ErrRefused) {
		t.Errorf("deliver to a replaced session: %v, want a refusal", err)
	}
	if b, _ := os.ReadFile(path); string(b) != other {
		t.Errorf("the replacement session was written to:\n%s", b)
	}
	if now, _ := tg.Incarnation(); now == inc {
		t.Errorf("incarnation %s unchanged after the file was replaced", now)
	}
	if o, err := f.Status(context.Background(), "x", inc); o != deliver.Unknown || err != nil {
		t.Errorf("Status of the old incarnation on the new file: %s %v, want unknown", o, err)
	}
}
