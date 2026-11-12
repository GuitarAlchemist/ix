package inventory

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// A FIFO where a session record, ixmux.json or the command log is expected, directly or through a
// symlink, would block a plain open forever: each is unreadable and the report comes back
// (review P2-5).
func TestAFIFOInPlaceOfAFileNeverHangs(t *testing.T) {
	home, dir := t.TempDir(), t.TempDir()
	os.MkdirAll(filepath.Join(home, "sessions"), 0o700)
	fifo := filepath.Join(t.TempDir(), "fifo")
	for _, p := range []string{fifo, filepath.Join(home, "sessions", "1.json"), filepath.Join(dir, "ixmux.json"), filepath.Join(dir, "commands.jsonl")} {
		if err := syscall.Mkfifo(p, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(fifo, filepath.Join(home, "sessions", "2.json")); err != nil {
		t.Fatal(err)
	}
	done := make(chan Report, 1)
	go func() { done <- collect(home, procs{}, dir) }()
	select {
	case r := <-done:
		for _, k := range [][2]string{{"claude-session", "1.json"}, {"claude-session", "2.json"}, {"ixmux", dir}, {"mandate-log", filepath.Join(dir, "commands.jsonl")}} {
			if es := find(r, k[0], k[1]); len(es) != 1 || es[0].Status != Unreadable {
				t.Errorf("%s %s: %+v", k[0], k[1], es)
			}
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the inventory hung on a FIFO")
	}
}
