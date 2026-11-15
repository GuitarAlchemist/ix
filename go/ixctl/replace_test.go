package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// Every writer of a file a reader trusts (the state file, the Workshop's state, the bookmarks)
// syncs the new content before it replaces the old: a sync that fails fails the write, and the
// file keeps its old content.
func TestReplaceFileSyncsBeforeTheRename(t *testing.T) {
	old := fileSync
	fileSync = func(*os.File) error { return errors.New("sync refused") }
	t.Cleanup(func() { fileSync = old })

	path := filepath.Join(t.TempDir(), "state.json")
	os.WriteFile(path, []byte("old"), 0o600)
	if err := writeFileAtomic(path, []byte("new")); err == nil {
		t.Fatal("the state file was replaced without a sync")
	}
	if b, _ := os.ReadFile(path); string(b) != "old" {
		t.Fatalf("after a failed sync the state file holds %q, want the old content", b)
	}

	bms := &bookmarkStore{path: filepath.Join(t.TempDir(), "b.bookmarks.json")}
	if err := bms.save(nil); err == nil {
		t.Fatal("the bookmarks were replaced without a sync")
	}
}

// holdFor opens path the way a reader does (no delete sharing on Windows, so a replace is
// refused) and lets it go after d.
func holdFor(t *testing.T, path string, d time.Duration) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	go func() { time.Sleep(d); f.Close() }()
	t.Cleanup(func() { f.Close() })
}

// On Windows a file another process reads cannot be replaced until it lets go; the state file
// and the need store ride that out as the bookmarks did, instead of failing at once.
func TestTheStateFileIsReplacedOnceItsReaderLetsGo(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("a reader holds a file against a replace on Windows only")
	}
	path := filepath.Join(t.TempDir(), "state.json")
	os.WriteFile(path, []byte("old"), 0o600)
	holdFor(t, path, 150*time.Millisecond)
	if err := writeFileAtomic(path, []byte("new")); err != nil {
		t.Fatalf("the state file, held 150 ms: %v", err)
	}
	if b, _ := os.ReadFile(path); string(b) != "new" {
		t.Fatalf("state file %q, want new", b)
	}

	s, clk, needPath := newNeedTest(t)
	s.submit(submitOf(clk, "n1", time.Hour))
	holdFor(t, needPath, 150*time.Millisecond)
	if _, err := s.claim("n1", "resolver-x", 60_000); err != nil {
		t.Fatalf("a claim while the need file was held 150 ms: %v", err)
	}
}
