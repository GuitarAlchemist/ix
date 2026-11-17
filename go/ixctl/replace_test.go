package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"testing"
	"time"
)

// A replace syncs the new content before it replaces the old: a sync that fails fails the write,
// and the file keeps its old content. Every writer of a file a reader trusts goes through
// replaceFile, which hands the OS's sync (the state file and the Workshop's state directly, the
// bookmarks and the need store through the replace their constructor sets):
// that wiring is by reading; the test reaches the bookmarks through the value it hands them.
func TestReplaceFileSyncsBeforeTheRename(t *testing.T) {
	t.Parallel()
	refuse := fileOps{sync: func(*os.File) error { return errors.New("sync refused") }, rename: os.Rename}

	path := filepath.Join(t.TempDir(), "state.json")
	os.WriteFile(path, []byte("old"), 0o600)
	if err := refuse.replace(path, []byte("new")); err == nil {
		t.Fatal("the state file was replaced without a sync")
	}
	if b, _ := os.ReadFile(path); string(b) != "old" {
		t.Fatalf("after a failed sync the state file holds %q, want the old content", b)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("after a failed sync the temp is still there (%v)", err)
	}

	bms := &bookmarkStore{path: filepath.Join(t.TempDir(), "b.bookmarks.json"), replace: refuse.replace}
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
	if err := replaceFile(path, []byte("new")); err != nil {
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

// Only Windows refuses a rename because a reader holds the target, so only there is a refusal
// worth a retry: elsewhere a permission error is the directory's or the file's own, and comes
// back at once instead of after the whole retry budget, under the store's lock.
func TestOnlyWindowsRetriesARenameRefusedForPermission(t *testing.T) {
	denied := &os.LinkError{Op: "rename", Old: "a.tmp", New: "a", Err: os.ErrPermission}
	sharing := &os.LinkError{Op: "rename", Old: "a.tmp", New: "a", Err: syscall.Errno(32)}
	missing := &os.LinkError{Op: "rename", Old: "a.tmp", New: "a", Err: os.ErrNotExist}
	for _, c := range []struct {
		goos string
		err  error
		want bool
	}{
		{"windows", denied, true},
		{"windows", sharing, true},
		{"windows", missing, false},
		{"linux", denied, false},
		{"darwin", denied, false},
		{"linux", missing, false},
		{"linux", sharing, false},
	} {
		if got := heldOn(c.goos, c.err); got != c.want {
			t.Errorf("heldOn(%s, %v) = %v, want %v", c.goos, c.err, got, c.want)
		}
	}
}

// os.Rename promises no atomicity off Unix: a rename may remove the file it replaces, then fail.
// The temp is then the only copy of either content, and the need store reads a lone temp as
// unknown, never as empty: replaceFile keeps it. A rename refused with the old file in place
// leaves no temp behind.
func TestARenameThatLostTheOldFileKeepsItsTemp(t *testing.T) {
	t.Parallel()
	s, clk, path := newNeedTest(t)
	s.submit(submitOf(clk, "n1", time.Hour))
	s.replace = fileOps{sync: (*os.File).Sync, rename: func(_, to string) error {
		os.Remove(to)
		return errors.New("rename failed after removing the target")
	}}.replace
	_, err := s.claim("n1", "resolver-x", 60_000)
	wantStatus(t, err, 503, "unknown")
	_, err = newNeedStore(path, clk.now).submit(submitOf(clk, "n1", time.Hour))
	wantStatus(t, err, 503, "missing")

	state := filepath.Join(t.TempDir(), "state.json")
	os.WriteFile(state, []byte("old"), 0o600)
	refused := fileOps{sync: (*os.File).Sync, rename: func(string, string) error { return errors.New("rename refused") }}
	if err := refused.replace(state, []byte("new")); err == nil {
		t.Fatal("a refused rename replaced the file")
	}
	if _, err := os.Stat(state + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("a rename refused with the old file in place left its temp (%v)", err)
	}
	if b, _ := os.ReadFile(state); string(b) != "old" {
		t.Fatalf("after a refused rename the file holds %q, want old", b)
	}
}
