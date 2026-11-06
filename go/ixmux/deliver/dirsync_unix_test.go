//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package deliver

import (
	"path/filepath"
	"testing"
)

type recordedDir struct {
	syncs *int
}

func (d recordedDir) Sync() error  { *d.syncs++; return nil }
func (d recordedDir) Close() error { return nil }

// S2D-2: creating the command log syncs its directory, so the new entry survives a power loss;
// reopening an existing log does not.
func TestNewLogSyncsItsDirectory(t *testing.T) {
	var opened []string
	syncs := 0
	openDir = func(name string) (dirSyncer, error) {
		opened = append(opened, name)
		return recordedDir{&syncs}, nil
	}
	t.Cleanup(func() { openDir = openDirOS })
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	e.Close()
	if len(opened) != 1 || opened[0] != filepath.Dir(path) || syncs != 1 {
		t.Fatalf("creating the log: directories opened %v, %d syncs; want [%s] and 1", opened, syncs, filepath.Dir(path))
	}
	e, _, err = Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	e.Close()
	if len(opened) != 1 || syncs != 1 {
		t.Errorf("reopening the log synced its directory again: %v, %d syncs", opened, syncs)
	}
}
