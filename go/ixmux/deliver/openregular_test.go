package deliver

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// OpenRegular refuses anything but a regular file. A FIFO is Linux-only
// (inventory/fifo_linux_test.go); a directory in a record's place runs this branch on every OS.
func TestOpenRegularRefusesADirectoryAndOpensAFile(t *testing.T) {
	dir := t.TempDir()
	if f, err := OpenRegular(dir); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		if f != nil {
			f.Close()
		}
		t.Errorf("a directory: %v, want refused as not a regular file", err)
	}
	p := filepath.Join(dir, "commands.jsonl")
	if err := os.WriteFile(p, []byte("x\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	f, err := OpenRegular(p)
	if err != nil {
		t.Fatalf("a regular file: %v", err)
	}
	f.Close()
	if _, err := OpenRegular(filepath.Join(dir, "missing")); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("a missing file: %v, want not exist", err)
	}
}
