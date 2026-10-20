package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Two startups of one name in one state dir: the second is refused by the
// OS lock, whether or not the first has written its state file yet, so one
// journal never carries two boots' records interleaved under one header.
// The lock goes with the instance: the name is free again after close.
func TestSecondInstanceOfANameIsRefused(t *testing.T) {
	stateDir := t.TempDir()
	in := newTestInstance(t, fakeSpec(t, "local", nil), stateDir)
	_, err := startInstance(fakeSpec(t, "local", nil), "", "127.0.0.1:0", stateDir)
	if err == nil || !strings.Contains(err.Error(), `"local"`) {
		t.Fatalf("second instance: err %v, want a refusal naming the instance", err)
	}
	// The window the state-file check cannot see: no state file, the lock held.
	os.Remove(filepath.Join(stateDir, "local.json"))
	if _, err := startInstance(fakeSpec(t, "local", nil), "", "127.0.0.1:0", stateDir); err == nil {
		t.Fatal("second instance started while the first holds the lock and no state file exists")
	}
	in.close()
	again := newTestInstance(t, fakeSpec(t, "local", nil), stateDir)
	if again.plane.Epoch() == in.plane.Epoch() {
		t.Fatal("the second boot drew the first boot's epoch")
	}
	j, err := os.ReadFile(filepath.Join(stateDir, "local.journal.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(string(j), `"kind":"boot"`); n != 2 {
		t.Fatalf("%d boot lines in the journal, want 2: the refused startups wrote none", n)
	}
}
