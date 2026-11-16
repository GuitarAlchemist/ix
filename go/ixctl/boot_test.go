package main

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// Every option reaches the plane before bootInstance returns: the probe kinds' settings and the
// budget, whichever manager filled them.
func TestBootInstanceAppliesEveryOption(t *testing.T) {
	in, err := bootInstance(instanceOptions{spec: fakeSpec(t, "boot", nil), addr: "127.0.0.1:0", stateDir: t.TempDir(),
		bookmarks: "memory", budget: &probeBudget{maxRate: 300, treeRate: 600},
		deltaJournal: "delta.jsonl", claudeBin: "claude.exe"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(in.close)
	tr := in.plane.probes
	tr.mu.Lock()
	cfg := tr.cfg
	tr.mu.Unlock()
	if cfg.deltaJournal != "delta.jsonl" || cfg.claudeBin != "claude.exe" {
		t.Fatalf("probe settings after boot: delta %q claude %q", cfg.deltaJournal, cfg.claudeBin)
	}
	if b := in.plane.probes.budgetNow(); b.maxRate != 300 || b.treeRate != 600 {
		t.Fatalf("probe budget after boot: %+v", b)
	}
}

// A Workshop that does not start fails the boot as such, and lets go of the name: the next boot
// of the same name in the same state directory starts.
func TestABootThatFailsAtTheWorkshopLetsGoOfTheName(t *testing.T) {
	dir := t.TempDir()
	o := instanceOptions{spec: fakeSpec(t, "ws", nil), addr: "127.0.0.1:0", stateDir: dir, bookmarks: "memory",
		workshop: workshopSettings{Version: 2, Enabled: true}}
	_, err := bootInstance(o)
	var ws errWorkshopStart
	if !errors.As(err, &ws) {
		t.Fatalf("a bad Workshop setting: %v, want a Workshop start failure", err)
	}
	o.workshop = workshopSettings{}
	in, err := bootInstance(o)
	if err != nil {
		t.Fatalf("the boot after a Workshop failure: %v", err)
	}
	in.close()
}

// The service host tells its three start failures apart in its log and its exit code: a
// Workshop that did not start is workshop.failed (exit 1), not host.failed nor host.refused.
func TestAHostWhoseWorkshopFailsSaysSo(t *testing.T) {
	log := &syncBuffer{}
	o := hostOpts(fakeSpec(t, "wsfail", nil), t.TempDir(), log)
	o.workshop = workshopSettings{Version: 2, Enabled: true}
	_, done := startHost(o)
	if code := waitExit(t, done, 10*time.Second); code != exitStartFailed {
		t.Fatalf("exit %d, want %d\n%s", code, exitStartFailed, log.String())
	}
	if s := log.String(); !strings.Contains(s, `"event":"workshop.failed"`) || strings.Contains(s, `"event":"host.failed"`) ||
		!strings.Contains(s, "Workshop config version must be 1") {
		t.Fatalf("log of a Workshop that did not start:\n%s", s)
	}
}
