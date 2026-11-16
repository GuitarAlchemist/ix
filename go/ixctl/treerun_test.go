package main

import (
	"strings"
	"testing"
	"time"
)

// fakeTree records the calls finishTree makes, in order, and answers from its fields.
type fakeTree struct {
	calls            []string
	gone             bool
	over             string // what the monitor says the tree went past; empty: nothing
	peak, cpu, procs int64
	closed           bool
}

func (f *fakeTree) kill() error { f.calls = append(f.calls, "kill"); return nil }
func (f *fakeTree) waitGone(time.Duration) bool {
	f.calls = append(f.calls, "waitGone")
	return f.gone
}
func (f *fakeTree) drain(time.Duration) { f.calls = append(f.calls, "drain") }
func (f *fakeTree) exceeded() <-chan string {
	f.calls = append(f.calls, "exceeded")
	ch := make(chan string, 1)
	if f.over != "" {
		ch <- f.over
	}
	return ch
}
func (f *fakeTree) usage() (int64, int64, int64) {
	f.calls = append(f.calls, "usage")
	if f.closed {
		return 0, 0, 0 // the job is gone after close
	}
	return f.peak, f.cpu, f.procs
}
func (f *fakeTree) close() { f.calls = append(f.calls, "close"); f.closed = true }

// The steps of a tree's end run in one order: the usage is read after the monitor is drained and
// before the tree is closed, and the end says whether the tree was seen gone.
func TestFinishTreeReadsInOrder(t *testing.T) {
	f := &fakeTree{gone: false, peak: 100, cpu: 20, procs: 3}
	end := finishTree(f, exitStat{code: 0, rss: 150, cpu: 10}, nil, "", false)
	if got := strings.Join(f.calls, " "); got != "kill waitGone drain exceeded usage close" {
		t.Fatalf("steps: %s", got)
	}
	if end.seenGone || end.peak != 150 || end.cpu != 20 || end.procs != 3 || end.reason != "" {
		t.Fatalf("end %+v, want not seen gone, peak raised to the leader's 150, cpu 20, 3 processes, no reason", end)
	}
	if end := finishTree(&fakeTree{gone: true}, exitStat{}, nil, "", false); !end.seenGone {
		t.Fatalf("a tree seen gone: end %+v", end)
	}
}

// The monitor's limit names the end of a run that ended on its own; the processor-time and
// process limits are read from the figures when the monitor named none.
func TestFinishTreeNamesTheLimitARunWentPast(t *testing.T) {
	if end := finishTree(&fakeTree{gone: true, over: reasonMemory}, exitStat{}, nil, "", false); end.reason != reasonMemory {
		t.Fatalf("monitor said memory: reason %q", end.reason)
	}
	lim := &Limits{CpuMs: 50, Processes: 4}
	if end := finishTree(&fakeTree{gone: true, cpu: 51}, exitStat{}, lim, "", false); end.reason != reasonCPU {
		t.Fatalf("cpu 51 over 50: reason %q", end.reason)
	}
	if end := finishTree(&fakeTree{gone: true, procs: 5}, exitStat{}, lim, "", false); end.reason != reasonProcesses {
		t.Fatalf("5 processes over 4: reason %q", end.reason)
	}
	if end := finishTree(&fakeTree{gone: true, over: reasonMemory}, exitStat{}, nil, reasonWall, false); end.reason != reasonWall {
		t.Fatalf("killed for the wall limit: reason %q, the plane's own", end.reason)
	}
}

// A run the plane ended (a requested stop) names no limit, whatever the monitor or the figures say.
func TestARequestedEndNamesNoLimit(t *testing.T) {
	f := &fakeTree{gone: true, over: reasonMemory, cpu: 99, procs: 9}
	if end := finishTree(f, exitStat{}, &Limits{CpuMs: 1, Processes: 1}, "", true); end.reason != "" {
		t.Fatalf("a requested end named %q", end.reason)
	}
}

// killAndWait gives the leader treeGoneLimit to report its exit after the kill; past it the exit
// is unknown. The give-up waits the whole bound (five seconds), so the test runs in parallel.
func TestKillAndWaitTakesTheExitOrGivesUp(t *testing.T) {
	t.Parallel()
	exited := make(chan exitStat, 1)
	exited <- exitStat{code: 3}
	f := &fakeTree{}
	if st := killAndWait(f, exited); st.code != 3 || strings.Join(f.calls, " ") != "kill" {
		t.Fatalf("exit %+v after %v, want code 3 after one kill", st, f.calls)
	}
	t0 := time.Now()
	if st := killAndWait(&fakeTree{}, make(chan exitStat)); st.code != -1 {
		t.Fatalf("no exit reported: %+v, want code -1", st)
	}
	if d := time.Since(t0); d < treeGoneLimit {
		t.Fatalf("gave up after %v, before treeGoneLimit (%v)", d, treeGoneLimit)
	}
}
