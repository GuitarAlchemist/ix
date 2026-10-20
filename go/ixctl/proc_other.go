//go:build !windows

package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// procTree is a service's process tree: its own process group.
type procTree struct {
	pgid int
	over chan string   // the limit the tree went past, once; nil when none is watched
	done chan struct{} // closed by close: ends the memory watch
	mu   sync.Mutex
	peak int64 // the largest resident set the watch has summed over the group
}

// memoryPoll is how often a watched group's resident set is read from /proc.
const memoryPoll = 100 * time.Millisecond

// startTree starts cmd as the leader of a new process group. A stop kills the
// group. A plane that dies hard takes the leader with it on Linux (Pdeathsig),
// and no more: a grandchild the leader forked keeps running (PR_SET_PDEATHSIG
// is per process and cleared across a privileged exec), a descendant that
// calls setsid leaves the group, and Go sends the signal when the thread that
// forked exits, which the runtime does not promise to keep alive. Containing
// the whole tree past the plane's death needs a cgroup or a death pipe the
// service cooperates with; Windows job objects do it today (proc_windows.go).
// With a memory limit, a goroutine sums the group's resident set from /proc
// every memoryPoll and reports the limit gone past on `exceeded`: nothing in
// the kernel bounds a group without a cgroup, so a spike shorter than one
// poll is not seen.
func startTree(cmd *exec.Cmd, limits *Limits) (*procTree, error) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	setPdeathsig(cmd.SysProcAttr)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	t := &procTree{pgid: cmd.Process.Pid, done: make(chan struct{})}
	if limits != nil && limits.MemoryMB > 0 {
		t.over = make(chan string, 1)
		go t.watchMemory(limits.MemoryMB << 20)
	}
	return t, nil
}

// watchMemory reads the group's resident set every memoryPoll until the tree is closed or
// has no live member, keeps the peak, and reports the limit gone past, once.
func (t *procTree) watchMemory(limit int64) {
	tick := time.NewTicker(memoryPoll)
	defer tick.Stop()
	for {
		select {
		case <-t.done:
			return
		case <-tick.C:
		}
		rss, live := groupRSS(t.pgid)
		if !live {
			return
		}
		t.mu.Lock()
		if rss > t.peak {
			t.peak = rss
		}
		t.mu.Unlock()
		if rss > limit {
			t.over <- reasonMemory
			return
		}
	}
}

// exceeded yields the limit the tree went past, once; nil when none is watched.
func (t *procTree) exceeded() <-chan string { return t.over }

// usage is the largest resident set the memory watch summed over the group (0 without a
// watch); nothing counts the group's processor time without a cgroup, so that is 0 here and
// comes from the leader's rusage (leaderUsage).
func (t *procTree) usage() (peakBytes, cpuMs int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.peak, 0
}

// leaderUsage is the leader's peak resident set (kilobytes on Linux, bytes elsewhere) and
// its processor time, user and system, with that of the children it waited for, as the
// kernel accounted them at exit (rusage); 0 when unknown.
func leaderUsage(ps *os.ProcessState) (rssBytes, cpuMs int64) {
	if ps == nil {
		return 0, 0
	}
	ru, ok := ps.SysUsage().(*syscall.Rusage)
	if !ok {
		return 0, 0
	}
	rssBytes = int64(ru.Maxrss)
	if runtime.GOOS == "linux" {
		rssBytes *= 1024
	}
	cpuMs = (int64(ru.Utime.Sec)+int64(ru.Stime.Sec))*1000 + (int64(ru.Utime.Usec)+int64(ru.Stime.Usec))/1000
	return rssBytes, cpuMs
}

// groupRSS sums the resident set of the live members of group pgid from /proc (statm's
// second field, in pages) and reports whether any live member was found.
func groupRSS(pgid int) (int64, bool) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0, true
	}
	page := int64(os.Getpagesize())
	var sum int64
	live := false
	for _, e := range entries {
		if _, err := strconv.Atoi(e.Name()); err != nil {
			continue
		}
		if g, zombie, ok := statOf(e.Name()); !ok || g != pgid || zombie {
			continue
		}
		live = true
		b, err := os.ReadFile("/proc/" + e.Name() + "/statm")
		if err != nil {
			continue
		}
		if f := strings.Fields(string(b)); len(f) >= 2 {
			if pages, err := strconv.ParseInt(f[1], 10, 64); err == nil {
				sum += pages * page
			}
		}
	}
	return sum, live
}

// statOf reads a process's group and zombie state from /proc/<pid>/stat (state and pgrp are
// the first and third fields after the command's closing parenthesis).
func statOf(pid string) (pgid int, zombie, ok bool) {
	b, err := os.ReadFile("/proc/" + pid + "/stat")
	if err != nil {
		return 0, false, false
	}
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return 0, false, false
	}
	f := strings.Fields(string(b[i+1:]))
	if len(f) < 3 {
		return 0, false, false
	}
	g, err := strconv.Atoi(f[2])
	if err != nil {
		return 0, false, false
	}
	return g, f[0] == "Z", true
}

func (t *procTree) kill() error {
	err := syscall.Kill(-t.pgid, syscall.SIGKILL)
	if errors.Is(err, syscall.ESRCH) {
		return nil
	}
	return err
}

// waitGone waits until no live process is left in the group. A zombie, dead
// but not yet reaped (a grandchild under a PID 1 that does not reap, as in a
// container), is not waited for: it would hold the group for the whole limit.
func (t *procTree) waitGone(limit time.Duration) bool {
	deadline := time.Now().Add(limit)
	for {
		if errors.Is(syscall.Kill(-t.pgid, 0), syscall.ESRCH) || !groupHasLive(t.pgid) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(time.Millisecond)
	}
}

// groupHasLive reports whether some process of group pgid is not a zombie,
// read from /proc (state and pgrp are the first and third fields after the
// command's closing parenthesis). Without /proc every member counts as live.
func groupHasLive(pgid int) bool {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return true
	}
	for _, e := range entries {
		if _, err := strconv.Atoi(e.Name()); err != nil {
			continue
		}
		if g, zombie, ok := statOf(e.Name()); ok && g == pgid && !zombie {
			return true
		}
	}
	return false
}

func (t *procTree) close() {
	select {
	case <-t.done:
	default:
		close(t.done)
	}
}

// exitCode is the shell's reading of how a child ended: its exit status, or
// 128 plus the signal that ended it (137 for SIGKILL), rather than -1 for
// every signal alike.
func exitCode(ee *exec.ExitError) int {
	if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
		return 128 + int(ws.Signal())
	}
	return ee.ExitCode()
}

// hrnow is a machine-wide clock in nanoseconds.
func hrnow() int64 { return time.Now().UnixNano() }

// msSinceProcessStart measures from package initialisation: close to, but
// a little after, the process start.
func msSinceProcessStart() float64 {
	return float64(time.Since(initTime).Microseconds()) / 1000
}
