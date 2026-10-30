//go:build !windows

package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// procTree is a service's process tree: its own process group.
type procTree struct {
	pgid      int
	over      chan string   // the limit the tree went past, once; nil when none is watched
	done      chan struct{} // closed by close: ends the watch
	gone      chan struct{} // closed when the watch has returned
	mu        sync.Mutex
	peak      int64 // the largest resident set the watch has summed over the group
	cpu       int64 // the largest processor time the watch has summed over the group, in milliseconds
	peakProcs int64 // the most live members the watch has counted in the group
}

// watchPoll is how often a watched group's resident set and processor time are read from /proc.
const watchPoll = 100 * time.Millisecond

// startTree starts cmd as the leader of a new process group. A stop kills the
// group. A plane that dies hard takes the leader with it on Linux (Pdeathsig),
// and no more: a grandchild the leader forked keeps running (PR_SET_PDEATHSIG
// is per process and cleared across a privileged exec), a descendant that
// calls setsid leaves the group, and Go sends the signal when the thread that
// forked exits, which the runtime does not promise to keep alive. Containing
// the whole tree past the plane's death needs a cgroup or a death pipe the
// service cooperates with; Windows job objects do it today (proc_windows.go).
// With a memory or processor-time limit, a goroutine sums the group's
// resident set and processor time from /proc every watchPoll and reports the
// limit gone past on `exceeded`: nothing in the kernel bounds a group without
// a cgroup, so a spike shorter than one poll is not seen.
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
	if limits != nil && (limits.MemoryMB > 0 || limits.CpuMs > 0 || limits.Processes > 0) {
		t.over, t.gone = make(chan string, 1), make(chan struct{})
		go t.watch(limits.MemoryMB<<20, limits.CpuMs, limits.Processes)
	}
	return t, nil
}

// watch reads the group's resident set, processor time and live member count every watchPoll
// until the tree is closed or has no live member, keeps their peaks, and reports a limit gone
// past, once (0 is no limit).
func (t *procTree) watch(memoryBytes, cpuMs, processes int64) {
	defer close(t.gone)
	tick := time.NewTicker(watchPoll)
	defer tick.Stop()
	for {
		select {
		case <-t.done:
			return
		case <-tick.C:
		}
		rss, cpu, count := groupUsage(t.pgid)
		if count == 0 {
			return
		}
		t.mu.Lock()
		if rss > t.peak {
			t.peak = rss
		}
		if cpu > t.cpu {
			// The sum can fall: a member that dies takes its time with it until its parent
			// waits for it (then it is in the parent's cutime), and a descendant reaped by
			// PID 1 never comes back. What was seen stays seen.
			t.cpu = cpu
		}
		if count > t.peakProcs {
			t.peakProcs = count
		}
		t.mu.Unlock()
		switch {
		case memoryBytes > 0 && rss > memoryBytes:
			t.over <- reasonMemory
			return
		case cpuMs > 0 && cpu > cpuMs:
			t.over <- reasonCPU
			return
		case processes > 0 && count > processes:
			t.over <- reasonProcesses
			return
		}
	}
}

// exceeded yields the limit the tree went past, once; nil when none is watched.
func (t *procTree) exceeded() <-chan string { return t.over }

// sample reads the group once more into the peaks, for a tree about to be killed before the
// watch's next tick.
func (t *procTree) sample() {
	if t.gone == nil {
		return
	}
	rss, cpu, count := groupUsage(t.pgid)
	t.mu.Lock()
	if rss > t.peak {
		t.peak = rss
	}
	if cpu > t.cpu {
		t.cpu = cpu
	}
	if count > t.peakProcs {
		t.peakProcs = count
	}
	t.mu.Unlock()
}

// drain waits, at most limit, until the watch has returned (it does within a poll of the
// group's last live member going), so what `exceeded` holds is final; at once without a
// watch.
func (t *procTree) drain(limit time.Duration) {
	if t.gone == nil {
		return
	}
	select {
	case <-t.gone:
	case <-time.After(limit):
	}
}

// usage is the largest resident set, the largest processor time and the most live members
// the watch counted over the group (0 without a watch: nothing counts a group without a
// cgroup, and the leader's rusage then speaks for the tree, leaderUsage). Time a descendant
// used between two polls and took with it (reaped by PID 1) is not seen.
func (t *procTree) usage() (peakBytes, cpuMs, peakProcs int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.peak, t.cpu, t.peakProcs
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

// groupUsage sums, over the live members of group pgid, the resident set (statm's second
// field, in pages) and the processor time (stat's utime, stime and the cutime, cstime of the
// children each has waited for, which are dead and so counted nowhere else), in milliseconds,
// and counts those members (0 when none is live; 1 without /proc, so the watch goes on).
func groupUsage(pgid int) (rssBytes, cpuMs, live int64) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return 0, 0, 1
	}
	page := int64(os.Getpagesize())
	var ticks int64
	for _, e := range entries {
		if _, err := strconv.Atoi(e.Name()); err != nil {
			continue
		}
		st, ok := statOf(e.Name())
		if !ok || st.pgid != pgid || st.zombie {
			continue
		}
		live++
		ticks += st.ticks
		b, err := os.ReadFile("/proc/" + e.Name() + "/statm")
		if err != nil {
			continue
		}
		if f := strings.Fields(string(b)); len(f) >= 2 {
			if pages, err := strconv.ParseInt(f[1], 10, 64); err == nil {
				rssBytes += pages * page
			}
		}
	}
	return rssBytes, ticks * 1000 / clockTicks(), live
}

// procStat is what the watch reads of /proc/<pid>/stat.
type procStat struct {
	pgid   int
	zombie bool
	ticks  int64 // utime + stime + cutime + cstime, in clock ticks
}

// statOf reads a process's group, zombie state and processor time from /proc/<pid>/stat:
// after the command's closing parenthesis, the state is field 1, the group field 3 and the
// times fields 12 to 15 (14 to 17 of the whole line).
func statOf(pid string) (procStat, bool) {
	b, err := os.ReadFile("/proc/" + pid + "/stat")
	if err != nil {
		return procStat{}, false
	}
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return procStat{}, false
	}
	f := strings.Fields(string(b[i+1:]))
	if len(f) < 15 {
		return procStat{}, false
	}
	g, err := strconv.Atoi(f[2])
	if err != nil {
		return procStat{}, false
	}
	st := procStat{pgid: g, zombie: f[0] == "Z"}
	for _, k := range []int{11, 12, 13, 14} {
		n, err := strconv.ParseInt(f[k], 10, 64)
		if err != nil {
			return procStat{}, false
		}
		st.ticks += n
	}
	return st, true
}

var (
	clockTicksOnce sync.Once
	clockTicksHz   int64
)

// clockTicks is the unit of /proc's processor times (sysconf(_SC_CLK_TCK)), read from the
// AT_CLKTCK entry of this process's auxiliary vector; 100, the kernel's USER_HZ everywhere
// that matters, when it cannot be read.
func clockTicks() int64 {
	clockTicksOnce.Do(func() {
		clockTicksHz = 100
		const atClktck = 17
		b, err := os.ReadFile("/proc/self/auxv")
		if err != nil {
			return
		}
		word := int(unsafe.Sizeof(uintptr(0)))
		for i := 0; i+2*word <= len(b); i += 2 * word {
			var key, val uint64
			if word == 8 {
				key, val = binary.NativeEndian.Uint64(b[i:]), binary.NativeEndian.Uint64(b[i+8:])
			} else {
				key, val = uint64(binary.NativeEndian.Uint32(b[i:])), uint64(binary.NativeEndian.Uint32(b[i+4:]))
			}
			if key == atClktck && val > 0 {
				clockTicksHz = int64(val)
				return
			}
		}
	})
	return clockTicksHz
}

// kill ends every process of the group, after a last look at what it holds (a peak the watch
// may not have ticked for).
func (t *procTree) kill() error {
	t.sample()
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
		if st, ok := statOf(e.Name()); ok && st.pgid == pgid && !st.zombie {
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
