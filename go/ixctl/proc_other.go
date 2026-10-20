//go:build !windows

package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// procTree is a service's process tree: its own process group.
type procTree struct {
	pgid int
}

// startTree starts cmd as the leader of a new process group. A stop kills the
// group. A plane that dies hard takes the leader with it on Linux (Pdeathsig),
// and no more: a grandchild the leader forked keeps running (PR_SET_PDEATHSIG
// is per process and cleared across a privileged exec), a descendant that
// calls setsid leaves the group, and Go sends the signal when the thread that
// forked exits, which the runtime does not promise to keep alive. Containing
// the whole tree past the plane's death needs a cgroup or a death pipe the
// service cooperates with; Windows job objects do it today (proc_windows.go).
func startTree(cmd *exec.Cmd) (*procTree, error) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	setPdeathsig(cmd.SysProcAttr)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &procTree{pgid: cmd.Process.Pid}, nil
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
		b, err := os.ReadFile("/proc/" + e.Name() + "/stat")
		if err != nil {
			continue
		}
		i := bytes.LastIndexByte(b, ')')
		if i < 0 {
			continue
		}
		f := strings.Fields(string(b[i+1:]))
		if len(f) < 3 {
			continue
		}
		if g, err := strconv.Atoi(f[2]); err == nil && g == pgid && f[0] != "Z" {
			return true
		}
	}
	return false
}

func (t *procTree) close() {}

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
