//go:build !windows

package main

import (
	"errors"
	"os/exec"
	"syscall"
	"time"
)

// procTree is a service's process tree: its own process group.
type procTree struct {
	pgid int
}

// startTree starts cmd as the leader of a new process group, killed with the
// plane (Pdeathsig, on Linux).
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

// waitGone waits until no process is left in the group.
func (t *procTree) waitGone(limit time.Duration) bool {
	deadline := time.Now().Add(limit)
	for {
		if errors.Is(syscall.Kill(-t.pgid, 0), syscall.ESRCH) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(time.Millisecond)
	}
}

func (t *procTree) close() {}

// hrnow is a machine-wide clock in nanoseconds.
func hrnow() int64 { return time.Now().UnixNano() }

// msSinceProcessStart measures from package initialisation: close to, but
// a little after, the process start.
func msSinceProcessStart() float64 {
	return float64(time.Since(initTime).Microseconds()) / 1000
}
