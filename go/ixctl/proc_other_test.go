//go:build !windows

package main

import (
	"bytes"
	"os"
	"os/exec"
	"strconv"
	"syscall"
	"testing"
	"time"
)

// A zombie in its own group (a child exited and not yet reaped) is not a live
// member: the group counts as gone at once, while kill(-pgid, 0) still finds it.
func TestZombieIsNotALiveGroupMember(t *testing.T) {
	if _, err := os.Stat("/proc/self/stat"); err != nil {
		t.Skip("no /proc")
	}
	if !groupHasLive(syscall.Getpgrp()) {
		t.Fatal("the test's own group reads as dead")
	}
	cmd := exec.Command("true")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	pid := cmd.Process.Pid
	deadline := time.Now().Add(5 * time.Second)
	for pidAlive(pid) && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if pidAlive(pid) {
		t.Fatal("true did not exit")
	}
	if syscall.Kill(-pid, 0) != nil {
		t.Fatal("the zombie's group is already gone to kill(-pgid, 0)")
	}
	if groupHasLive(pid) {
		t.Fatal("a zombie counts as a live member")
	}
	began := time.Now()
	if !(&procTree{pgid: pid}).waitGone(5 * time.Second) {
		t.Fatal("waitGone waited the whole limit for a zombie")
	}
	if took := time.Since(began); took > time.Second {
		t.Fatalf("waitGone took %v on a zombie", took)
	}
	cmd.Wait()
}

// A child ended by a signal reports 128 + the signal, the shell's convention,
// not -1.
func TestSignalledExitIs128PlusSignal(t *testing.T) {
	cmd := exec.Command("sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Skip("no sleep:", err)
	}
	cmd.Process.Kill()
	err := cmd.Wait()
	ee, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("wait: %v", err)
	}
	if code := exitCode(ee); code != 137 {
		t.Fatalf("exit code %d, want 137 (128 + SIGKILL)", code)
	}
}

// pidAlive treats a zombie as gone: it is dead, only not yet reaped.
func pidAlive(pid int) bool {
	if syscall.Kill(pid, 0) != nil {
		return false
	}
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return true
	}
	if i := bytes.LastIndexByte(b, ')'); i >= 0 && i+2 < len(b) {
		return b[i+2] != 'Z'
	}
	return true
}

// leaveTree makes cmd the leader of a session of its own, as a daemonising descendant does: it
// leaves the plane's process group, so a group kill does not reach it.
func leaveTree(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true} }
