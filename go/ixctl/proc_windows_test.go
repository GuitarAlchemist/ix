//go:build windows

package main

import (
	"os/exec"

	"golang.org/x/sys/windows"
)

const stillActive = 259

func pidAlive(pid int) bool {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(h)
	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		return false
	}
	return code == stillActive
}

// leaveTree leaves cmd as it is: the job object holds every descendant and does not allow
// breakaway, so no descendant can leave the tree on Windows.
func leaveTree(cmd *exec.Cmd) {}
