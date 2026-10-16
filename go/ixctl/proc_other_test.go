//go:build !windows

package main

import (
	"bytes"
	"os"
	"strconv"
	"syscall"
)

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
