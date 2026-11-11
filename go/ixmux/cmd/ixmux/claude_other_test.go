//go:build !windows

package main

import "syscall"

// alive: the process pid exists.
func alive(pid int) bool { return syscall.Kill(pid, 0) == nil }
