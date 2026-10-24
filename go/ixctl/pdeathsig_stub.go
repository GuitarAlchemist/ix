//go:build !windows && !linux

package main

import "syscall"

func setPdeathsig(*syscall.SysProcAttr) {}
