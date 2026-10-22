//go:build !windows

package main

import (
	"errors"
	"os"
	"syscall"
)

// lockFile takes an exclusive, non-blocking lock on path, held until the
// returned file is closed or the process ends. held is false when another
// process holds it; the file is then already closed.
func lockFile(path string) (f *os.File, held bool, err error) {
	f, err = os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, false, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, false, nil
		}
		return nil, false, err
	}
	return f, true, nil
}
