//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd

package deliver

import (
	"errors"
	"io"
	"os"
	"syscall"
)

// lockFile takes an exclusive flock on path, released when the file is closed or the process
// dies. The test that binds it has run on Windows only: P, not T, here.
// @ai:invariant one engine per command log under flock [P:inferred conf:0.6 src:go/ixmux/deliver/hardening_test.go:TestSecondWriterIsRefused]
func lockFile(path string) (io.Closer, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, errLocked
		}
		return nil, err
	}
	return f, nil
}
