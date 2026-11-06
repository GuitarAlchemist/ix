package deliver

import (
	"errors"
	"io"
	"os"
	"syscall"
)

const errorSharingViolation = syscall.Errno(32) // ERROR_SHARING_VIOLATION

// lockFile holds path open with no sharing: any other open of it, from this process or another,
// fails until the handle is closed, which the system does when the process dies.
// @ai:invariant one engine per command log: a second Open is refused while the first holds the lock [T:test conf:0.85 src:go/ixmux/deliver/hardening_test.go:TestSecondWriterIsRefused]
// @ai:invariant the lock is released when its process is killed: the log reopens after the owner is killed mid-delivery [T:test conf:0.85 src:go/ixmux/deliver/kill_test.go:TestKillAfterEffectThenReconcileReceived]
func lockFile(path string) (io.Closer, error) {
	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	h, err := syscall.CreateFile(p, syscall.GENERIC_READ|syscall.GENERIC_WRITE, 0, nil,
		syscall.OPEN_ALWAYS, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if errors.Is(err, errorSharingViolation) {
		return nil, errLocked
	}
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(h), path), nil
}
