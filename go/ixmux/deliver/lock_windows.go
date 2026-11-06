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
