//go:build unix

package deliver

import (
	"os"
	"syscall"
)

// openShared opens path for reading without blocking: a FIFO put where a file is expected would
// block a plain open until a writer comes (OpenRegular then refuses it). A rename over the file
// does not wait for readers here.
func openShared(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
}
