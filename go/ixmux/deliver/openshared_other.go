//go:build !windows

package deliver

import "os"

// openShared opens path for reading; on these systems a rename over it does not wait for readers.
func openShared(path string) (*os.File, error) { return os.Open(path) }
