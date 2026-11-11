//go:build !windows && !unix

package deliver

import "os"

// openShared opens path for reading.
func openShared(path string) (*os.File, error) { return os.Open(path) }
