//go:build unix

package deliver

import "os"

type dirSyncer interface {
	Sync() error
	Close() error
}

func openDirOS(name string) (dirSyncer, error) { return os.Open(name) }

// openDir opens a directory to sync it; a test replaces it to see the calls.
var openDir = openDirOS

// syncDir fsyncs dir, so a file just created in it keeps its directory entry across a power loss.
// The test below compiles under GOOS=linux but has not run yet (no Unix runner): P, not T.
// @ai:invariant creating the command log fsyncs its directory before any record is written [P:inferred conf:0.6 src:go/ixmux/deliver/dirsync_unix_test.go:TestNewLogSyncsItsDirectory]
func syncDir(dir string) error {
	d, err := openDir(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
