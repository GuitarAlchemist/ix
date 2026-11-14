//go:build !windows

package server

import (
	"errors"
	"os"
)

// writeSecret creates path holding b with mode 0600 (the user alone; a umask can only narrow
// it). Any failure leaves no file and refuses the start.
// @ai:assumption mode 0600 on a fresh O_EXCL file is the user-only equivalent of the Windows DACL [P:inferred conf:0.6]
func writeSecret(path string, b []byte) (err error) {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			os.Remove(path)
		}
	}()
	if _, err := f.Write(b); err != nil {
		return err
	}
	return f.Sync()
}
