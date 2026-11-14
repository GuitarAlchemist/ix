//go:build !windows && !unix

package server

import (
	"errors"
	"os"
)

// fileIdentity is not known on this system: a fake session has no incarnation and refuses.
func fileIdentity(f *os.File) (string, error) {
	return "", errors.New("no file identity on this system")
}
