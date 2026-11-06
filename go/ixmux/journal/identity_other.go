//go:build unix

package journal

import (
	"fmt"
	"os"
	"syscall"
)

// fileIdentity names the file behind f (device and inode), so another file later written under
// the same path is another incarnation. Same format as the V2 prototype.
func fileIdentity(f *os.File) (string, error) {
	st, err := f.Stat()
	if err != nil {
		return "", err
	}
	s, ok := st.Sys().(*syscall.Stat_t)
	if !ok {
		return "", fmt.Errorf("no device and inode for %s", f.Name())
	}
	return fmt.Sprintf("%x:%x", uint64(s.Dev), uint64(s.Ino)), nil
}
