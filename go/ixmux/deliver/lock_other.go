//go:build !windows && !(darwin || dragonfly || freebsd || linux || netbsd || openbsd)

package deliver

import "io"

// lockFile has no implementation here (AIX, Solaris, illumos and other systems without flock in
// package syscall): Open is refused lock-unsupported rather than run unlocked, since two engines
// on one log could deliver a mandate twice.
func lockFile(path string) (io.Closer, error) { return nil, errLockUnsupported }
