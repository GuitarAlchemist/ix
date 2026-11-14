//go:build !windows && !unix

package deliver

import "errors"

// syncDir is not available here; Open is refused lock-unsupported before it is reached.
func syncDir(dir string) error { return errors.New("directory sync unsupported on this system") }

func renameDurable(oldpath, newpath string) error {
	return errors.New("rename unsupported on this system")
}
