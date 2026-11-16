package main

import (
	"errors"
	"os"
	"runtime"
	"syscall"
	"time"
)

// replaceError says at which stage replaceFile failed: stageTemp (writing or syncing the new
// content beside the file: the file is as it was) or stageRename (the replace itself: the file
// is as it was, unless the rename went through and only its answer was lost, or removed the
// file and then failed, which only the temp then holds).
type replaceError struct {
	stage string
	err   error
}

const (
	stageTemp   = "temp"
	stageRename = "rename"
)

func (e *replaceError) Error() string { return e.err.Error() }
func (e *replaceError) Unwrap() error { return e.err }

// tempOf is the file a replace of path writes first: a reader that finds it without path
// beside it is looking at a replace that removed the file and then failed.
func tempOf(path string) string { return path + ".tmp" }

// fileOps is what a replace needs from the file system: sync makes a file's content durable,
// rename replaces one file with another. replaceFile uses the OS's; a test builds its own value
// with the step it wants to fail, so no test changes what another sees.
type fileOps struct {
	sync   func(*os.File) error
	rename func(oldpath, newpath string) error
}

// replaceFile replaces path's content with b by a rename: where the OS's rename is atomic
// (Unix), a reader sees the old content or the new, never a part; Windows does not promise it.
// The new content is written to path.tmp and synced before the rename (the file's content, not
// the directory: what a power loss can still lose is in the README's need section); on Windows
// the rename is refused while another process holds the file it replaces (an indexer or an
// antivirus reading what the last write left), so a few short retries, about a second in all,
// ride that out. On a failure the temp is removed when it can be, unless the rename failed and
// path cannot be seen (gone, or a Stat refused): then the temp may be the only copy, and is
// kept. A temp with no file beside it is what the need store reads as unknown on its next load.
//
// @ai:invariant replaceFile renames only a synced temp, and retries a rename refused because the target is held [P:assumed conf:0.8] (fileOps.replace pinned by TestReplaceFileSyncsBeforeTheRename; replaceFile's own retry by TestTheStateFileIsReplacedOnceItsReaderLetsGo; that replaceFile hands the OS's sync, by reading)
func replaceFile(path string, b []byte) error {
	return fileOps{sync: (*os.File).Sync, rename: os.Rename}.replace(path, b)
}

// replace is replaceFile with these file operations.
func (o fileOps) replace(path string, b []byte) error {
	tmp := tempOf(path)
	if err := o.writeSynced(tmp, b); err != nil {
		os.Remove(tmp)
		return &replaceError{stageTemp, err}
	}
	if err := o.renameOver(tmp, path); err != nil {
		// A rename may remove path, then fail (os.Rename promises no atomicity off Unix): the
		// temp is then the only copy, and the evidence a reader such as the need store goes by.
		if _, serr := os.Stat(path); serr == nil {
			os.Remove(tmp)
		}
		return &replaceError{stageRename, err}
	}
	return nil
}

func (o fileOps) writeSynced(path string, b []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err := o.sync(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// renameOver replaces path with tmp. On Windows the replace is refused (access denied, or
// a sharing violation) for as long as another process holds the file it replaces; a few
// short retries, about a second in all, ride that out. Any other error, and a file that stays
// held, come back at once: the caller holds its store's lock meanwhile.
func (o fileOps) renameOver(tmp, path string) error {
	const attempts = 20
	for i := 1; ; i++ {
		err := o.rename(tmp, path)
		if err == nil || i == attempts || !heldByAnother(err) {
			return err
		}
		time.Sleep(time.Duration(5*i) * time.Millisecond)
	}
}

// heldByAnother says whether a rename failed because the target is open elsewhere:
// ERROR_ACCESS_DENIED (os.IsPermission) or ERROR_SHARING_VIOLATION (32), on Windows only. A
// Windows ACL that denies the rename reads the same as a held file and takes the whole retry
// budget before it comes back.
func heldByAnother(err error) bool { return heldOn(runtime.GOOS, err) }

func heldOn(goos string, err error) bool {
	if goos != "windows" {
		return false
	}
	var errno syscall.Errno
	return errors.As(err, &errno) && errno == 32 || os.IsPermission(err)
}
