package main

import (
	"errors"
	"os"
	"runtime"
	"syscall"
	"time"
)

// replaceError says at which stage replaceFile failed: "temp" (writing or syncing the new
// content beside the file: the file is as it was) or "rename" (the replace itself: the file is
// as it was, unless the rename went through and only its answer was lost).
type replaceError struct {
	stage string
	err   error
}

func (e *replaceError) Error() string { return e.err.Error() }
func (e *replaceError) Unwrap() error { return e.err }

// fileSync makes a file's content durable (a test seam).
var fileSync = (*os.File).Sync

// replaceFile replaces path's content with b, whole: a reader sees the old content or the new,
// never a part. The new content is written to path.tmp and synced before the rename (the file's
// content, not the directory: what a power loss can still lose is in the README's need
// section); on Windows the rename is refused while another process holds the file it replaces
// (an indexer or an antivirus reading what the last write left), so a few short retries, about
// a second in all, ride that out. On a failure the temp file is removed when it can be; one
// left behind (a remove refused) is what the need store reads as unknown on its next load.
//
// @ai:invariant replaceFile renames only a synced temp, and retries a rename refused because the target is held [P:assumed conf:0.8] (pinned by TestReplaceFileSyncsBeforeTheRename, TestTheStateFileIsReplacedOnceItsReaderLetsGo)
func replaceFile(path string, b []byte) error {
	tmp := path + ".tmp"
	if err := writeSynced(tmp, b); err != nil {
		os.Remove(tmp)
		return &replaceError{"temp", err}
	}
	if err := renameOver(tmp, path); err != nil {
		os.Remove(tmp)
		return &replaceError{"rename", err}
	}
	return nil
}

func writeSynced(path string, b []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err := fileSync(f); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// renameOver replaces path with tmp. On Windows the replace is refused (access denied, or
// a sharing violation) for as long as another process holds the file it replaces; a few
// short retries, about a second in all, ride that out. Any other error, and a file that stays
// held, come back at once: the caller holds its store's lock meanwhile.
func renameOver(tmp, path string) error {
	const attempts = 20
	for i := 1; ; i++ {
		err := os.Rename(tmp, path)
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
