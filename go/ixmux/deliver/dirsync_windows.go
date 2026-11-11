package deliver

import "golang.org/x/sys/windows"

// syncDir does nothing on Windows. Go cannot fsync a directory there (os.File.Sync is
// FlushFileBuffers, which needs a handle with write access that os.Open does not give a
// directory), and NTFS records the creation of a directory entry as a metadata change in its
// own log ($LogFile), made durable by the log's commit rather than by a call on the directory.
// Whether the file's own FlushFileBuffers (cmdLog.append) forces that commit for the new entry
// is not verified here: no test can cut the power.
// @ai:assumption on NTFS the new log's directory entry is durable once its first record is flushed, without a directory sync [U:uncertain conf:0.5]
func syncDir(dir string) error { return nil }

// renameDurable replaces newpath by oldpath with MoveFileEx(REPLACE_EXISTING|WRITE_THROUGH): the
// call returns once the rename is flushed to disk, so appends to the new log, each flushed, cannot
// outlive a rename a power cut would undo (os.Rename does not ask for WRITE_THROUGH).
// @ai:assumption MOVEFILE_WRITE_THROUGH makes the rename durable before the first append to the compacted log [U:uncertain conf:0.6]
func renameDurable(oldpath, newpath string) error {
	from, err := windows.UTF16PtrFromString(oldpath)
	if err != nil {
		return err
	}
	to, err := windows.UTF16PtrFromString(newpath)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(from, to, windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH)
}
