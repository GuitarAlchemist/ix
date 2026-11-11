package deliver

// syncDir does nothing on Windows. Go cannot fsync a directory there (os.File.Sync is
// FlushFileBuffers, which needs a handle with write access that os.Open does not give a
// directory), and NTFS records the creation of a directory entry as a metadata change in its
// own log ($LogFile), made durable by the log's commit rather than by a call on the directory.
// Whether the file's own FlushFileBuffers (cmdLog.append) forces that commit for the new entry
// is not verified here: no test can cut the power.
// @ai:assumption on NTFS the new log's directory entry is durable once its first record is flushed, without a directory sync [U:uncertain conf:0.5]
func syncDir(dir string) error { return nil }
