package deliver

import "os"

// openShared opens path for reading. A running ixmux can still append to it; its compaction,
// which renames a copy over the log, fails while the file is open here (MoveFileEx replaces no
// open file, delete sharing or not): that failure is no break, the log is kept as it was, and the
// next compaction is tried a minute later.
func openShared(path string) (*os.File, error) { return os.Open(path) }
