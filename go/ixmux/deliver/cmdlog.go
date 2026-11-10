package deliver

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// logRecord is one transition in the command log: one JSON line, synced before it counts.
type logRecord struct {
	At          time.Time `json:"at"`
	Target      string    `json:"target"`
	MandateID   string    `json:"mandate_id"`
	Incarnation string    `json:"incarnation,omitempty"`
	State       State     `json:"state"`
	Reason      string    `json:"reason,omitempty"`
	Cause       string    `json:"cause,omitempty"` // of an uncertain state: one of the Cause* values
	Ack         *Ack      `json:"ack,omitempty"`
	Fingerprint string    `json:"fp,omitempty"` // of the mandate, on its accepted records
	// Written by a compaction only, which keeps one record per mandate: the state before its last
	// transition, and when its first record was written.
	Prev    State      `json:"prev,omitempty"`
	Created *time.Time `json:"created,omitempty"`
}

// cmdLog is the append-only command log, written by one engine at a time: lock is held on
// <log>.lock from Open to close.
type cmdLog struct {
	mu     sync.Mutex
	f      logFile
	lock   io.Closer
	broken error // set by a failed write or sync: no more appends
	// size is the log's length in bytes; max the bound Open reads back (MaxLogBytes). No append
	// takes size past max, so the next Open can read whatever this one wrote.
	size, max int64
}

// logFile is the append handle of the log (an *os.File; a test may wrap it).
type logFile interface {
	Write(b []byte) (int, error)
	Sync() error
	Close() error
}

var (
	errLocked          = errors.New("locked")
	errLockUnsupported = errors.New("no file lock on this system")
	// errLogFull: a record after accepted does not fit; the outcome of its delivery is unknown.
	errLogFull = errors.New("the command log is full")
	// errNoGain: a compaction would not make the log smaller; it is not done.
	errNoGain = errors.New("compaction would not shrink the log")
)

// logFormat names the command log format; logHeader is its first line. A log without it, or
// with another version, is refused (log-format-unsupported): no log of an earlier format was
// ever deployed, and its records would lack the fingerprints the engine relies on.
const (
	logFormat = "ixmux.cmdlog.v1"
	logHeader = `{"format":"` + logFormat + `"}` + "\n"
)

// dirSync syncs the log's directory (syncDir); a test replaces it, on every system.

// renameLog replaces the log by its compacted copy (renameDurable); a test replaces it.
var renameLog = renameDurable
var dirSync = syncDir

// openLog reads the log at path back, at most maxBytes of it. A last record without its '\n'
// (a crash while writing it) is cut off and reported as torn; any other unreadable record is
// log-corrupt.
func openLog(path string, maxBytes int64) (*cmdLog, []logRecord, *Torn, error) {
	// Read and cut through a handle without O_APPEND: on Windows an append-only handle may not
	// truncate. Appends then go through a second, append-only handle.
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()
	fail := func(err error) (*cmdLog, []logRecord, *Torn, error) { return nil, nil, nil, err }
	st, err := f.Stat()
	if err != nil {
		return fail(err)
	}
	if st.Size() > maxBytes {
		return fail(&Refusal{LogTooLarge, fmt.Sprintf("%s holds %d bytes, bound %d", path, st.Size(), maxBytes)})
	}
	data := make([]byte, st.Size())
	if _, err := f.ReadAt(data, 0); err != nil && err != io.EOF {
		return fail(err)
	}
	// The first line names the format. With no complete first line (a new log, or a crash while
	// its header was written) the header is (re)written; any other first line is refused.
	start, needHeader := int64(0), true
	if nl := bytes.IndexByte(data, '\n'); nl >= 0 {
		var h struct {
			Format string `json:"format"`
		}
		if json.Unmarshal(data[:nl], &h) != nil || h.Format != logFormat {
			return fail(&Refusal{LogFormatUnsupported, fmt.Sprintf("%s does not start with the %s header", path, logFormat)})
		}
		start, needHeader = int64(nl)+1, false
	}
	var recs []logRecord
	var torn *Torn
	for at := start; at < int64(len(data)); {
		i := bytes.IndexByte(data[at:], '\n')
		if i < 0 {
			torn = &Torn{At: at, Bytes: int64(len(data)) - at}
			break
		}
		var r logRecord
		// A v1 accepted or delivering record carries the mandate's fingerprint (Deliver writes it):
		// without it, the same mandate would later be refused mandate-id-reuse.
		if err := json.Unmarshal(data[at:at+int64(i)], &r); err != nil || r.Target == "" || r.MandateID == "" || r.State == "" ||
			((r.State == Accepted || r.State == Delivering) && r.Fingerprint == "") {
			return fail(&Refusal{LogCorrupt, fmt.Sprintf("record at offset %d of %s is unreadable", at, path)})
		}
		recs = append(recs, r)
		at += int64(i) + 1
	}
	if torn != nil {
		if err := f.Truncate(torn.At); err != nil {
			return fail(err)
		}
		if err := f.Sync(); err != nil {
			return fail(err)
		}
	}
	if needHeader {
		if _, err := f.WriteAt([]byte(logHeader), 0); err != nil {
			return fail(err)
		}
		if err := f.Sync(); err != nil {
			return fail(err)
		}
	}
	// The directory is synced at every Open, before any transition: a previous Open that created
	// the log may have failed its own sync and left the file behind.
	// @ai:invariant no transition is appended before an Open has synced the log's directory [T:test conf:0.85 src:go/ixmux/deliver/s2f_test.go:TestFailedDirectorySyncIsRetried]
	if err := dirSync(filepath.Dir(path)); err != nil {
		return fail(fmt.Errorf("sync the directory of %s: %w", path, err))
	}
	af, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return fail(err)
	}
	size := int64(len(data))
	switch {
	case needHeader: // the file was empty, or cut at 0 as a torn header
		size = int64(len(logHeader))
	case torn != nil:
		size = torn.At
	}
	return &cmdLog{f: af, size: size, max: maxBytes}, recs, torn, nil
}

// append writes r as one line and syncs it. A failed write or sync breaks the log: a fragment
// may be on disk, and any later line would be glued to it into one corrupt record. Every later
// append is then refused (log-broken) until the engine is reopened, which cuts the fragment as a
// torn tail.
//
// A line that would leave less than room bytes before max is not written: the log stays whole and
// readable at the next Open. For an accepted record (room > 0) that is a refusal, log-full:
// nothing was sent. For any later record the adapter may already have acted: errLogFull, which is
// no refusal.
// @ai:invariant a transition counts only once its whole line is written, and nothing is appended after a partial write [T:test conf:0.85 src:go/ixmux/deliver/s2d_test.go:TestPartialWriteBreaksTheLog]
// @ai:invariant a transition counts only once its line is synced, and nothing is appended after a failed sync [T:test conf:0.85 src:go/ixmux/deliver/s2f_test.go:TestFailedSyncBreaksTheLog]
// @ai:invariant no append takes the log past the bound Open reads back, so an engine can always reopen the log it wrote [T:test conf:0.85 src:go/ixmux/deliver/logbound_test.go:TestALogFilledByItsOwnAppendsStillOpens]
func (l *cmdLog) append(r logRecord, room int64) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.broken != nil {
		return &Refusal{LogBroken, "the command log stopped after a failed write (" + l.broken.Error() + "); reopen the engine"}
	}
	if l.size+int64(len(b))+room > l.max {
		if room > 0 {
			return &Refusal{LogFull, fmt.Sprintf("the command log holds %d bytes of %d, no room for a new mandate: nothing was sent (a restart compacts the log's history, never its mandates)", l.size, l.max)}
		}
		return fmt.Errorf("%w: it holds %d bytes of %d, a %d-byte record does not fit", errLogFull, l.size, l.max, len(b))
	}
	if n, err := l.f.Write(b); err != nil || n != len(b) {
		l.broken = fmt.Errorf("wrote %d of %d bytes: %v", n, len(b), err)
		return l.broken
	}
	l.size += int64(len(b))
	if err := l.f.Sync(); err != nil {
		l.broken = fmt.Errorf("sync: %w", err)
		return l.broken
	}
	return nil
}

// rewrite replaces the log at path with its header and recs. They are written whole and synced
// to path+".compact" first, which then replaces the log by a rename: a crash leaves the old log or
// the new one, never part of either. On a failure before the rename the old log is kept and
// appends go on to it (fatal false). After the rename, a failure to sync the directory or to
// reopen the log breaks it (fatal true): an append to the new log could otherwise be lost to a
// crash that brings the old one back.
func (l *cmdLog) rewrite(path string, recs []logRecord) (fatal bool, err error) {
	var buf bytes.Buffer
	buf.WriteString(logHeader)
	for _, r := range recs {
		b, err := json.Marshal(r)
		if err != nil {
			return false, err
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if int64(buf.Len()) >= l.size {
		return false, errNoGain
	}
	tmp := path + ".compact"
	if err := writeSynced(tmp, buf.Bytes()); err != nil {
		os.Remove(tmp)
		return false, err
	}
	// Windows renames over no open file: the append handle is closed first, and reopened on the
	// old log if the rename fails.
	if err := l.f.Close(); err != nil {
		l.broken = fmt.Errorf("close before the compaction: %w", err)
		return true, l.broken
	}
	if err := renameLog(tmp, path); err != nil {
		os.Remove(tmp)
		af, oerr := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
		if oerr != nil {
			l.broken = fmt.Errorf("reopen the log after a failed compaction: %w", oerr)
			return true, l.broken
		}
		l.f = af
		return false, err
	}
	if err := dirSync(filepath.Dir(path)); err != nil {
		l.broken = fmt.Errorf("sync the directory after the compaction: %w", err)
		return true, l.broken
	}
	af, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		l.broken = fmt.Errorf("reopen the compacted log: %w", err)
		return true, l.broken
	}
	l.f, l.size = af, int64(buf.Len())
	return false, nil
}

// writeSynced creates (or empties) path, writes b to it and syncs it.
func writeSynced(path string, b []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func (l *cmdLog) close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	err := l.f.Close()
	if l.lock != nil {
		if lerr := l.lock.Close(); lerr != nil && err == nil {
			err = fmt.Errorf("release the lock of the command log: %w", lerr)
		}
		l.lock = nil
	}
	return err
}
