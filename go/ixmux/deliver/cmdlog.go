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
	Ack         *Ack      `json:"ack,omitempty"`
	Fingerprint string    `json:"fp,omitempty"` // of the mandate, on its accepted records
}

// cmdLog is the append-only command log, written by one engine at a time: lock is held on
// <log>.lock from Open to close.
type cmdLog struct {
	mu     sync.Mutex
	f      logFile
	lock   io.Closer
	broken error // set by a failed write or sync: no more appends
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
)

// logFormat names the command log format; logHeader is its first line. A log without it, or
// with another version, is refused (log-format-unsupported): no log of an earlier format was
// ever deployed, and its records would lack the fingerprints the engine relies on.
const (
	logFormat = "ixmux.cmdlog.v1"
	logHeader = `{"format":"` + logFormat + `"}` + "\n"
)

// openLog reads the log at path back, at most maxBytes of it. A last record without its '\n'
// (a crash while writing it) is cut off and reported as torn; any other unreadable record is
// log-corrupt.
func openLog(path string, maxBytes int64) (*cmdLog, []logRecord, *Torn, error) {
	// Read and cut through a handle without O_APPEND: on Windows an append-only handle may not
	// truncate. Appends then go through a second, append-only handle.
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	created := err == nil
	if errors.Is(err, os.ErrExist) {
		f, err = os.OpenFile(path, os.O_RDWR, 0)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()
	fail := func(err error) (*cmdLog, []logRecord, *Torn, error) { return nil, nil, nil, err }
	if created {
		if err := syncDir(filepath.Dir(path)); err != nil {
			return fail(fmt.Errorf("sync the directory of the new %s: %w", path, err))
		}
	}
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
		if err := json.Unmarshal(data[at:at+int64(i)], &r); err != nil || r.Target == "" || r.MandateID == "" || r.State == "" {
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
	af, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return fail(err)
	}
	return &cmdLog{f: af}, recs, torn, nil
}

// append writes r as one line and syncs it. A failed write or sync breaks the log: a fragment
// may be on disk, and any later line would be glued to it into one corrupt record. Every later
// append is then refused (log-broken) until the engine is reopened, which cuts the fragment as a
// torn tail.
// @ai:invariant a transition counts only once its whole line is written and synced, and nothing is appended after a failed write [T:test conf:0.85 src:go/ixmux/deliver/s2d_test.go:TestPartialWriteBreaksTheLog]
func (l *cmdLog) append(r logRecord) error {
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
	if n, err := l.f.Write(b); err != nil || n != len(b) {
		l.broken = fmt.Errorf("wrote %d of %d bytes: %v", n, len(b), err)
		return l.broken
	}
	if err := l.f.Sync(); err != nil {
		l.broken = fmt.Errorf("sync: %w", err)
		return l.broken
	}
	return nil
}

func (l *cmdLog) close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	err := l.f.Close()
	if l.lock != nil {
		l.lock.Close()
		l.lock = nil
	}
	return err
}
