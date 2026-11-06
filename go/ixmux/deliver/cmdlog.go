package deliver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
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

// cmdLog is the append-only command log.
type cmdLog struct {
	mu sync.Mutex
	f  *os.File
}

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
	var recs []logRecord
	var torn *Torn
	for at := int64(0); at < int64(len(data)); {
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
	af, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return fail(err)
	}
	return &cmdLog{f: af}, recs, torn, nil
}

// append writes r as one line and syncs it.
func (l *cmdLog) append(r logRecord) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, err := l.f.Write(b); err != nil {
		return err
	}
	return l.f.Sync()
}

func (l *cmdLog) close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.f.Close()
}
