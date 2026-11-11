package deliver

import (
	"fmt"
	"io"
	"time"
)

// LogMandate is one mandate as a command log last recorded it.
type LogMandate struct {
	Target      string
	MandateID   string
	Incarnation string
	State       State
	Cause       string    // of an uncertain state
	Reason      string    // of a refused or uncertain state
	Since       time.Time // its last transition
	Created     time.Time // its first record
	Answered    bool      // its record carries the target's answer (delivered, or settled late)
}

// testHookReadLogOpen runs in ReadLog while the log is open, before it is read.
var testHookReadLogOpen func()

// ReadLog reads the command log at path, at most maxBytes of it, without its lock and without
// changing it: what another process may be writing at the same moment. It returns each mandate in
// its last recorded state, in the order of their first records. A last record without its '\n'
// (one being written, or torn by a crash) is left out and reported. On Windows a running ixmux's
// compaction fails while the file is open here (openShared): not fatal, retried a minute later.
// @ai:invariant ReadLog never writes, truncates or locks the log, and never breaks a running engine [T:test conf:0.85 src:go/ixmux/deliver/readlog_test.go:TestReadLogNeverBreaksARunningEngine]
func ReadLog(path string, maxBytes int64) ([]LogMandate, *Torn, error) {
	if maxBytes <= 0 {
		maxBytes = MaxLogBytes
	}
	f, err := openShared(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	if testHookReadLogOpen != nil {
		testHookReadLogOpen()
	}
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if st.Size() > maxBytes {
		return nil, nil, &Refusal{LogTooLarge, fmt.Sprintf("%s holds %d bytes, bound %d", path, st.Size(), maxBytes)}
	}
	data := make([]byte, st.Size())
	n, err := f.ReadAt(data, 0)
	if err != nil && err != io.EOF {
		return nil, nil, err
	}
	recs, torn, _, err := parseLog(data[:n], path)
	if err != nil {
		return nil, nil, err
	}
	at := map[key]int{}
	var out []LogMandate
	for _, r := range recs {
		k := key{r.Target, r.MandateID}
		i, ok := at[k]
		if !ok {
			i = len(out)
			at[k] = i
			created := r.At
			if r.Created != nil {
				created = *r.Created
			}
			out = append(out, LogMandate{Target: r.Target, MandateID: r.MandateID, Created: created})
		}
		m := &out[i]
		m.State, m.Since = r.State, r.At
		if r.Incarnation != "" {
			m.Incarnation = r.Incarnation
		}
		m.Cause, m.Reason = r.Cause, r.Reason
		if r.Ack != nil {
			m.Answered = true
		}
	}
	return out, torn, nil
}
