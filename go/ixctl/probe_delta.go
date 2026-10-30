package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

// The delta@1 kind tails the JSONL journal of Delta, the voice robot (brickyard's brain.exe):
// one JSON object per line, as its writer appends them, with `t`, `t_ms`, `event`, `pid`,
// `journal_seq` and the event's own fields. The plane samples the file; brain.exe stays the
// only writer, and nothing is pushed into the plane. One line per tick, so a node's declared
// rate (1000 / every_ms) stays the bound whatever the journal's own pace: a burst in the
// journal drains at the node's rate, nothing is dropped, and the budget D10 (ix#416) counts it
// like any other kind. Every top-level field of the line passes through as it is; the record
// adds `offset`, the line's byte position in the file, a resume key that belongs to the file
// and not to the writer (the writer's own `journal_seq` restarts with each brain, so its dedup
// key is the pair `pid`, `journal_seq`). `node`, `kind` and `seq` are the tree's stamps on every
// record: they win over a line field of the same name.

// deltaFields is the known union of the journal's fields, in the order `probe.kinds` lists
// them: the first six on every line or added here, the rest per event (heard: utterance_id,
// text, asr_ms, route, triage, triage_ms, parrot, promise; say: say_id, utterance_id, text, by,
// request, cost; speaking and spoken: say_id, cost, muted; start: version; stop: replaced_by).
// A field the writer adds later passes through too: the list documents, it does not filter.
var deltaFields = []string{
	"event", "journal_seq", "t_ms", "t", "pid", "offset",
	"utterance_id", "text", "asr_ms", "route", "triage", "triage_ms", "parrot", "promise",
	"say_id", "by", "request", "cost", "muted", "version", "replaced_by",
}

const (
	// journalReadChunk bounds one refill: a journal that grew by more is read over several ticks.
	journalReadChunk = 64 << 10
	// journalMaxLine bounds one line, newline excluded. A longer line is one error record at its
	// offset, and the tail skips to the next newline without keeping its bytes: a line that never
	// ends cannot grow the sampler's memory. A journal line of Delta is a few hundred bytes.
	journalMaxLine = 1 << 20
)

// tailEntry is one complete line read and not yet emitted, or the error that stands in for a
// line the tail refused (oversized), in file order.
type tailEntry struct {
	line []byte
	at   int64 // the byte offset of the line's first byte
	err  error
}

// journalTail is the delta@1 sampler: it reads the journal from where it stopped, keeps the
// complete lines it has not emitted yet and the bytes of a line still being written, and
// emits one line per sample.
type journalTail struct {
	path     string
	offset   int64       // the file position the next refill reads from (every byte seen so far)
	partial  []byte      // the bytes after the last newline seen: a line still being written
	lineAt   int64       // the offset of the line `partial` (or the skipped line) began at
	skipping bool        // inside an oversized line, already said: drop bytes until its newline
	pending  []tailEntry // read and not yet emitted, in file order
	missing  bool        // the file was absent at the last refill and that was said already
}

func newJournalTail(path string) *journalTail { return &journalTail{path: path} }

// sample emits the next complete line as a record, refilling from the file when none is
// pending. No new line is `errNoDelta` (nothing emitted). A line that is not a JSON object is
// an error record naming its offset; the lines after it still flow. A missing file is said once
// and then waited for; a file shorter than what was read is a new journal under the same name:
// said once, then read again from offset 0.
func (j *journalTail) sample() (map[string]any, error) {
	if len(j.pending) == 0 {
		if err := j.refill(); err != nil {
			return nil, err
		}
		if len(j.pending) == 0 {
			return nil, errNoDelta
		}
	}
	e := j.pending[0]
	j.pending = j.pending[1:]
	if e.err != nil {
		return nil, e.err
	}
	var rec map[string]any
	if err := json.Unmarshal(e.line, &rec); err != nil || rec == nil {
		return nil, fmt.Errorf("journal line at offset %d is not a JSON object", e.at)
	}
	rec["offset"] = e.at
	return rec, nil
}

// refill reads what the file holds past the offset, up to journalReadChunk bytes, and splits
// it into complete lines; the remainder waits for its newline.
func (j *journalTail) refill() error {
	st, err := os.Stat(j.path)
	if errors.Is(err, os.ErrNotExist) {
		if j.missing {
			return errNoDelta
		}
		j.missing = true
		return fmt.Errorf("journal %s is missing; waiting for it", j.path)
	}
	if err != nil {
		return err
	}
	j.missing = false
	if st.Size() < j.offset {
		j.offset, j.partial, j.lineAt, j.skipping = 0, nil, 0, false
		return fmt.Errorf("journal %s shrank to %d bytes; reading it again from offset 0", j.path, st.Size())
	}
	if st.Size() == j.offset {
		return errNoDelta
	}
	f, err := os.Open(j.path)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := make([]byte, journalReadChunk)
	n, err := f.ReadAt(buf, j.offset)
	if err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	pos := j.offset // the file offset of chunk[0]
	j.offset += int64(n)
	chunk := buf[:n]
	for len(chunk) > 0 {
		i := bytes.IndexByte(chunk, '\n')
		if i < 0 {
			j.hold(chunk)
			return nil
		}
		if !j.skipping {
			j.hold(chunk[:i])
		}
		if !j.skipping {
			if line := bytes.TrimSpace(j.partial); len(line) > 0 {
				j.pending = append(j.pending, tailEntry{line: append([]byte(nil), line...), at: j.lineAt})
			}
		}
		pos += int64(i + 1)
		chunk = chunk[i+1:]
		j.partial, j.lineAt, j.skipping = nil, pos, false
	}
	return nil
}

// hold appends bytes of the line in progress; past journalMaxLine the line becomes one error
// entry at its offset and the rest of it is skipped up to its newline.
func (j *journalTail) hold(b []byte) {
	if j.skipping {
		return
	}
	if len(j.partial)+len(b) > journalMaxLine {
		j.pending = append(j.pending, tailEntry{at: j.lineAt,
			err: fmt.Errorf("journal line at offset %d is longer than max_line %d bytes; skipped", j.lineAt, journalMaxLine)})
		j.partial, j.skipping = nil, true
		return
	}
	j.partial = append(j.partial, b...)
}
