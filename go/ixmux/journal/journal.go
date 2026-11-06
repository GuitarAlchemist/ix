// Package journal reads JSONL session journals by byte offset: the complete records after a
// consumer's checkpoint, each over its exact byte range, in bounded memory.
package journal

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
)

// Bounds of one read. A record's length counts its bytes up to and including its '\n', so a
// record of exactly MaxLine bytes fits in a ReadMax buffer and is accepted.
const (
	MaxLine  = 1 << 20 // a longer record is one line-too-long event
	ReadMax  = 1 << 20 // what one Read keeps in memory at most
	MaxBytes = ReadMax // the default budget of one Read

	scanChunk = 64 << 10 // the smallest buffer a Read scans with
)

// LineTooLong marks a record longer than MaxLine or than the request's budget.
const LineTooLong = "line-too-long"

// Refusal codes.
const (
	StaleIncarnation = "stale-incarnation" // the checkpoint is of another journal
	Truncated        = "truncated"         // the checkpoint lies past the end of the journal
	BadCheckpoint    = "bad-checkpoint"    // the checkpoint is not a record boundary
)

// Refusal is a typed refusal of a checkpoint; nothing was read.
type Refusal struct {
	Code   string
	Detail string
}

func (r *Refusal) Error() string { return r.Code + ": " + r.Detail }

// Event is one complete record of the journal over its exact byte range [Offset, End). Line is
// the record without its "\n" (and "\r"); it aliases the page's buffer. A line-too-long event
// carries no Line.
type Event struct {
	Offset  int64  `json:"offset"`
	End     int64  `json:"end"`
	Line    []byte `json:"-"`
	Blank   bool   `json:"blank,omitempty"`
	Invalid string `json:"invalid,omitempty"`
}

// Pending says the record at At has no '\n' yet: Bytes of it are written. Next does not pass it.
type Pending struct {
	At    int64 `json:"at"`
	Bytes int64 `json:"bytes"`
}

// Page is the answer of one Read. Next == Size with no Pending means the consumer is caught up.
type Page struct {
	Incarnation string   `json:"incarnation"`
	Events      []Event  `json:"events"`
	Next        int64    `json:"next"`
	Size        int64    `json:"size"`
	Pending     *Pending `json:"pending,omitempty"`
}

// EventID names an event the same way on every read: source/session@incarnation:offset.
func EventID(key, incarnation string, offset int64) string {
	return fmt.Sprintf("%s@%s:%d", key, incarnation, offset)
}

// EventID names e, read in p, for the session key (source/session).
func (p Page) EventID(key string, e Event) string { return EventID(key, p.Incarnation, e.Offset) }

// Identify returns the journal's incarnation and size: 12 hex of SHA-256 over the file's identity
// and its first complete record ("" while there is none).
func Identify(path string) (incarnation string, size int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	return identify(f, make([]byte, scanChunk))
}

// identify hashes the first record in chunks of buf, so a first record of any length costs no
// more memory than buf.
func identify(f *os.File, buf []byte) (string, int64, error) {
	id, err := fileIdentity(f)
	if err != nil {
		return "", 0, err
	}
	st, err := f.Stat()
	if err != nil {
		return "", 0, err
	}
	size := st.Size()
	h := sha256.New()
	h.Write([]byte(id + "\n"))
	for pos := int64(0); pos < size; {
		got, err := f.ReadAt(buf[:min(int64(len(buf)), size-pos)], pos)
		if i := bytes.IndexByte(buf[:got], '\n'); i >= 0 {
			h.Write(buf[:i])
			return fmt.Sprintf("%x", h.Sum(nil)[:6]), size, nil
		}
		h.Write(buf[:got])
		pos += int64(got)
		if err != nil && !errors.Is(err, io.EOF) {
			return "", 0, err
		}
		if got == 0 {
			break
		}
	}
	return "", size, nil
}

// Read returns the complete records of the journal at path from the checkpoint from, within
// maxBytes (MaxBytes when <= 0, at most ReadMax). A record longer than MaxLine or than the
// budget becomes one line-too-long event over its exact range, found by scanning in bounded
// chunks. A record without its '\n' yet stops the page with Pending. It refuses a checkpoint of
// another incarnation, one past the end of the journal, and one that is not a record boundary.
func Read(path, incarnation string, from, maxBytes int64) (Page, error) {
	budget := maxBytes
	if budget <= 0 {
		budget = MaxBytes
	}
	budget = min(budget, ReadMax)
	f, err := os.Open(path)
	if err != nil {
		return Page{}, err
	}
	defer f.Close()
	buf := make([]byte, max(budget, min(scanChunk, ReadMax))) // the only buffer of this read
	inc, size, err := identify(f, buf)
	if err != nil {
		return Page{}, err
	}
	switch {
	case inc != incarnation:
		return Page{}, &Refusal{StaleIncarnation, fmt.Sprintf("checkpoint is of incarnation %q, the journal is now %q", incarnation, inc)}
	case from < 0 || from > size:
		return Page{}, &Refusal{Truncated, fmt.Sprintf("checkpoint offset %d, the journal holds %d bytes", from, size)}
	}
	if from > 0 {
		if _, err := f.ReadAt(buf[:1], from-1); err != nil || buf[0] != '\n' {
			return Page{}, &Refusal{BadCheckpoint, fmt.Sprintf("offset %d is not a record boundary", from)}
		}
	}
	page := Page{Incarnation: inc, Events: []Event{}, Next: from, Size: size}
	got, err := f.ReadAt(buf[:min(budget, size-from)], from)
	if err != nil && !errors.Is(err, io.EOF) {
		return Page{}, err
	}
	rest := buf[:got]
	for {
		i := bytes.IndexByte(rest, '\n')
		if i < 0 {
			break
		}
		end := page.Next + int64(i) + 1
		page.Events = append(page.Events, lineEvent(rest[:i], page.Next, end))
		rest, page.Next = rest[i+1:], end
	}
	switch {
	case len(rest) == 0:
	case page.Next+int64(len(rest)) == size:
		page.Pending = &Pending{At: page.Next, Bytes: size - page.Next}
	case len(page.Events) == 0:
		// The record at from is longer than the budget: find its end without keeping it.
		end, err := recordEnd(f, buf, from+int64(len(rest)), size)
		if err != nil {
			return Page{}, err
		}
		if end < 0 {
			page.Pending = &Pending{At: from, Bytes: size - from}
			break
		}
		page.Events = append(page.Events, Event{Offset: from, End: end, Invalid: LineTooLong})
		page.Next = end
	}
	return page, nil
}

// recordEnd scans [pos, size) in chunks of buf for the next '\n' and returns the offset after it,
// or -1 when the record is not complete before size.
func recordEnd(f *os.File, buf []byte, pos, size int64) (int64, error) {
	for pos < size {
		got, err := f.ReadAt(buf[:min(int64(len(buf)), size-pos)], pos)
		if i := bytes.IndexByte(buf[:got], '\n'); i >= 0 {
			return pos + int64(i) + 1, nil
		}
		pos += int64(got)
		if err != nil && !errors.Is(err, io.EOF) {
			return 0, err
		}
		if got == 0 {
			break
		}
	}
	return -1, nil
}

func lineEvent(line []byte, at, end int64) Event {
	if end-at > MaxLine {
		return Event{Offset: at, End: end, Invalid: LineTooLong}
	}
	line = bytes.TrimSuffix(line, []byte("\r"))
	return Event{Offset: at, End: end, Line: line, Blank: len(bytes.TrimSpace(line)) == 0}
}
