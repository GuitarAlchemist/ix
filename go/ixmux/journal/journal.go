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

	// MaxEvents is the most events one page holds: ReadMax bounds the buffer, not the events
	// over it (a page of one-byte records would otherwise hold a million of them).
	MaxEvents = 4096

	scanChunk = 64 << 10 // the smallest buffer a Read scans with
)

// LineTooLong marks a record longer than MaxLine, whatever the request's budget.
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

// Page is the answer of one Read. A page whose Next equals the checkpoint always says why:
// Pending (a record without its '\n' yet) or AtEnd (the consumer is caught up).
type Page struct {
	Incarnation string   `json:"incarnation"`
	Events      []Event  `json:"events"`
	Next        int64    `json:"next"`
	Size        int64    `json:"size"`
	Pending     *Pending `json:"pending,omitempty"`
	AtEnd       bool     `json:"at_end,omitempty"`
	// Provisional: the journal has no first record yet; Incarnation names the file alone.
	Provisional bool `json:"provisional,omitempty"`
	// PromotedFrom: the checkpoint's provisional incarnation, accepted because the first record
	// of the same file has completed since; continue with Incarnation.
	PromotedFrom string `json:"promoted_from,omitempty"`
}

// EventID names an event the same way on every read: source/session@incarnation:offset.
func EventID(key, incarnation string, offset int64) string {
	return fmt.Sprintf("%s@%s:%d", key, incarnation, offset)
}

// EventID names e, read in p, for the session key (source/session).
func (p Page) EventID(key string, e Event) string { return EventID(key, p.Incarnation, e.Offset) }

// Identify returns the journal's incarnation and size.
//
// The incarnation is 12 hex of SHA-256 over the file's identity and its first record (the bytes
// before the first '\n', or the first MaxLine bytes of a longer first record): another file
// under the same path, or another first record, is another incarnation. While the journal has
// neither (it is empty, or its first record is shorter than MaxLine and has no '\n' yet), the
// incarnation is provisional: 12 hex of SHA-256 over the file's identity alone. Once the first
// record completes, Read accepts a provisional checkpoint of the same file and says so
// (Page.PromotedFrom); a provisional checkpoint of another file is stale.
func Identify(path string) (incarnation string, size int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	inc, _, size, err := identify(f, make([]byte, scanChunk))
	return inc, size, err
}

func hash12(parts ...[]byte) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write(p)
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:6])
}

// identify returns the incarnation, the provisional incarnation of the same file, and the size.
// It reads at most MaxLine bytes, in chunks of buf.
func identify(f *os.File, buf []byte) (inc, provisional string, size int64, err error) {
	id, err := fileIdentity(f)
	if err != nil {
		return "", "", 0, err
	}
	st, err := f.Stat()
	if err != nil {
		return "", "", 0, err
	}
	size = st.Size()
	provisional = hash12([]byte("provisional\n" + id))
	h := sha256.New()
	h.Write([]byte(id + "\n"))
	limit := min(size, MaxLine)
	for pos := int64(0); pos < limit; {
		chunk := buf[:min(int64(len(buf)), limit-pos)]
		if err := readAt(f, chunk, pos); err != nil {
			return "", "", 0, err
		}
		if i := bytes.IndexByte(chunk, '\n'); i >= 0 {
			h.Write(chunk[:i])
			return fmt.Sprintf("%x", h.Sum(nil)[:6]), provisional, size, nil
		}
		h.Write(chunk)
		pos += int64(len(chunk))
	}
	if size >= MaxLine { // a first record longer than MaxLine: its first MaxLine bytes name it
		return fmt.Sprintf("%x", h.Sum(nil)[:6]), provisional, size, nil
	}
	return provisional, provisional, size, nil
}

// Read returns the complete records of the journal at path from the checkpoint from, within
// maxBytes (MaxBytes when <= 0, at most ReadMax). The first record of a page is returned even
// when it exceeds the budget, up to MaxLine, so an event never depends on the budget. A record
// longer than MaxLine becomes one line-too-long event over its exact range, found by scanning in
// bounded chunks. A record without its '\n' yet stops the page with Pending. It refuses a checkpoint of
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
	inc, provisional, size, err := identify(f, buf)
	if err != nil {
		return Page{}, err
	}
	if testHookAfterIdentify != nil {
		testHookAfterIdentify(path)
	}
	promoted := ""
	switch {
	case incarnation == inc:
	case incarnation == provisional: // the same file, whose first record has completed since
		promoted = provisional
	default:
		return Page{}, &Refusal{StaleIncarnation, fmt.Sprintf("checkpoint is of incarnation %q, the journal is now %q", incarnation, inc)}
	}
	switch {
	case from < 0 || from > size:
		return Page{}, &Refusal{Truncated, fmt.Sprintf("checkpoint offset %d, the journal holds %d bytes", from, size)}
	}
	if from > 0 {
		if err := readAt(f, buf[:1], from-1); err != nil {
			return Page{}, err
		}
		if buf[0] != '\n' {
			return Page{}, &Refusal{BadCheckpoint, fmt.Sprintf("offset %d is not a record boundary", from)}
		}
	}
	page := Page{Incarnation: inc, Next: from, Size: size, Provisional: inc == provisional, PromotedFrom: promoted}
	rest := buf[:min(budget, size-from)]
	if err := readAt(f, rest, from); err != nil {
		return Page{}, err
	}
	page.Events = make([]Event, 0, min(bytes.Count(rest, []byte{'\n'}), MaxEvents)+1) // one allocation
	for len(page.Events) < MaxEvents {
		i := bytes.IndexByte(rest, '\n')
		if i < 0 {
			break
		}
		end := page.Next + int64(i) + 1
		page.Events = append(page.Events, lineEvent(rest[:i], page.Next, end))
		rest, page.Next = rest[i+1:], end
	}
	switch {
	case len(rest) == 0, len(page.Events) == MaxEvents: // read all of it, or the page is full
	case page.Next+int64(len(rest)) == size:
		page.Pending = &Pending{At: page.Next, Bytes: size - page.Next}
	case len(page.Events) == 0:
		// The first record is longer than the budget: find its end without keeping it. Up to
		// MaxLine it is returned whole anyway, so an event never depends on the budget.
		end, err := recordEnd(f, buf, from+int64(len(rest)), size)
		if err != nil {
			return Page{}, err
		}
		if end < 0 {
			page.Pending = &Pending{At: from, Bytes: size - from}
			break
		}
		if end-from <= MaxLine {
			if end-from > int64(len(buf)) {
				buf = make([]byte, end-from) // replaces the page buffer: still <= ReadMax
			}
			rec := buf[:end-from]
			if err := readAt(f, rec, from); err != nil {
				return Page{}, err
			}
			page.Events = append(page.Events, lineEvent(rec[:len(rec)-1], from, end))
		} else {
			page.Events = append(page.Events, Event{Offset: from, End: end, Invalid: LineTooLong})
		}
		page.Next = end
	}
	page.AtEnd = page.Next == size && page.Pending == nil
	return page, nil
}

// readAt fills b from offset off. Every read of Read lies inside the size snapshot, so fewer
// bytes than asked means the journal shrank meanwhile: a typed truncated refusal.
func readAt(f *os.File, b []byte, off int64) error {
	got, err := f.ReadAt(b, off)
	switch {
	case got == len(b):
		return nil
	case err == nil || errors.Is(err, io.EOF):
		return &Refusal{Truncated, fmt.Sprintf("the journal shrank during the read: %d of %d bytes at offset %d", got, len(b), off)}
	}
	return err
}

// testHookAfterIdentify, when set by a test, runs between the size snapshot and the data reads.
var testHookAfterIdentify func(path string)

// recordEnd scans [pos, size) in chunks of buf for the next '\n' and returns the offset after it,
// or -1 when the record is not complete before size.
func recordEnd(f *os.File, buf []byte, pos, size int64) (int64, error) {
	for pos < size {
		chunk := buf[:min(int64(len(buf)), size-pos)]
		if err := readAt(f, chunk, pos); err != nil {
			return 0, err
		}
		if i := bytes.IndexByte(chunk, '\n'); i >= 0 {
			return pos + int64(i) + 1, nil
		}
		pos += int64(len(chunk))
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
