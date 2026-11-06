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
// @ai:invariant an event's content and id never depend on the read budget [T:test conf:0.9 src:go/ixmux/journal/journal_test.go:TestEventDoesNotDependOnBudget]
const (
	MaxLine = 1 << 20 // a longer record is one line-too-long event
	// ReadMax bounds the page buffer of one Read. A Read holds at most two buffers: the page
	// buffer (<= ReadMax) and, for a first record longer than the budget, that record's buffer
	// (<= MaxLine): ReadMax+MaxLine bytes in all, even facing a record of several MiB.
	// @ai:invariant one Read allocates at most two buffers, ReadMax + MaxLine bytes, even facing a record of several MiB [T:test conf:0.85 src:go/ixmux/journal/s2e_test.go:TestReadMemoryIsTwoBuffersAtMost]
	ReadMax  = 1 << 20 // the page buffer of one Read
	MaxBytes = ReadMax // the default budget of one Read

	// MaxEvents is the most events one page holds: ReadMax bounds the buffer, not the events
	// over it (a page of one-byte records would otherwise hold a million of them).
	// @ai:invariant a page holds at most MaxEvents events in one allocation [T:test conf:0.9 src:go/ixmux/journal/hardening_test.go:TestPageEventsAreBounded]
	MaxEvents = 4096

	// ScanMax is the most bytes one Read reads from the file, identity included.
	// @ai:invariant one Read reads at most ScanMax bytes of the file [T:test conf:0.9 src:go/ixmux/journal/hardening_test.go:TestHugeFirstRecordScanIsBoundedPerCall]
	ScanMax = 4 << 20

	scanChunk = 64 << 10 // the smallest buffer a Read scans with
)

// LineTooLong marks a record longer than MaxLine, whatever the request's budget.
const LineTooLong = "line-too-long"

// Refusal codes.
const (
	StaleIncarnation = "stale-incarnation" // the checkpoint is of another journal
	Truncated        = "truncated"         // the checkpoint lies past the end of the journal
	BadCheckpoint    = "bad-checkpoint"    // the checkpoint is not a record boundary
	BadCursor        = "bad-cursor"        // the resume hint belongs to another From or incarnation
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

// Pending says no '\n' is known after the record at At: Bytes of the journal follow it, and
// [At, Scanned) holds no '\n'. Next does not pass it. Scanned == At+Bytes: the record is
// incomplete, wait for more. Scanned < At+Bytes: the scan stopped at ScanMax, call again at once.
// Either way, pass Resume back to ReadCursor, unchanged, to resume the scan at Scanned instead of
// at At. Resume names the checkpoint its hint was found for; any other From or incarnation is
// refused bad-cursor.
type Pending struct {
	At      int64  `json:"at"`
	Bytes   int64  `json:"bytes"`
	Scanned int64  `json:"scanned"`
	Resume  Cursor `json:"resume"`
}

// Cursor is a checkpoint (Incarnation, From) with an optional resume hint: Scanned, from the
// Pending of a previous page read at ScanFrom under ScanIncarnation, says [From, Scanned) holds
// no '\n'. A hint whose ScanFrom or ScanIncarnation differs from the checkpoint is refused.
// @ai:assumption a caller passes Pending.Resume back unchanged: the hint fields are compared, not authenticated (S3 signs them) [U:uncertain conf:0.5]
type Cursor struct {
	Incarnation string `json:"incarnation"`
	From        int64  `json:"from"`
	Scanned     int64  `json:"scanned,omitempty"`
	// ScanFrom and ScanIncarnation say which checkpoint Scanned was found for.
	ScanFrom        int64  `json:"scan_from,omitempty"`
	ScanIncarnation string `json:"scan_incarnation,omitempty"`
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
//
// A first record longer than MaxLine is named once its '\n' is found, as Read names it. Identify
// looks for that '\n' within ScanMax bytes; a longer record needs IdentifyFrom.
func Identify(path string) (incarnation string, size int64, err error) {
	id, err := IdentifyFrom(path, 0)
	return id.Incarnation, id.Size, err
}

// Identity is what IdentifyFrom found.
type Identity struct {
	Incarnation string `json:"incarnation"`
	Size        int64  `json:"size"`
	// Provisional: the first record has no '\n' yet, or none was found within this call's scan.
	Provisional bool `json:"provisional,omitempty"`
	// Scanned, for a provisional journal whose first record is longer than MaxLine: [0, Scanned)
	// holds no '\n'. Pass it back to IdentifyFrom to resume the scan there.
	Scanned int64 `json:"scanned,omitempty"`
}

// IdentifyFrom is Identify resuming the scan for the '\n' of a first record longer than MaxLine at
// scanned (0, or an Identity.Scanned of the same file). One call reads at most ScanMax bytes.
// @ai:invariant Identify and Read give the same incarnation for the same complete file [T:test conf:0.85 src:go/ixmux/journal/s2g_test.go:TestIdentifyPromotesAnOversizedFirstRecord]
func IdentifyFrom(path string, scanned int64) (Identity, error) {
	f, err := os.Open(path)
	if err != nil {
		return Identity{}, err
	}
	defer f.Close()
	s := &src{f: f}
	buf := make([]byte, scanChunk)
	named, provisional, complete, size, err := identify(s, buf)
	if err != nil {
		return Identity{}, err
	}
	if complete {
		return Identity{Incarnation: named, Size: size}, nil
	}
	if size < MaxLine { // a short first record without its '\n' yet
		return Identity{Incarnation: provisional, Size: size, Provisional: true}, nil
	}
	// [0, MaxLine) holds no '\n': the first record is complete once any '\n' follows.
	pos := scanned
	if pos < MaxLine || pos > size {
		pos = MaxLine
	}
	for pos < size && s.n < ScanMax {
		chunk := buf[:min(int64(len(buf)), size-pos, ScanMax-s.n)]
		if err := s.readAt(chunk, pos); err != nil {
			return Identity{}, err
		}
		if bytes.IndexByte(chunk, '\n') >= 0 {
			return Identity{Incarnation: named, Size: size}, nil
		}
		pos += int64(len(chunk))
	}
	return Identity{Incarnation: provisional, Size: size, Provisional: true, Scanned: pos}, nil
}

func hash12(parts ...[]byte) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write(p)
	}
	return fmt.Sprintf("%x", h.Sum(nil)[:6])
}

// identify returns the named incarnation (over the first record, or its first MaxLine bytes),
// the provisional incarnation of the same file, whether the first record's '\n' was seen, and the
// size. It reads at most MaxLine bytes, in chunks of buf. A first record longer than MaxLine is
// named by its first MaxLine bytes but stays provisional until its '\n' is found (by Read).
func identify(s *src, buf []byte) (named, provisional string, complete bool, size int64, err error) {
	id, err := fileIdentity(s.f)
	if err != nil {
		return "", "", false, 0, err
	}
	st, err := s.f.Stat()
	if err != nil {
		return "", "", false, 0, err
	}
	size = st.Size()
	provisional = hash12([]byte("provisional\n" + id))
	h := sha256.New()
	h.Write([]byte(id + "\n"))
	limit := min(size, MaxLine)
	for pos := int64(0); pos < limit; {
		chunk := buf[:min(int64(len(buf)), limit-pos)]
		if err := s.readAt(chunk, pos); err != nil {
			return "", "", false, 0, err
		}
		if i := bytes.IndexByte(chunk, '\n'); i >= 0 {
			h.Write(chunk[:i])
			return fmt.Sprintf("%x", h.Sum(nil)[:6]), provisional, true, size, nil
		}
		h.Write(chunk)
		pos += int64(len(chunk))
	}
	if size >= MaxLine { // a first record longer than MaxLine: its first MaxLine bytes name it
		return fmt.Sprintf("%x", h.Sum(nil)[:6]), provisional, false, size, nil
	}
	return provisional, provisional, false, size, nil
}

// Read returns the complete records of the journal at path from the checkpoint from, within
// maxBytes (MaxBytes when <= 0, at most ReadMax) and at most MaxEvents records. It holds at most
// two buffers: the page (<= ReadMax) and an over-budget first record (<= MaxLine). The first record
// of a page is returned even when it exceeds the budget, up to MaxLine, so an event never
// depends on the budget. A record longer than MaxLine becomes one line-too-long event over its
// exact range, found by scanning without keeping it. One call reads at most ScanMax bytes of the
// file: a scan not finished within it, or a record without its '\n' yet, stops the page with
// Pending, whose Scanned resumes the scan (see ReadCursor). It refuses a checkpoint of another
// incarnation, one past the end of the journal, and one that is not a record boundary.
func Read(path, incarnation string, from, maxBytes int64) (Page, error) {
	return ReadCursor(path, Cursor{Incarnation: incarnation, From: from}, maxBytes)
}

// ReadCursor is Read from a cursor whose Scanned hint skips the bytes a previous call has
// already scanned for the end of the record at From.
func ReadCursor(path string, c Cursor, maxBytes int64) (Page, error) {
	from := c.From
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
	s := &src{f: f}
	buf := make([]byte, max(budget, min(scanChunk, ReadMax))) // the page buffer
	named, provisional, complete, size, err := identify(s, buf)
	if err != nil {
		return Page{}, err
	}
	if testHookAfterIdentify != nil {
		testHookAfterIdentify(path)
	}
	if c.Incarnation != named && c.Incarnation != provisional {
		now := provisional
		if complete {
			now = named
		}
		return Page{}, &Refusal{StaleIncarnation, fmt.Sprintf("checkpoint is of incarnation %q, the journal is now %q", c.Incarnation, now)}
	}
	if from < 0 || from > size || c.Scanned > size {
		return Page{}, &Refusal{Truncated, fmt.Sprintf("checkpoint offset %d (scanned %d), the journal holds %d bytes", from, c.Scanned, size)}
	}
	if c.Scanned != 0 && (c.ScanFrom != from || c.ScanIncarnation != c.Incarnation || c.Scanned < from) {
		// A hint found for another checkpoint could start the scan inside a later record.
		return Page{}, &Refusal{BadCursor, fmt.Sprintf("scanned %d was found from %d (incarnation %q), not from %d (incarnation %q)",
			c.Scanned, c.ScanFrom, c.ScanIncarnation, from, c.Incarnation)}
	}
	if from > 0 {
		if err := s.readAt(buf[:1], from-1); err != nil {
			return Page{}, err
		}
		if buf[0] != '\n' {
			return Page{}, &Refusal{BadCheckpoint, fmt.Sprintf("offset %d is not a record boundary", from)}
		}
	}
	page := Page{Next: from, Size: size}
	rest := buf[:min(budget, size-from)]
	if err := s.readAt(rest, from); err != nil {
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
		page.Pending = &Pending{At: page.Next, Bytes: size - page.Next, Scanned: size}
	case len(page.Events) == 0:
		if err := s.overBudget(&page, buf, max(from+int64(len(rest)), c.Scanned)); err != nil {
			return Page{}, err
		}
	}
	page.AtEnd = page.Next == size && page.Pending == nil
	// The first record is complete once its '\n' is known: seen by identify, implied by a
	// checkpoint past it, or found by this page. Only then is the journal named and a provisional
	// checkpoint promoted.
	if complete || from > 0 || len(page.Events) > 0 {
		page.Incarnation = named
		if c.Incarnation == provisional && named != provisional {
			page.PromotedFrom = provisional
		}
	} else {
		page.Incarnation, page.Provisional = provisional, true
	}
	if p := page.Pending; p != nil {
		p.Resume = Cursor{Incarnation: page.Incarnation, From: p.At, Scanned: p.Scanned, ScanFrom: p.At, ScanIncarnation: page.Incarnation}
	}
	return page, nil
}

// overBudget handles a first record at page.Next longer than the budget; [page.Next, known)
// holds no '\n'. Up to MaxLine it is read whole; longer, its end is scanned for within ScanMax.
func (s *src) overBudget(page *Page, buf []byte, known int64) error {
	from, size := page.Next, page.Size
	if known-from < MaxLine {
		buf = make([]byte, min(MaxLine, size-from)) // the second buffer, beside the page's: <= MaxLine
		if err := s.readAt(buf, from); err != nil {
			return err
		}
		if i := bytes.IndexByte(buf, '\n'); i >= 0 {
			end := from + int64(i) + 1
			page.Events = append(page.Events, lineEvent(buf[:i], from, end))
			page.Next = end
			return nil
		}
		known = from + int64(len(buf))
	}
	end, scanned, err := s.recordEnd(buf, known, size)
	if err != nil {
		return err
	}
	if end < 0 {
		page.Pending = &Pending{At: from, Bytes: size - from, Scanned: scanned}
		return nil
	}
	page.Events = append(page.Events, Event{Offset: from, End: end, Invalid: LineTooLong})
	page.Next = end
	return nil
}

// src is the journal file of one call, counting the bytes read from it.
type src struct {
	f *os.File
	n int64
}

// readAt fills b from offset off. Every read of a call lies inside its size snapshot, so fewer
// bytes than asked means the journal shrank meanwhile: a typed truncated refusal.
func (s *src) readAt(b []byte, off int64) error {
	got, err := s.f.ReadAt(b, off)
	s.n += int64(got)
	if testHookRead != nil {
		testHookRead(int64(got))
	}
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

// testHookRead, when set by a test, sees the bytes of every file read.
var testHookRead func(n int64)

// recordEnd scans [pos, size) in chunks of buf for the next '\n', until the call has read
// ScanMax bytes. It returns the offset after the '\n', or -1 and where the scan stopped.
func (s *src) recordEnd(buf []byte, pos, size int64) (end, scanned int64, err error) {
	for pos < size && s.n < ScanMax {
		chunk := buf[:min(int64(len(buf)), size-pos, ScanMax-s.n)]
		if err := s.readAt(chunk, pos); err != nil {
			return 0, 0, err
		}
		if i := bytes.IndexByte(chunk, '\n'); i >= 0 {
			return pos + int64(i) + 1, 0, nil
		}
		pos += int64(len(chunk))
	}
	return -1, pos, nil
}

func lineEvent(line []byte, at, end int64) Event {
	if end-at > MaxLine {
		return Event{Offset: at, End: end, Invalid: LineTooLong}
	}
	line = bytes.TrimSuffix(line, []byte("\r"))
	return Event{Offset: at, End: end, Line: line, Blank: len(bytes.TrimSpace(line)) == 0}
}
