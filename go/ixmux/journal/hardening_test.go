package journal

import (
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// S5-3 (CI Linux, run 42326800534; was P1-2): a file system may give a new file the identity of
// the one it replaces (Linux reuses a freed inode), so an empty journal names no file. Every
// empty journal has EmptyIncarnation, and a checkpoint taken on one reads whatever file is at the
// path from 0: nothing is lost or repeated, since no record had completed. The same on every OS.
func TestEmptyJournalCheckpointPromisesNothing(t *testing.T) {
	path, incA := writeJournal(t, "")
	_, incOther := writeJournal(t, "")
	if len(incA) != 12 || incA != EmptyIncarnation || incOther != EmptyIncarnation {
		t.Errorf("two empty journals: %q and %q, want both %q (12 hex)", incA, incOther, EmptyIncarnation)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("abc"), 0o600); err != nil {
		t.Fatal(err)
	}
	page, err := Read(path, incA, 0, 0)
	if err != nil || !page.Provisional || page.Incarnation == EmptyIncarnation || page.PromotedFrom != EmptyIncarnation ||
		page.Next != 0 || page.Pending == nil || !pendingIs(page.Pending, 0, 3, 3) {
		t.Errorf("A (empty) replaced by B (incomplete): %+v %v, want B's provisional page from 0, promoted from %s", page, err, EmptyIncarnation)
	}
	if err := os.WriteFile(path, []byte("abc\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	named, _, _ := Identify(path)
	page, err = Read(path, incA, 0, 0)
	if err != nil || page.Provisional || page.Incarnation != named || page.PromotedFrom != EmptyIncarnation ||
		len(page.Events) != 1 || string(page.Events[0].Line) != "abc" {
		t.Errorf("A (empty), then B complete: %+v %v, want B's first record under %s, promoted from %s", page, err, named, EmptyIncarnation)
	}
	// A journal emptied in place is empty too: a checkpoint of its former incarnation is stale.
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = Read(path, named, 0, 0)
	var r *Refusal
	if !errors.As(err, &r) || r.Code != StaleIncarnation {
		t.Errorf("a named checkpoint on the emptied journal: %v, want %s", err, StaleIncarnation)
	}
}

// P1-2: the first record completing does not make a provisional checkpoint stale, and says so.
func TestFirstRecordCompletionPromotesTheCheckpoint(t *testing.T) {
	path, incP := writeJournal(t, "ab")
	page := mustRead(t, path, incP, 0, 0)
	if !page.Provisional || page.Pending == nil || page.Incarnation != incP {
		t.Fatalf("incomplete first record: %+v", page)
	}
	fh, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	fh.WriteString("c\n")
	fh.Close()
	incF, _, err := Identify(path)
	if err != nil {
		t.Fatal(err)
	}
	page, err = Read(path, incP, 0, 0)
	if err != nil {
		t.Fatalf("provisional checkpoint after completion: %v", err)
	}
	if page.Incarnation != incF || incF == incP || page.PromotedFrom != incP || page.Provisional || len(page.Events) != 1 {
		t.Errorf("promotion: %+v, want incarnation %s promoted from %s and 1 event", page, incF, incP)
	}
	if got := page.EventID(key, page.Events[0]); got != key+"@"+incF+":0" {
		t.Errorf("event id %s", got)
	}
}

// P1-4: a 64 MiB first record is scanned in calls of at most ScanMax bytes each, resumed from the
// hint of the previous call, and ends as one line-too-long event.
func TestHugeFirstRecordScanIsBoundedPerCall(t *testing.T) {
	const huge = 64 << 20
	path, inc := writeJournal(t, strings.Repeat("x", huge-1)+"\n{}\n")
	var n int64
	testHookRead = func(b int64) { n += b }
	defer func() { testHookRead = nil }()
	c := Cursor{Incarnation: inc}
	for calls := 1; ; calls++ {
		n = 0
		page, err := ReadCursor(path, c, 0)
		if err != nil {
			t.Fatal(err)
		}
		if n > ScanMax {
			t.Fatalf("call %d read %d bytes, ScanMax %d", calls, n, ScanMax)
		}
		if len(page.Events) > 0 {
			ev := page.Events[0]
			if len(page.Events) != 1 || ev.Invalid != LineTooLong || ev.Offset != 0 || ev.End != huge || page.Next != huge {
				t.Fatalf("call %d: %d events, first %+v, next %d", calls, len(page.Events), ev, page.Next)
			}
			t.Logf("64 MiB first record: one line-too-long after %d calls", calls)
			return
		}
		if page.Pending == nil || page.Pending.Scanned <= c.Scanned || page.Next != 0 {
			t.Fatalf("call %d: no event and no progressing resume hint: %+v", calls, page.Pending)
		}
		c = page.Pending.Resume
		if calls > huge/(ScanMax/4) {
			t.Fatalf("still scanning after %d calls", calls)
		}
	}
}

// P1-3: a page of tiny records is bounded by MaxEvents, in events and in measured memory.
func TestPageEventsAreBounded(t *testing.T) {
	path, inc := writeJournal(t, strings.Repeat("\n", 1<<20))
	// The bound: the page buffer, one events slice of MaxEvents+1, and 64 KiB of slack.
	bound := uint64(ReadMax + (MaxEvents+1)*int(unsafe.Sizeof(Event{})) + 64<<10)
	next, reads := int64(0), 0
	for next < 1<<20 {
		var before, after runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&before)
		page := mustRead(t, path, inc, next, 0)
		runtime.ReadMemStats(&after)
		if alloc := after.TotalAlloc - before.TotalAlloc; alloc > bound {
			t.Fatalf("read %d: %d events, %d bytes allocated, bound %d", reads, len(page.Events), alloc, bound)
		}
		if len(page.Events) > MaxEvents || len(page.Events) == 0 {
			t.Fatalf("read %d: %d events, want 1..%d", reads, len(page.Events), MaxEvents)
		}
		for i, ev := range page.Events {
			if ev.Offset != next+int64(i) || ev.End != ev.Offset+1 || !ev.Blank {
				t.Fatalf("read %d, event %d: %+v (gap or duplicate)", reads, i, ev)
			}
		}
		next = page.Next
		reads++
	}
	if want := (1 << 20) / MaxEvents; reads != want {
		t.Errorf("%d reads, want %d", reads, want)
	}
}

// typed reports whether a page that does not advance says why.
func typed(p Page, from int64) bool {
	return p.Next > from || p.Pending != nil || p.AtEnd
}

// P1-1: the journal is cut between the size snapshot and the data reads.
func TestConcurrentTruncationIsTyped(t *testing.T) {
	for _, cut := range []int64{2, 1, 0} {
		path, inc := writeJournal(t, "a\nb\n")
		testHookAfterIdentify = func(p string) {
			if err := os.Truncate(p, cut); err != nil {
				t.Fatal(err)
			}
		}
		page, err := Read(path, inc, 2, 0)
		testHookAfterIdentify = nil
		var r *Refusal
		switch {
		case errors.As(err, &r):
			if r.Code != Truncated {
				t.Errorf("cut to %d: refusal %v, want %s", cut, err, Truncated)
			}
		case err != nil:
			t.Errorf("cut to %d: %v", cut, err)
		case !typed(page, 2):
			t.Errorf("cut to %d: next %d == from with no typed indication: %+v", cut, page.Next, page)
		}
	}
}
