package journal

import (
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// P1-2: an empty journal replaced by another, incomplete one is another incarnation.
func TestReplacedEmptyJournalIsStale(t *testing.T) {
	path, incA := writeJournal(t, "")
	if len(incA) != 12 {
		t.Errorf("empty journal: incarnation %q, want 12 hex naming the file", incA)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("abc"), 0o600); err != nil {
		t.Fatal(err)
	}
	page, err := Read(path, incA, 0, 0)
	var r *Refusal
	if !errors.As(err, &r) || r.Code != StaleIncarnation {
		t.Errorf("A (empty) replaced by B (incomplete): %+v %v, want %s", page, err, StaleIncarnation)
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
