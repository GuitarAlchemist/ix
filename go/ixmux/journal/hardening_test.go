package journal

import (
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

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
