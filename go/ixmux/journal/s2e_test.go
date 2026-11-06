package journal

import (
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
)

// S2E-5 (P2): a first record that reaches MaxLine bytes without its '\n' keeps the provisional
// incarnation; the promotion comes with the '\n'.
func TestPromotionWaitsForTheFirstNewline(t *testing.T) {
	path, incP := writeJournal(t, "ab")
	appendTo := func(s string) {
		fh, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
		if err != nil {
			t.Fatal(err)
		}
		fh.WriteString(s)
		fh.Close()
	}
	appendTo(strings.Repeat("x", MaxLine+10)) // past MaxLine, still no '\n'
	if inc, _, _ := Identify(path); inc != incP {
		t.Errorf("first record of %d bytes without '\\n': Identify %s, want the provisional %s", MaxLine+12, inc, incP)
	}
	page := mustRead(t, path, incP, 0, 0)
	if page.PromotedFrom != "" || !page.Provisional || page.Incarnation != incP || page.Pending == nil {
		t.Errorf("still pending past MaxLine: %+v, want provisional %s and no promotion", page, incP)
	}
	appendTo("\n")
	page = mustRead(t, path, incP, 0, 0)
	if page.PromotedFrom != incP || page.Provisional || page.Incarnation == incP || len(page.Events) != 1 || page.Events[0].Invalid != LineTooLong {
		t.Errorf("after the '\\n': %+v, want a promotion and one line-too-long event", page)
	}
}

// S2E-1 (Codex round 3): a resume hint found while scanning the second of two records longer
// than MaxLine, reused with the first record's From, would merge both records into one event
// and skip the second. The cursor is refused instead.
func TestCursorReusedWithAnotherFromIsRefused(t *testing.T) {
	const rec1, rec2 = 2 << 20, 6 << 20
	path, inc := writeJournal(t, "{}\n"+record(rec1)+record(rec2)+"{}\n")
	from1, from2 := int64(3), int64(3+rec1)

	// A genuine pending cursor, while scanning the second record.
	page := mustRead(t, path, inc, from2, 16384)
	if page.Pending == nil || page.Pending.Scanned <= from2 || page.Pending.Scanned >= from2+rec2 {
		t.Fatalf("no pending scan inside the second record: %+v", page.Pending)
	}
	hint := Cursor{Incarnation: inc, From: from2, Scanned: page.Pending.Scanned,
		ScanFrom: page.Pending.Resume.ScanFrom, ScanIncarnation: page.Pending.Resume.ScanIncarnation}

	// Reused with the first record's From.
	c := hint
	c.From = from1
	for calls := 1; calls < 20; calls++ {
		got, err := ReadCursor(path, c, 16384)
		var r *Refusal
		if errors.As(err, &r) && r.Code == BadCursor {
			// Refused. Read honestly from the first record: two events, one per record.
			c, ends := Cursor{Incarnation: inc, From: from1}, []int64{}
			for len(ends) < 2 {
				got, err := ReadCursor(path, c, 16384)
				if err != nil {
					t.Fatal(err)
				}
				if got.Pending != nil {
					c = got.Pending.Resume
					continue
				}
				for _, ev := range got.Events {
					ends = append(ends, ev.End)
				}
				c = Cursor{Incarnation: inc, From: got.Next}
			}
			if ends[0] != from1+rec1 || ends[1] != from2+rec2 {
				t.Errorf("honest reads: ends %v, want [%d %d]", ends, from1+rec1, from2+rec2)
			}
			return
		}
		if err != nil {
			t.Fatalf("call %d: %v", calls, err)
		}
		if len(got.Events) > 0 {
			ev := got.Events[0]
			if ev.Offset != from1 || ev.End != from1+rec1 {
				t.Fatalf("reused cursor: event [%d, %d) spans both records (want [%d, %d)); next %d skips the second",
					ev.Offset, ev.End, from1, from1+rec1, got.Next)
			}
			t.Fatalf("reused cursor was accepted: %+v", ev)
		}
		c.Scanned = got.Pending.Scanned
	}
	t.Fatal("no answer after 20 calls")
}

// S2E-5 (P2): the true memory bound of one Read is two buffers, the page buffer (at most
// ReadMax) and the buffer of a first record longer than the budget (at most MaxLine). A small
// budget and a record just under MaxLine allocate both: more than ReadMax alone.
func TestReadMemoryIsTwoBuffersAtMost(t *testing.T) {
	path, inc := writeJournal(t, "{}\n"+record(MaxLine-10)+"{}\n")
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	page, err := ReadCursor(path, Cursor{Incarnation: inc, From: 3}, 100)
	runtime.ReadMemStats(&after)
	if err != nil {
		t.Fatal(err)
	}
	alloc := after.TotalAlloc - before.TotalAlloc
	t.Logf("budget 100, record of %d bytes: %d bytes allocated (ReadMax %d, ReadMax+MaxLine %d)", MaxLine-10, alloc, ReadMax, ReadMax+MaxLine)
	if len(page.Events) != 1 || page.Events[0].Invalid != "" {
		t.Fatalf("want the record as one event: %+v", page)
	}
	if alloc > ReadMax+MaxLine+64<<10 {
		t.Errorf("allocated %d bytes, beyond two buffers (%d)", alloc, ReadMax+MaxLine)
	}
}
