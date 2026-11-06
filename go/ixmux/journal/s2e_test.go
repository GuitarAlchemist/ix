package journal

import (
	"errors"
	"testing"
)

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
