package journal

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const key = "fixture/s1"

func writeJournal(t *testing.T, content string) (path, inc string) {
	t.Helper()
	path = filepath.Join(t.TempDir(), "journal.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	inc, _, err := Identify(path)
	if err != nil {
		t.Fatal(err)
	}
	return path, inc
}

func mustRead(t *testing.T, path, inc string, from, budget int64) Page {
	t.Helper()
	page, err := Read(path, inc, from, budget)
	if err != nil {
		t.Fatalf("Read(from=%d, max_bytes=%d): %v", from, budget, err)
	}
	return page
}

// record is one record of n bytes including its '\n'.
func record(n int) string { return strings.Repeat("x", n-1) + "\n" }

func TestChainedReadsCoverTheFileOnce(t *testing.T) {
	path, _ := issue511Journal(t)
	inc, size, err := Identify(path)
	if err != nil {
		t.Fatal(err)
	}
	var ids [2][]string
	for b, budget := range []int64{16384, 0} {
		var reads int
		end := int64(0)
		for from := int64(0); from < size; reads++ {
			page := mustRead(t, path, inc, from, budget)
			if page.Next == from {
				t.Fatalf("max_bytes=%d: read %d does not advance from %d", budget, reads, from)
			}
			for _, ev := range page.Events {
				if ev.Offset != end {
					t.Fatalf("max_bytes=%d: event at %d, want %d (gap or duplicate)", budget, ev.Offset, end)
				}
				end = ev.End
				ids[b] = append(ids[b], page.EventID(key, ev))
			}
			// The same checkpoint read again gives the same events.
			again := mustRead(t, path, inc, from, budget)
			if len(again.Events) != len(page.Events) || again.Next != page.Next {
				t.Fatalf("max_bytes=%d: rereading %d differs", budget, from)
			}
			from = page.Next
		}
		if reads != 3 || end != size || len(ids[b]) != 3 {
			t.Fatalf("max_bytes=%d: %d reads, %d events, covered %d of %d", budget, reads, len(ids[b]), end, size)
		}
	}
	for i := range ids[0] {
		if ids[0][i] != ids[1][i] {
			t.Errorf("event %d: id %q at 16 KiB, %q at default", i, ids[0][i], ids[1][i])
		}
	}
	if want := key + "@" + inc + ":8"; ids[0][1] != want {
		t.Errorf("id %q, want %q", ids[0][1], want)
	}
}

func TestMaxLineBoundary(t *testing.T) {
	for _, tc := range []struct {
		n       int
		invalid string
	}{{MaxLine, ""}, {MaxLine + 1, LineTooLong}} {
		path, inc := writeJournal(t, "{}\n"+record(tc.n)+"{}\n")
		page := mustRead(t, path, inc, 3, 0)
		if len(page.Events) != 1 {
			t.Fatalf("record of %d bytes: %d events", tc.n, len(page.Events))
		}
		ev := page.Events[0]
		if ev.Invalid != tc.invalid || ev.Offset != 3 || ev.End != 3+int64(tc.n) || page.Next != ev.End {
			t.Errorf("record of %d bytes: invalid %q over [%d, %d) next %d", tc.n, ev.Invalid, ev.Offset, ev.End, page.Next)
		}
		if tc.invalid == "" && len(ev.Line) != tc.n-1 {
			t.Errorf("record of %d bytes: line of %d bytes", tc.n, len(ev.Line))
		}
	}
}

// TestEventDoesNotDependOnBudget: a complete record of at most MaxLine bytes, larger than the
// budget, is still returned whole when it is the first of the page.
func TestEventDoesNotDependOnBudget(t *testing.T) {
	path, inc := writeJournal(t, "{}\n"+record(20<<10)+"{}\n")
	var got [2]Page
	for i, budget := range []int64{16384, 0} {
		got[i] = mustRead(t, path, inc, 3, budget)
		if len(got[i].Events) == 0 {
			t.Fatalf("max_bytes=%d: no event", budget)
		}
	}
	a, b := got[0].Events[0], got[1].Events[0]
	if a.Invalid != "" || string(a.Line) != string(b.Line) || got[0].EventID(key, a) != got[1].EventID(key, b) || a.End != b.End {
		t.Errorf("16 KiB: invalid %q, %d bytes, id %s, end %d; default: invalid %q, %d bytes, id %s, end %d",
			a.Invalid, len(a.Line), got[0].EventID(key, a), a.End, b.Invalid, len(b.Line), got[1].EventID(key, b), b.End)
	}
	if got[0].Next != 3+20<<10 {
		t.Errorf("16 KiB: next %d, want %d (only the first record)", got[0].Next, 3+20<<10)
	}
}

func TestCRLFAndBlankLines(t *testing.T) {
	path, inc := writeJournal(t, "a\r\n\r\n\nb\n")
	page := mustRead(t, path, inc, 0, 0)
	want := []struct {
		line       string
		blank      bool
		offset, to int64
	}{{"a", false, 0, 3}, {"", true, 3, 5}, {"", true, 5, 6}, {"b", false, 6, 8}}
	if len(page.Events) != len(want) || page.Next != 8 || page.Pending != nil {
		t.Fatalf("%d events, next %d, pending %v", len(page.Events), page.Next, page.Pending)
	}
	for i, w := range want {
		ev := page.Events[i]
		if string(ev.Line) != w.line || ev.Blank != w.blank || ev.Offset != w.offset || ev.End != w.to {
			t.Errorf("event %d: %q blank %v [%d, %d), want %+v", i, ev.Line, ev.Blank, ev.Offset, ev.End, w)
		}
	}
}

func TestEmptyFile(t *testing.T) {
	path, inc := writeJournal(t, "")
	page := mustRead(t, path, inc, 0, 0)
	if len(page.Events) != 0 || page.Next != 0 || page.Size != 0 || page.Pending != nil || !page.AtEnd || !page.Provisional {
		t.Errorf("empty journal: %+v", page)
	}
}

func TestIncompleteLastLineIsPending(t *testing.T) {
	path, inc := writeJournal(t, "a\nbc")
	page := mustRead(t, path, inc, 0, 0)
	if len(page.Events) != 1 || page.Next != 2 || page.Pending == nil || *page.Pending != (Pending{At: 2, Bytes: 2}) {
		t.Fatalf("from 0: %d events, next %d, pending %v", len(page.Events), page.Next, page.Pending)
	}
	for range 2 {
		page = mustRead(t, path, inc, 2, 0)
		if len(page.Events) != 0 || page.Next != 2 || page.Pending == nil || *page.Pending != (Pending{At: 2, Bytes: 2}) {
			t.Fatalf("from 2: %d events, next %d, pending %v", len(page.Events), page.Next, page.Pending)
		}
	}
	// An incomplete first record: a provisional incarnation, and it is pending.
	path, inc = writeJournal(t, "abc")
	page = mustRead(t, path, inc, 0, 0)
	if !page.Provisional || page.Pending == nil || *page.Pending != (Pending{At: 0, Bytes: 3}) {
		t.Errorf("incomplete first record: provisional %v, pending %v", page.Provisional, page.Pending)
	}
}

func TestIncompleteLineLongerThanBudgetIsPending(t *testing.T) {
	path, inc := writeJournal(t, "a\n"+strings.Repeat("x", 20000))
	page := mustRead(t, path, inc, 2, 16384)
	if len(page.Events) != 0 || page.Next != 2 || page.Pending == nil || *page.Pending != (Pending{At: 2, Bytes: 20000}) {
		t.Errorf("%d events, next %d, pending %v", len(page.Events), page.Next, page.Pending)
	}
}

func TestRefusals(t *testing.T) {
	path, inc := writeJournal(t, "{\"n\":1}\n{\"n\":2}\n")
	other, otherInc := writeJournal(t, "{\"n\":9}\n")
	if otherInc == inc {
		t.Fatal("two journals share an incarnation")
	}
	for _, tc := range []struct {
		name, path, inc string
		from            int64
		code            string
	}{
		{"another journal", other, inc, 0, StaleIncarnation},
		{"unknown incarnation", path, "000000000000", 0, StaleIncarnation},
		{"past the end", path, inc, 17, Truncated},
		{"negative", path, inc, -1, Truncated},
		{"mid-record", path, inc, 3, BadCheckpoint},
	} {
		_, err := Read(tc.path, tc.inc, tc.from, 0)
		var r *Refusal
		if !errors.As(err, &r) || r.Code != tc.code {
			t.Errorf("%s: %v, want %s", tc.name, err, tc.code)
		}
	}
}

// TestReadMemoryIsBounded: a record of several MiB costs one buffer of at most ReadMax.
func TestReadMemoryIsBounded(t *testing.T) {
	const big = 5 << 20
	path, inc := writeJournal(t, "{}\n"+record(big)+"{}\n")
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	page := mustRead(t, path, inc, 3, 0)
	runtime.ReadMemStats(&after)
	if alloc := after.TotalAlloc - before.TotalAlloc; alloc > ReadMax+64<<10 {
		t.Errorf("one Read allocated %d bytes, bound %d", alloc, ReadMax)
	}
	if len(page.Events) != 1 || page.Events[0].Invalid != LineTooLong || page.Next != 3+big {
		t.Errorf("%d events, next %d", len(page.Events), page.Next)
	}
}
