package journal

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The fixture of ix#511: one record of 1,114,550 bytes between two small ones.
const bigRecord = 1_114_550

func issue511Journal(t *testing.T) (path string, from int64) {
	t.Helper()
	first := []byte(`{"n":1}` + "\n")
	big := []byte(`{"pad":"` + strings.Repeat("x", bigRecord-len(`{"pad":""}`)-1) + `"}` + "\n")
	if len(big) != bigRecord {
		t.Fatalf("fixture record is %d bytes, want %d", len(big), bigRecord)
	}
	last := []byte(`{"n":3}` + "\n")
	path = filepath.Join(t.TempDir(), "journal.jsonl")
	if err := os.WriteFile(path, bytes.Join([][]byte{first, big, last}, nil), 0o600); err != nil {
		t.Fatal(err)
	}
	return path, int64(len(first))
}

type readFunc func(path, incarnation string, from, maxBytes int64) (Page, error)

// v2Read is a minimal copy of the V2 prototype's session.read loop (sessionmux.go, read), without
// the fix: it looks for '\n' only inside one buffer of the budget.
func v2Read(path, incarnation string, from, maxBytes int64) (Page, error) {
	f, err := os.Open(path)
	if err != nil {
		return Page{}, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return Page{}, err
	}
	size := st.Size()
	max := maxBytes
	if max <= 0 || max > ReadMax {
		max = ReadMax
	}
	buf := make([]byte, min(max, size-from))
	got, _ := f.ReadAt(buf, from)
	buf = buf[:got]
	page := Page{Incarnation: incarnation, Events: []Event{}, Next: from, Size: size}
	for {
		i := bytes.IndexByte(buf, '\n')
		if i < 0 {
			break
		}
		end := page.Next + int64(i) + 1
		ev := Event{Offset: page.Next, End: end, Line: buf[:i]}
		if i > MaxLine {
			ev.Invalid = LineTooLong
		}
		page.Events = append(page.Events, ev)
		buf, page.Next = buf[i+1:], end
	}
	return page, nil
}

// underTest is the reader the oracle checks. It was v2Read for the red run of ix#511.
var underTest readFunc = Read

// TestIssue511V2Stalls keeps the defect reproduced: V2 returns no event and does not advance.
func TestIssue511V2Stalls(t *testing.T) {
	path, from := issue511Journal(t)
	for _, budget := range []int64{16384, 0} {
		page, err := v2Read(path, "", from, budget)
		if err != nil {
			t.Fatal(err)
		}
		if len(page.Events) != 0 || page.Next != from {
			t.Errorf("max_bytes=%d: V2 gave %d events, next %d; ix#511 is no longer reproduced", budget, len(page.Events), page.Next)
		}
	}
}

// TestIssue511ResumePastOversizedLine: from a checkpoint at the start of the oversized record,
// both budgets of the issue must advance past it with one line-too-long event over its range.
func TestIssue511ResumePastOversizedLine(t *testing.T) {
	path, from := issue511Journal(t)
	inc, _, err := Identify(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, budget := range []int64{16384, 0} {
		page, err := underTest(path, inc, from, budget)
		if err != nil {
			t.Fatalf("max_bytes=%d: %v", budget, err)
		}
		if len(page.Events) != 1 || page.Next == from {
			t.Errorf("max_bytes=%d: %d events, next %d (from %d): the read does not advance", budget, len(page.Events), page.Next, from)
			continue
		}
		ev := page.Events[0]
		if ev.Invalid != LineTooLong || ev.Offset != from || ev.End != from+bigRecord || page.Next != ev.End {
			t.Errorf("max_bytes=%d: event %+v next %d, want line-too-long over [%d, %d)", budget, ev, page.Next, from, from+bigRecord)
		}
	}
}
