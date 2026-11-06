package journal

import (
	"os"
	"strings"
	"testing"
)

// S2G-2 (Codex round 5): once a first record longer than MaxLine has its '\n', Identify names the
// journal as Read does, instead of staying provisional forever.
func TestIdentifyPromotesAnOversizedFirstRecord(t *testing.T) {
	path, incP := writeJournal(t, strings.Repeat("x", 2<<20)) // 2 MiB, no '\n' yet
	if inc, _, _ := Identify(path); inc != incP {
		t.Fatalf("incomplete: Identify %s, want the provisional %s", inc, incP)
	}
	fh, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	fh.WriteString("\n{}\n")
	fh.Close()
	c, read := Cursor{Incarnation: incP}, ""
	for calls := 0; read == "" && calls < 10; calls++ {
		page, err := ReadCursor(path, c, 0)
		if err != nil {
			t.Fatal(err)
		}
		if page.Pending != nil {
			c = page.Pending.Resume
			continue
		}
		read = page.Incarnation
	}
	inc, _, _ := Identify(path)
	if read == "" || read == incP || inc != read {
		t.Errorf("after the '\\n': Read names %q, Identify %q (provisional %s); want the same named incarnation", read, inc, incP)
	}
}

// S2G-2: a first record longer than ScanMax is named after as many IdentifyFrom calls as needed,
// each reading at most ScanMax bytes, and then matches Read.
func TestIdentifyFromResumesPastScanMax(t *testing.T) {
	path, incP := writeJournal(t, strings.Repeat("x", 9<<20)+"\n{}\n")
	read := ""
	for c, calls := (Cursor{Incarnation: incP}), 0; read == "" && calls < 20; calls++ {
		page, err := ReadCursor(path, c, 0)
		if err != nil {
			t.Fatal(err)
		}
		if page.Pending != nil {
			c = page.Pending.Resume
			continue
		}
		read = page.Incarnation
	}
	var reads []int64
	testHookRead = func(n int64) { reads = append(reads, n) }
	defer func() { testHookRead = nil }()
	id, calls := Identity{}, 0
	for id.Incarnation == "" || id.Provisional {
		if calls++; calls > 10 {
			t.Fatalf("still provisional after 10 calls: %+v", id)
		}
		var total int64
		reads = reads[:0]
		var err error
		if id, err = IdentifyFrom(path, id.Scanned); err != nil {
			t.Fatal(err)
		}
		for _, n := range reads {
			total += n
		}
		if total > ScanMax {
			t.Errorf("call %d read %d bytes, bound %d", calls, total, ScanMax)
		}
		if id.Provisional && id.Incarnation != incP {
			t.Errorf("call %d: provisional %s, want %s", calls, id.Incarnation, incP)
		}
	}
	if calls < 3 || id.Incarnation != read || read == incP {
		t.Errorf("after %d calls: IdentifyFrom %s, Read %s; want the same named incarnation, past ScanMax", calls, id.Incarnation, read)
	}
}
