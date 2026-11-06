package journal

import (
	"errors"
	"os"
	"testing"
)

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
