package main

import (
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// R2-2: the start line says how long a finished mandate is remembered, and how many a compaction
// forgot.
func TestRecoveryLineSaysTheWindowAndWhatWasForgotten(t *testing.T) {
	line := recoveryLine(deliver.Recovery{Records: 9, Compacted: &deliver.Compaction{From: 900, To: 300, Forgotten: 4}, DedupWindow: 720 * time.Hour})
	for _, want := range []string{"compacted from 900 to 300 bytes (4 finished mandates past the deduplication window forgotten)", "a finished mandate is remembered at least 720h0m0s"} {
		if !strings.Contains(line, want) {
			t.Errorf("%q does not say %q", line, want)
		}
	}
}

// Review R2 P3-4: a negative --dedup-window is a bad command line, not the default.
func TestANegativeDedupWindowIsRefused(t *testing.T) {
	var out strings.Builder
	if code := run([]string{"serve", "--state", t.TempDir(), "--dedup-window", "-1h"}, &out); code != 2 || !strings.Contains(out.String(), "negative") {
		t.Errorf("serve --dedup-window -1h: exit %d, %q; want exit 2 saying it is negative", code, out.String())
	}
}
