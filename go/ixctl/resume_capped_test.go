package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Codex round 1 on the resume chain: a capped read is never the whole answer, a receipt is named
// receipt.json, a command line starts with its executable, and "at most 5 redirects" means five.
// Fake transports and temp files only.

// padTo returns s, then spaces up to n bytes, then tail: valid JSON as far as a reader capped at
// n bytes can see.
func padTo(s string, n int, tail string) string {
	return s + strings.Repeat(" ", n-len(s)) + tail
}

func TestResumeHealthBodyOverTheCapIsDown(t *testing.T) {
	inv := mustInventory(t, `{"components":[{"name":"workshop","health":"`+brandingURL+`","health_fields":{"name":"IxForge","version":"0.1.1"}}]}`)
	branding := `{"name":"IxForge","short_name":"IxForge","version":"0.1.1"}`
	for _, c := range []struct {
		name, body, verdict, line string
	}{
		{"exactly the cap", padTo(branding, 64<<10, ""), verdictOK, `answers with name "IxForge"`},
		{"one byte over", padTo(branding, 64<<10, "x"), verdictDown, "larger than 65536 bytes"},
		{"valid prefix, then garbage", padTo(branding, 64<<10, `,"version":"6.6.6"}`), verdictDown, "larger than 65536 bytes"},
	} {
		rt := roundTrip(func(r *http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(c.body)), Request: r}, nil
		})
		r := checkResume(inv, resumeProbes{health: healthProbe(rt, resumeHealthTimeout)})[0]
		if r.Verdict != c.verdict || !strings.Contains(resumeChecks(r), c.line) {
			t.Errorf("%s: %s %q, want %s with %q", c.name, r.Verdict, r.Checks, c.verdict, c.line)
		}
	}
}

func TestResumeReceiptOverTheCapIsUnreadable(t *testing.T) {
	dir := t.TempDir()
	read := realResumeProbesOver(nil).readFile
	receipt := receiptJSON(receiptRun1, candidateSHA, "completed", "2026-11-10T20:00:00Z")
	for _, c := range []struct {
		name, body, err string
	}{
		{"exactly the cap", padTo(receipt, 1<<20, ""), ""},
		{"valid prefix, then garbage", padTo(receipt, 1<<20, `"State":"reconciliation_required"}`), "larger than 1048576 bytes"},
	} {
		path := filepath.Join(dir, strings.ReplaceAll(c.name, " ", "-")+".json")
		if err := os.WriteFile(path, []byte(c.body), 0o600); err != nil {
			t.Fatal(err)
		}
		b, err := read(path)
		switch {
		case c.err == "" && (err != nil || len(b) != 1<<20):
			t.Errorf("%s: %d bytes, %v; want the whole file", c.name, len(b), err)
		case c.err != "" && (err == nil || !strings.Contains(err.Error(), c.err)):
			t.Errorf("%s: %d bytes, %v; want %q", c.name, len(b), err, c.err)
		}
	}
}

// Only <DataDir>\update-<RunID>\receipt.json is a receipt: another file in the run's directory,
// however well correlated its content, decides nothing.
func TestResumeReceiptMustBeNamedReceiptJSON(t *testing.T) {
	for _, name := range []string{"draft.json", "receipt.json.bak", "Receipt.JSON"} {
		path := receiptData + "/update-" + receiptRun1 + "/" + name
		inv := mustInventory(t, strings.Replace(receiptInventory, receiptPath, path, 1))
		files := map[string]string{path: receiptJSON(receiptRun1, candidateSHA, "reconciliation_required", "2026-11-10T20:00:00Z")}
		r := checkResume(inv, receiptProbes(t, files, []string{"update-" + receiptRun1}, nil))[0]
		text := resumeChecks(r)
		if name == "Receipt.JSON" {
			// Windows names are case-insensitive: the same file.
			if r.Verdict != verdictDown {
				t.Errorf("%s: %s %q, want the state applied (down)", name, r.Verdict, r.Checks)
			}
			continue
		}
		if r.Verdict != verdictUnknown || !strings.Contains(text, "it is not named receipt.json") {
			t.Errorf("%s: %s %q, want unknown, not named receipt.json", name, r.Verdict, r.Checks)
		}
	}
}

// The documented "at most 5" redirects: the fifth is followed, the sixth is refused.
func TestResumeHealthFollowsAtMostFiveRedirects(t *testing.T) {
	for _, c := range []struct {
		hops int
		ok   bool
	}{{5, true}, {6, false}} {
		rt := roundTrip(func(r *http.Request) (*http.Response, error) {
			var n int
			fmt.Sscanf(r.URL.Path, "/hop%d", &n)
			if n < c.hops {
				h := http.Header{}
				h.Set("Location", fmt.Sprintf("http://127.0.0.1:8761/hop%d", n+1))
				return &http.Response{StatusCode: http.StatusFound, Header: h, Body: http.NoBody, Request: r}, nil
			}
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("ok")), Request: r}, nil
		})
		_, err := healthProbe(rt, resumeHealthTimeout)("http://127.0.0.1:8761/hop0", "")
		if (err == nil) != c.ok {
			t.Errorf("%d redirects: %v, want followed=%v", c.hops, err, c.ok)
		}
	}
}
