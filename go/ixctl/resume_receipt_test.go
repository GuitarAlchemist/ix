package main

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// IxForge's update receipts (COMPAT-R9; GAIA-WINSVC-BRANDING-R7, RECEIPT-R8). Fake files only.

// A fake receipt store: the receipt under test in run 1, and other runs beside it.
const (
	receiptRun1  = "0123456789abcdef0123456789abcdef"
	receiptRun2  = "fedcba9876543210fedcba9876543210"
	receiptData  = "C:/data"
	receiptPath  = receiptData + "/update-" + receiptRun1 + "/receipt.json"
	candidateSHA = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
	otherSHA     = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
)

func receiptJSON(run, sha, state, updated string) string {
	return `{"Schema":1,"RunID":"` + run + `","CandidateSHA256":"` + sha + `","State":"` + state + `","UpdatedUTC":"` + updated + `"}`
}

// receiptProbes serves files (path to content; a missing path is os.ErrNotExist), lists the data
// directory's entries, and answers health with healthErr.
func receiptProbes(t *testing.T, files map[string]string, entries []string, healthErr error) resumeProbes {
	return resumeProbes{
		health: func(string, string) ([]byte, error) { return []byte(`{}`), healthErr },
		readFile: func(path string) ([]byte, error) {
			b, ok := files[path]
			if !ok {
				return nil, os.ErrNotExist
			}
			return []byte(b), nil
		},
		listDir: func(dir string) ([]string, error) {
			if dir != receiptData {
				t.Fatalf("listed %s, want %s", dir, receiptData)
			}
			return entries, nil
		},
	}
}

var receiptInventory = `{"components":[{"name":"workshop","health":"` + brandingURL + `","receipt":"` + receiptPath +
	`","receipt_candidate_sha256":"` + candidateSHA + `"}]}`

// The updater's receipt is a record, never proof of the current state: once correlated, it can
// only lower a verdict. reconciliation_required is down; only completed and rolled_back are end
// states.
func TestResumeReceiptNeverProvesHealth(t *testing.T) {
	inv := mustInventory(t, receiptInventory)
	run := func(healthErr error, file string, present bool) resumeResult {
		files := map[string]string{}
		if present {
			files[receiptPath] = file
		}
		return checkResume(inv, receiptProbes(t, files, []string{"update-" + receiptRun1}, healthErr))[0]
	}
	state := func(s string) string {
		return receiptJSON(receiptRun1, candidateSHA, s, "2026-11-10T20:00:00.0000000Z")
	}
	for _, c := range []struct {
		name, file string
		present    bool
		verdict    string
		line       string
	}{
		{"reconciliation_required", state("reconciliation_required"), true, verdictDown, "state reconciliation_required: identity or health not proven, a human must reconcile"},
		{"completed", state("completed"), true, verdictOK, "state completed (a record, not proof of the current state)"},
		{"rolled_back", state("rolled_back"), true, verdictOK, "state rolled_back (a record, not proof of the current state)"},
		{"installed", state("installed"), true, verdictDegraded, "state installed: the update did not reach an end state"},
		{"failed_before_replacement", state("failed_before_replacement"), true, verdictDegraded, "state failed_before_replacement: the update did not reach an end state"},
		{"unknown state", state("healthy"), true, verdictUnknown, `unknown state "healthy"`},
		{"no state", state(""), true, verdictUnknown, `unknown state ""`},
		{"truncated", `{"Schema":1,"State":"compl`, true, verdictUnknown, "is not JSON"},
		{"missing", "", false, verdictUnknown, "receipt " + receiptPath + ": "},
	} {
		r := run(nil, c.file, c.present)
		if r.Verdict != c.verdict || !strings.Contains(resumeChecks(r), c.line) {
			t.Errorf("%s: %s %q, want %s with %q", c.name, r.Verdict, r.Checks, c.verdict, c.line)
		}
	}
	// A completed receipt does not lift a component whose health fails.
	if r := run(errors.New("HTTP 503"), state("completed"), true); r.Verdict != verdictDown {
		t.Errorf("completed receipt over failed health: %s %q, want down", r.Verdict, r.Checks)
	}
	for _, c := range []struct{ js, want string }{
		{`{"components":[{"name":"a","receipt":"` + receiptPath + `","receipt_candidate_sha256":"` + candidateSHA + `"}]}`, "nothing to observe"},
		{`{"components":[{"name":"a","health":"` + brandingURL + `","receipt":"` + receiptPath + `"}]}`, "needs receipt_candidate_sha256"},
		{`{"components":[{"name":"a","health":"` + brandingURL + `","receipt":"` + receiptPath + `","receipt_candidate_sha256":"abc"}]}`, "needs receipt_candidate_sha256"},
		{`{"components":[{"name":"a","health":"` + brandingURL + `","receipt_candidate_sha256":"` + candidateSHA + `"}]}`, "without receipt"},
	} {
		if _, err := parseResumeInventory([]byte(c.js)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", c.js, err, c.want)
		}
	}
}

// A receipt decides nothing until it is tied to the current install. Each uncorrelated receipt
// below says reconciliation_required and is unknown, never down nor ok: the old or foreign state
// is shown, not applied.
func TestResumeReceiptNeedsCorrelation(t *testing.T) {
	inv := mustInventory(t, receiptInventory)
	const at, earlier, later = "2026-11-10T20:00:00Z", "2026-11-09T08:00:00Z", "2026-11-11T08:00:00Z"
	own, run2 := "update-"+receiptRun1, "update-"+receiptRun2
	other := receiptData + "/" + run2 + "/receipt.json"
	const bad = "reconciliation_required"
	for _, c := range []struct {
		name    string
		receipt string
		extra   map[string]string
		entries []string
		line    string
	}{
		{"schema 2", strings.Replace(receiptJSON(receiptRun1, candidateSHA, bad, at), `"Schema":1`, `"Schema":2`, 1), nil, nil, "schema 2, not 1"},
		{"no schema", strings.Replace(receiptJSON(receiptRun1, candidateSHA, bad, at), `"Schema":1,`, ``, 1), nil, nil, "schema 0, not 1"},
		{"RunID not hex", receiptJSON("R1", candidateSHA, bad, at), nil, nil, `RunID "R1" is not 32 lowercase hex digits`},
		{"RunID of another directory", receiptJSON(receiptRun2, candidateSHA, bad, at), nil, nil, "RunID " + receiptRun2 + " does not name its directory " + own},
		{"another candidate", receiptJSON(receiptRun1, otherSHA, bad, at), nil, nil, "another candidate than receipt_candidate_sha256"},
		{"no candidate", receiptJSON(receiptRun1, "", bad, at), nil, nil, "another candidate"},
		{"no time", receiptJSON(receiptRun1, candidateSHA, bad, ""), nil, nil, `UpdatedUTC "" is not a time`},
		{"superseded by a newer run", receiptJSON(receiptRun1, candidateSHA, bad, at),
			map[string]string{other: receiptJSON(receiptRun2, candidateSHA, "completed", later)}, []string{own, run2}, "run " + run2 + " is as recent or newer"},
		{"a run at the same time", receiptJSON(receiptRun1, candidateSHA, bad, at),
			map[string]string{other: receiptJSON(receiptRun2, candidateSHA, "completed", at)}, []string{own, run2}, "is as recent or newer"},
		{"a run that cannot be read", receiptJSON(receiptRun1, candidateSHA, bad, at),
			nil, []string{own, run2}, "run " + run2 + " cannot be compared"},
		{"a run with no time", receiptJSON(receiptRun1, candidateSHA, bad, at),
			map[string]string{other: receiptJSON(receiptRun2, candidateSHA, "completed", "")}, []string{own, run2}, "is as recent or newer"},
	} {
		files := map[string]string{receiptPath: c.receipt}
		for k, v := range c.extra {
			files[k] = v
		}
		entries := c.entries
		if entries == nil {
			entries = []string{own}
		}
		r := checkResume(inv, receiptProbes(t, files, entries, nil))[0]
		text := resumeChecks(r)
		if r.Verdict != verdictUnknown || !strings.Contains(text, c.line) ||
			!strings.Contains(text, "not correlated with the current install") || !strings.Contains(text, `its state "reconciliation_required" is not applied`) {
			t.Errorf("%s: %s %q, want unknown, not correlated: %q", c.name, r.Verdict, r.Checks, c.line)
		}
	}

	// Correlated, with an older run of another candidate and unrelated entries beside it: the
	// state applies.
	files := map[string]string{
		receiptPath: receiptJSON(receiptRun1, candidateSHA, bad, at),
		other:       receiptJSON(receiptRun2, otherSHA, "completed", earlier),
	}
	r := checkResume(inv, receiptProbes(t, files, []string{"logs", own, run2, "workshop"}, nil))[0]
	if r.Verdict != verdictDown || !strings.Contains(resumeChecks(r), "state reconciliation_required: identity or health not proven") {
		t.Errorf("correlated: %s %q, want down", r.Verdict, r.Checks)
	}
	// Correlated completed over a healthy answer: ok, because health answered.
	files[receiptPath] = receiptJSON(receiptRun1, candidateSHA, "completed", at)
	if r := checkResume(inv, receiptProbes(t, files, []string{own, run2}, nil))[0]; r.Verdict != verdictOK {
		t.Errorf("correlated completed: %s %q, want ok", r.Verdict, r.Checks)
	}
	// The other runs cannot be listed: unknown.
	p := receiptProbes(t, files, nil, nil)
	p.listDir = func(string) ([]string, error) { return nil, errors.New("Access is denied.") }
	if r := checkResume(inv, p)[0]; r.Verdict != verdictUnknown || !strings.Contains(resumeChecks(r), "the other runs cannot be listed: Access is denied.") {
		t.Errorf("unlistable: %s %q, want unknown", r.Verdict, r.Checks)
	}
}
