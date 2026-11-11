package main

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// IxForge's branding and receipt contract (COMPAT-R9, GAIA-WINSVC-BRANDING-R7). Fake probes only.
// branding.json is {"name":"IxForge","short_name":"IxForge","version":"0.1.1"}: version is the
// web client's (PWA) version, not the ixctl binary's ("ixctl version", 0.1.0-ixforge.4).

const brandingURL = "http://127.0.0.1:8761/workshop/branding.json"

func healthReturning(body string) resumeProbes {
	return resumeProbes{health: func(string, string) ([]byte, error) { return []byte(body), nil }}
}

func TestResumeHealthFieldsChecksNameAndVersion(t *testing.T) {
	inv := mustInventory(t, `{"components":[{"name":"workshop","health":"`+brandingURL+`",
		"health_host":"aw.example.ts.net","health_fields":{"name":"IxForge","version":"0.1.1"}}]}`)
	for _, c := range []struct {
		body, verdict, line string
	}{
		{`{"name":"IxForge","short_name":"IxForge","version":"0.1.1"}`, verdictOK, `health ` + brandingURL + ` answers with name "IxForge", version "0.1.1"`},
		{`{"name":"IxForge","short_name":"IxForge","version":"0.1.0"}`, verdictDown, `version is "0.1.0", not "0.1.1"`},
		{`{"name":"IxForge","short_name":"IxForge","version":"0.1.0-ixforge.4"}`, verdictDown, `version is "0.1.0-ixforge.4", not "0.1.1"`},
		{`{"name":"IxForge","short_name":"IxForge"}`, verdictDown, `body has no version`},
		{`{"name":"IxForge","version":0.1}`, verdictDown, `version is not a string`},
		{`{"name":"ixforge","version":"0.1.1"}`, verdictDown, `name is "ixforge", not "IxForge"`},
		{`{"name":"Impostor","version":"0.1.1","x":"\"name\":\"IxForge\""}`, verdictDown, `name is "Impostor", not "IxForge"`},
		{`<!doctype html><title>IxForge</title>`, verdictDown, `body is not a JSON object`},
		{`["IxForge","0.1.1"]`, verdictDown, `body is not a JSON object`},
		{`{"name":"IxForge","version":"0.1.1"} trailing`, verdictDown, `body is not a JSON object`},
		{``, verdictDown, `body is not a JSON object`},
	} {
		r := checkResume(inv, healthReturning(c.body))[0]
		if r.Verdict != c.verdict || !strings.Contains(resumeChecks(r), c.line) {
			t.Errorf("body %s: %s %q, want %s with %q", c.body, r.Verdict, r.Checks, c.verdict, c.line)
		}
	}
}

func TestResumeHealthFieldsRefusals(t *testing.T) {
	for _, c := range []struct{ js, want string }{
		{`{"components":[{"name":"a","listen":"127.0.0.1:1","health_fields":{"version":"0.1.1"}}]}`, "without health"},
		{`{"components":[{"name":"a","health":"` + brandingURL + `","health_fields":{}}]}`, "health_fields is empty"},
		{`{"components":[{"name":"a","health":"` + brandingURL + `","health_fields":{"version":""}}]}`, "health_fields version is empty"},
	} {
		if _, err := parseResumeInventory([]byte(c.js)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want %q", c.js, err, c.want)
		}
	}
}

// The updater's receipt is a record, never proof of the current state: it can only lower a
// verdict. reconciliation_required is down; only completed and rolled_back are end states.
func TestResumeReceiptNeverProvesHealth(t *testing.T) {
	const receipt = `C:/data/update-r1/receipt.json`
	inv := mustInventory(t, `{"components":[{"name":"workshop","health":"`+brandingURL+`","receipt":"`+receipt+`"}]}`)
	run := func(healthErr error, file string, readErr error) resumeResult {
		p := resumeProbes{
			health: func(string, string) ([]byte, error) { return []byte(`{}`), healthErr },
			readFile: func(path string) ([]byte, error) {
				if path != receipt {
					t.Fatalf("read %s, want %s", path, receipt)
				}
				return []byte(file), readErr
			},
		}
		return checkResume(inv, p)[0]
	}
	state := func(s string) string { return `{"Schema":1,"State":"` + s + `","RunID":"r1"}` }
	for _, c := range []struct {
		name, file string
		readErr    error
		verdict    string
		line       string
	}{
		{"reconciliation_required", state("reconciliation_required"), nil, verdictDown, "state reconciliation_required: identity or health not proven, a human must reconcile"},
		{"completed", state("completed"), nil, verdictOK, "state completed (a record, not proof of the current state)"},
		{"rolled_back", state("rolled_back"), nil, verdictOK, "state rolled_back (a record, not proof of the current state)"},
		{"installed", state("installed"), nil, verdictDegraded, "state installed: the update did not reach an end state"},
		{"failed_before_replacement", state("failed_before_replacement"), nil, verdictDegraded, "state failed_before_replacement: the update did not reach an end state"},
		{"unknown state", state("healthy"), nil, verdictUnknown, `unknown state "healthy"`},
		{"no state", `{"Schema":1}`, nil, verdictUnknown, `unknown state ""`},
		{"truncated", `{"Schema":1,"State":"compl`, nil, verdictUnknown, "is not JSON"},
		{"missing", "", os.ErrNotExist, verdictUnknown, "receipt " + receipt + ": "},
	} {
		r := run(nil, c.file, c.readErr)
		if r.Verdict != c.verdict || !strings.Contains(resumeChecks(r), c.line) {
			t.Errorf("%s: %s %q, want %s with %q", c.name, r.Verdict, r.Checks, c.verdict, c.line)
		}
	}
	// A completed receipt does not lift a component whose health fails.
	if r := run(errors.New("HTTP 503"), state("completed"), nil); r.Verdict != verdictDown {
		t.Errorf("completed receipt over failed health: %s %q, want down", r.Verdict, r.Checks)
	}
	// A receipt alone observes nothing current.
	if _, err := parseResumeInventory([]byte(`{"components":[{"name":"a","receipt":"` + receipt + `"}]}`)); err == nil || !strings.Contains(err.Error(), "nothing to observe") {
		t.Errorf("receipt alone accepted: %v", err)
	}
}
