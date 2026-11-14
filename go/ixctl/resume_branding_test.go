package main

import (
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
