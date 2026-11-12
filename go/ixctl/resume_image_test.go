package main

import (
	"errors"
	"strings"
	"testing"
)

// GAIA-WINSVC-IMAGE-R9: a receipt is about this install only when its ExpectedImagePath is the
// service's configured command line. Fake SCM and fake files only.

func TestSplitCommandLine(t *testing.T) {
	for _, c := range []struct {
		line, exe string
		args      []string
		err       string
	}{
		{`"C:\Program Files\ixctl\ixctl.exe" service run "ixctl-trial"`, `C:\Program Files\ixctl\ixctl.exe`, []string{"service", "run", "ixctl-trial"}, ""},
		{`C:\ixctl\ixctl.exe service   run`, `C:\ixctl\ixctl.exe`, []string{"service", "run"}, ""},
		{`"C:\ixctl\ixctl.exe"  `, `C:\ixctl\ixctl.exe`, []string{}, ""},
		// CommandLineToArgvW makes a leading blank an empty first argument: refused, not trimmed.
		{`  "C:\ixctl\ixctl.exe"`, "", nil, "starts with a blank"},
		{"\t" + `C:\ixctl\ixctl.exe service`, "", nil, "starts with a blank"},
		{`"C:\ixctl\ixctl.exe" --config "C:\a b\c.json"`, `C:\ixctl\ixctl.exe`, []string{"--config", `C:\a b\c.json`}, ""},
		{`C:\Program Files\ixctl\ixctl.exe service`, "", nil, `unquoted executable "C:\\Program" is ambiguous`},
		{`"C:\ixctl\ixctl.exe service`, "", nil, "a quote is not closed"},
		{`"C:\ixctl\ixctl.exe"service`, "", nil, "text follows a closing quote"},
		{`"C:\ixctl\ixctl.exe" --name \"x\"`, "", nil, "a backslash escapes a quote"},
		{``, "", nil, "no executable"},
		{`""`, "", nil, "no executable"},
	} {
		exe, args, err := splitCommandLine(c.line)
		switch {
		case c.err != "":
			if err == nil || !strings.Contains(err.Error(), c.err) {
				t.Errorf("%s: %q %q %v, want error %q", c.line, exe, args, err, c.err)
			}
		case err != nil || exe != c.exe || strings.Join(args, "|") != strings.Join(c.args, "|"):
			t.Errorf("%s: %q %q %v, want %q %q", c.line, exe, args, err, c.exe, c.args)
		}
	}
}

func TestResumeReceiptMatchesTheConfiguredImage(t *testing.T) {
	inv := mustInventory(t, receiptInventory)
	const at = "2026-11-10T20:00:00Z"
	run := func(service, expected, configured string, svcErr error) resumeResult {
		files := map[string]string{receiptPath: receiptJSONFor(receiptRun1, candidateSHA, service, expected, "reconciliation_required", at)}
		p := receiptProbes(t, files, []string{"update-" + receiptRun1}, nil)
		p.service = func(string) (string, string, string, error) {
			if svcErr != nil {
				return "", "", "", svcErr
			}
			return "running", "automatic", configured, nil
		}
		return checkResume(inv, p)[0]
	}
	// The same command line, written differently where Windows does not care: the state applies.
	for _, expected := range []string{
		configuredImage,
		`"c:\programdata\IXCTL\ixctl.EXE" service run "ixctl-trial" --workshop-config "C:\ProgramData\ixctl\workshop\config.json"`,
		`"C:/ProgramData/ixctl/ixctl.exe" service run ixctl-trial --workshop-config "C:\ProgramData\ixctl\workshop\config.json"`,
		`C:\ProgramData\ixctl\ixctl.exe  service run "ixctl-trial"  --workshop-config "C:\ProgramData\ixctl\workshop\config.json" `,
	} {
		if r := run("IXCTL", expected, configuredImage, nil); r.Verdict != verdictDown || !strings.Contains(resumeChecks(r), "state reconciliation_required: identity") {
			t.Errorf("same image %s: %s %q, want the state applied (down)", expected, r.Verdict, r.Checks)
		}
	}
	// Anything else is unknown: the state is shown, not applied.
	for _, c := range []struct {
		name, service, expected, configured string
		svcErr                              error
		line                                string
	}{
		{"another program", "ixctl", `"C:\Other\impostor.exe" service run "ixctl-trial" --workshop-config "C:\ProgramData\ixctl\workshop\config.json"`, configuredImage, nil,
			`it expects C:\Other\impostor.exe, the service runs C:\ProgramData\ixctl\ixctl.exe`},
		{"another instance", "ixctl", strings.Replace(configuredImage, `"ixctl-trial"`, `"ixctl-prod"`, 1), configuredImage, nil, "it expects the arguments"},
		{"argument case", "ixctl", strings.Replace(configuredImage, `ixctl-trial`, `IXCTL-TRIAL`, 1), configuredImage, nil, "it expects the arguments"},
		{"an argument more", "ixctl", configuredImage + " --debug", configuredImage, nil, "it expects the arguments"},
		{"an argument less", "ixctl", `"C:\ProgramData\ixctl\ixctl.exe" service run "ixctl-trial"`, configuredImage, nil, "it expects the arguments"},
		{"another service", "piper-voice", configuredImage, configuredImage, nil, `it is about service "piper-voice", not ixctl`},
		{"no ExpectedImagePath", "ixctl", "", configuredImage, nil, "its ExpectedImagePath cannot be read: no executable"},
		{"ambiguous expected", "ixctl", `C:\Program Files\ixctl\ixctl.exe service run`, configuredImage, nil, "its ExpectedImagePath cannot be read: unquoted executable"},
		{"ambiguous configured", "ixctl", configuredImage, `C:\Program Files\ixctl\ixctl.exe service run`, nil, "the service's configured command line cannot be read: unquoted executable"},
		{"malformed configured", "ixctl", configuredImage, `"C:\ProgramData\ixctl\ixctl.exe service run`, nil, "the service's configured command line cannot be read: a quote is not closed"},
		{"not installed", "ixctl", configuredImage, "", nil, "the service has no configured command line (not installed)"},
		{"SCM refused", "ixctl", configuredImage, "", errors.New("Access is denied."), "the service's configured command line cannot be read: Access is denied."},
	} {
		r := run(c.service, c.expected, c.configured, c.svcErr)
		text := resumeChecks(r)
		if r.Verdict != verdictUnknown || !strings.Contains(text, c.line) || !strings.Contains(text, `its state "reconciliation_required" is not applied`) {
			t.Errorf("%s: %s %q, want unknown with %q", c.name, r.Verdict, r.Checks, c.line)
		}
	}
	// A receipt is compared with a service, so it needs one.
	js := `{"components":[{"name":"a","health":"` + brandingURL + `","receipt":"` + receiptPath + `","receipt_candidate_sha256":"` + candidateSHA + `"}]}`
	if _, err := parseResumeInventory([]byte(js)); err == nil || !strings.Contains(err.Error(), "receipt needs service") {
		t.Errorf("receipt without service: %v", err)
	}
}
