package main

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

// call runs the validator's real stdin -> stdout/stderr path in process.
func call(t *testing.T, in string) (code int, out report, stderr string) {
	t.Helper()
	var so, se bytes.Buffer
	code = run(strings.NewReader(in), &so, &se)
	if code == 0 {
		if err := json.Unmarshal(so.Bytes(), &out); err != nil {
			t.Fatalf("stdout is not a report: %v: %q", err, so.String())
		}
	} else if so.Len() != 0 {
		t.Fatalf("exit %d but stdout is not empty: %q", code, so.String())
	}
	return code, out, se.String()
}

func fixture(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestValidFixture(t *testing.T) {
	code, r, se := call(t, fixture(t, "../fixtures/candidate-valid.json"))
	if code != 0 || !r.Valid || r.Validator != name || len(r.Errors) != 0 || se != "" {
		t.Fatalf("code=%d report=%+v stderr=%q", code, r, se)
	}
	// "errors" is an empty array, never null: consumers index it.
	var so bytes.Buffer
	run(strings.NewReader(fixture(t, "../fixtures/candidate-valid.json")), &so, &bytes.Buffer{})
	if !strings.Contains(so.String(), `"errors":[]`) {
		t.Fatalf("errors must serialise as []: %s", so.String())
	}
}

// The invalid fixture's stdout is byte-for-byte the golden file run.sh asserts under Wasmtime.
func TestInvalidFixtureMatchesGolden(t *testing.T) {
	var so, se bytes.Buffer
	code := run(strings.NewReader(fixture(t, "../fixtures/candidate-invalid.json")), &so, &se)
	want := fixture(t, "../p3-spike/expected/validator-invalid.stdout")
	if code != 0 || so.String() != want || se.Len() != 0 {
		t.Fatalf("code=%d\n got %q\nwant %q\nstderr %q", code, so.String(), want, se.String())
	}
}

func TestFieldErrors(t *testing.T) {
	base := `"target":{"repo":"r"},"verdict":"pass"`
	cases := []struct {
		name, in string
		want     []string
	}{
		{"all fields valid", `{"schema_version":1,` + base + `}`, []string{}},
		{"notes allowed", `{"schema_version":1,` + base + `,"notes":"n"}`, []string{}},
		{"wrong schema_version", `{"schema_version":3,` + base + `}`, []string{"schema_version: must be 1, got 3"}},
		{"schema_version as string", `{"schema_version":"1",` + base + `}`, []string{"schema_version: required number"}},
		{"schema_version missing", `{` + base + `}`, []string{"schema_version: required number"}},
		{"target missing", `{"schema_version":1,"verdict":"pass"}`, []string{"target: required object"}},
		{"target not an object", `{"schema_version":1,"target":"r","verdict":"pass"}`, []string{"target: required object"}},
		{"repo empty", `{"schema_version":1,"target":{"repo":""},"verdict":"pass"}`, []string{"target.repo: required non-empty string"}},
		{"repo not a string", `{"schema_version":1,"target":{"repo":7},"verdict":"pass"}`, []string{"target.repo: required non-empty string"}},
		{"verdict unknown", `{"schema_version":1,"target":{"repo":"r"},"verdict":"maybe"}`, []string{"verdict: must be one of informational|pass|fail, got maybe"}},
		{"verdict missing", `{"schema_version":1,"target":{"repo":"r"}}`, []string{"verdict: must be one of informational|pass|fail, got <nil>"}},
		{"unknown field", `{"schema_version":1,` + base + `,"extra":true}`, []string{"unknown field: extra"}},
		{"not an object: array", `[1,2,3]`, []string{"candidate: must be an object"}},
		{"not an object: null", `null`, []string{"candidate: must be an object"}},
		{"not an object: string", `"x"`, []string{"candidate: must be an object"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			code, r, se := call(t, c.in)
			if code != 0 || se != "" {
				t.Fatalf("code=%d stderr=%q", code, se)
			}
			if r.Valid != (len(c.want) == 0) || !reflect.DeepEqual(r.Errors, c.want) {
				t.Fatalf("got valid=%v %q, want %q", r.Valid, r.Errors, c.want)
			}
		})
	}
}

// Go randomises map iteration; the sorted errors must not depend on it.
func TestErrorOrderIsStable(t *testing.T) {
	in := `{"zeta":1,"alpha":2,"mid":3,"schema_version":9,"target":{},"verdict":"x","beta":4}`
	want := []string{
		"schema_version: must be 1, got 9",
		"target.repo: required non-empty string",
		"unknown field: alpha",
		"unknown field: beta",
		"unknown field: mid",
		"unknown field: zeta",
		"verdict: must be one of informational|pass|fail, got x",
	}
	for i := 0; i < 200; i++ {
		_, r, _ := call(t, in)
		if !reflect.DeepEqual(r.Errors, want) {
			t.Fatalf("run %d: got %q, want %q", i, r.Errors, want)
		}
	}
}

// Input that is not exactly one JSON value exits 2, writes nothing to stdout and says why on stderr.
func TestRejectedInput(t *testing.T) {
	valid := `{"schema_version":1,"target":{"repo":"r"},"verdict":"pass"}`
	cases := []struct{ name, in, stderr string }{
		{"empty", "", "not JSON: EOF"},
		{"whitespace only", " \n\t ", "not JSON: EOF"},
		{"malformed", "nope", "not JSON: invalid character"},
		{"truncated object", `{"schema_version":1,`, "not JSON: unexpected EOF"},
		{"second JSON value", valid + ` {"x":1}`, "not JSON: trailing data after the first value"},
		{"trailing garbage", valid + `garbage`, "not JSON: trailing data after the first value"},
		{"two documents on two lines", valid + "\n" + valid + "\n", "not JSON: trailing data after the first value"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			code, _, se := call(t, c.in)
			if code != 2 || !strings.HasPrefix(se, c.stderr) {
				t.Fatalf("code=%d stderr=%q, want 2 and prefix %q", code, se, c.stderr)
			}
		})
	}
}

// Whitespace after the single value is not trailing data: the fixtures end with a newline.
func TestTrailingWhitespaceAccepted(t *testing.T) {
	valid := `{"schema_version":1,"target":{"repo":"r"},"verdict":"pass"}`
	for _, tail := range []string{"", "\n", "\r\n", "  \n\t"} {
		if code, r, se := call(t, valid+tail); code != 0 || !r.Valid || se != "" {
			t.Fatalf("tail %q: code=%d report=%+v stderr=%q", tail, code, r, se)
		}
	}
}
