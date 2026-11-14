// FIXTURE VALIDATOR — written for the ixql-wasi-validator prototype only.
//
// It is NOT the previously validated Go package referred to in the prototype's README: no such
// package exists in this repository. It checks a deliberately small, made-up "sandbox candidate"
// shape so the IXQL -> WASI -> Go path can be exercised end to end.
//
// Contract (one JSON value each way, the ix-wasm stage convention):
//
//	stdin : exactly one JSON value (the candidate), optionally followed by whitespace
//	stdout: {"validator": "...", "valid": bool, "errors": ["..."]}   exit 0
//	stderr: a message, exit 2, when stdin is not exactly one JSON value
//
// Standard library only: ix-wasm builds with GOPROXY=off, so no module can be fetched.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

const name = "ix-prototype-fixture-validator/0.1.0"

var allowedVerdicts = map[string]bool{"informational": true, "pass": true, "fail": true}

type report struct {
	Validator string   `json:"validator"`
	Valid     bool     `json:"valid"`
	Errors    []string `json:"errors"`
}

func validate(v any) []string {
	errs := []string{}
	obj, ok := v.(map[string]any)
	if !ok {
		return append(errs, "candidate: must be an object")
	}
	switch sv := obj["schema_version"].(type) {
	case float64:
		if sv != 1 {
			errs = append(errs, fmt.Sprintf("schema_version: must be 1, got %v", sv))
		}
	default:
		errs = append(errs, "schema_version: required number")
	}
	target, ok := obj["target"].(map[string]any)
	if !ok {
		errs = append(errs, "target: required object")
	} else if repo, ok := target["repo"].(string); !ok || repo == "" {
		errs = append(errs, "target.repo: required non-empty string")
	}
	if verdict, ok := obj["verdict"].(string); !ok || !allowedVerdicts[verdict] {
		errs = append(errs, fmt.Sprintf("verdict: must be one of informational|pass|fail, got %v", obj["verdict"]))
	}
	known := map[string]bool{"schema_version": true, "target": true, "verdict": true, "notes": true}
	for k := range obj {
		if !known[k] {
			errs = append(errs, "unknown field: "+k)
		}
	}
	sort.Strings(errs) // map iteration order must not leak into the output
	return errs
}

// run is main with its streams and exit code made explicit, so tests drive the real path.
func run(stdin io.Reader, stdout, stderr io.Writer) int {
	var v any
	dec := json.NewDecoder(stdin)
	if err := dec.Decode(&v); err != nil {
		io.WriteString(stderr, "not JSON: "+err.Error()+"\n")
		return 2
	}
	// Exactly one value: a Decoder stops after the first, so `{...} {...}` or `{...}junk` would
	// otherwise be judged on the first value alone. Trailing whitespace is fine.
	// @ai:invariant stdin holding anything but one JSON value plus whitespace exits 2 with no stdout [T:test conf:0.95 src:main_test.go:115]
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		io.WriteString(stderr, "not JSON: trailing data after the first value\n")
		return 2
	}
	errs := validate(v)
	json.NewEncoder(stdout).Encode(report{Validator: name, Valid: len(errs) == 0, Errors: errs})
	return 0
}

func main() { os.Exit(run(os.Stdin, os.Stdout, os.Stderr)) }
