// FIXTURE VALIDATOR — written for the ixql-wasi-validator prototype only.
//
// It is NOT the previously validated Go package referred to in the prototype's README: no such
// package exists in this repository. It checks a deliberately small, made-up "sandbox candidate"
// shape so the IXQL -> WASI -> Go path can be exercised end to end.
//
// Contract (one JSON value each way, the ix-wasm stage convention):
//
//	stdin : any JSON value (the candidate)
//	stdout: {"validator": "...", "valid": bool, "errors": ["..."]}   exit 0
//	stderr: a message, exit 2, when stdin is not JSON at all
//
// Standard library only: ix-wasm builds with GOPROXY=off, so no module can be fetched.
package main

import (
	"encoding/json"
	"fmt"
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

func main() {
	var v any
	if err := json.NewDecoder(os.Stdin).Decode(&v); err != nil {
		os.Stderr.WriteString("not JSON: " + err.Error() + "\n")
		os.Exit(2)
	}
	errs := validate(v)
	json.NewEncoder(os.Stdout).Encode(report{Validator: name, Valid: len(errs) == 0, Errors: errs})
}
