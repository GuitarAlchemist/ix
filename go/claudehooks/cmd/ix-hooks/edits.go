package main

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

// pipelineValidate warns when a Write or Edit leaves a pipeline file whose node seems to take
// itself as input (was pipeline-validate.sh, PreToolUse on Write|Edit). It never blocks.
//
// The script grepped each line for `\.input\("[^"]+",\s*"([^"]+)"\).*\.node\("\1"`; RE2 has no
// back-reference, so the capture is matched first and the `.node("<same>"` looked for after it.
func pipelineValidate(e *env) int {
	p, _ := e.payload()
	tool := field(p, "tool_name")
	path := field(p, "tool_input", "file_path")
	if (tool != "Write" && tool != "Edit") || !strings.Contains(path, "pipeline") {
		return 0
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if selfReferencingNode(line) {
			fmt.Fprintln(e.stderr, "[ix-pipeline] WARNING: Possible self-referencing node detected")
			break
		}
	}
	return 0
}

var pipelineInput = regexp.MustCompile(`\.input\("[^"]+",\s*"([^"]+)"\)`)

func selfReferencingNode(line string) bool {
	for _, m := range pipelineInput.FindAllStringSubmatchIndex(line, -1) {
		name := line[m[2]:m[3]]
		if strings.Contains(line[m[1]:], `.node("`+name+`"`) {
			return true
		}
	}
	return false
}

// rustCheck runs `cargo check -p <crate>` after a Write or Edit of a .rs file under crates/<crate>/
// and prints the first error and warning lines when it fails (was rust-check.sh, PostToolUse on
// Write|Edit). It never blocks.
//
// The crate is read from a forward-slash path, as the script did: a Windows path with
// backslashes names no crate and the hook stays silent there, so it does not start a cargo
// build after every edit on this machine (one heavy build at a time).
func rustCheck(e *env) int {
	p, _ := e.payload()
	tool := field(p, "tool_name")
	path := field(p, "tool_input", "file_path")
	if tool != "Write" && tool != "Edit" {
		return 0
	}
	if !strings.HasSuffix(path, ".rs") {
		return 0
	}
	m := crateOfPath.FindStringSubmatch(path)
	if m == nil {
		return 0
	}
	crate := m[1]
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	out, ok := e.sys.combined(root, "cargo", "check", "-p", crate, "--message-format=short")
	if !ok {
		fmt.Fprintf(e.stderr, "[rust-check] cargo check -p %s failed:\n", crate)
		n := 0
		for _, l := range strings.Split(out, "\n") {
			if n < 10 && (strings.HasPrefix(l, "error") || strings.HasPrefix(l, "warning")) {
				fmt.Fprintln(e.stderr, strings.TrimRight(l, "\r"))
				n++
			}
		}
	}
	return 0
}

var crateOfPath = regexp.MustCompile(`/crates/([^/]+)/`)
