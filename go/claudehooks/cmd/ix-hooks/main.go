// ix-hooks is every Claude Code hook of this repository, one subcommand per hook. It replaces the
// bash scripts that lived in .claude/hooks/ (the user, 2026-11-05: "je veux jamais de shell dans
// les hooks"): under Git Bash on Windows each of them cost seconds per tool call, crashed with a
// bash.exe.stackdump, or silently lost its JSON parser (python3 resolving to the WindowsApps alias).
//
// .claude/settings.json runs `session-start` through `go run` (Go caches the built executable),
// and that subcommand installs the binary as .claude/hooks/bin/ix-hooks[.exe], which every other
// hook calls directly. The bin directory is gitignored.
//
// Each subcommand keeps its script's contract: the same inputs, the same exit codes, the same
// files written, the same text on stdout and stderr.
package main

import (
	"fmt"
	"io"
	"os"
	"sort"
)

// env is what a hook may touch: its streams, the clock, the environment and the commands it
// runs. Tests replace it.
type env struct {
	stdin  io.Reader
	stdout io.Writer
	stderr io.Writer
	args   []string
	sys    *system
}

var hooks = map[string]func(*env) int{
	"governance-check":        governanceCheck,
	"pipeline-validate":       pipelineValidate,
	"rust-check":              rustCheck,
	"correction-detector":     correctionDetector,
	"digest-staleness-nudge":  digestStalenessNudge,
	"digest-activity-tracker": digestActivityTracker,
	"digest-stop-finalize":    digestStopFinalize,
	"digest-validate":         digestValidate,
	"precompact-digest":       precompactDigest,
	"sessionstart-digest":     sessionstartDigest,
	"session-start":           sessionStart,
	"pr-rationale-capture":    prRationaleCapture,
}

func main() {
	os.Exit(dispatch(os.Args[1:], &env{stdin: os.Stdin, stdout: os.Stdout, stderr: os.Stderr, sys: realSystem()}))
}

func dispatch(args []string, e *env) int {
	if len(args) == 0 || hooks[args[0]] == nil {
		names := make([]string, 0, len(hooks))
		for n := range hooks {
			names = append(names, n)
		}
		sort.Strings(names)
		fmt.Fprintf(e.stderr, "usage: ix-hooks <hook> [args]\nhooks: %v\n", names)
		return 1 // never 2: a misnamed PreToolUse hook must not block every tool call
	}
	e.args = args[1:]
	return hooks[args[0]](e)
}
