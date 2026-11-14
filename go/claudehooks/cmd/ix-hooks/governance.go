package main

import (
	"fmt"
	"regexp"
)

// governanceCheck is the Demerzel governance PreToolUse hook (was governance-check.sh).
//
// Contract (code.claude.com/docs/en/hooks): event data arrives as JSON on stdin
// ({tool_name, tool_input:{command|file_path,...}}); BLOCK = exit 2 (stderr is fed back to Claude
// and the tool call is prevented); WARN = stderr + exit 0. References: Demerzel Constitution v2.0.0.
//
// The patterns are the script's bash `=~` regexes; `(?s)` because a POSIX `.` crosses newlines.
func governanceCheck(e *env) int {
	p, _ := e.payload()
	tool := field(p, "tool_name")
	command := field(p, "tool_input", "command")
	path := field(p, "tool_input", "file_path")

	// Fail open with a visible warning when parsing yielded nothing: never block every tool call
	// because the harness or the JSON decoder changed under us.
	if tool == "" {
		fmt.Fprintln(e.stderr, "[demerzel] WARN: governance hook received no parseable tool_name on stdin — rules not evaluated.")
		return 0
	}

	if tool == "Bash" {
		for _, b := range governanceBlocks {
			if anyMatch(b.patterns, command) {
				fmt.Fprintln(e.stderr, b.msg)
				return 2
			}
		}
		for _, w := range governanceWarns {
			if w.pattern.MatchString(command) {
				fmt.Fprintln(e.stderr, w.msg)
			}
		}
	}

	// Article 4: Proportionality — warn on sensitive file writes.
	if tool == "Write" && sensitivePath.MatchString(path) {
		fmt.Fprintln(e.stderr, "[demerzel] WARN: Article 4 (Proportionality) — writing to sensitive file.")
	}
	return 0
}

type govBlock struct {
	patterns []*regexp.Regexp
	msg      string
}

type govWarn struct {
	pattern *regexp.Regexp
	msg     string
}

func res(ps ...string) []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(ps))
	for i, p := range ps {
		out[i] = regexp.MustCompile(`(?s)` + p)
	}
	return out
}

func anyMatch(rs []*regexp.Regexp, s string) bool {
	for _, r := range rs {
		if r.MatchString(s) {
			return true
		}
	}
	return false
}

// Article 3 blocks are anchored to root ("/", "/*") or a bare wildcard: a plain substring match
// on "rm -rf /" would block every absolute path (rm -rf /tmp/build-cache), which the warning
// below already covers.
var governanceBlocks = []govBlock{
	{res(`rm -rf +/+([[:space:]]|$)`, `rm -rf +/+\*`, `rm -rf +\*`),
		"[demerzel] BLOCKED: Article 3 (Reversibility) — recursive delete of root or wildcard is catastrophic."},
	{res(`git push --force.*(main|master)`, `git push -f.*(main|master)`),
		"[demerzel] BLOCKED: Article 3 (Reversibility) — force push to main/master destroys shared history."},
	{res(`DROP DATABASE`, `drop database`),
		"[demerzel] BLOCKED: Article 3 (Reversibility) — DROP DATABASE is irreversible."},
	{res(`chmod 777 /`),
		"[demerzel] BLOCKED: Article 9 (Bounded Autonomy) — chmod 777 on root is unsafe."},
}

var governanceWarns = []govWarn{
	{res(`(rm -rf|git reset --hard|git clean -fd)`)[0],
		"[demerzel] WARN: Article 3 (Reversibility) — destructive command detected. Confirm before proceeding."},
	{res(`(git push --force|git push -f)`)[0],
		"[demerzel] WARN: Article 3 (Reversibility) — force push detected. Verify target branch."},
	{res(`(chmod 777|sudo|chown root|--no-verify)`)[0],
		"[demerzel] WARN: Article 9 (Bounded Autonomy) — permission escalation detected."},
	{res(`(DROP TABLE|TRUNCATE|DELETE FROM.*WHERE 1)`)[0],
		"[demerzel] WARN: Article 3 (Reversibility) — destructive SQL detected."},
}

var sensitivePath = regexp.MustCompile(`(/etc/|\.env$|credentials|secret|\.pem$|\.key$)`)
