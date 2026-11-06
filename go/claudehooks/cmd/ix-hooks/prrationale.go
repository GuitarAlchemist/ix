package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// prRationaleCapture (PostToolUse on Bash) snapshots a `gh pr create`'s title, body and diff
// stats to state/digests/pr-<num>-<slug>.md, so the rationale survives later edits of the pull
// request (was pr-rationale-capture.sh; Cherny's PR rationale capture).
func prRationaleCapture(e *env) int {
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	p, raw := e.payload()
	if p == nil || len(raw) == 0 {
		return 0
	}
	cmd := field(p, "tool_input", "command")
	if !ghPRCreate.MatchString(cmd) {
		return 0
	}
	output := toolOutput(p["tool_response"])

	title := quotedFlag(cmd, "title")
	body := quotedFlag(cmd, "body")
	num := "unknown"
	if m := prURL.FindStringSubmatch(output); m != nil {
		num = m[1]
	}
	slug := prSlug(title)

	dir := digestsOf(root).dir
	os.MkdirAll(dir, 0o755)
	ts := e.sys.now().UTC().Format("2006-01-02T15:04:05Z")
	branch := e.gitOr(root, "unknown", "rev-parse", "--abbrev-ref", "HEAD")
	shortStat, _ := e.sys.output("", "git", "-C", root, "diff", "--shortstat", "HEAD~1")
	if shortStat != "" {
		shortStat = " " + shortStat // git prints it with a leading space; the script kept it
	}
	doc := fmt.Sprintf(`---
schema_version: 1
trigger: pr-rationale-capture
captured_at: %s
branch: %s
pr_number: %s
diff_shortstat:%s
---

# PR #%s — %s

**Branch:** %s
**Captured:** %s
**Diff:**%s

## Title

%s

## Body

%s
`, ts, branch, num, shortStat, num, title, branch, ts, shortStat, title, body)
	os.WriteFile(filepath.Join(dir, "pr-"+num+"-"+slug+".md"), []byte(doc), 0o644)
	return 0
}

var (
	ghPRCreate = regexp.MustCompile(`gh[[:space:]]+pr[[:space:]]+create`)
	prURL      = regexp.MustCompile(`pull/([0-9]+)`)
	nonSlug    = regexp.MustCompile(`[^a-z0-9]+`)
)

// toolOutput is jq's `.tool_response.output // .tool_response // empty`, raw.
func toolOutput(v any) string {
	if m, ok := v.(map[string]any); ok {
		if s, ok := m["output"].(string); ok && s != "" {
			return s
		}
		if out, ok := m["output"]; ok && out != nil && out != false {
			b, _ := json.Marshal(out)
			return string(b)
		}
	}
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	}
	b, _ := json.Marshal(v)
	return string(b)
}

// quotedFlag is the first `--<name> "…"` or `--<name> '…'` value in cmd, without its quotes.
func quotedFlag(cmd, name string) string {
	re := regexp.MustCompile(`--` + name + `[[:space:]]+("[^"]*"|'[^']*')`)
	m := re.FindStringSubmatch(cmd)
	if m == nil {
		return ""
	}
	return m[1][1 : len(m[1])-1]
}

// prSlug: lowercased, every run of other characters one dash, no dash at either end, at most
// 40 bytes; "untitled" when nothing is left.
func prSlug(title string) string {
	s := strings.Trim(nonSlug.ReplaceAllString(strings.ToLower(title), "-"), "-")
	if len(s) > 40 {
		s = s[:40]
	}
	if s == "" {
		return "untitled"
	}
	return s
}
