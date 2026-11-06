package main

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// The digest hooks keep state/digests/latest.md: a nudge when it goes stale, a metadata stub
// after enough activity, at Stop and before a compaction, and its injection at SessionStart.
// Karpathy R1+R4, Cherny's session continuity (see .claude/skills/digest/SKILL.md).

type digestPaths struct {
	dir, archive, latest string
}

func digestsOf(root string) digestPaths {
	dir := filepath.Join(root, "state", "digests")
	return digestPaths{dir, filepath.Join(dir, "archive"), filepath.Join(dir, "latest.md")}
}

// safeYAML quotes an untrusted string as a single-quoted YAML scalar: newlines become spaces, at
// most max bytes, quotes doubled; "null" when empty. Closes the YAML-injection findings of the
// 2026-05-15 review.
func safeYAML(v string, max int) string {
	if v == "" {
		return "null"
	}
	c := strings.NewReplacer("\r", " ", "\n", " ").Replace(v)
	if len(c) > max {
		c = c[:max]
	}
	return "'" + strings.ReplaceAll(c, "'", "''") + "'"
}

var safeIDChars = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// safeID is v when, stripped of CR, LF and tabs and cut to max bytes, it is a plain identifier;
// fallback otherwise. Closes the path-traversal finding of the same review.
func safeID(v, fallback string, max int) string {
	if v == "" {
		return fallback
	}
	c := strings.NewReplacer("\r", "", "\n", "", "\t", "").Replace(v)
	if len(c) > max {
		c = c[:max]
	}
	if !safeIDChars.MatchString(c) {
		return fallback
	}
	return c
}

// counter reads a whole-number file, 0 when absent or not a number.
func counter(path string) int {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	n, err := strconv.Atoi(strings.Join(strings.Fields(string(raw)), ""))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

func (e *env) head(root string) (branch, sha, subject string) {
	return e.gitOr(root, "unknown", "rev-parse", "--abbrev-ref", "HEAD"),
		e.gitOr(root, "unknown", "rev-parse", "--short", "HEAD"),
		e.gitOr(root, "unknown", "log", "-1", "--format=%s")
}

// openPR is "#<n>" for the branch's pull request, "" without gh or without one.
func (e *env) openPR(root string) string {
	n, ok := e.sys.output(root, "gh", "pr", "view", "--json", "number", "-q", ".number")
	if !ok || n == "" {
		return ""
	}
	return "#" + n
}

// archiveLatest copies latest.md into the archive before a hook replaces it, when there is one.
// On an error the previous digest would be lost: the caller keeps latest.md as it is (the
// scripts' `set -e` stopped at a failed `cp` the same way).
func archiveLatest(e *env, d digestPaths, name, hook string) bool {
	data, err := os.ReadFile(d.latest)
	if errors.Is(err, fs.ErrNotExist) {
		return true
	}
	if err == nil {
		err = os.WriteFile(filepath.Join(d.archive, name), data, 0o644)
	}
	if err != nil {
		fmt.Fprintf(e.stderr, "[%s] could not archive latest.md, kept as it is: %v\n", hook, err)
		return false
	}
	return true
}

// replaceLatest writes latest.md through a temporary file and a rename, so a failed write never
// leaves it truncated; a failure keeps the previous latest.md and is said on stderr.
func replaceLatest(e *env, d digestPaths, body, hook string) {
	tmp, err := os.CreateTemp(d.dir, ".latest-*.tmp")
	if err == nil {
		_, err = tmp.WriteString(body)
		if cerr := tmp.Close(); err == nil {
			err = cerr
		}
		if err == nil {
			err = os.Rename(tmp.Name(), d.latest)
		}
		if err != nil {
			_ = os.Remove(tmp.Name()) // a leftover temporary file is harmless
		}
	}
	if err != nil {
		fmt.Fprintf(e.stderr, "[%s] could not write latest.md, kept as it is: %v\n", hook, err)
	}
}

// digestStalenessNudge (UserPromptSubmit) says when the digest is missing or has drifted;
// silent when it is fresh.
func digestStalenessNudge(e *env) int {
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	d := digestsOf(root)
	age, exists := e.ageMinutes(d.latest)
	mutations := counter(filepath.Join(d.dir, ".activity-counter"))

	var reason string
	switch {
	case !exists:
		reason = "No session digest exists yet. Invoke /digest at your next natural breakpoint."
	case age > 30 && mutations > 10:
		reason = fmt.Sprintf("Last digest %d min ago; %d mutations since. Karpathy R4: task complete != goal achieved — consider /digest to mark success criteria status.", age, mutations)
	case age > 90:
		reason = fmt.Sprintf("Last digest %d min ago. Session drifting — invoke /digest before the next compaction.", age)
	default:
		return 0
	}
	writeContext(e.stdout, "[digest-nudge] "+reason)
	return 0
}

// digestActivityTracker (PostToolUse on Edit|Write|Bash) counts mutations, and after
// IX_DIGEST_MID_COUNT of them (20), or IX_DIGEST_MID_MIN minutes (30) with at least 3, writes a
// metadata-only mid-session digest, so a crash or a network drop does not lose the session's
// place before Stop or PreCompact.
func digestActivityTracker(e *env) int {
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	d := digestsOf(root)
	_ = os.MkdirAll(d.archive, 0o755) // a failure surfaces in archiveLatest

	// The counters are best effort: a lost count only delays the next stub.
	total := counter(filepath.Join(d.dir, ".activity-counter")) + 1
	_ = os.WriteFile(filepath.Join(d.dir, ".activity-counter"), []byte(strconv.Itoa(total)+"\n"), 0o644)
	midPath := filepath.Join(d.dir, ".activity-count")
	mid := counter(midPath) + 1
	_ = os.WriteFile(midPath, []byte(strconv.Itoa(mid)+"\n"), 0o644)

	thresholdCount := envInt(e, "IX_DIGEST_MID_COUNT", 20)
	thresholdMin := envInt(e, "IX_DIGEST_MID_MIN", 30)
	ageMin := int64(99999)
	if a, ok := e.ageMinutes(d.latest); ok {
		ageMin = a
	}
	if !(mid >= thresholdCount || (ageMin >= int64(thresholdMin) && mid >= 3)) {
		return 0
	}
	_ = os.WriteFile(midPath, []byte("0\n"), 0o644)

	now := e.sys.now().UTC()
	tsIso, tsFile := now.Format("2006-01-02T15:04:05Z"), now.Format("2006-01-02T15-04-05Z")
	if !archiveLatest(e, d, tsFile+"-pre-mid.md", "digest-activity-tracker") {
		return 0
	}
	branch, sha, subject := e.head(root)
	body := fmt.Sprintf(`---
schema_version: 1
session_id: mid-session-auto
written_at: %s
trigger: activity-tracker-mid-session
branch: %s
head_sha: %s
head_subject: %s
mutations_since_last: %d
---

# Session digest (mid-session auto — activity threshold reached)

**Branch:** %s @ %s — %s

## Model-driven sections

_Auto-written by digest-activity-tracker after %d mutations / %dm since last digest.
Invoke `+"`/digest`"+` at your next natural breakpoint to populate **Next action**,
**In-flight**, **Live hypotheses**, **Open questions**, **Do NOT carry forward**,
and **Success criteria**._
`, tsIso, branch, sha, subject, mid, branch, sha, subject, mid, ageMin)
	_ = os.WriteFile(filepath.Join(d.dir, "mid-"+tsFile+".md"), []byte(body), 0o644) // a copy of latest.md
	replaceLatest(e, d, body, "digest-activity-tracker")
	return 0
}

func envInt(e *env, name string, def int) int {
	if n, err := strconv.Atoi(e.sys.getenv(name)); err == nil {
		return n
	}
	return def
}

// digestStopFinalize (Stop) writes a metadata digest when /digest has not written one in the
// last 10 minutes, archiving the previous one first.
func digestStopFinalize(e *env) int {
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	d := digestsOf(root)
	if age, ok := e.ageMinutes(d.latest); ok && age < 10 {
		return 0
	}
	_ = os.MkdirAll(d.archive, 0o755) // a failure surfaces in archiveLatest or replaceLatest
	now := e.sys.now().UTC()
	tsIso, tsFile := now.Format("2006-01-02T15:04:05Z"), now.Format("2006-01-02T15-04-05Z")
	if !archiveLatest(e, d, tsFile+"-stop.md", "digest-stop-finalize") {
		return 0
	}
	branch, sha, subject := e.head(root)
	pr := e.openPR(root)
	prLine := ""
	if pr != "" {
		prLine = "**Open PR:** " + pr + "\n"
	}
	body := fmt.Sprintf(`---
schema_version: 1
session_id: stop-finalize
written_at: %s
trigger: stop-hook-finalize
branch: %s
head_sha: %s
head_subject: %s
open_pr: %s
---

# Session digest (Stop-hook finalize — /digest not invoked in last 10 min)

**Branch:** %s @ %s — %s
%s
## Model-driven sections

_Session ended without a recent `+"`/digest`"+`. Next session: re-orient from
`+"`git log`"+` + open PR. Prior digests (if any) are in `+"`state/digests/archive/`"+`._
`, tsIso, safeYAML(branch, 200), safeYAML(sha, 200), safeYAML(subject, 200), safeYAML(pr, 200),
		branch, sha, subject, prLine)
	replaceLatest(e, d, body, "digest-stop-finalize")
	validateDigest(d.latest, discard{}) // the script ran digest-validate and ignored its verdict
	return 0
}

// precompactDigest (PreCompact) archives latest.md under the session's id and, when /digest has
// not written it in the last 30 minutes, replaces it with a metadata-only fallback.
func precompactDigest(e *env) int {
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	d := digestsOf(root)
	_ = os.MkdirAll(d.archive, 0o755) // a failure surfaces in archiveLatest or replaceLatest
	p, _ := e.payload()
	session := field(p, "session_id")
	if session == "" {
		session = "unknown"
	}
	now := e.sys.now().UTC()
	ts, tsFile := now.Format("2006-01-02T15:04:05Z"), now.Format("2006-01-02T15-04-05Z")
	if !archiveLatest(e, d, tsFile+"-"+safeID(session, "unknown", 64)+".md", "precompact-digest") {
		return 0
	}
	if _, err := os.Stat(d.latest); err == nil {
		if age, _ := e.ageMinutes(d.latest); age < 30 {
			return 0
		}
	}
	branch, sha, subject := e.head(root)
	pr := e.openPR(root)
	prLine := ""
	if pr != "" {
		prLine = "**Open PR:** " + pr + "\n"
	}
	body := fmt.Sprintf(`---
schema_version: 1
session_id: %s
written_at: %s
trigger: precompact-hook-fallback
branch: %s
head_sha: %s
head_subject: %s
open_pr: %s
---

# Session digest (fallback — /digest was not invoked before compaction)

**Branch:** %s @ %s — %s
%s
## Model-driven sections

_No `+"`/digest`"+` invocation was captured before this compaction. Re-orient from
`+"`git log`"+` and the open PR. Invoke `+"`/digest`"+` mid-session to populate the
**Next action**, **In-flight**, **Live hypotheses**, **Open questions**, and
**Do NOT carry forward** sections before the next compaction event._
`, safeYAML(session, 200), ts, safeYAML(branch, 200), safeYAML(sha, 200), safeYAML(subject, 200), safeYAML(pr, 200),
		branch, sha, subject, prLine)
	replaceLatest(e, d, body, "precompact-digest")
	return 0
}

// digestValidate checks a digest's frontmatter against docs/contracts/digest-schema.json's
// required fields (Karpathy R11). Argument: the digest path, state/digests/latest.md by default.
// Exit 1 with the reason on stderr when it does not conform.
func digestValidate(e *env) int {
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	path := digestsOf(root).latest
	if len(e.args) > 0 && e.args[0] != "" {
		path = e.args[0]
	}
	if _, err := os.Stat(path); err != nil {
		return 0
	}
	return validateDigest(path, e.stderr)
}

type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }

var validTriggers = map[string]bool{
	"digest-skill": true, "precompact-hook-fallback": true, "stop-hook-finalize": true, "auto-write-routine": true,
}

func validateDigest(path string, stderr interface{ Write([]byte) (int, error) }) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	// awk '/^---$/{c++; next} c==1{print}': the lines between the first two `---`.
	var fm []string
	c := 0
	for _, l := range strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n") {
		if l == "---" {
			c++
			continue
		}
		if c == 1 {
			fm = append(fm, l)
		}
	}
	if len(strings.TrimSpace(strings.Join(fm, ""))) == 0 {
		fmt.Fprintf(stderr, "digest-validate: missing or malformed YAML frontmatter in %s\n", path)
		return 1
	}
	value := func(k string) (string, bool) {
		re := regexp.MustCompile(`^` + k + `:[[:space:]]*`)
		for _, l := range fm {
			if loc := re.FindStringIndex(l); loc != nil {
				return l[loc[1]:], true
			}
		}
		return "", false
	}
	for _, k := range []string{"schema_version", "session_id", "written_at", "trigger", "branch", "head_sha", "head_subject"} {
		v, ok := value(k)
		if !ok || strings.TrimSpace(v) == "" || v[0] == ' ' || v[0] == '\t' {
			fmt.Fprintf(stderr, "digest-validate: required field '%s' missing or null\n", k)
			return 1
		}
	}
	if sv, _ := value("schema_version"); strings.Join(strings.Fields(sv), "") != "1" {
		fmt.Fprintf(stderr, "digest-validate: schema_version must be 1 (got '%s')\n", strings.Join(strings.Fields(sv), ""))
		return 1
	}
	if tr, _ := value("trigger"); !validTriggers[strings.Join(strings.Fields(tr), "")] {
		fmt.Fprintf(stderr, "digest-validate: trigger '%s' not valid\n", strings.Join(strings.Fields(tr), ""))
		return 1
	}
	return 0
}
