package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeSystem: a repository at root (git answers from a table), no gh, no network, a fixed clock.
func fakeSystem(root string, now time.Time) *system {
	return &system{
		now:    func() time.Time { return now },
		getenv: func(string) string { return "" },
		output: func(dir, name string, args ...string) (string, bool) {
			key := name + " " + strings.Join(args, " ")
			switch {
			case strings.HasSuffix(key, "rev-parse --abbrev-ref HEAD"):
				return "feat/x", true
			case strings.HasSuffix(key, "rev-parse --short HEAD"):
				return "abc1234", true
			case strings.HasSuffix(key, "log -1 --format=%s"):
				return "feat: it's done", true
			}
			return "", false
		},
		combined:   func(string, string, ...string) (string, bool) { return "", true },
		fetch:      func(string, time.Duration) ([]byte, error) { return nil, errors.New("offline") },
		executable: func() (string, error) { return "", errors.New("no executable in tests") },
		toplevel:   func() (string, bool) { return root, root != "" },
	}
}

// The repository root is found without starting git, from a nested directory, for a checkout
// (.git directory) and a worktree (.git file).
func TestFindTopLevel(t *testing.T) {
	for _, worktree := range []bool{false, true} {
		root := t.TempDir()
		if worktree {
			os.WriteFile(filepath.Join(root, ".git"), []byte("gitdir: elsewhere\n"), 0o644)
		} else {
			os.Mkdir(filepath.Join(root, ".git"), 0o755)
		}
		nested := filepath.Join(root, "a", "b")
		os.MkdirAll(nested, 0o755)
		for _, from := range []string{root, nested} {
			got, ok := findTopLevel(from)
			if !ok || got != filepath.ToSlash(root) {
				t.Errorf("worktree=%v from %s: got %q %v, want %q", worktree, from, got, ok, filepath.ToSlash(root))
			}
		}
	}
}

func runHook(t *testing.T, sys *system, name, stdin string, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := dispatch(append([]string{name}, args...), &env{stdin: strings.NewReader(stdin), stdout: &out, stderr: &errb, sys: sys})
	return code, out.String(), errb.String()
}

func bashPayload(command string) string {
	b, _ := json.Marshal(map[string]any{"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": map[string]any{"command": command}})
	return string(b)
}

// The cases of the former .claude/hooks/test-governance-check.sh (ix#228 P0-1), plus the ones it lacked.
func TestGovernanceCheck(t *testing.T) {
	sys := fakeSystem("", time.Now())
	for _, c := range []struct {
		desc, stdin string
		want        int
		warn        string
	}{
		{"rm -rf / is blocked", bashPayload("rm -rf /"), 2, "BLOCKED: Article 3"},
		{"ls is allowed", bashPayload("ls -la"), 0, ""},
		{"rm -rf /tmp/build-cache is allowed (warn only)", bashPayload("rm -rf /tmp/build-cache"), 0, "destructive command"},
		{"rm -rf /* is blocked", bashPayload("rm -rf /*"), 2, "recursive delete"},
		{"rm -rf * is blocked", bashPayload("rm -rf *"), 2, "recursive delete"},
		{"git push --force origin main is blocked", bashPayload("git push --force origin main"), 2, "force push to main"},
		{"force push to main across a newline is blocked", bashPayload("git push -f origin\nmain"), 2, "force push to main"},
		{"DROP DATABASE is blocked", bashPayload(`psql -c "DROP DATABASE prod"`), 2, "DROP DATABASE"},
		{"chmod 777 / is blocked", bashPayload("chmod 777 /"), 2, "Article 9"},
		{"git reset --hard warns without blocking", bashPayload("git reset --hard HEAD~1"), 0, "destructive command"},
		{"--no-verify warns", bashPayload("git commit --no-verify"), 0, "permission escalation"},
		{"Write to .env warns without blocking", `{"tool_name":"Write","tool_input":{"file_path":"/app/.env"}}`, 0, "sensitive file"},
		{"unparseable stdin fails open", "not json at all", 0, "rules not evaluated"},
	} {
		code, _, errs := runHook(t, sys, "governance-check", c.stdin)
		if code != c.want || (c.warn != "" && !strings.Contains(errs, c.warn)) || (c.warn == "" && errs != "") {
			t.Errorf("%s: exit %d (want %d), stderr %q (want %q)", c.desc, code, c.want, errs, c.warn)
		}
	}
}

// The cases of the former .claude/hooks/test-correction-detector.sh.
func TestCorrectionDetector(t *testing.T) {
	sys := fakeSystem("", time.Now())
	fires := map[string]string{
		"non, pas comme ça":          "Non, pas comme ça, on avait dit PowerShell.",
		"pas ça":                     "pas ça, je veux le worktree",
		"arrête":                     "Arrête de faire ça à chaque fois.",
		"arrete (no accent)":         "arrete, on relit le fichier d'abord",
		"plutôt":                     "Plutôt la commande slash rétro, pas celle-là.",
		"attends":                    "Attends, ce n'est pas ce que je t'ai demandé.",
		"c'est faux":                 "C'est faux, le port est 5176.",
		"on en a déjà parlé (mid)":   "Le fichier est bon mais on en a déjà parlé : jamais de stash.",
		"je t'ai déjà dit (mid)":     "Bon, je t'ai déjà dit de ne pas toucher au main tree.",
		"on avait dit (mid)":         "Pour le commit, on avait dit des chemins explicites.",
		"no, don't":                  "No, don't amend after CI passed.",
		"stop":                       "Stop rewriting the README.",
		"actually":                   "Actually we discussed this before.",
		"second correction in a row": "Non, encore raté.",
		"opener on a later line":     "Merci.\nnon, pas ce fichier", // grep was line-based: ^ is any line's start
	}
	for desc, prompt := range fires {
		in, _ := json.Marshal(map[string]string{"prompt": prompt})
		code, out, _ := runHook(t, sys, "correction-detector", string(in))
		if code != 0 || !strings.Contains(out, "/correct") || !strings.Contains(out, "teaching line") {
			t.Errorf("fires: %s: exit %d, out %q", desc, code, out)
		}
		var v map[string]string
		if json.Unmarshal([]byte(out), &v) != nil || !strings.HasPrefix(v["additionalContext"], "[correction-nudge] ") {
			t.Errorf("fires: %s: not the {additionalContext} JSON: %q", desc, out)
		}
	}
	silent := map[string]string{
		"a request":                `{"prompt": "Peux-tu lancer les tests sur Windows ?"}`,
		"a status question":        `{"prompt": "On en est où du développement ?"}`,
		"a word that contains non": `{"prompt": "Le nonce est bon, continue."}`,
		"no as part of a word":     `{"prompt": "Nothing to change, go on."}`,
		"empty prompt":             `{"prompt": ""}`,
		"empty stdin":              ``,
		"user_prompt is read too":  `{"user_prompt": "Le build passe."}`,
	}
	for desc, in := range silent {
		if code, out, _ := runHook(t, sys, "correction-detector", in); code != 0 || out != "" {
			t.Errorf("silent: %s: exit %d, out %q", desc, code, out)
		}
	}
	if _, out, _ := runHook(t, sys, "correction-detector", "non, raw text"); !strings.Contains(out, "/correct") {
		t.Error("raw (non-JSON) stdin is read as the prompt")
	}
}

func writeDigest(t *testing.T, root, trigger string, age time.Duration, now time.Time) string {
	t.Helper()
	d := digestsOf(root)
	os.MkdirAll(d.dir, 0o755)
	os.WriteFile(d.latest, []byte("---\nschema_version: 1\ntrigger: "+trigger+"\n---\n\nDIGEST-MARKER\n"), 0o644)
	mt := now.Add(-age)
	os.Chtimes(d.latest, mt, mt)
	return d.latest
}

// The cases of the former .claude/hooks/test-sessionstart-digest.sh.
func TestSessionstartDigest(t *testing.T) {
	now := time.Now()
	for _, c := range []struct {
		desc, trigger, source    string
		wantManifest, wantDigest bool
	}{
		{"startup injects the manifest and a written digest", "manual", "startup", true, true},
		{"resume injects the manifest", "manual", "resume", true, true},
		{"an empty payload is a startup", "manual", "", true, true},
		{"compact injects no manifest, keeps the digest", "manual", "compact", false, true},
		{"compact skips a tracker stub", "activity-tracker-mid-session", "compact", false, false},
		{"startup skips a tracker stub", "activity-tracker-mid-session", "startup", true, false},
		{"compact skips a precompact stub", "precompact-hook-fallback", "compact", false, false},
	} {
		root := t.TempDir()
		os.MkdirAll(filepath.Join(root, ".claude"), 0o755)
		os.WriteFile(filepath.Join(root, ".claude", "manifest-bootstrap.md"), []byte("MANIFEST-MARKER\n"), 0o644)
		writeDigest(t, root, c.trigger, time.Minute, now)
		payload := "{}"
		if c.source != "" {
			payload = `{"source": "` + c.source + `"}`
		}
		code, out, _ := runHook(t, fakeSystem(root, now), "sessionstart-digest", payload)
		m, d := strings.Contains(out, "MANIFEST-MARKER"), strings.Contains(out, "DIGEST-MARKER")
		if code != 0 || m != c.wantManifest || d != c.wantDigest {
			t.Errorf("%s: exit %d, manifest %v want %v, digest %v want %v", c.desc, code, m, c.wantManifest, d, c.wantDigest)
		}
		if strings.Contains(c.trigger, "-") && c.trigger != "manual" && !strings.Contains(out, "automatic stub") {
			t.Errorf("%s: a skipped stub is not named in one line: %q", c.desc, out)
		}
	}

	root := t.TempDir()
	writeDigest(t, root, "digest-skill", 25*time.Hour, now)
	if _, out, _ := runHook(t, fakeSystem(root, now), "sessionstart-digest", `{"source":"compact"}`); strings.Contains(out, "DIGEST-MARKER") || !strings.Contains(out, ">24h old") {
		t.Errorf("a day-old digest is injected: %q", out)
	}
	writeDigest(t, root, "digest-skill", 90*time.Minute, now)
	if _, out, _ := runHook(t, fakeSystem(root, now), "sessionstart-digest", `{"source":"compact"}`); !strings.Contains(out, "last written 1h 30m ago") {
		t.Errorf("age line: %q", out)
	}
}

func TestSessionStartInstallsTheBinary(t *testing.T) {
	root := t.TempDir()
	exe := filepath.Join(t.TempDir(), "built")
	os.WriteFile(exe, []byte("v1"), 0o755)
	sys := fakeSystem(root, time.Now())
	sys.executable = func() (string, error) { return exe, nil }
	runHook(t, sys, "session-start", "{}")
	if got, _ := os.ReadFile(binPath(root)); string(got) != "v1" {
		t.Fatalf("installed %q", got)
	}
	os.WriteFile(exe, []byte("v2"), 0o755)
	runHook(t, sys, "session-start", "{}")
	if got, _ := os.ReadFile(binPath(root)); string(got) != "v2" {
		t.Fatalf("not refreshed: %q", got)
	}
	left, _ := filepath.Glob(filepath.Join(filepath.Dir(binPath(root)), ".ix-hooks-*.tmp"))
	if len(left) != 0 {
		t.Fatalf("temporary files left: %v", left)
	}
}

func TestDigestStalenessNudge(t *testing.T) {
	now := time.Now()
	root := t.TempDir()
	if _, out, _ := runHook(t, fakeSystem(root, now), "digest-staleness-nudge", ""); !strings.Contains(out, "No session digest exists yet") {
		t.Errorf("missing digest: %q", out)
	}
	writeDigest(t, root, "digest-skill", 5*time.Minute, now)
	if _, out, _ := runHook(t, fakeSystem(root, now), "digest-staleness-nudge", ""); out != "" {
		t.Errorf("fresh digest nudged: %q", out)
	}
	writeDigest(t, root, "digest-skill", 45*time.Minute, now)
	os.WriteFile(filepath.Join(root, "state", "digests", ".activity-counter"), []byte("11\n"), 0o644)
	if _, out, _ := runHook(t, fakeSystem(root, now), "digest-staleness-nudge", ""); !strings.Contains(out, "Last digest 45 min ago; 11 mutations since") || !strings.Contains(out, "!=") {
		t.Errorf("drifted digest: %q", out)
	}
	writeDigest(t, root, "digest-skill", 100*time.Minute, now)
	os.WriteFile(filepath.Join(root, "state", "digests", ".activity-counter"), []byte("x"), 0o644)
	if _, out, _ := runHook(t, fakeSystem(root, now), "digest-staleness-nudge", ""); !strings.Contains(out, "Session drifting") {
		t.Errorf("old digest: %q", out)
	}
	if _, out, _ := runHook(t, fakeSystem("", now), "digest-staleness-nudge", ""); out != "" {
		t.Errorf("outside a repository: %q", out)
	}
}

func TestDigestActivityTrackerWritesAStubAtTheThreshold(t *testing.T) {
	now := time.Date(2026, 11, 5, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	writeDigest(t, root, "digest-skill", time.Minute, now)
	sys := fakeSystem(root, now)
	for i := 0; i < 19; i++ {
		runHook(t, sys, "digest-activity-tracker", "")
	}
	d := digestsOf(root)
	if data, _ := os.ReadFile(d.latest); !strings.Contains(string(data), "DIGEST-MARKER") {
		t.Fatal("stub written before the threshold")
	}
	runHook(t, sys, "digest-activity-tracker", "")
	data, _ := os.ReadFile(d.latest)
	if !strings.Contains(string(data), "trigger: activity-tracker-mid-session") || !strings.Contains(string(data), "mutations_since_last: 20") || !strings.Contains(string(data), "feat/x @ abc1234 — feat: it's done") {
		t.Fatalf("stub: %s", data)
	}
	if counter(filepath.Join(d.dir, ".activity-count")) != 0 || counter(filepath.Join(d.dir, ".activity-counter")) != 20 {
		t.Fatal("counters not reset/kept")
	}
	if _, err := os.Stat(filepath.Join(d.archive, "2026-11-05T12-00-00Z-pre-mid.md")); err != nil {
		t.Fatal("previous digest not archived")
	}
}

func TestStopAndPrecompactDigestsAreValidAndQuoted(t *testing.T) {
	now := time.Date(2026, 11, 5, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	sys := fakeSystem(root, now)
	runHook(t, sys, "digest-stop-finalize", "")
	d := digestsOf(root)
	data, _ := os.ReadFile(d.latest)
	if !strings.Contains(string(data), "head_subject: 'feat: it''s done'") || !strings.Contains(string(data), "open_pr: null") {
		t.Fatalf("stop digest: %s", data)
	}
	if code, _, errs := runHook(t, sys, "digest-validate", ""); code != 0 {
		t.Fatalf("stop digest invalid: %s", errs)
	}
	// Written a minute ago: Stop leaves it, PreCompact archives it and keeps it.
	os.Chtimes(d.latest, now.Add(-time.Minute), now.Add(-time.Minute))
	runHook(t, sys, "precompact-digest", `{"session_id":"../../evil"}`)
	if _, err := os.Stat(filepath.Join(d.archive, "2026-11-05T12-00-00Z-unknown.md")); err != nil {
		t.Fatal("a traversing session id was not replaced by 'unknown'")
	}
	if data2, _ := os.ReadFile(d.latest); !bytes.Equal(data, data2) {
		t.Fatal("a fresh digest was overwritten before compaction")
	}
	os.Chtimes(d.latest, now.Add(-time.Hour), now.Add(-time.Hour))
	runHook(t, sys, "precompact-digest", `{"session_id":"s-1"}`)
	data, _ = os.ReadFile(d.latest)
	if !strings.Contains(string(data), "session_id: 's-1'") || !strings.Contains(string(data), "trigger: precompact-hook-fallback") {
		t.Fatalf("precompact digest: %s", data)
	}
}

func TestDigestValidate(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "d.md")
	for body, want := range map[string]string{
		"---\nschema_version: 1\nsession_id: s\nwritten_at: t\ntrigger: digest-skill\nbranch: b\nhead_sha: h\nhead_subject: x\n---\n": "",
		"no frontmatter\n": "missing or malformed",
		"---\nschema_version: 1\nsession_id: s\nwritten_at: t\ntrigger: digest-skill\nbranch: b\nhead_sha: h\n---\n":                  "'head_subject' missing",
		"---\nschema_version: 2\nsession_id: s\nwritten_at: t\ntrigger: digest-skill\nbranch: b\nhead_sha: h\nhead_subject: x\n---\n": "schema_version must be 1",
		"---\nschema_version: 1\nsession_id: s\nwritten_at: t\ntrigger: other\nbranch: b\nhead_sha: h\nhead_subject: x\n---\n":        "trigger 'other' not valid",
	} {
		os.WriteFile(p, []byte(body), 0o644)
		code, _, errs := runHook(t, fakeSystem(root, time.Now()), "digest-validate", "", p)
		if (want == "") != (code == 0) || !strings.Contains(errs, want) {
			t.Errorf("%q: exit %d, stderr %q, want %q", body, code, errs, want)
		}
	}
}

func TestPipelineValidate(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "my_pipeline.rs")
	os.WriteFile(p, []byte("ok\nb.input(\"x\", \"loop\").then().node(\"loop\", f)\n"), 0o644)
	in, _ := json.Marshal(map[string]any{"tool_name": "Edit", "tool_input": map[string]any{"file_path": p}})
	if _, _, errs := runHook(t, fakeSystem("", time.Now()), "pipeline-validate", string(in)); !strings.Contains(errs, "self-referencing node") {
		t.Errorf("not warned: %q", errs)
	}
	os.WriteFile(p, []byte("b.input(\"x\", \"a\").node(\"b\", f)\n"), 0o644)
	if code, _, errs := runHook(t, fakeSystem("", time.Now()), "pipeline-validate", string(in)); code != 0 || errs != "" {
		t.Errorf("warned on a sound pipeline: %d %q", code, errs)
	}
}

func TestRustCheckOnlyForForwardSlashCratePaths(t *testing.T) {
	var ran []string
	sys := fakeSystem(t.TempDir(), time.Now())
	sys.combined = func(dir, name string, args ...string) (string, bool) {
		ran = append(ran, name+" "+strings.Join(args, " "))
		return "warning: unused\nerror[E0425]: x\n  --> a.rs\n", false
	}
	in := func(tool, path string) string {
		b, _ := json.Marshal(map[string]any{"tool_name": tool, "tool_input": map[string]any{"file_path": path}})
		return string(b)
	}
	_, _, errs := runHook(t, sys, "rust-check", in("Edit", "/r/crates/ix-math/src/lib.rs"))
	if len(ran) != 1 || ran[0] != "cargo check -p ix-math --message-format=short" || !strings.Contains(errs, "error[E0425]") || strings.Contains(errs, "-->") {
		t.Fatalf("ran %v, stderr %q", ran, errs)
	}
	for _, c := range []string{in("Read", "/r/crates/a/b.rs"), in("Edit", "/r/crates/a/b.toml"), in("Edit", `C:\r\crates\a\b.rs`)} {
		runHook(t, sys, "rust-check", c)
	}
	if len(ran) != 1 {
		t.Fatalf("cargo ran for a non-matching edit: %v", ran)
	}
}

func TestPRRationaleCapture(t *testing.T) {
	root := t.TempDir()
	in, _ := json.Marshal(map[string]any{
		"tool_input":    map[string]any{"command": `gh pr create --title "Fix: the Thing!" --body 'why it matters'`},
		"tool_response": map[string]any{"output": "https://github.com/o/r/pull/42\n"},
	})
	runHook(t, fakeSystem(root, time.Date(2026, 11, 5, 12, 0, 0, 0, time.UTC)), "pr-rationale-capture", string(in))
	data, err := os.ReadFile(filepath.Join(root, "state", "digests", "pr-42-fix-the-thing.md"))
	if err != nil || !strings.Contains(string(data), "# PR #42 — Fix: the Thing!") || !strings.Contains(string(data), "why it matters") {
		t.Fatalf("%v %s", err, data)
	}
	other, _ := json.Marshal(map[string]any{"tool_input": map[string]any{"command": "gh pr view 1"}})
	runHook(t, fakeSystem(root, time.Now()), "pr-rationale-capture", string(other))
	if all, _ := filepath.Glob(filepath.Join(root, "state", "digests", "pr-*.md")); len(all) != 1 {
		t.Fatalf("captured a non-create command: %v", all)
	}
}

func TestUnknownHookDoesNotBlock(t *testing.T) {
	if code, _, errs := runHook(t, fakeSystem("", time.Now()), "no-such-hook", ""); code == 2 || !strings.Contains(errs, "usage") {
		t.Fatalf("exit %d %q", code, errs)
	}
}

func TestManifestRender(t *testing.T) {
	raw := []byte(`{"repo":"ga","generated_at":"g","quality":{"domains":{"zeta":{"source":"s1","data":{"oracle_status":"ok","metric_value":0.5,"summary":"fine"}},"alpha":{"source":"s2","data":{"oracle_status":"bad","totalFailures":2}}},"regressions":["r1"]},"services":[{"name":"api","port":5232,"public_path":"/api","expected":"200"}],"backlog":{"overall_progress_pct":40,"total_shipped":4,"total_items":10,"total_epics":2,"epics":[{"title":"E","shipped":1,"active":true,"backlog":0,"progress_pct":50}]},"activity":[{"short_sha":"abc","author":"me","date":"d","subject":"s"}]}`)
	md, ok := renderManifest(raw, time.Date(2026, 11, 5, 9, 8, 7, 0, time.UTC))
	if !ok {
		t.Fatal("not rendered")
	}
	for _, want := range []string{
		"**Source Repository:** ga\n**Fetched At:** 2026-11-05 09:08:07\n",
		"### zeta (🟢 OK)\n- **Source:** s1\n- **Metric Value:** 0.5\n- **Summary:** fine\n\n### alpha (🔴 DEGRADED)\n- **Source:** s2\n- **Failures/Warnings:** 2 fail(s), 0 warn(s)\n\n",
		"### ⚠️ ACTIVE REGRESSIONS\n- r1\n\n",
		"|---|---|---|---|\n| api | 5232 | /api | 200 |\n\n",
		"**Overall Progress:** 40% Shipped (4 of 10 items across 2 epics)",
		"\n| E | 1 | True | 0 | 50% |\n\n",
		"|---|---|---|---|\n| abc | me | d | s |",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("missing %q in\n%s", want, md)
		}
	}
	if _, ok := renderManifest([]byte(`not json`), time.Now()); ok {
		t.Error("rendered garbage")
	}
}
