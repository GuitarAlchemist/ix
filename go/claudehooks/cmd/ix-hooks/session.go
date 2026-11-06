package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

// sessionStart is the one SessionStart hook: it installs this binary for the other hooks, then
// injects the digest (sessionstartDigest). settings.json runs it through `go run`, so a session
// always starts with the hooks built from the checked-out source.
func sessionStart(e *env) int {
	if root := e.repoRoot(); root != "" {
		if err := e.install(root); err != nil {
			fmt.Fprintf(e.stderr, "[ix-hooks] could not install .claude/hooks/bin: %v (the previous build stays)\n", err)
		}
	}
	return sessionstartDigest(e)
}

// binPath is where the hooks other than session-start find the binary.
func binPath(root string) string {
	name := "ix-hooks"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(root, ".claude", "hooks", "bin", name)
}

// install copies the running executable to binPath when its content differs. The copy goes to a
// temporary name first and is renamed over the target, so a hook starting meanwhile runs either
// the old binary or the new one, never half of one. On Windows the rename fails while another
// session runs the old binary; that session's next start retries.
func (e *env) install(root string) error {
	self, err := e.sys.executable()
	if err != nil {
		return err
	}
	want, err := os.ReadFile(self)
	if err != nil {
		return err
	}
	target := binPath(root)
	if have, err := os.ReadFile(target); err == nil && sha256.Sum256(have) == sha256.Sum256(want) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(target), ".ix-hooks-*.tmp")
	if err != nil {
		return err
	}
	_, werr := io.Copy(tmp, bytes.NewReader(want))
	cerr := tmp.Close()
	if werr != nil || cerr != nil {
		_ = os.Remove(tmp.Name()) // best-effort cleanup; the write error is the one reported
		return fmt.Errorf("writing %s: %v %v", tmp.Name(), werr, cerr)
	}
	err = os.Chmod(tmp.Name(), 0o755)
	if err == nil {
		err = os.Rename(tmp.Name(), target)
	}
	if err != nil {
		_ = os.Remove(tmp.Name()) // best-effort cleanup; err is the one reported
		return err
	}
	return nil
}

var sourceMember = regexp.MustCompile(`"source"[[:space:]]*:[[:space:]]*"([a-z]*)"`)

// sessionstartDigest prints state/digests/latest.md for the session's context (was
// sessionstart-digest.sh). It says in one line when the digest is more than 24 hours old or an
// automatic stub. The ecosystem manifest is synced and injected on every source but compact:
// after a compaction the summary carries the session, and the manifest (another repo's snapshot,
// already injected at startup) would only take room from it.
func sessionstartDigest(e *env) int {
	root := e.repoRoot()
	if root == "" {
		return 0
	}
	_, raw := e.payload()
	source := ""
	if m := sourceMember.FindSubmatch(raw); m != nil {
		source = string(m[1])
	}
	latest := digestsOf(root).latest
	bootstrap := filepath.Join(root, ".claude", "manifest-bootstrap.md")

	if source != "compact" {
		e.syncManifest(bootstrap)
		if data, err := os.ReadFile(bootstrap); err == nil {
			fmt.Fprint(e.stdout, "\n=== Ecosystem Manifest (Vite Dev Data) ===\n")
			_, _ = e.stdout.Write(data) // stdout, as the fmt.Fprint calls around it
			if len(data) > 0 && data[len(data)-1] != '\n' {
				fmt.Fprintln(e.stdout)
			}
			fmt.Fprint(e.stdout, "=== End Manifest ===\n\n")
		}
	}

	age, ok := e.ageMinutes(latest)
	if !ok {
		return 0
	}
	if age >= 1440 {
		fmt.Fprint(e.stdout, "\nSession digest exists but is >24h old — skipping injection. Run /digest to refresh.\n\n")
		return 0
	}
	data, err := os.ReadFile(latest)
	if err != nil {
		return 0
	}
	// The tracker's and the pre-compaction stubs hold the branch and HEAD only; nothing in them
	// was written by a model, so they are named, not injected.
	for _, l := range strings.Split(string(data), "\n") {
		if t, found := strings.CutPrefix(l, "trigger:"); found {
			t = strings.TrimSpace(t)
			if t == "activity-tracker-mid-session" || t == "precompact-hook-fallback" {
				fmt.Fprintf(e.stdout, "\nstate/digests/latest.md is an automatic stub (%s): nothing model-written to carry forward.\n\n", t)
				return 0
			}
			break
		}
	}
	ageStr := fmt.Sprintf("%d min", age)
	if age >= 60 {
		ageStr = fmt.Sprintf("%dh %dm", age/60, age%60)
	}
	fmt.Fprintf(e.stdout, "\n=== Session digest (last written %s ago) ===\n", ageStr)
	_, _ = e.stdout.Write(data) // stdout, as the fmt.Fprint calls around it
	if len(data) > 0 && data[len(data)-1] != '\n' {
		fmt.Fprintln(e.stdout)
	}
	fmt.Fprint(e.stdout, "=== End digest ===\n\n")
	return 0
}
