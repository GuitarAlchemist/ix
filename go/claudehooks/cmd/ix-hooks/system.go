package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// system is the outside world a hook reads: the clock, the environment, the processes it runs
// and the network. Files are touched directly; tests give each hook its own temp repository.
type system struct {
	now    func() time.Time
	getenv func(string) string
	// output runs a program and returns its stdout, trimmed, and whether it exited 0 (the
	// scripts' `$(cmd 2>/dev/null)`).
	output func(dir, name string, args ...string) (string, bool)
	// combined runs a program and returns stdout and stderr together (`2>&1`).
	combined func(dir, name string, args ...string) (string, bool)
	// fetch GETs a URL within the timeout.
	fetch func(url string, timeout time.Duration) ([]byte, error)
	// executable is this program's own path.
	executable func() (string, error)
	// toplevel is the repository holding the working directory (`git rev-parse --show-toplevel`).
	toplevel func() (string, bool)
}

func realSystem() *system {
	return &system{
		now:    time.Now,
		getenv: os.Getenv,
		output: func(dir, name string, args ...string) (string, bool) {
			cmd := exec.Command(name, args...)
			cmd.Dir = dir
			out, err := cmd.Output()
			return strings.TrimSpace(string(out)), err == nil
		},
		combined: func(dir, name string, args ...string) (string, bool) {
			cmd := exec.Command(name, args...)
			cmd.Dir = dir
			out, err := cmd.CombinedOutput()
			return string(out), err == nil
		},
		fetch: func(url string, timeout time.Duration) ([]byte, error) {
			c := &http.Client{Timeout: timeout}
			resp, err := c.Get(url)
			if err != nil {
				return nil, err
			}
			defer resp.Body.Close()
			if resp.StatusCode/100 != 2 {
				return nil, &httpStatus{resp.StatusCode}
			}
			return io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		},
		executable: os.Executable,
		toplevel: func() (string, bool) {
			dir, err := os.Getwd()
			if err != nil {
				return "", false
			}
			return findTopLevel(dir)
		},
	}
}

// findTopLevel is the nearest ancestor of dir (itself included) holding a `.git` directory or
// worktree file, with forward slashes as git prints it. It replaces a `git rev-parse` process:
// each one cost 1 to 2.5 s on a loaded Windows machine, on every prompt and every edit.
func findTopLevel(dir string) (string, bool) {
	for {
		if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
			return filepath.ToSlash(dir), true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

type httpStatus struct{ code int }

func (h *httpStatus) Error() string { return http.StatusText(h.code) }

// repoRoot is the repository holding the hook's working directory, "" outside one (every
// digest hook then exits 0, as the scripts did).
func (e *env) repoRoot() string {
	root, ok := e.sys.toplevel()
	if !ok {
		return ""
	}
	return root
}

// gitOr is a git query in root, or fallback when git fails (the scripts' `|| echo unknown`).
func (e *env) gitOr(root, fallback string, args ...string) string {
	out, ok := e.sys.output("", "git", append([]string{"-C", root}, args...)...)
	if !ok {
		return fallback
	}
	return out
}

// payload reads stdin as a JSON object; nil when it is empty or not an object.
func (e *env) payload() (map[string]any, []byte) {
	raw, _ := io.ReadAll(e.stdin)
	var m map[string]any
	if json.Unmarshal(bytes.TrimSpace(raw), &m) != nil {
		return nil, raw
	}
	return m, raw
}

// field is a nested string member, "" when absent or not a string (python's _field, jq's `// ""`).
func field(m map[string]any, path ...string) string {
	var v any = m
	for _, k := range path {
		obj, ok := v.(map[string]any)
		if !ok {
			return ""
		}
		v = obj[k]
	}
	s, _ := v.(string)
	return s
}

// ageMinutes is whole minutes since path was modified, as `(now - mtime) / 60`; ok false when
// the file does not exist.
func (e *env) ageMinutes(path string) (int64, bool) {
	st, err := os.Stat(path)
	if err != nil {
		return 0, false
	}
	return int64(e.sys.now().Sub(st.ModTime()).Seconds()) / 60, true
}

// writeContext prints the JSON the scripts printed with `jq -n --arg ctx … '{additionalContext: $ctx}'`.
func writeContext(w io.Writer, ctx string) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false) // jq writes < > & as they are
	enc.SetIndent("", "  ")
	_ = enc.Encode(map[string]string{"additionalContext": ctx}) // stdout, as fmt.Print
}
