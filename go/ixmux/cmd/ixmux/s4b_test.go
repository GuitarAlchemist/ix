package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// S4B-6 (Codex round 7, P2): a --claude-target named twice is refused at start, with code 2 and
// a clear message, before any child is launched.
func TestDuplicateClaudeTargetIsRefused(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state")
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "serve", "--addr", "127.0.0.1:0", "--state", state,
		"--claude-target", "c", "--claude-target", "c", "--claude-bin", fakeClaude)
	cmd.Env = append(os.Environ(), "FAKE_EFFECTS="+filepath.Join(dir, "effects"))
	out, err := cmd.CombinedOutput()
	var ee *exec.ExitError
	if !errors.As(err, &ee) || ee.ExitCode() != 2 || !strings.Contains(string(out), `"c"`) {
		t.Errorf("--claude-target c twice: %v\n%s\nwant exit code 2 naming \"c\"", err, out)
	}
	if _, err := os.Stat(filepath.Join(state, "claude-c")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("claude-c exists (%v): a child was launched", err)
	}
}

// S4B-3 (Codex round 7, P1), end to end: a variable of ixmux's environment reaches the claude
// child only when --claude-env names it. The fake writes the names it got, never the values.
func TestClaudeEnvFlag(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want bool
	}{
		{"not named", nil, false},
		{"named", []string{"--claude-env", "IXMUX_TEST_SECRET"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			names := filepath.Join(dir, "names")
			args := append([]string{"--claude-target", "c", "--claude-bin", fakeClaude, "--claude-env", "FAKE_ENVNAMES"}, tc.args...)
			p := start(t, filepath.Join(dir, "state"), []string{"IXMUX_TEST_SECRET=x", "FAKE_ENVNAMES=" + names}, args...)
			p.do("POST", "/v1/shutdown", nil, nil)
			<-p.exited
			b, err := os.ReadFile(names)
			if err != nil {
				t.Fatalf("the fake wrote no names: %v\n%s", err, p.out)
			}
			got := strings.Fields(string(b))
			if has := slices.ContainsFunc(got, func(n string) bool { return strings.EqualFold(n, "IXMUX_TEST_SECRET") }); has != tc.want {
				t.Errorf("IXMUX_TEST_SECRET in the child's environment = %v, want %v (names %v)", has, tc.want, got)
			}
		})
	}
}
