package claudestream

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// S4B-3 (Codex round 7, P1): the child's environment is a whitelist. A variable of the parent
// reaches it only when named (PassEnv); the fake writes the names it got, never the values.
func TestEnvironmentIsAWhitelist(t *testing.T) {
	t.Setenv("IXMUX_TEST_SECRET", "x")
	for _, tc := range []struct {
		name string
		pass []string
		want bool
	}{
		{"not named", nil, false},
		{"named", []string{"IXMUX_TEST_SECRET"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			names := filepath.Join(dir, "names")
			a, err := Start(Config{Name: "c", Dir: filepath.Join(dir, "claude-c"), Bin: fake, Env: []string{"FAKE_ENVNAMES=" + names},
				PassEnv: tc.pass, InitWait: 5 * time.Second, StopWait: 5 * time.Second})
			if err != nil {
				t.Fatal(err)
			}
			a.Stop()
			b, _ := os.ReadFile(names)
			got := strings.Fields(string(b))
			has := slices.ContainsFunc(got, func(n string) bool { return strings.EqualFold(n, "IXMUX_TEST_SECRET") })
			if has != tc.want {
				t.Errorf("IXMUX_TEST_SECRET %s: in the child's environment = %v, want %v (names %v)", tc.name, has, tc.want, got)
			}
			if !slices.ContainsFunc(got, func(n string) bool { return strings.EqualFold(n, "PATH") }) {
				t.Errorf("PATH missing from the child's environment: %v", got)
			}
		})
	}
}
