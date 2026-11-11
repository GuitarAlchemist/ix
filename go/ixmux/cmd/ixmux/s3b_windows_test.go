package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// S3B-5 (Codex round 6, P1): a handle held on ixmux.token without delete sharing (os.Open on
// Windows does not share delete) makes the stop fail to remove it: the exit code is not 0, and
// stderr does not hold the token. Once the handle is closed, the next start rewrites the token.
func TestStopWithTheTokenHeldIsNotZero(t *testing.T) {
	state := filepath.Join(t.TempDir(), "state")
	p := start(t, state, nil)
	fh, err := os.Open(filepath.Join(state, server.TokenFile))
	if err != nil {
		t.Fatal(err)
	}
	p.do("POST", "/v1/shutdown", nil, nil)
	<-p.exited
	fh.Close()
	if code := p.cmd.ProcessState.ExitCode(); code == server.ExitOK {
		t.Errorf("stop with the token left behind: exit code 0, want non-zero\n%s", p.out)
	}
	if strings.Contains(p.out.String(), p.token) {
		t.Error("the token appears in ixmux's output")
	}
	q := start(t, state, nil)
	if q.token == p.token {
		t.Error("the next start kept the token left behind")
	}
	q.do("POST", "/v1/shutdown", nil, nil)
	<-q.exited
	if code := q.cmd.ProcessState.ExitCode(); code != server.ExitOK {
		t.Errorf("the next stop: exit code %d, want 0\n%s", code, q.out)
	}
}
