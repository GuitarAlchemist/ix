package workshop

import (
	"context"
	"errors"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestACommandWaitingItsTurnRunsOnlyIfStillAllowed: commands run one at a time, so a command can wait for the previous
// one. If the owner withdraws the terminal, or revokes the device, while it waits, it never starts.
func TestACommandWaitingItsTurnRunsOnlyIfStillAllowed(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the terminal runs Windows PowerShell")
	}
	s, err := Open(filepath.Join(t.TempDir(), "terminal.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	s.terminalTimeout = defaultTerminalLimit
	link, err := s.NewLinkInvite("https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	pc, pcToken, err := s.Claim(link.Code, "PC")
	if err != nil {
		t.Fatal(err)
	}
	tabletToken, tabletID := paired(t, s)
	if err := s.AllowTerminal(tabletID, true); err != nil {
		t.Fatal(err)
	}

	for _, c := range []struct {
		name, token string
		withdraw    func() error
		want        error
	}{
		{"terminal withdrawn", tabletToken, func() error { return s.AllowTerminal(tabletID, false) }, ErrNoTerminal},
		{"device revoked", pcToken, func() error { return s.Decide(pc.ID, "revoked") }, ErrRevoked},
	} {
		s.terminalRun.Lock() // the previous command is still running
		done := make(chan error, 1)
		go func() {
			_, err := s.RunTerminal(context.Background(), c.token, "Write-Output 1")
			done <- err
		}()
		time.Sleep(300 * time.Millisecond) // the command has passed its checks and waits its turn
		if err := c.withdraw(); err != nil {
			t.Fatal(err)
		}
		s.terminalRun.Unlock()
		select {
		case err := <-done:
			if !errors.Is(err, c.want) {
				t.Errorf("%s while the command waited: it must not run, got %v", c.name, err)
			}
		case <-time.After(defaultTerminalLimit + 10*time.Second):
			t.Fatalf("%s: the command never returned", c.name)
		}
	}
	if log, err := s.TerminalLog(10); err != nil || len(log) != 0 {
		t.Fatalf("no command ran, so none is journaled: %+v %v", log, err)
	}
}
