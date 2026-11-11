package claudestream

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

func TestBlockedStdinIsBounded(t *testing.T) {
	testBlockedStdin(t)
}

func TestEarlyResultCannotRetireWriteWatchdog(t *testing.T) {
	testBlockedStdin(t, "FAKE_EARLY_ACK=blocked")
}

func testBlockedStdin(t *testing.T, env ...string) {
	a, _, _ := setupCfg(t, 100*time.Millisecond, func(c *Config) {
		c.TurnMax = 500 * time.Millisecond
		c.StopWait = 200 * time.Millisecond
	}, append([]string{"FAKE_NOREAD=1"}, env...)...)
	c := a.live()
	defer c.cmd.Process.Kill() // also frees the pre-fix writer when this regression fails
	inc := a.sessionID()
	done := make(chan error, 1)
	go func() {
		_, err := a.Deliver(context.Background(), deliver.Mandate{MandateID: "blocked", Target: "c", Incarnation: inc, Text: strings.Repeat("x", deliver.MaxText)})
		done <- err
	}()
	time.Sleep(100 * time.Millisecond)
	status := make(chan struct{})
	go func() { a.Incarnation(); a.Alive(); a.Tools(); close(status) }()
	select {
	case <-status:
	case <-time.After(200 * time.Millisecond):
		t.Error("target status waits on blocked stdin")
	}
	select {
	case err := <-done:
		if err == nil {
			t.Error("unread mandate was delivered")
		}
	case <-time.After(4 * time.Second):
		t.Error("TurnMax did not bound blocked stdin")
	}
}

func TestBlockedStdinWithFailedKill(t *testing.T) {
	a, _, _ := setupCfg(t, 100*time.Millisecond, func(c *Config) {
		c.TurnMax = 300 * time.Millisecond
		c.StopWait = 200 * time.Millisecond
	}, "FAKE_NOREAD=1")
	failKills(t, a.live())
	inc := a.sessionID()
	done := make(chan error, 1)
	go func() {
		_, err := a.Deliver(context.Background(), deliver.Mandate{MandateID: "blocked", Target: "c", Incarnation: inc, Text: strings.Repeat("x", deliver.MaxText)})
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Error("unread mandate was delivered")
		}
		if !a.Lost() {
			t.Error("failed kill did not mark the target lost")
		}
	case <-time.After(4 * time.Second):
		t.Error("failed kill did not release blocked stdin")
	}
}

func TestStopWithBlockedStdin(t *testing.T) {
	a, _, _ := setupCfg(t, 100*time.Millisecond, func(c *Config) {
		c.TurnMax = time.Minute
		c.StopWait = 200 * time.Millisecond
	}, "FAKE_NOREAD=1")
	c := a.live()
	defer c.cmd.Process.Kill()
	inc := a.sessionID()
	done := make(chan error, 1)
	go func() {
		_, err := a.Deliver(context.Background(), deliver.Mandate{MandateID: "blocked", Target: "c", Incarnation: inc, Text: strings.Repeat("x", deliver.MaxText)})
		done <- err
	}()
	time.Sleep(100 * time.Millisecond)
	stopped := make(chan error, 1)
	go func() { stopped <- a.Stop() }()
	select {
	case err := <-stopped:
		if err != nil {
			t.Error(err)
		}
	case <-time.After(4 * time.Second):
		t.Error("Stop waits on blocked stdin")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Error("delivery still running after Stop")
	}
}

// startChild must get its opening turn back to select on InitWait, even if
// the child cannot consume the opening message. Closing stdin on failure ends it.
func TestOpeningSendDoesNotHoldInitWait(t *testing.T) {
	stream, err := os.CreateTemp(t.TempDir(), "stream")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	c := &child{a: &Adapter{stream: stream}, session: "opening", stdin: writer}
	result := make(chan *turn, 1)
	go func() {
		p, _ := c.send("", "opening message", -1, "")
		result <- p
	}()
	var p *turn
	select {
	case p = <-result:
		if p == nil {
			t.Fatal("opening send failed before the pipe was closed")
		}
	case <-time.After(time.Second):
		t.Fatal("opening send prevents startChild from observing InitWait")
	}
	select {
	case <-p.done:
		t.Fatal("unread opening message resolved before stdin closed")
	default:
	}
	writer.Close() // startChild's InitWait failure calls stop, which does this
	select {
	case <-p.done:
		if p.res.err == nil {
			t.Error("unread opening message succeeded")
		}
	case <-time.After(time.Second):
		t.Fatal("closing stdin did not release the opening turn")
	}
}
