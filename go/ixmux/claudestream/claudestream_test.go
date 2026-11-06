package claudestream

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// The tests drive a fake claude (internal/fakeclaude): no model is ever called here.
var fake string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "fakeclaude")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	exe := ""
	if runtime.GOOS == "windows" {
		exe = ".exe"
	}
	fake = filepath.Join(dir, "claude"+exe)
	cmd := exec.Command(filepath.Join(runtime.GOROOT(), "bin", "go"+exe), "build", "-o", fake, "./internal/fakeclaude")
	if out, err := cmd.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "build fakeclaude: %v\n%s", err, out)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func effects(t *testing.T, path string) int {
	t.Helper()
	b, _ := os.ReadFile(path)
	return bytes.Count(b, []byte("\n"))
}

// setup starts an adapter on the fake with env, and an engine with it as target "c".
func setup(t *testing.T, timeout time.Duration, env ...string) (*Adapter, *deliver.Engine, string) {
	t.Helper()
	dir := t.TempDir()
	eff := filepath.Join(dir, "effects")
	a, err := Start(Config{Name: "c", Dir: filepath.Join(dir, "claude-c"), Bin: fake, Env: append([]string{"FAKE_EFFECTS=" + eff}, env...),
		InitWait: 5 * time.Second, StopWait: 5 * time.Second})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { a.Stop() })
	e, _, err := deliver.Open(filepath.Join(dir, "cmd.jsonl"), deliver.Options{Timeout: timeout})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	e.Register("c", a.Target())
	return a, e, eff
}

func send(t *testing.T, a *Adapter, e *deliver.Engine, id, text string) (deliver.Result, error) {
	t.Helper()
	inc, err := a.Incarnation()
	if err != nil {
		t.Fatalf("incarnation: %v", err)
	}
	return e.Deliver(context.Background(), deliver.Mandate{MandateID: id, Target: "c", Incarnation: inc, Text: text})
}

// Oracle: the incarnation is the session_id of system/init; an init announcing another session
// than the one asked for is refused.
func TestIncarnationComesFromInit(t *testing.T) {
	for _, mode := range []string{"start", "first"} {
		t.Run(mode, func(t *testing.T) {
			a, _, eff := setup(t, 5*time.Second, "FAKE_INIT="+mode)
			inc, err := a.Incarnation()
			if err != nil || len(inc) != 36 || inc != a.sessionID() {
				t.Errorf("incarnation %q (%v), want the init's session_id %q", inc, err, a.sessionID())
			}
			if want := map[string]int{"start": 0, "first": 1}[mode]; effects(t, eff) != want {
				t.Errorf("init %s: %d turns before any mandate, want %d", mode, effects(t, eff), want)
			}
		})
	}
	dir := t.TempDir()
	if a, err := Start(Config{Name: "c", Dir: dir, Bin: fake, Env: []string{"FAKE_SESSION=another"}, InitWait: 5 * time.Second, StopWait: 5 * time.Second}); err == nil {
		a.Stop()
		t.Error("an init of another session was accepted")
	}
}

// Oracle: each result is correlated to its mandate; the Ack carries the bounded answer and
// whether it began with ACK <mandate_id>.
func TestAckIsCorrelated(t *testing.T) {
	a, e, eff := setup(t, 5*time.Second)
	for _, id := range []string{"m1", "m2"} {
		res, err := send(t, a, e, id, "Dis bonjour.")
		if err != nil || res.State != deliver.Delivered || res.Ack == nil || !res.Ack.AckSeen || !strings.HasPrefix(res.Ack.Text, "ACK "+id) {
			t.Fatalf("%s: %+v %v (ack %+v)", id, res, err, res.Ack)
		}
	}
	if n := effects(t, eff); n != 2 {
		t.Errorf("%d turns, want 2", n)
	}
}

func TestAckNotSeen(t *testing.T) {
	a, e, _ := setup(t, 5*time.Second, "FAKE_NOACK=1")
	res, err := send(t, a, e, "m1", "Dis bonjour.")
	if err != nil || res.State != deliver.Delivered || res.Ack == nil || res.Ack.AckSeen {
		t.Errorf("an answer without ACK: %+v %v, want delivered with ack_seen false", res, err)
	}
}

// Oracle: a child whose init lists tools is never sent a mandate.
func TestToolsListedIsRefused(t *testing.T) {
	a, e, eff := setup(t, 5*time.Second, "FAKE_TOOLS=Bash,Read")
	res, err := e.Deliver(context.Background(), deliver.Mandate{MandateID: "m1", Target: "c", Incarnation: a.sessionID(), Text: "x"})
	if res.State != deliver.Refused || !strings.Contains(res.Reason, "tools") {
		t.Errorf("tools listed: %+v %v, want refused for its tools", res, err)
	}
	if n := effects(t, eff); n != 0 {
		t.Errorf("%d turns, want 0", n)
	}
}

// Oracle: the child dies mid-mandate: the delivery is uncertain, and the child is not relaunched.
func TestChildDeathMakesUncertain(t *testing.T) {
	a, e, eff := setup(t, 5*time.Second)
	res, _ := send(t, a, e, "m1", "die")
	if res.State != deliver.Uncertain {
		t.Errorf("the child died mid-mandate: %+v, want uncertain", res)
	}
	if inc, err := a.Incarnation(); err == nil {
		t.Errorf("incarnation %q after the child's death, want an error (no relaunch)", inc)
	}
	if n := effects(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
	if o, _ := a.Status(context.Background(), "m1", a.sessionID()); o != deliver.Unknown {
		t.Errorf("Status of a sent mandate without result: %s, want unknown", o)
	}
}

// Oracle: Status reads the stream journal: no send is not received, a send with its result is
// received, a send without result is unknown, another incarnation is unknown.
func TestStatusReadsTheStream(t *testing.T) {
	hold := filepath.Join(t.TempDir(), "hold")
	os.WriteFile(hold, nil, 0o600)
	a, e, _ := setup(t, 100*time.Millisecond, "FAKE_HOLD="+hold)
	inc := a.sessionID()
	ctx := context.Background()
	if o, err := a.Status(ctx, "never", inc); o != deliver.NotReceived {
		t.Errorf("never sent: %s %v, want not-received", o, err)
	}
	if res, _ := send(t, a, e, "s1", "slow"); res.State != deliver.Uncertain {
		t.Fatalf("held past the deadline: %+v", res)
	}
	if o, _ := a.Status(ctx, "s1", inc); o != deliver.Unknown {
		t.Errorf("sent, no result yet: %s, want unknown", o)
	}
	if o, _ := a.Status(ctx, "s1", "another"); o != deliver.Unknown {
		t.Errorf("another incarnation: %s, want unknown", o)
	}
	os.Remove(hold)
	deadline := time.Now().Add(5 * time.Second)
	for o := deliver.Unknown; o != deliver.Received; o, _ = a.Status(ctx, "s1", inc) {
		if time.Now().After(deadline) {
			t.Fatal("never received after the hold")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// Oracle: Stop closes stdin and the child exits.
func TestStopEndsTheChild(t *testing.T) {
	a, _, _ := setup(t, 5*time.Second)
	if err := a.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if alive(a.PID()) {
		t.Errorf("child %d alive after Stop", a.PID())
	}
}
