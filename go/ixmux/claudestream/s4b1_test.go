package claudestream

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// setupCfg is setup with a change to the adapter's Config.
func setupCfg(t *testing.T, timeout time.Duration, mod func(*Config), env ...string) (*Adapter, *deliver.Engine, string) {
	t.Helper()
	dir := t.TempDir()
	eff := filepath.Join(dir, "effects")
	cfg := Config{Name: "c", Dir: filepath.Join(dir, "claude-c"), Bin: fake, Env: append([]string{"FAKE_EFFECTS=" + eff}, env...),
		InitWait: 5 * time.Second, StopWait: 5 * time.Second}
	if mod != nil {
		mod(&cfg)
	}
	a, err := Start(cfg)
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

// noteKinds counts the ixmux lines of a stream.jsonl by kind.
func noteKinds(t *testing.T, path string) map[string]int {
	t.Helper()
	out := map[string]int{}
	for _, m := range stopStepsAll(t, path) {
		if k, ok := m["ixmux"].(string); ok {
			out[k]++
		}
	}
	return out
}

func stopStepsAll(t *testing.T, path string) []map[string]any {
	t.Helper()
	b, _ := os.ReadFile(path)
	var out []map[string]any
	for _, l := range strings.Split(string(b), "\n") {
		if m := parseNote(l); m != nil {
			out = append(out, m)
		}
	}
	return out
}

// parseNote is an ixmux line of the stream (it has an "ixmux" key), or nil.
func parseNote(line string) map[string]any {
	var m map[string]any
	if json.Unmarshal([]byte(line), &m) != nil || m["ixmux"] == nil {
		return nil
	}
	return m
}

// S4B-1 (Codex round 7, P1): a result is kept only if it is of the live incarnation, with a turn
// pending, and its ACK is exact. None of these results makes m2 delivered wrongly.
func TestResultsAreCorrelated(t *testing.T) {
	for _, tc := range []struct {
		mode      string
		delivered bool // m2's own answer then comes and is right to deliver it
	}{
		{"othersession", true},
		{"duplicate", false},
		{"prefix", false},
		{"othermandate", false},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			a, e, _ := setup(t, 5*time.Second, "FAKE_M2="+tc.mode)
			if res, err := send(t, a, e, "m1", "Dis bonjour."); err != nil || res.State != deliver.Delivered {
				t.Fatalf("m1: %+v %v", res, err)
			}
			res, err := send(t, a, e, "m2", "Dis bonjour.")
			switch {
			case tc.delivered && (res.State != deliver.Delivered || res.Ack == nil || res.Ack.Text != "ACK m2\nBONJOUR"):
				t.Errorf("m2 with a result of another session first: %+v %v (ack %+v), want delivered by its own answer", res, err, res.Ack)
			case !tc.delivered && res.State == deliver.Delivered:
				t.Errorf("m2 with a %s answer: delivered (ack %+v), want uncertain", tc.mode, res.Ack)
			case !tc.delivered && res.State != deliver.Uncertain:
				t.Errorf("m2 with a %s answer: %+v %v, want uncertain", tc.mode, res, err)
			}
			kinds := noteKinds(t, a.StreamPath())
			if want := map[string]string{"othersession": "ignored-result", "duplicate": "ack-mismatch", "prefix": "ack-mismatch", "othermandate": "ack-mismatch"}[tc.mode]; kinds[want] == 0 {
				t.Errorf("no %s line in the stream: %v", want, kinds)
			}
		})
	}
}

// S4B-1: a result with no turn pending is journaled and ignored.
func TestResultWithoutTurnIsIgnored(t *testing.T) {
	a, e, _ := setup(t, 5*time.Second, "FAKE_M2=duplicate")
	send(t, a, e, "m1", "Dis bonjour.")
	send(t, a, e, "m2", "Dis bonjour.") // the duplicate takes the turn: m2's own result has none
	deadline := time.Now().Add(5 * time.Second)
	for noteKinds(t, a.StreamPath())["ignored-result"] == 0 {
		if time.Now().After(deadline) {
			t.Fatalf("the result without a turn left no ignored-result line: %v", noteKinds(t, a.StreamPath()))
		}
		time.Sleep(20 * time.Millisecond)
	}
	if o, _ := a.Status(context.Background(), "m2", a.sessionID()); o == deliver.Received {
		t.Error("Status says m2 received from a result that was ignored")
	}
}
