package claudestream

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// S4B-5 (Codex round 7, P1): a child that floods its output reaches the incarnation's cap: it is
// killed, the pending turn is uncertain for stream-too-large, and the disk stays under the cap.
func TestFloodIsCapped(t *testing.T) {
	const cap = 1 << 20
	a, e, _ := setupCfg(t, 10*time.Second, func(c *Config) { c.MaxStream = cap }, "FAKE_FLOOD=4000000")
	res, _ := send(t, a, e, "m1", "flood")
	if res.State != deliver.Uncertain || !strings.Contains(res.Reason, "stream-too-large") {
		t.Errorf("a flooding child: %+v, want uncertain for stream-too-large", res)
	}
	if st, err := os.Stat(a.StreamPath()); err != nil || st.Size() > cap+4<<10 {
		t.Errorf("stream.jsonl: %v %v, want at most the %d-byte cap (plus its notes)", st.Size(), err, cap)
	}
	if a.Alive() {
		t.Error("the flooding child is still alive")
	}
}

// S4B-5 (Codex round 7, P1): a result line longer than the line bound resolves the turn as
// uncertain; it never leaves it pending.
func TestResultLineTooLong(t *testing.T) {
	a, e, _ := setupCfg(t, 5*time.Second, func(c *Config) { c.MaxLine = 64 << 10 }, "FAKE_REPLY=200000")
	begun := time.Now()
	res, _ := send(t, a, e, "m1", "x")
	if res.State != deliver.Uncertain || time.Since(begun) > 4*time.Second {
		t.Errorf("a result over the line bound: %+v after %s, want uncertain at once", res, time.Since(begun))
	}
}

// S4B-5 (Codex round 7, P1): Status honours its context, and resumes where it stopped reading
// instead of reading the stream again from byte 0.
func TestStatusHonoursItsContextAndCursor(t *testing.T) {
	a, e, _ := setup(t, 5*time.Second)
	inc := a.sessionID()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := a.Status(ctx, "m1", inc); !errors.Is(err, context.Canceled) {
		t.Errorf("Status with a cancelled context: %v, want context.Canceled", err)
	}
	var froms []int64
	testHookStatusFrom = func(from int64) { froms = append(froms, from) }
	t.Cleanup(func() { testHookStatusFrom = nil })
	a.Status(context.Background(), "m1", inc)
	send(t, a, e, "m1", "x")
	if o, _ := a.Status(context.Background(), "m1", inc); o != deliver.Received {
		t.Errorf("Status after m1's result: %s, want received", o)
	}
	if len(froms) < 2 || froms[len(froms)-1] == 0 {
		t.Errorf("Status read offsets %v, want the second read to resume past 0", froms)
	}
}
