package main

import (
	"bytes"
	"context"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// listCounter is a frame transport that counts the probe.list requests its client sends.
type listCounter struct {
	frameConn
	lists *atomic.Int64
}

func (c listCounter) Write(ctx context.Context, frame []byte) error {
	if bytes.Contains(frame, []byte(`"op":"probe.list"`)) {
		c.lists.Add(1)
	}
	return c.frameConn.Write(ctx, frame)
}

// A jump of ctl at session start (a reconnect past the allowance, with a record to skip) asks no
// list of its own: the session reads the list once its control subscriptions stand, and that list
// is past the jump. A jump later in the session asks the loop for one (resnapshot).
func TestAJumpAtSessionStartAsksNoSecondList(t *testing.T) {
	b := memPlane(t, fakeSpec(t, "b", nil), "b")
	a := memPlane(t, fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(t.TempDir(), "never-read.json"), Probes: []string{"cpu"}}}), "a")
	l := a.peers["b"]
	var mu sync.Mutex
	var conns []*Client
	var lists []*atomic.Int64
	l.connect = func(context.Context) (*Client, stateFile, error) {
		n := &atomic.Int64{}
		c := newClient(listCounter{newMemFrames(newMemSession(b.ctx, b)), n})
		mu.Lock()
		conns, lists = append(conns, c), append(lists, n)
		mu.Unlock()
		return c, stateFile{Name: b.name, Addr: "mem:" + b.name, Token: "mem:" + b.Epoch()}, nil
	}
	b.Start()
	a.Start()
	waitLinkStatus(t, l, func(st PeerStatus) bool { return st.State == "up" && st.Live && st.Reconciled })
	b.Publish("ctl", event("note.before", map[string]any{"n": 1}))
	deadline := time.Now().Add(5 * time.Second)
	for {
		l.mu.Lock()
		_, mirrored := l.since["ctl"]
		l.mu.Unlock()
		if mirrored {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("nothing of b's ctl was mirrored")
		}
		time.Sleep(10 * time.Millisecond)
	}

	l.allowFrames.Store(0) // the reconnect may not catch up: it jumps
	mu.Lock()
	first := conns[0]
	mu.Unlock()
	first.Close()
	b.Publish("ctl", event("note.skipped", map[string]any{"n": 2})) // what the jump skips
	waitLinkStatus(t, l, func(st PeerStatus) bool { return st.State == "up" && st.Live && st.Reconciled && st.Jumps == 1 })
	time.Sleep(300 * time.Millisecond) // room for a second list, were one asked for
	mu.Lock()
	defer mu.Unlock()
	if len(conns) != 2 {
		t.Fatalf("%d connections, want 2", len(conns))
	}
	if n := lists[1].Load(); n != 1 {
		t.Fatalf("the session that jumped at start asked for the list %d times, want once", n)
	}
}
