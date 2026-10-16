package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// subscribe with since replays from the ring, then goes live, in order,
// with no hole and no repeat, while the publisher keeps going.
func TestWSSubscribeSinceReplaysThenLiveNoHole(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	base := in.plane.hub.Head()
	publishN(in.plane, "svc/x", 100)
	in.plane.Publish("ctl", event("noise", map[string]any{}))
	c := dial(t, in)

	const live = 1000
	go func() {
		for i := 0; i < live; i++ {
			in.plane.Publish("svc/x", event("tick", map[string]any{"i": i}))
			if i%10 == 9 {
				time.Sleep(time.Millisecond)
			}
		}
	}()
	since := base + 50
	w := subscribe(t, c, "svc/", &since)
	var seqs []uint64
	for len(seqs) < 50+live {
		f, err := w.next(10*time.Second, func(*Envelope) bool { return true })
		if err != nil {
			t.Fatalf("after %d events: %v", len(seqs), err)
		}
		seqs = append(seqs, f.Env.Seq)
	}
	for i := 1; i < len(seqs); i++ {
		want := seqs[i-1] + 1
		if seqs[i-1] == base+100 {
			want++ // the ctl event in between is not under svc/
		}
		if seqs[i] != want {
			t.Fatalf("hole or repeat at %d: %d then %d", i, seqs[i-1], seqs[i])
		}
	}
	if seqs[0] != base+51 {
		t.Fatalf("replay starts at %d, want %d", seqs[0], base+51)
	}
}

// A client that stops reading: its live subscription overflows the bounded
// queue and is dropped with a gap frame naming the first seq it missed.
func TestWSOverflowGivesGapFrame(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	c := dial(t, in)
	_, frames, err := c.Subscribe(context.Background(), "svc/", nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	pad := strings.Repeat("x", 1024)
	for i := 0; i < 20000; i++ {
		in.plane.Publish("svc/x", event("big", map[string]any{"pad": pad}))
	}
	var last uint64
	var gap *Gap
	for gap == nil {
		select {
		case f := <-frames:
			switch {
			case f.Gap != nil:
				gap = f.Gap
			case f.Env != nil:
				if last != 0 && f.Env.Seq != last+1 {
					t.Fatalf("hole before the gap: %d then %d", last, f.Env.Seq)
				}
				last = f.Env.Seq
			}
		case <-time.After(10 * time.Second):
			t.Fatalf("no gap frame; last seq %d", last)
		}
	}
	if gap.Reason != "overflow" || gap.From != last+1 {
		t.Fatalf("gap %+v after seq %d", gap, last)
	}
	t.Logf("overflow after %d events", last)
	in.plane.Publish("svc/x", event("after", map[string]any{}))
	select {
	case f := <-frames:
		if f.Env != nil {
			t.Fatalf("dropped subscription still delivers seq %d", f.Env.Seq)
		}
	case <-time.After(200 * time.Millisecond):
	}
	// The connection itself is still fine.
	var p map[string]any
	json.Unmarshal(do(t, c, Request{Op: "ping"}), &p)
	if p["pong"] != true {
		t.Fatalf("ping %v", p)
	}
}

func TestWSSessionErrors(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	c := dial(t, in)
	ctx := context.Background()
	if _, _, err := c.Subscribe(ctx, "ctl", nil, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Do(ctx, Request{Op: "subscribe", Ch: "ctl", Peer: "b"}); err == nil {
		t.Error("subscribe with a peer accepted")
	}
	if _, err := c.Do(ctx, Request{Op: "unsubscribe", Sub: 999}); err == nil {
		t.Error("unsubscribe of nothing accepted")
	}
	if _, err := c.Do(ctx, Request{Op: "unsubscribe", Sub: 1}); err != nil {
		t.Errorf("unsubscribe: %v", err)
	}
	if _, err := c.Do(ctx, Request{Op: "apply", Peer: "zz", Desired: running("x")}); err == nil || !strings.Contains(err.Error(), "unknown peer") {
		t.Errorf("unknown peer: %v", err)
	}
}
