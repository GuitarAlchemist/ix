package main

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// S2 slice 3, criterion 6 of ix#408: the link's queues are bounded per class, in records and
// in bytes, and control goes first. A durably slow parent (the mirroring loop spends 5 ms on
// every pipe frame while eight 10 ms pipes offer 800 a second) fills the pipe queue; from then
// on pipe frames are dropped, not queued without end and not held back on the reader: each run
// of drops is marked on the mirrored pipe before the record that followed, counted per pipe in
// the status, while a probe.created of the peer still crosses on the mirrored ctl within a
// second and the link stays up.
func TestControlStillFlowsWhenPipesOverloadTheLink(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	names := []string{"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"}
	specB := fakeSpec(t, "b", nil)
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json"), Probes: names}})
	b := newTestInstance(t, specB, dirB)
	a := newTestInstance(t, specA, dirA)
	waitStatus(t, a, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })
	a.plane.peers["b"].pipeDelay.Store(int64(5 * time.Millisecond))

	c := dial(t, a)
	var zero uint64
	pipes := subscribe(t, c, "peer/b/probe/", &zero)
	ctl := subscribe(t, c, "peer/b/ctl", &zero)
	own := subscribe(t, c, "ctl", &zero)
	for _, n := range names {
		probeOn(t, c, n, "heartbeat@1", true)
	}

	// The first run of drops, marked on its pipe: the parent's own envelope, saying how many
	// records the link itself discarded (an exact local count, not an inference).
	last := map[string]uint64{} // per pipe, the src_seq last read: the marker's successor is later
	markerAt := time.Time{}
	m := waitEnv(t, pipes, 20*time.Second, "a dropped marker", func(e *Envelope) bool {
		if e.Ev.Kind == "sample" {
			last[e.Ch] = e.SrcSeq
		}
		return e.Ev.Kind == linkGapKind && dataOf(e)["reason"] == "dropped"
	})
	markerAt = time.Now()
	g := gapOf(t, m)
	if m.Origin != "a" || m.SrcSeq != 0 || g.Peer != "b" || g.Dropped == 0 || g.SrcEpoch != b.plane.Epoch() {
		t.Fatalf("dropped marker: %s", m.Raw())
	}
	before, seen := last[m.Ch]
	if !seen {
		t.Fatalf("a dropped marker on %s before any record of it was read: the queue cannot have been full of them", m.Ch)
	}
	after := waitEnv(t, pipes, 5*time.Second, "the record after the marker", func(e *Envelope) bool {
		return e.Ch == m.Ch && e.Ev.Kind == "sample"
	})
	if after.SrcSeq <= before {
		t.Fatalf("after the marker on %s: src_seq %d, the last before was %d", m.Ch, after.SrcSeq, before)
	}

	// Control under the load: a node created on b now is announced on the mirrored ctl within
	// two seconds (measured in milliseconds; without the priority it waits behind up to 1024
	// pipe frames at 5 ms each, five seconds), not behind the queue of pipe frames.
	t0 := time.Now()
	do(t, c, Request{Op: "probe.create", Peer: "b", Probe: node("mark", "heartbeat@1", 1000, nil)})
	waitEnv(t, ctl, 2*time.Second, "the mirrored probe.created under load", func(e *Envelope) bool {
		return e.Ev.Kind == "probe.created" && dataOf(e)["node"] == "mark"
	})
	t.Logf("probe.created crossed in %s under load, %s after the first dropped marker", time.Since(t0), t0.Sub(markerAt))

	// The status counts the drops per pipe and the link is up; nothing said peer.down.
	st := waitStatus(t, a, 5*time.Second, "drops in the status", func(st StatusReply) bool {
		for _, p := range st.Peers[0].Probes {
			if p.Dropped > 0 {
				return true
			}
		}
		return false
	})
	if st.Peers[0].State != "up" {
		t.Fatalf("link %s under load", st.Peers[0].State)
	}
	for _, p := range st.Peers[0].Probes {
		if p.State != pipeMirrored {
			t.Fatalf("pipe %s is %s under load", p.Name, p.State)
		}
	}
	if f, err := own.next(200*time.Millisecond, func(e *Envelope) bool { return e.Ev.Kind == "peer.down" }); err == nil {
		t.Fatalf("the link went down under load: %s", f.Env.Raw())
	} else if !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("waiting for no peer.down: %v", err)
	}
}

// The peer dropping a pipe's subscription (its queue for the link overflowed) no longer ends
// the session, which took ctl and svc/ down with it: the pipe is marked `overflow` on its
// mirrored channel with what the peer said and opened live again, marked at that live start as
// any; a frame of the dropped subscription still queued is not mirrored. An overflow on ctl or
// svc/ still ends the session: those the link must resume.
func TestAnOverflowedPipeReopensLiveAndIsMarked(t *testing.T) {
	b := newTestInstance(t, fakeSpec(t, "b", nil), "")
	c := dial(t, b)
	l := newPeerLink(PeerSpec{Name: "b", Probes: []string{"cpu"}}, NewHub("a"))
	l.boot = b.plane.Epoch()
	l.pipes["cpu"].state, l.pipes["cpu"].sub = pipeMirrored, 7
	cur := l.hub.Subscribe("peer/b/", nil, "")
	refresh := make(chan struct{}, 1)
	q := newPipeQueue(1)
	fail := make(chan error, 1)
	ctx := context.Background()

	over := Gap{Reason: "overflow", From: 9, Epoch: b.plane.Epoch()}
	if err := l.apply(ctx, c, q, fail, refresh, 1, "b", peerFrame{prefix: "probe/cpu", f: Frame{Gap: &over}, sub: 7}); err != nil {
		t.Fatalf("a pipe's overflow ended the session: %v", err)
	}
	e := nextEnv(t, cur)
	g := gapOf(t, e)
	if e.Ch != "peer/b/probe/cpu" || e.Origin != "a" || g.Peer != "b" || g.Reason != "overflow" || g.From != 9 || g.SrcEpoch != b.plane.Epoch() {
		t.Fatalf("overflow marker: %s", e.Raw())
	}
	e2 := nextEnv(t, cur)
	if g2 := gapOf(t, e2); e2.Ch != "peer/b/probe/cpu" || g2.Reason != "live-start" || g2.SrcEpoch != b.plane.Epoch() {
		t.Fatalf("after the overflow: %s, want the reopened pipe's live start", e2.Raw())
	}
	l.mu.Lock()
	p := *l.pipes["cpu"]
	l.mu.Unlock()
	if p.state != pipeMirrored || p.sub == 7 || p.sub == 0 {
		t.Fatalf("pipe after the overflow: %+v, want mirrored under a new subscription", p)
	}
	stale := &Envelope{Ch: "probe/cpu", Seq: 8, Epoch: b.plane.Epoch(), Origin: "b", Ev: Event{Kind: "sample"}}
	if err := l.apply(ctx, c, q, fail, refresh, 1, "b", peerFrame{prefix: "probe/cpu", f: Frame{Env: stale}, sub: 7}); err != nil {
		t.Fatal(err)
	}
	wctx, cancel := context.WithTimeout(ctx, 300*time.Millisecond)
	defer cancel()
	if envs, _, _ := cur.Next(wctx, 1); len(envs) != 0 {
		t.Fatalf("a frame of the dropped subscription was mirrored: %s", envs[0].Raw())
	}
	if err := l.apply(ctx, c, q, fail, refresh, 1, "b", peerFrame{prefix: "svc/", f: Frame{Gap: &over}}); err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Fatalf("svc/ overflow: %v, want the session to end", err)
	}
	select {
	case err := <-fail:
		t.Fatalf("reopening failed: %v", err)
	default:
	}
}

// The pipe queue is bounded in bytes as in records: a record past the byte bound is refused and
// the bytes it reserved given back; what the loop took is given back too. The out-of-band slots
// for a pipe's overflow are one per wanted pipe and never wait: a send past them is refused.
func TestThePipeQueueIsBoundedInBytesToo(t *testing.T) {
	q := newPipeQueue(2)
	big := peerFrame{prefix: "probe/x", f: Frame{Bytes: 3 << 20}, sub: 1}
	if !q.offer(big) || q.bytes.Load() != 3<<20 {
		t.Fatalf("a first 3 MiB record: offered %v, bytes %d", false, q.bytes.Load())
	}
	two := peerFrame{prefix: "probe/y", f: Frame{Bytes: 2 << 20}, sub: 2}
	if q.offer(two) || q.bytes.Load() != 3<<20 || len(q.ch) != 1 {
		t.Fatalf("a 2 MiB record past the 4 MiB bound was queued, or its bytes kept: bytes %d, queued %d", q.bytes.Load(), len(q.ch))
	}
	q.took(<-q.ch)
	if q.bytes.Load() != 0 || !q.offer(two) || q.bytes.Load() != 2<<20 {
		t.Fatalf("after the loop took the first: bytes %d", q.bytes.Load())
	}
	small := peerFrame{prefix: "probe/z", f: Frame{Bytes: 100}, sub: 3}
	for i := 0; i < maxPipeQueue-1; i++ {
		if !q.offer(small) {
			t.Fatalf("record %d of %d refused under both bounds", i+2, maxPipeQueue)
		}
	}
	if q.offer(small) || len(q.ch) != maxPipeQueue {
		t.Fatalf("a record past the record bound was queued: %d", len(q.ch))
	}
	over := peerFrame{prefix: "probe/x", f: Frame{Gap: &Gap{Reason: "overflow", From: 9}}, sub: 1}
	sent := 0
	for i := 0; i < 3; i++ {
		select {
		case q.over <- over:
			sent++
		default:
		}
	}
	if sent != 2 {
		t.Fatalf("%d overflows slotted for two pipes", sent)
	}
}
