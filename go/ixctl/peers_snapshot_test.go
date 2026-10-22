package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"
)

// S2 slice 4 (criteria 3 and 4 of ix#408, Gaia's precision IX-S2-20261022-0119): the probe list
// is a snapshot bound to a cursor of the plane's own stream, with creation identities. The list
// taken with `with_cursor` says the head it stands at (`cursor`: `seq`, `epoch`) and, per node,
// the seq of its `probe.created` (`created_seq`): read under the tree lock, where every lifecycle
// event is published, so the list is exactly what ctl says up to the cursor and nothing of what
// follows. The link reads its list so, skips the lifecycle events the snapshot already holds
// (seq at or before the cursor), keeps each pipe's identity (the incarnation it mirrors), and
// takes a new snapshot when it jumped on ctl (the lifecycle events of the jumped span are lost):
// a node gone from the list is dropped, a name whose identity changed is reopened as the new
// incarnation, and the state is reconciled again.

// The list is exact at its cursor while the tree churns: for every snapshot, folding the plane's
// own stream from the start to the cursor (created, started, stopped, removed) gives the list's
// nodes, identities, export flags and states; a leaf whose stop is in progress (its probe.stopped
// not out yet) says `stopping`, never `stopped` ahead of ctl.
func TestTheListIsExactAtItsCursor(t *testing.T) {
	b := newTestInstance(t, fakeSpec(t, "b", nil), "")
	c := dial(t, b)
	p := b.plane
	cur := p.Subscribe("", uint64p(0), "") // every channel, from the start: folded on from one cursor to the next
	type folded struct {
		created uint64
		export  bool
		state   string
	}
	want := map[string]folded{}
	last := uint64(0)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // the churn: leaves under a mux and at the root, created, started, stopped, removed
		defer wg.Done()
		op := func(req Request) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			c.Do(ctx, req) // a refusal (a name taken, a node gone) is part of the churn
		}
		op(Request{Op: "probe.create", Probe: node("agg", "mux@1", 0, nil)})
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			name := fmt.Sprintf("leaf%d", i%3)
			extra := map[string]any{"parent": "agg"}
			if i%2 == 0 {
				extra["export"] = true
			}
			if i%3 == 2 {
				extra = nil // a root leaf
			}
			op(Request{Op: "probe.create", Probe: node(name, "heartbeat@1", 100, extra)})
			op(Request{Op: "probe.start", Node: name})
			if i%2 == 1 {
				op(Request{Op: "probe.stop", Node: name})
			}
			op(Request{Op: "probe.remove", Node: fmt.Sprintf("leaf%d", (i+1)%3)})
		}
	}()
	for k := 0; k < 25; k++ {
		time.Sleep(15 * time.Millisecond)
		var list struct {
			Cursor struct {
				Seq   uint64 `json:"seq"`
				Epoch string `json:"epoch"`
			} `json:"cursor"`
			Nodes []struct {
				Name       string `json:"name"`
				State      string `json:"state"`
				Export     bool   `json:"export"`
				CreatedSeq uint64 `json:"created_seq"`
			} `json:"nodes"`
		}
		if err := json.Unmarshal(do(t, c, Request{Op: "probe.list", WithCursor: true}), &list); err != nil {
			t.Fatal(err)
		}
		if list.Cursor.Epoch != p.Epoch() || list.Cursor.Seq == 0 {
			t.Fatalf("snapshot %d: cursor %+v, want this boot %s and a seq", k, list.Cursor, p.Epoch())
		}
		got := map[string]folded{}
		for _, n := range list.Nodes {
			state := n.State
			if state == "stopping" { // its probe.stopped is not out yet: ctl still says running
				state = probeRunning
			}
			got[n.Name] = folded{n.CreatedSeq, n.Export, state}
		}
		for last < list.Cursor.Seq {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			envs, gap, err := cur.Next(ctx, int(min(64, list.Cursor.Seq-last))) // never past the cursor
			cancel()
			if err != nil || gap != nil {
				t.Fatalf("snapshot %d: folding the stream to %d: gap %v err %v (the ring must hold the whole run)", k, list.Cursor.Seq, gap, err)
			}
			for _, e := range envs {
				last = e.Seq
				if e.Seq > list.Cursor.Seq {
					t.Fatalf("snapshot %d: the read ran past the cursor %d to %d", k, list.Cursor.Seq, e.Seq)
				}
				if e.Ch != "ctl" {
					continue
				}
				d := dataOf(e)
				name, _ := d["node"].(string)
				switch e.Ev.Kind {
				case "probe.created":
					want[name] = folded{e.Seq, d["export"] == true, probeStopped}
				case "probe.started":
					f := want[name]
					f.state = probeRunning
					want[name] = f
				case "probe.stopped":
					f := want[name]
					f.state = probeStopped
					want[name] = f
				case "probe.removed":
					delete(want, name)
				}
			}
		}
		if len(got) != len(want) {
			t.Fatalf("snapshot %d at %d: list %v, the stream folded to the cursor says %v", k, list.Cursor.Seq, got, want)
		}
		for name, w := range want {
			if g, ok := got[name]; !ok || g != w {
				t.Fatalf("snapshot %d at %d: %s is %+v in the list, %+v by the stream (list %v)", k, list.Cursor.Seq, name, got[name], w, got)
			}
		}
	}
	close(stop)
	wg.Wait()
}

// The link reads the list as a snapshot: a lifecycle event at or before the cursor is already
// reflected and is not followed (it would reopen a pipe the list no longer has, or close one it
// has); one after it is. Each pipe carries the identity of the incarnation it mirrors (the seq and
// boot of its probe.created), from the list or from the followed probe.created, and the status
// says it. A new snapshot (what the link takes after a jump on ctl) reconciles: a name whose
// identity changed is reopened as the new incarnation under a new subscription, a name gone from
// the list is dropped, and the state is reconciled again.
func TestTheLinkReadsTheListAsASnapshotWithIdentities(t *testing.T) {
	b := newTestInstance(t, fakeSpec(t, "b", nil), "")
	c := dial(t, b)
	epoch := b.plane.Epoch()
	l := newPeerLink(PeerSpec{Name: "b", Probes: []string{"cpu"}}, NewHub("a"))
	l.boot = epoch
	q := newPipeQueue()
	merged := make(chan peerFrame, 16)
	fail := make(chan error, 1)
	refresh := make(chan struct{}, 1)
	ctx := context.Background()
	pipe := func() pipeSub {
		l.mu.Lock()
		defer l.mu.Unlock()
		return *l.pipes["cpu"]
	}
	waitPipe := func(what string, ok func(pipeSub) bool) pipeSub {
		t.Helper()
		deadline := time.Now().Add(5 * time.Second)
		for {
			p := pipe()
			if ok(p) {
				return p
			}
			if time.Now().After(deadline) {
				t.Fatalf("waiting for %s: pipe %+v", what, p)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	list := func(cursor uint64, nodes string) json.RawMessage {
		return json.RawMessage(fmt.Sprintf(`{"cursor":{"seq":%d,"epoch":%q},"nodes":[%s]}`, cursor, epoch, nodes))
	}
	ctl := func(seq uint64, kind, node string, extra map[string]any) peerFrame {
		d := map[string]any{"node": node, "kind": "heartbeat@1"}
		for k, v := range extra {
			d[k] = v
		}
		return peerFrame{prefix: "ctl", f: Frame{Env: &Envelope{Ch: "ctl", Seq: seq, Epoch: epoch, Origin: "b", Ev: event(kind, d)}}}
	}

	// The snapshot: cpu exported, created at 7, the list standing at 10.
	if err := l.snapshot(ctx, c, q, fail, list(10, `{"name":"cpu","export":true,"created_seq":7}`)); err != nil {
		t.Fatal(err)
	}
	first := waitPipe("cpu mirrored from the list", func(p pipeSub) bool { return p.state == pipeMirrored })
	if first.created != (srcPoint{7, epoch}) {
		t.Fatalf("cpu's identity from the list: %+v, want seq 7 of %s", first.created, epoch)
	}
	if st := l.status().Probes[0]; st.CreatedSeq != 7 || st.SrcEpoch != epoch {
		t.Fatalf("status identity: %+v", st)
	}
	// A probe.removed of cpu at 9 is before the cursor: the list already reflects what followed
	// it (cpu is there), so it is not followed.
	if err := l.apply(ctx, c, q, merged, fail, refresh, 1, "b", ctl(9, "probe.removed", "cpu", map[string]any{"reason": "removed"})); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if p := pipe(); p.state != pipeMirrored || p.sub != first.sub {
		t.Fatalf("a probe.removed before the cursor was followed: %+v", p)
	}
	// One at 11 is after the cursor: followed.
	if err := l.apply(ctx, c, q, merged, fail, refresh, 1, "b", ctl(11, "probe.removed", "cpu", map[string]any{"reason": "removed"})); err != nil {
		t.Fatal(err)
	}
	waitPipe("cpu absent", func(p pipeSub) bool { return p.state == pipeAbsent })
	// A probe.created at 12, exported: followed, and the identity is the event's.
	if err := l.apply(ctx, c, q, merged, fail, refresh, 1, "b", ctl(12, "probe.created", "cpu", map[string]any{"export": true})); err != nil {
		t.Fatal(err)
	}
	second := waitPipe("cpu mirrored from the event", func(p pipeSub) bool { return p.state == pipeMirrored })
	if second.created != (srcPoint{12, epoch}) || second.sub == first.sub {
		t.Fatalf("cpu after the followed probe.created: %+v (was %+v)", second, first)
	}

	// A new snapshot with another incarnation of the name (created at 15): the pipe is reopened
	// as that incarnation, under a new subscription; the cursor moves, the state is reconciled.
	if err := l.snapshot(ctx, c, q, fail, list(20, `{"name":"cpu","export":true,"created_seq":15}`)); err != nil {
		t.Fatal(err)
	}
	third := waitPipe("cpu reopened as the new incarnation", func(p pipeSub) bool {
		return p.state == pipeMirrored && p.created == srcPoint{15, epoch}
	})
	if third.sub == second.sub || third.sub == 0 {
		t.Fatalf("the new incarnation kept the old subscription: %+v (was %+v)", third, second)
	}
	l.mu.Lock()
	cursor, listed := l.cursor, l.listed
	l.mu.Unlock()
	if cursor != (srcPoint{20, epoch}) || !listed {
		t.Fatalf("after the snapshot: cursor %+v listed %v", cursor, listed)
	}
	// The same identity again: nothing moves.
	if err := l.snapshot(ctx, c, q, fail, list(22, `{"name":"cpu","export":true,"created_seq":15}`)); err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)
	if p := pipe(); p.sub != third.sub {
		t.Fatalf("a snapshot with the same identity reopened the pipe: %+v", p)
	}
	// A snapshot without the name: dropped.
	if err := l.snapshot(ctx, c, q, fail, list(30, ``)); err != nil {
		t.Fatal(err)
	}
	waitPipe("cpu dropped by the snapshot", func(p pipeSub) bool { return p.state == pipeAbsent && p.created == srcPoint{} })
	select {
	case err := <-fail:
		t.Fatalf("the link failed: %v", err)
	default:
	}
}

// A jump on ctl loses the lifecycle events of its span, so the link takes a new snapshot right
// after it: here the peer holds an exported cpu the link never heard of (no list was read); the
// jump is marked, the list is read with a cursor, cpu is mirrored and the state is reconciled
// again.
func TestAJumpOnCtlTakesANewSnapshot(t *testing.T) {
	b := newTestInstance(t, fakeSpec(t, "b", nil), "")
	c := dial(t, b)
	do(t, c, Request{Op: "probe.create", Probe: node("cpu", "heartbeat@1", 10, map[string]any{"export": true})})
	l := newPeerLink(PeerSpec{Name: "b", Probes: []string{"cpu"}}, NewHub("a"))
	l.boot = b.plane.Epoch()
	cur := l.hub.Subscribe("peer/b/", nil, "")
	q := newPipeQueue()
	merged := make(chan peerFrame, 64)
	fail := make(chan error, 1)
	l.mu.Lock()
	l.client = c // the session's client: a new snapshot applies only while it still is
	l.ctl["ctl"].sub, l.ctl["ctl"].live = 5, false
	l.since["ctl"] = 3
	l.mu.Unlock()
	l.jumpControl(context.Background(), c, q, merged, fail, "ctl", 5)
	m := nextEnvWithin(t, cur, 5*time.Second)
	if g := gapOf(t, m); m.Ch != "peer/b/ctl" || g.Reason != "jumped" || g.Prefix != "ctl" || g.From != 3 || g.To == 0 {
		t.Fatalf("after the jump: %s, want the jumped marker on the mirrored ctl from 3", m.Raw())
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		l.mu.Lock()
		p, listed, cursor := *l.pipes["cpu"], l.listed, l.cursor
		l.mu.Unlock()
		if p.state == pipeMirrored && listed && cursor.seq >= 1 && cursor.epoch == b.plane.Epoch() && p.created.epoch == b.plane.Epoch() && p.created.seq != 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("after the jump: pipe %+v listed %v cursor %+v, want cpu mirrored from a new snapshot", p, listed, cursor)
		}
		time.Sleep(10 * time.Millisecond)
	}
	e := nextEnvWithin(t, cur, 5*time.Second)
	if g := gapOf(t, e); e.Ch != "peer/b/probe/cpu" || g.Reason != "live-start" {
		t.Fatalf("after the snapshot: %s, want cpu's live start", e.Raw())
	}
	select {
	case err := <-fail:
		t.Fatalf("the link failed: %v", err)
	default:
	}
}
