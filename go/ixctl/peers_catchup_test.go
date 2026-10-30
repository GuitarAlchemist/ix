package main

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

// S2 slice 3, criterion 7 of ix#408: a parent that cannot keep up with its peer's ctl / svc/
// stream catches up once, within a bound the link owns (frames, bytes and time per link, refilled
// slowly, never reset by a reconnect), then jumps to live with a dated marker on the mirrored
// ctl (`link.gap`, reason `jumped`, `from` the last seq mirrored, `to` the head the live
// subscription starts from) instead of replaying without end or growing memory; a peer that
// dropped the link's ctl / svc/ subscription (its queue toward the link overflowed) has that
// subscription reopened in the same session, not the session torn down; and the status tells a
// live flow from a reconciled state, lifting `degraded` only when both hold.

// The plane says when a resumed subscription is live: a frame {"sub", "live": true} once the
// replay has passed the head the subscription started from (at once when nothing was to be
// replayed), and the subscribe reply says that head. Only a subscriber that asked for it
// (`live: true`) gets the frame: every other consumer, the S7 fixtures included, is unchanged.
func TestAResumedSubscriptionIsToldWhenItIsLive(t *testing.T) {
	b := newTestInstance(t, fakeSpec(t, "b", nil), "")
	c := dial(t, b)
	for i := 1; i <= 3; i++ {
		b.plane.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	var zero uint64
	s, err := c.SubscribeFromLiveMarkers(context.Background(), "svc/", &zero, b.plane.Epoch(), runMarkers, 64)
	if err != nil {
		t.Fatal(err)
	}
	if s.Markers != runMarkers {
		t.Fatalf("the reply acknowledged markers %q, want %q", s.Markers, runMarkers)
	}
	if s.Head != b.plane.hub.Head() {
		t.Fatalf("the reply says head %d, the plane's is %d", s.Head, b.plane.hub.Head())
	}
	got := readFrames(t, s.Frames, 4)
	for i := 0; i < 3; i++ {
		if got[i].Env == nil || got[i].Env.Seq > s.Head || got[i].Live {
			t.Fatalf("replayed frame %d: %+v, want an envelope up to the head %d", i, got[i], s.Head)
		}
	}
	if !got[3].Live || got[3].Env != nil || got[3].Gap != nil {
		t.Fatalf("after the replay: %+v, want the live frame", got[3])
	}
	b.plane.Publish("svc/x", event("stats", map[string]any{"n": 4}))
	if f := readFrames(t, s.Frames, 1)[0]; f.Env == nil || f.Env.Seq <= s.Head {
		t.Fatalf("after the live frame: %+v, want an envelope past the head %d", f, s.Head)
	}

	// Nothing to replay: live at once, before any event.
	s2, err := c.SubscribeFromLive(context.Background(), "svc/", uint64p(b.plane.hub.Head()), b.plane.Epoch(), 64)
	if err != nil {
		t.Fatal(err)
	}
	if f := readFrames(t, s2.Frames, 1)[0]; !f.Live {
		t.Fatalf("with nothing to replay the first frame is %+v, want the live frame", f)
	}

	// A subscriber that did not ask is told nothing: the four envelopes, then silence.
	_, frames, err := c.SubscribeFrom(context.Background(), "svc/", &zero, b.plane.Epoch(), 64)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range readFrames(t, frames, 4) {
		if f.Env == nil {
			t.Fatalf("a subscriber that did not ask got %+v", f)
		}
	}
	select {
	case f := <-frames:
		t.Fatalf("a subscriber that did not ask got %+v after its replay", f)
	case <-time.After(200 * time.Millisecond):
	}
}

func readFrames(t *testing.T, frames <-chan Frame, n int) []Frame {
	t.Helper()
	var got []Frame
	for len(got) < n {
		select {
		case f, ok := <-frames:
			if !ok {
				t.Fatalf("the connection ended after %d frames", len(got))
			}
			got = append(got, f)
		case <-time.After(5 * time.Second):
			t.Fatalf("only %d of %d frames: %+v", len(got), n, got)
		}
	}
	return got
}

// A catch-up past the link's allowance jumps: the link replays at most the allowance (fifty
// frames here, the test knob), marks the jump on the peer's mirrored ctl, from the last seq it
// mirrored to the head its live subscription starts from, dated and naming the source boot, and
// mirrors what the peer publishes from then on. The allowance is the link's, slowly refilled: a
// second outage of the same size jumps at once, since a reconnect does not refill it.
func TestACatchUpPastTheAllowanceJumpsLiveAndIsMarked(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil), dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	l.allowFrames.Store(50)
	cur := l.hub.Subscribe("peer/b/", nil, "")
	run := func() context.CancelFunc {
		ctx, cancel := context.WithCancel(context.Background())
		go l.run(ctx)
		waitLink(t, l, "up")
		return cancel
	}
	cancel := run()
	b.plane.Publish("svc/x", event("stats", map[string]any{"n": 0}))
	var mirrored *Envelope
	for mirrored == nil {
		if e := nextEnvWithin(t, cur, 10*time.Second); e.Ch == "peer/b/svc/x" {
			mirrored = e
		}
	}
	cancel()
	waitLink(t, l, "down")
	for i := 1; i <= 500; i++ {
		b.plane.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	headAfterBurst := b.plane.hub.Head()
	cancel = run()

	// Up to the allowance replayed, in order, then the marker.
	readUntilJump := func(last uint64) (replayed int, m linkGap, mEnv *Envelope, lastSeq uint64) {
		t.Helper()
		lastSeq = last
		for m.Reason != "jumped" {
			e := nextEnvWithin(t, cur, 10*time.Second)
			switch {
			case e.Ch == "peer/b/svc/x":
				replayed++
				if e.SrcSeq <= lastSeq {
					t.Fatalf("mirrored src_seq %d after %d", e.SrcSeq, lastSeq)
				}
				lastSeq = e.SrcSeq
			case e.Ev.Kind == linkGapKind:
				m, mEnv = gapOf(t, e), e
			}
		}
		return
	}
	replayed, m, mEnv, lastSeq := readUntilJump(mirrored.SrcSeq)
	t0 := time.Now() // the allowance refills by wall time from here: a slow run earns a few frames back
	if replayed == 0 || replayed > 50 {
		t.Fatalf("%d frames replayed before the jump, want up to the allowance of 50 and more than none", replayed)
	}
	if mEnv.Ch != "peer/b/ctl" || mEnv.Origin != "a" || mEnv.SrcSeq != 0 || mEnv.T == 0 {
		t.Fatalf("the jump marker is %s, want the parent's own dated envelope on the mirrored ctl", mEnv.Raw())
	}
	if m.Peer != "b" || m.Prefix != "svc/" || m.From != lastSeq || m.To < headAfterBurst || m.SrcEpoch != b.plane.Epoch() {
		t.Fatalf("jump marker %+v, want prefix svc/ from %d (the last mirrored) to at least %d, boot %s", m, lastSeq, headAfterBurst, b.plane.Epoch())
	}
	// Live from the jump on: what the peer publishes now is mirrored, past the marker's `to`.
	b.plane.Publish("svc/x", event("stats", map[string]any{"n": 501}))
	var after *Envelope
	for after == nil {
		if e := nextEnvWithin(t, cur, 10*time.Second); e.Ch == "peer/b/svc/x" {
			after = e
		}
	}
	if after.SrcSeq <= m.To {
		t.Fatalf("after the jump: src_seq %d, the marker said live from %d", after.SrcSeq, m.To)
	}
	st := waitLinkStatus(t, l, func(st PeerStatus) bool { return st.Live && st.Reconciled && !st.Degraded })
	if st.Jumps != 1 {
		t.Fatalf("status after the jump: %+v, want one jump counted", st)
	}

	// The allowance is the link's, not the session's.
	cancel()
	waitLink(t, l, "down")
	for i := 502; i <= 700; i++ {
		b.plane.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	cancel = run()
	defer cancel()
	replayed, _, _, _ = readUntilJump(after.SrcSeq)
	if earned := 2 + time.Since(t0).Seconds()*50/replayRefill.Seconds(); float64(replayed) > earned {
		t.Fatalf("%d frames replayed after the second outage, more than the %.0f the time since the first jump refilled: the reconnect refilled the allowance", replayed, earned)
	}
	if st := l.status(); st.Jumps != 2 {
		t.Fatalf("status after the second jump: %+v, want two jumps counted", st)
	}
}

func waitLink(t *testing.T, l *peerLink, state string) {
	t.Helper()
	waitLinkStatus(t, l, func(st PeerStatus) bool { return st.State == state })
}

func waitLinkStatus(t *testing.T, l *peerLink, ok func(PeerStatus) bool) PeerStatus {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for {
		st := l.status()
		if ok(st) {
			return st
		}
		if time.Now().After(deadline) {
			t.Fatalf("link status stayed %+v", st)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// degraded is lifted only when the flow is live (both control subscriptions past their replay)
// and the state reconciled (the peer's list applied this session, no wanted pipe still opening);
// the two are told apart, and a link that is down is neither.
func TestDegradedIsLiftedOnlyWhenLiveAndReconciled(t *testing.T) {
	l := newPeerLink(PeerSpec{Name: "b", Probes: []string{"cpu"}}, NewHub("a"))
	want := func(what string, live, reconciled, degraded bool, jumps uint64) {
		t.Helper()
		st := l.status()
		if st.Live != live || st.Reconciled != reconciled || st.Degraded != degraded || st.Jumps != jumps {
			t.Fatalf("%s: live %v reconciled %v degraded %v jumps %d, want %v %v %v %d", what, st.Live, st.Reconciled, st.Degraded, st.Jumps, live, reconciled, degraded, jumps)
		}
	}
	set := func(f func()) { l.mu.Lock(); f(); l.mu.Unlock() }
	want("down", false, false, true, 0)
	set(func() { l.up = true })
	want("up, both catching up", false, false, true, 0)
	set(func() { l.ctl["ctl"].live = true })
	want("svc/ still catching up", false, false, true, 0)
	set(func() { l.ctl["svc/"].live = true })
	want("live, list not applied", true, false, true, 0)
	set(func() { l.listed = true; l.pipes["cpu"].state = pipeOpening })
	want("live, a pipe opening", true, false, true, 0)
	set(func() { l.pipes["cpu"].state = pipeMirrored })
	want("live and reconciled", true, true, false, 0)
	set(func() { l.pipes["cpu"].state = pipeNotExportable })
	want("a declined pipe is reconciled", true, true, false, 0)
	set(func() { l.ctl["svc/"].live = false; l.jumps = 1 })
	want("catching up again after a jump", false, true, true, 1)
	set(func() { l.ctl["svc/"].live = true; l.up = false })
	want("down again", false, false, true, 1)
}

// A sustained control stream above the reader's capacity: eight thousand svc/ events in a few
// milliseconds at a parent that spends half a millisecond on each, more than the session queue,
// the subscription channel and the link's queue hold together. The peer drops the link's svc/
// subscription (overflow); the link reopens it in the same session, catches up within its
// allowance (a hundred frames here) and jumps, marked on the mirrored ctl; it never reconnects
// (no peer.down on the parent's own ctl, the peer up throughout); once the burst is through the
// status settles live and reconciled, degraded lifted, the jumps counted, and a live event
// crosses.
func TestASustainedControlStreamAboveTheReaderJumpsWithoutReconnecting(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	specB := fakeSpec(t, "b", nil)
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json"), Probes: []string{"cpu"}}})
	b := newTestInstance(t, specB, dirB)
	a := newTestInstance(t, specA, dirA)
	waitStatus(t, a, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })
	l := a.plane.peers["b"]
	l.ctlDelay.Store(int64(500 * time.Microsecond))
	l.allowFrames.Store(100)
	b.web.writeDelay.Store(int64(time.Millisecond)) // the overflow is the peer's queue filling behind its writer, whatever the kernel buffers hold

	c := dial(t, a)
	var zero uint64
	own := subscribe(t, c, "ctl", &zero)
	ctl := subscribe(t, c, "peer/b/ctl", &zero)
	svc := subscribe(t, c, "peer/b/svc/", &zero)
	for i := 1; i <= 8000; i++ {
		b.plane.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	m := waitEnv(t, ctl, 30*time.Second, "a jumped marker", func(e *Envelope) bool {
		return e.Ev.Kind == linkGapKind && dataOf(e)["reason"] == "jumped"
	})
	g := gapOf(t, m)
	if m.Origin != "a" || m.T == 0 || g.Prefix != "svc/" || g.To <= g.From || g.SrcEpoch != b.plane.Epoch() {
		t.Fatalf("jump marker: %s, want svc/ from the last mirrored seq to a later head, dated, naming b's boot", m.Raw())
	}
	st := waitStatus(t, a, 30*time.Second, "the link live and reconciled again", func(st StatusReply) bool {
		p := st.Peers[0]
		return p.State == "up" && p.Live && p.Reconciled && !p.Degraded
	})
	if st.Peers[0].Jumps == 0 {
		t.Fatalf("status after the load: %+v, want the jumps counted", st.Peers[0])
	}
	if f, err := own.next(200*time.Millisecond, func(e *Envelope) bool { return e.Ev.Kind == "peer.down" }); err == nil {
		t.Fatalf("the link reconnected under the load: %s", f.Env.Raw())
	}
	b.plane.Publish("svc/x", event("stats", map[string]any{"n": 8001}))
	e := waitEnv(t, svc, 5*time.Second, "a live event after the load", func(e *Envelope) bool {
		return dataOf(e)["n"] == float64(8001)
	})
	if e.SrcSeq <= g.To {
		t.Fatalf("the live event has src_seq %d, the first jump said live from %d", e.SrcSeq, g.To)
	}
	t.Logf("%d jumps under the load", st.Peers[0].Jumps)
}

// The live frame comes after what the plane answers first and comes even when nothing of the
// prefix is to be replayed (Codex round 43): a resume with a stale epoch gets the `epoch-changed`
// gap first, then the live frame; a resume on a prefix with no record while the head advanced on
// other channels gets the live frame without waiting for a record of the prefix.
func TestTheLiveFrameFollowsTheGapAndComesOnAQuietPrefix(t *testing.T) {
	b := newTestInstance(t, fakeSpec(t, "b", nil), "")
	c := dial(t, b)
	s, err := c.SubscribeFromLive(context.Background(), "svc/", uint64p(5), "not-this-boot", 64)
	if err != nil {
		t.Fatal(err)
	}
	got := readFrames(t, s.Frames, 2)
	if got[0].Gap == nil || got[0].Gap.Reason != "epoch-changed" || got[0].Live {
		t.Fatalf("a resume with a stale epoch got %+v first, want the epoch-changed gap", got[0])
	}
	if !got[1].Live {
		t.Fatalf("after the gap: %+v, want the live frame", got[1])
	}

	for i := 1; i <= 3; i++ {
		b.plane.Publish("ctl", event("noise", map[string]any{"n": i}))
	}
	var zero uint64
	s2, err := c.SubscribeFromLive(context.Background(), "svc/", &zero, b.plane.Epoch(), 64)
	if err != nil {
		t.Fatal(err)
	}
	if f := readFrames(t, s2.Frames, 1)[0]; !f.Live {
		t.Fatalf("a resume on a quiet prefix got %+v, want the live frame once the replay is past the head", f)
	}
}

// A reconnect past the allowance that has nothing to skip marks no jump (Codex round 43): the
// peer's head is the last seq mirrored on the prefix, in the same boot, so a live subscription
// misses nothing, and `jumped` means something was skipped. Once the peer published again, the
// same reopen jumps and is marked from that seq.
func TestAnEmptyReconnectPastTheAllowanceMarksNoJump(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil), dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	cur := l.hub.Subscribe("peer/b/", nil, "")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go l.run(ctx)
	waitLink(t, l, "up")
	first := b.plane.Publish("svc/x", event("stats", map[string]any{"n": 0}))
	for e := nextEnvWithin(t, cur, 10*time.Second); e.Ch != "peer/b/svc/x"; e = nextEnvWithin(t, cur, 10*time.Second) {
	}
	l.allowFrames.Store(0) // no catch-up may begin from now on
	l.mu.Lock()
	c, sub := l.client, l.ctl["svc/"].sub
	l.mu.Unlock()
	merged := make(chan peerFrame, 64)
	fail := make(chan error, 1)
	reopen := func(prefix string, sub int64) int64 {
		t.Helper()
		l.reopenControl(ctx, c, nil, merged, fail, peerFrame{prefix: prefix, sub: sub, f: Frame{Gap: &Gap{Reason: "overflow"}}})
		deadline := time.Now().Add(5 * time.Second)
		for {
			l.mu.Lock()
			now := l.ctl[prefix].sub
			l.mu.Unlock()
			if now != 0 && now != sub {
				return now
			}
			if time.Now().After(deadline) {
				t.Fatalf("%s was not reopened after %d", prefix, sub)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	markers := func(d time.Duration) (ms []linkGap) {
		nctx, ncancel := context.WithTimeout(ctx, d)
		defer ncancel()
		for {
			envs, _, err := cur.Next(nctx, 16)
			if err != nil {
				return ms
			}
			for _, e := range envs {
				if e.Ev.Kind == linkGapKind {
					ms = append(ms, gapOf(t, e))
				}
			}
		}
	}
	sub = reopen("svc/", sub)
	if ms := markers(300 * time.Millisecond); len(ms) != 0 {
		t.Fatalf("a reopen with nothing to skip was marked: %+v", ms)
	}
	if st := l.status(); st.Jumps != 0 {
		t.Fatalf("a reopen with nothing to skip was counted as a jump: %+v", st)
	}
	select {
	case err := <-fail:
		t.Fatalf("reopening failed: %v", err)
	default:
	}

	// The record the reopen must skip is published once the prefix's subscription is taken away
	// from the link (a sentinel id no frame carries, the old one forgotten by the client), so no
	// schedule can mirror it first: a frame of the old subscription fails `current` whenever it
	// arrives, and the reopen finds the sentinel it is given. Codex rounds 47 and 48: a writer hold
	// around the publish only narrowed the window, this is the happens-before edge.
	standDown := func(prefix string, sub int64) int64 {
		l.mu.Lock()
		l.ctl[prefix].sub = -1
		l.mu.Unlock()
		c.forget(sub)
		return -1
	}
	sub = standDown("svc/", sub)
	second := b.plane.Publish("svc/x", event("stats", map[string]any{"n": 1}))
	reopen("svc/", sub)
	ms := markers(2 * time.Second)
	if len(ms) != 1 || ms[0].Reason != "jumped" || ms[0].Prefix != "svc/" || ms[0].From != first.Seq || ms[0].To < second.Seq {
		t.Fatalf("a reopen past a new record: markers %+v, want one jumped on svc/ from %d to at least %d", ms, first.Seq, second.Seq)
	}
	if st := l.status(); st.Jumps != 1 {
		t.Fatalf("status after the real jump: %+v, want one jump counted", st)
	}

	// The same on ctl, the prefix whose jump costs the reconciled state (Codex round 46): a reopen
	// with nothing to skip keeps it, a reopen past a new record clears it once.
	firstCtl := b.plane.Publish("ctl", event("noise", map[string]any{"n": 0}))
	for e := nextEnvWithin(t, cur, 10*time.Second); e.Ch != "peer/b/ctl"; e = nextEnvWithin(t, cur, 10*time.Second) {
	}
	state := func() (int64, bool, uint64) {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.ctl["ctl"].sub, l.listed, l.unlisted
	}
	ctlSub, listed0, gen0 := state()
	if !listed0 {
		t.Fatal("the link is not reconciled before the ctl reopen")
	}
	ctlSub = reopen("ctl", ctlSub)
	if ms := markers(300 * time.Millisecond); len(ms) != 0 {
		t.Fatalf("a ctl reopen with nothing to skip was marked: %+v", ms)
	}
	if _, listed1, gen1 := state(); !listed1 || gen1 != gen0 {
		t.Fatalf("a ctl reopen with nothing to skip cost the reconciled state: listed %v, generation %d (was %d)", listed1, gen1, gen0)
	}
	ctlSub = standDown("ctl", ctlSub) // as above: the record is published once the link has let the subscription go
	secondCtl := b.plane.Publish("ctl", event("noise", map[string]any{"n": 1}))
	reopen("ctl", ctlSub)
	ms = markers(2 * time.Second)
	if len(ms) != 1 || ms[0].Reason != "jumped" || ms[0].Prefix != "ctl" || ms[0].From != firstCtl.Seq || ms[0].To < secondCtl.Seq {
		t.Fatalf("a ctl reopen past a new record: markers %+v, want one jumped on ctl from %d to at least %d", ms, firstCtl.Seq, secondCtl.Seq)
	}
	if _, listed2, gen2 := state(); listed2 || gen2 != gen0+1 {
		t.Fatalf("a jump on ctl did not clear the reconciled state once: listed %v, generation %d (was %d)", listed2, gen2, gen0)
	}
	if st := l.status(); st.Jumps != 2 {
		t.Fatalf("status after the ctl jump: %+v, want two jumps counted", st)
	}
}
