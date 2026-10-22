package main

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// S2 slice 3 (D4 ix#402 Q6, criteria 4 and 5 of ix#408): what the mirror skips is marked in
// the mirrored stream itself, as a `link.gap` envelope the parent publishes (its own origin, no
// provenance), distinct from a local gap frame and carrying no count. A pipe starts live on
// every session and after every drop, so each live start is marked on the pipe before its first
// mirrored frame, naming the source boot it now reads and the last frame it mirrored before;
// a gap the peer answers on ctl / svc/ (its history expired, its restart) is relayed on the
// peer's mirrored ctl.
func TestAConsumerOnTheParentIsToldWhatTheMirrorSkipped(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	specB := fakeSpec(t, "b", nil)
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json"), Probes: []string{"cpu"}}})
	b := newTestInstance(t, specB, dirB)
	a := newTestInstance(t, specA, dirA)
	waitStatus(t, a, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })
	c := dial(t, a)
	var zero uint64
	w := subscribe(t, c, "peer/b/", &zero)
	epoch1 := b.plane.Epoch()

	// First live start: the first envelope on the pipe is the marker, then the samples.
	probeOn(t, c, "cpu", "heartbeat@1", true)
	first := waitEnv(t, w, 5*time.Second, "first envelope on the mirrored pipe", func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu"
	})
	m := gapOf(t, first)
	if first.Origin != "a" || first.SrcSeq != 0 || first.SrcEpoch != "" || m.Peer != "b" || m.Reason != "live-start" || m.SrcEpoch != epoch1 || m.LastSrcSeq != 0 {
		t.Fatalf("first envelope on the pipe: %s, want the parent's live-start marker for b's boot %s with nothing mirrored before", first.Raw(), epoch1)
	}
	s := waitEnv(t, w, 5*time.Second, "a mirrored sample", func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu" && e.Ev.Kind == "sample"
	})
	if s.SrcEpoch != epoch1 {
		t.Fatalf("sample from boot %s, want %s", s.SrcEpoch, epoch1)
	}

	// B restarts under its name: the link finds a new instance (it resumes nothing from it, so
	// the peer answers no gap) and says epoch-changed itself on peer/b/ctl, old boot to new; the
	// tree is gone with the old boot (not persisted), so the pipe is absent until the node is
	// created again, and that live start is marked with the last frame mirrored of the old boot.
	b.close()
	waitStatus(t, a, 10*time.Second, "peer b down", func(st StatusReply) bool { return st.Peers[0].State == "down" })
	b2 := newTestInstance(t, specB, dirB)
	epoch2 := b2.plane.Epoch()
	waitStatus(t, a, 15*time.Second, "peer b up again", func(st StatusReply) bool { return st.Peers[0].State == "up" })
	relayed := waitEnv(t, w, 10*time.Second, "the relayed epoch change", func(e *Envelope) bool {
		return e.Ch == "peer/b/ctl" && e.Ev.Kind == "link.gap"
	})
	r := gapOf(t, relayed)
	if relayed.Origin != "a" || r.Peer != "b" || r.Reason != "epoch-changed" || r.SrcEpoch != epoch2 || r.LastSrcEpoch != epoch1 {
		t.Fatalf("relayed gap: %s, want epoch-changed from %s to %s", relayed.Raw(), epoch1, epoch2)
	}
	probeOn(t, c, "cpu", "heartbeat@1", true)
	again := waitEnv(t, w, 10*time.Second, "the pipe's envelope after the restart", func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu" && e.SrcEpoch != epoch1
	})
	m2 := gapOf(t, again)
	if m2.Reason != "live-start" || m2.SrcEpoch != epoch2 || m2.LastSrcSeq == 0 || m2.LastSrcEpoch != epoch1 {
		t.Fatalf("the restarted pipe's first envelope: %s, want a live-start marker for %s naming the last frame of %s", again.Raw(), epoch2, epoch1)
	}
	if m2.LastSrcSeq < s.SrcSeq {
		t.Fatalf("marker says the last mirrored frame was %d, but %d was read", m2.LastSrcSeq, s.SrcSeq)
	}
	s2 := waitEnv(t, w, 5*time.Second, "a sample of the new boot", func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu" && e.Ev.Kind == "sample"
	})
	if s2.SrcEpoch != epoch2 {
		t.Fatalf("sample from boot %s after the restart, want %s", s2.SrcEpoch, epoch2)
	}
}

// linkGap is the data of a link.gap envelope.
type linkGap struct {
	Peer         string `json:"peer"`
	Prefix       string `json:"prefix,omitempty"`
	Reason       string `json:"reason"`
	SrcEpoch     string `json:"src_epoch,omitempty"`
	From         uint64 `json:"from,omitempty"`
	Oldest       uint64 `json:"oldest,omitempty"`
	LastSrcSeq   uint64 `json:"last_src_seq,omitempty"`
	LastSrcEpoch string `json:"last_src_epoch,omitempty"`
	Dropped      uint64 `json:"dropped,omitempty"`
}

func gapOf(t *testing.T, e *Envelope) linkGap {
	t.Helper()
	if e.Ev.Kind != "link.gap" {
		t.Fatalf("not a continuity marker: %s", e.Raw())
	}
	var g linkGap
	if err := json.Unmarshal(e.Ev.Data, &g); err != nil {
		t.Fatalf("link.gap data: %v", err)
	}
	return g
}

// A new instance of the peer (its token changed) is a boot the link has not read yet: a marker
// published before anything of it was seen omits src_epoch rather than name the previous boot
// (Codex round 41: the boot survived the token change, so a live start behind a failed status
// and an empty backlog named the old boot). The first thing that names the new boot marks the
// boundary once, from the last boot ever named; an observation of an older session (its status
// refresher answering late) is ignored, so it can neither regress the boot nor mark a false
// boundary.
func TestANewInstanceNamesNoBootUntilRead(t *testing.T) {
	l := newPeerLink(PeerSpec{Name: "b", Probes: []string{"cpu"}}, NewHub("a"))
	l.token, l.boot, l.lastBoot, l.epoch = "t1", "aaaaaaaaaaaa", "aaaaaaaaaaaa", "aaaaaaaaaaaa"
	l.since["ctl"] = 40
	l.last["probe/cpu"] = srcPoint{39, "aaaaaaaaaaaa"}
	cur := l.hub.Subscribe("peer/b/", nil, "")

	since, _, sess := l.beginSession("t2")
	if len(since) != 0 {
		t.Fatalf("a new instance resumed from %v", since)
	}
	l.mu.Lock()
	l.markLiveStartLocked("probe/cpu")
	l.mu.Unlock()
	g := gapOf(t, nextEnv(t, cur))
	if g.Reason != "live-start" || g.SrcEpoch != "" || g.LastSrcSeq != 39 || g.LastSrcEpoch != "aaaaaaaaaaaa" {
		t.Fatalf("live start on a boot not read yet: %+v, want no src_epoch and the last frame of the old boot", g)
	}

	// An older session's late status cannot regress the boot or mark anything.
	l.setServices(sess-1, json.RawMessage(`{"services":[],"epoch":"aaaaaaaaaaaa"}`))
	l.noteBoot(sess, "bbbbbbbbbbbb")
	b := gapOf(t, nextEnv(t, cur))
	if b.Reason != "epoch-changed" || b.SrcEpoch != "bbbbbbbbbbbb" || b.LastSrcEpoch != "aaaaaaaaaaaa" {
		t.Fatalf("boundary: %+v", b)
	}
	l.setServices(sess-1, json.RawMessage(`{"services":[],"epoch":"aaaaaaaaaaaa"}`))
	l.noteBoot(sess, "bbbbbbbbbbbb")
	l.mu.Lock()
	l.markLiveStartLocked("probe/cpu")
	boot := l.boot
	l.mu.Unlock()
	if boot != "bbbbbbbbbbbb" {
		t.Fatalf("boot %q after a stale observation of the old one", boot)
	}
	if m := gapOf(t, nextEnv(t, cur)); m.Reason != "live-start" || m.SrcEpoch != "bbbbbbbbbbbb" {
		t.Fatalf("after the boundary: %+v, want a live start naming the new boot, nothing else", m)
	}
}

// A gap the peer answers on a resumed ctl or svc/ subscription is relayed on the peer's
// mirrored ctl as a marker, with what the peer said (history expired: since-too-old and the
// oldest kept; a restart: epoch-changed and the new boot); an overflow still ends the session,
// since the link itself fell behind and reconnects to resume.
func TestAPeersGapIsRelayedOnItsMirroredCtl(t *testing.T) {
	l := &peerLink{spec: PeerSpec{Name: "b"}, hub: NewHub("a"), since: map[string]uint64{}}
	refresh := make(chan struct{}, 1)
	cur := l.hub.Subscribe("peer/b/", nil, "")
	for _, tc := range []struct {
		prefix string
		gap    Gap
	}{
		{"ctl", Gap{Reason: "since-too-old", From: 1, Oldest: 4000, Epoch: "b0b0b0b0b0b0"}},
		{"svc/", Gap{Reason: "epoch-changed", From: 1, Oldest: 1, Epoch: "c1c1c1c1c1c1"}},
	} {
		gap := tc.gap
		if err := l.apply(context.Background(), nil, nil, nil, refresh, 1, "b", peerFrame{prefix: tc.prefix, f: Frame{Gap: &gap}}); err != nil {
			t.Fatalf("%s %s: %v", tc.prefix, tc.gap.Reason, err)
		}
		e := nextEnv(t, cur)
		g := gapOf(t, e)
		if e.Ch != "peer/b/ctl" || e.Origin != "a" || g.Peer != "b" || g.Prefix != tc.prefix || g.Reason != tc.gap.Reason || g.SrcEpoch != tc.gap.Epoch || g.From != tc.gap.From || g.Oldest != tc.gap.Oldest {
			t.Fatalf("%s %s relayed as %s", tc.prefix, tc.gap.Reason, e.Raw())
		}
	}
	over := Gap{Reason: "overflow", From: 9}
	if err := l.apply(context.Background(), nil, nil, nil, refresh, 1, "b", peerFrame{prefix: "ctl", f: Frame{Gap: &over}}); err == nil || !strings.Contains(err.Error(), "overflow") {
		t.Fatalf("overflow: %v, want the session to end", err)
	}
}

// nextEnv reads one envelope from a hub cursor, within a second.
func nextEnv(t *testing.T, cur *Cursor) *Envelope {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	envs, gap, err := cur.Next(ctx, 1)
	if err != nil || gap != nil || len(envs) != 1 {
		t.Fatalf("cursor: envs %v gap %v err %v", envs, gap, err)
	}
	return envs[0]
}

// The boot boundary is noticed whichever says it first: the status, a queued frame, or the
// first frame mirrored (the status can fail and the backlog be empty), and never twice. Codex
// round 40: the detection read the status only, so a restart behind a failed status went
// unmarked, and a later pipe marker named the previous boot.
func TestTheBootBoundaryIsNoticedOnTheFirstMirroredFrame(t *testing.T) {
	l := &peerLink{spec: PeerSpec{Name: "b"}, hub: NewHub("a"), since: map[string]uint64{}, last: map[string]srcPoint{}}
	l.boot, l.lastBoot, l.epoch = "aaaaaaaaaaaa", "aaaaaaaaaaaa", "aaaaaaaaaaaa" // the boot known is the last named
	refresh := make(chan struct{}, 1)
	cur := l.hub.Subscribe("peer/b/", nil, "")
	frame := func(seq uint64, epoch string) peerFrame {
		return peerFrame{prefix: "svc/", f: Frame{Env: &Envelope{Ch: "svc/x", Seq: seq, Epoch: epoch, Origin: "b", Ev: Event{Kind: "stats"}}}}
	}
	if err := l.apply(context.Background(), nil, nil, nil, refresh, 1, "b", frame(1, "bbbbbbbbbbbb")); err != nil {
		t.Fatal(err)
	}
	g := gapOf(t, nextEnv(t, cur))
	if g.Reason != "epoch-changed" || g.SrcEpoch != "bbbbbbbbbbbb" || g.LastSrcEpoch != "aaaaaaaaaaaa" {
		t.Fatalf("boundary: %+v", g)
	}
	if e := nextEnv(t, cur); e.Ch != "peer/b/svc/x" || e.SrcEpoch != "bbbbbbbbbbbb" {
		t.Fatalf("after the boundary: %s", e.Raw())
	}
	if err := l.apply(context.Background(), nil, nil, nil, refresh, 1, "b", frame(2, "bbbbbbbbbbbb")); err != nil {
		t.Fatal(err)
	}
	if e := nextEnv(t, cur); e.Ev.Kind != "stats" {
		t.Fatalf("a second frame of the same boot marked again: %s", e.Raw())
	}
	l.mu.Lock()
	boot := l.boot
	l.mu.Unlock()
	if boot != "bbbbbbbbbbbb" {
		t.Fatalf("boot kept as %q", boot)
	}
}

// The peer's history expired while the link was down: the resumed subscription is answered
// since-too-old, and the link marks it on the peer's mirrored ctl with what the peer said
// (criterion 5 of ix#408), against a real plane whose ring turned over.
func TestAnExpiredHistoryOnThePeerIsMarkedOnItsMirroredCtl(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil), dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	cur := l.hub.Subscribe("peer/b/", nil, "")
	ctx1, cancel1 := context.WithCancel(context.Background())
	go l.run(ctx1)
	deadline := time.Now().Add(10 * time.Second)
	for l.status().State != "up" { // the svc/ subscription is live: publish once the link reads
		if time.Now().After(deadline) {
			t.Fatal("the link did not come up")
		}
		time.Sleep(10 * time.Millisecond)
	}
	b.plane.Publish("svc/x", event("stats", map[string]any{"n": 0}))
	var mirrored *Envelope
	for mirrored == nil {
		if e := nextEnvWithin(t, cur, 10*time.Second); e.Ch == "peer/b/svc/x" {
			mirrored = e
		}
	}
	cancel1()
	deadline = time.Now().Add(5 * time.Second)
	for l.status().State != "down" {
		if time.Now().After(deadline) {
			t.Fatal("the link did not go down")
		}
		time.Sleep(10 * time.Millisecond)
	}
	for i := 1; i <= ringSize+64; i++ {
		b.plane.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	go l.run(ctx2)
	var m linkGap
	for m.Reason == "" {
		if e := nextEnvWithin(t, cur, 10*time.Second); e.Ev.Kind == linkGapKind {
			m = gapOf(t, e)
		}
	}
	// The link resumed svc/ from the frame it mirrored (ctl had none, so it starts live); the
	// peer answers from the next seq, with the oldest it still keeps, past the resume point.
	if m.Prefix != "svc/" || m.Reason != "since-too-old" || m.SrcEpoch != b.plane.Epoch() || m.From != mirrored.SrcSeq+1 || m.Oldest <= m.From {
		t.Fatalf("expired history marked as %+v, want since-too-old on svc/ from %d with the oldest kept past it", m, mirrored.SrcSeq+1)
	}
}

func nextEnvWithin(t *testing.T, cur *Cursor, d time.Duration) *Envelope {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	envs, gap, err := cur.Next(ctx, 1)
	if err != nil || gap != nil || len(envs) != 1 {
		t.Fatalf("cursor: envs %v gap %v err %v", envs, gap, err)
	}
	return envs[0]
}
