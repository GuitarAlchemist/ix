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
		if err := l.apply(context.Background(), nil, nil, nil, refresh, "b", peerFrame{prefix: tc.prefix, f: Frame{Gap: &gap}}); err != nil {
			t.Fatalf("%s %s: %v", tc.prefix, tc.gap.Reason, err)
		}
		e := nextEnv(t, cur)
		g := gapOf(t, e)
		if e.Ch != "peer/b/ctl" || e.Origin != "a" || g.Peer != "b" || g.Prefix != tc.prefix || g.Reason != tc.gap.Reason || g.SrcEpoch != tc.gap.Epoch || g.From != tc.gap.From || g.Oldest != tc.gap.Oldest {
			t.Fatalf("%s %s relayed as %s", tc.prefix, tc.gap.Reason, e.Raw())
		}
	}
	over := Gap{Reason: "overflow", From: 9}
	if err := l.apply(context.Background(), nil, nil, nil, refresh, "b", peerFrame{prefix: "ctl", f: Frame{Gap: &over}}); err == nil || !strings.Contains(err.Error(), "overflow") {
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
