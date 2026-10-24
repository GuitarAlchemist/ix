package main

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func waitMirroredEnvelope(t *testing.T, cur *Cursor, d time.Duration, what string, match func(*Envelope) bool) *Envelope {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	for {
		envs, gap, err := cur.Next(ctx, 1)
		if err != nil {
			t.Fatalf("waiting for %s: %v", what, err)
		}
		if gap != nil {
			t.Fatalf("waiting for %s: gap %+v", what, gap)
		}
		if len(envs) == 1 && match(envs[0]) {
			return envs[0]
		}
	}
}

func startRunLink(t *testing.T, l *peerLink, acknowledged bool) context.CancelFunc {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	go l.run(ctx)
	waitLinkStatus(t, l, func(st PeerStatus) bool {
		return st.State == "up" && st.Live && st.RunMarkers == acknowledged
	})
	return cancel
}

// A7: federation requests markers and preserves the source run on the marker bracket, every
// service line and attributable ctl event. Marker filtering remains the local subscriber's choice.
func TestFederationKeepsSourceRunAndMarkers(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil, fakeSvc{
		name: "pod", args: []string{"--crash-after-ms", "700", "--crash-code", "0"},
		restart: restartNever,
	}), dirB)
	a := newTestInstance(t, fakeSpec(t, "a", []PeerSpec{{
		Name: "b", StateFile: filepath.Join(dirB, "b.json"),
	}}), dirA)

	waitStatus(t, a, 10*time.Second, "peer b live with run markers", func(st StatusReply) bool {
		return st.Peers[0].State == "up" && st.Peers[0].Live && st.Peers[0].RunMarkers
	})
	c := dial(t, a)
	marked := subscribeMarkers(t, c, "peer/b/svc/pod", runMarkers)
	plain := subscribe(t, c, "peer/b/svc/pod", nil)
	ctl := subscribe(t, c, "peer/b/ctl", nil)
	do(t, c, Request{Op: "apply", Peer: "b", Desired: running("pod")})

	begin := waitEnv(t, marked, 10*time.Second, "mirrored run begin", func(e *Envelope) bool {
		return e.Ev.Kind == runBeginKind
	})
	line := waitEnv(t, marked, 10*time.Second, "mirrored worker line", func(e *Envelope) bool {
		return e.Ev.Kind == "log"
	})
	end := waitEnv(t, marked, 10*time.Second, "mirrored run end", func(e *Envelope) bool {
		return e.Ev.Kind == runEndKind
	})
	if begin.Run == nil {
		t.Fatalf("begin has no run: %s", begin.Raw())
	}
	run := *begin.Run
	for _, env := range []*Envelope{line, end} {
		if env.Run == nil || *env.Run != run {
			t.Fatalf("%s run %+v, want source run %+v", env.Ev.Kind, env.Run, run)
		}
	}
	if !(begin.SrcSeq < line.SrcSeq && line.SrcSeq < end.SrcSeq) ||
		!(begin.Seq < line.Seq && line.Seq < end.Seq) {
		t.Fatalf("mirrored order begin %d/%d, line %d/%d, end %d/%d",
			begin.SrcSeq, begin.Seq, line.SrcSeq, line.Seq, end.SrcSeq, end.Seq)
	}

	for _, kind := range []string{"service.starting", "service.started", "service.exited"} {
		env := waitEnv(t, ctl, 10*time.Second, "mirrored "+kind, func(e *Envelope) bool {
			return e.Ev.Kind == kind
		})
		if env.Run == nil || *env.Run != run {
			t.Fatalf("%s run %+v, want source run %+v", kind, env.Run, run)
		}
	}

	waitEnv(t, plain, 10*time.Second, "unmarked subscriber's worker line", func(e *Envelope) bool {
		return e.Ev.Kind == "log"
	})
	if f, err := plain.next(300*time.Millisecond, isRunMarker); err == nil {
		t.Fatalf("subscriber without markers received %s", f.Env.Raw())
	}

	source := waitStatus(t, b, 10*time.Second, "source run exit", func(st StatusReply) bool {
		s := svcStatus(st, "pod")
		return s.State == stateStopped && len(s.RecentExits) == 1
	})
	svc := svcStatus(source, "pod")
	if svc.Run == nil || *svc.Run != run ||
		svc.RecentExits[0].Run == nil || *svc.RecentExits[0].Run != run {
		t.Fatalf("source status run %+v recent exit %+v, mirrored run %+v",
			svc.Run, svc.RecentExits[0].Run, run)
	}
}

// A resumed svc/ subscription replays a run.end published while the link was down, retaining
// the run that ended rather than constructing one from the mirroring hub.
func TestFederationReconnectReplaysRunEndWithSourceRun(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil, fakeSvc{name: "pod", restart: restartNever}), dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	cur := l.hub.Subscribe("peer/b/", nil, "")

	cancel := startRunLink(t, l, true)
	c := dial(t, b)
	do(t, c, Request{Op: "apply", Desired: running("pod")})
	begin := waitMirroredEnvelope(t, cur, 10*time.Second, "run begin", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == runBeginKind
	})
	waitMirroredEnvelope(t, cur, 10*time.Second, "worker line", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == "log"
	})
	cancel()
	waitLink(t, l, "down")

	do(t, c, Request{Op: "stop", Service: "pod"})
	source := waitStatus(t, b, 10*time.Second, "source stopped", func(st StatusReply) bool {
		return svcStatus(st, "pod").State == stateStopped
	})
	run := svcStatus(source, "pod").Run
	if begin.Run == nil || run == nil || *begin.Run != *run {
		t.Fatalf("source run %+v, mirrored begin %+v", run, begin.Run)
	}

	cancel = startRunLink(t, l, true)
	defer cancel()
	end := waitMirroredEnvelope(t, cur, 10*time.Second, "replayed run end", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == runEndKind
	})
	if end.Run == nil || *end.Run != *run {
		t.Fatalf("replayed end run %+v, want source run %+v", end.Run, *run)
	}
}

// A peer boot is part of the source run identity; a restarted peer's run must never be merged
// with one from its preceding instance.
func TestFederationPeerRestartKeepsNewSourceRun(t *testing.T) {
	dirB := t.TempDir()
	specB := fakeSpec(t, "b", nil, fakeSvc{name: "pod", restart: restartNever})
	b := newTestInstance(t, specB, dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	cur := l.hub.Subscribe("peer/b/", nil, "")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go l.run(ctx)
	waitLinkStatus(t, l, func(st PeerStatus) bool { return st.State == "up" && st.RunMarkers })

	do(t, dial(t, b), Request{Op: "apply", Desired: running("pod")})
	first := waitMirroredEnvelope(t, cur, 10*time.Second, "first boot run", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == runBeginKind
	})
	if first.Run == nil {
		t.Fatalf("first boot begin has no run: %s", first.Raw())
	}
	b.close()
	waitLink(t, l, "down")

	b2 := newTestInstance(t, specB, dirB)
	waitLinkStatus(t, l, func(st PeerStatus) bool {
		return st.State == "up" && st.Live && st.RunMarkers
	})
	do(t, dial(t, b2), Request{Op: "apply", Desired: running("pod")})
	second := waitMirroredEnvelope(t, cur, 10*time.Second, "second boot run", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == runBeginKind &&
			e.Run != nil && e.Run.Epoch == b2.plane.Epoch()
	})
	if second.Run == nil || second.Run.Epoch != b2.plane.Epoch() ||
		second.Run.Epoch == first.Run.Epoch || *second.Run == *first.Run {
		t.Fatalf("first run %+v, restarted peer run %+v, new epoch %s",
			first.Run, second.Run, b2.plane.Epoch())
	}
}

// Stopping and launching the declared service again under the same name creates another source
// run. Federation follows the envelope; it never keys a run by service name.
func TestFederationSameNameRelaunchKeepsDistinctSourceRuns(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil, fakeSvc{name: "pod", restart: restartNever}), dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	cur := l.hub.Subscribe("peer/b/", nil, "")
	cancel := startRunLink(t, l, true)
	defer cancel()
	c := dial(t, b)

	do(t, c, Request{Op: "apply", Desired: running("pod")})
	first := waitMirroredEnvelope(t, cur, 10*time.Second, "first run", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == runBeginKind
	})
	do(t, c, Request{Op: "stop", Service: "pod"})
	waitMirroredEnvelope(t, cur, 10*time.Second, "first run end", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == runEndKind
	})
	do(t, c, Request{Op: "apply", Desired: running("pod")})
	second := waitMirroredEnvelope(t, cur, 10*time.Second, "second run", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod" && e.Ev.Kind == runBeginKind
	})
	if first.Run == nil || second.Run == nil ||
		first.Run.Origin != second.Run.Origin || first.Run.Epoch != second.Run.Epoch ||
		second.Run.Seq <= first.Run.Seq || *second.Run == *first.Run {
		t.Fatalf("same-name runs first %+v, second %+v", first.Run, second.Run)
	}
}

// A deterministic allowance knob makes the resumed svc/ subscription jump. Markers in the
// skipped span stay lost; events and markers published after the jump retain their source run.
func TestFederationJumpKeepsLaterRunsAndInventsNoMarkers(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil), dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	cur := l.hub.Subscribe("peer/b/", nil, "")
	cancel := startRunLink(t, l, true)

	// As the supervisor does: the run is named by its service.starting on ctl, its markers and
	// lines go to svc/<name>.
	starting := func() *Envelope {
		return b.plane.hub.PublishNewRun("ctl", event("service.starting", map[string]any{"service": "x"}))
	}
	first := starting()
	b.plane.hub.PublishRun("svc/x", event(runBeginKind, map[string]any{}), *first.Run)
	waitMirroredEnvelope(t, cur, 10*time.Second, "first run begin", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/x" && e.Ev.Kind == runBeginKind
	})
	cancel()
	waitLink(t, l, "down")

	skipped := starting()
	b.plane.hub.PublishRun("svc/x", event(runBeginKind, map[string]any{}), *skipped.Run)
	b.plane.hub.PublishRun("svc/x", event(runEndKind, map[string]any{"output": "complete"}), *skipped.Run)
	l.allowFrames.Store(0)
	cancel = startRunLink(t, l, true)
	defer cancel()

	for {
		env := nextEnvWithin(t, cur, 10*time.Second)
		if isRunMarker(env) {
			t.Fatalf("the jumped span fabricated or replayed marker %s", env.Raw())
		}
		if env.Ev.Kind == linkGapKind &&
			ctlData(env)["reason"] == "jumped" && ctlData(env)["prefix"] == "svc/" {
			break
		}
	}

	after := starting()
	b.plane.hub.PublishRun("svc/x", event(runBeginKind, map[string]any{}), *after.Run)
	b.plane.hub.PublishRun("svc/x", event("log", map[string]any{"line": "after jump"}), *after.Run)
	begin := waitMirroredEnvelope(t, cur, 10*time.Second, "begin after jump", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/x" && e.Ev.Kind == runBeginKind
	})
	line := waitMirroredEnvelope(t, cur, 10*time.Second, "line after jump", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/x" && e.Ev.Kind == "log"
	})
	if begin.Run == nil || line.Run == nil || *begin.Run != *after.Run || *line.Run != *after.Run {
		t.Fatalf("after-jump source run %+v, begin %+v, line %+v", after.Run, begin.Run, line.Run)
	}
}

// run_markers is the guarantee of the two subscriptions that stand now, not of one they replaced.
// Inside one live session, b's queue toward the link overflows and the link reopens svc/ alone
// (a jump: the allowance is zero). While b answers that subscribe as an older plane would, the flag
// falls though the link stays up and ctl keeps its acknowledgement; once b acknowledges again, the
// next reopen of svc/ brings it back. Neither step reconnects.
func TestRunMarkersFollowTheSubscriptionsThatStandNow(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil), dirB)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	l.ctlDelay.Store(int64(500 * time.Microsecond))
	l.allowFrames.Store(0)
	b.web.writeDelay.Store(int64(time.Millisecond)) // the overflow is b's queue filling behind its writer, whatever the kernel buffers hold
	locked := func(f func()) {
		l.mu.Lock()
		defer l.mu.Unlock()
		f()
	}
	var before uint64
	cancel := startRunLink(t, l, true)
	defer cancel()
	locked(func() { before = l.sess })
	reopenSvc := func(from int, acknowledged bool) PeerStatus {
		t.Helper()
		jumps := l.status().Jumps
		for i := from; i < from+8000; i++ {
			b.plane.Publish("svc/x", event("stats", map[string]any{"n": i}))
		}
		return waitLinkStatus(t, l, func(st PeerStatus) bool {
			return st.State == "up" && st.Live && st.Jumps > jumps && st.RunMarkers == acknowledged
		})
	}

	b.web.runMarkersUnsupported.Store(true) // the next subscribes b answers are an older plane's
	reopenSvc(1, false)
	locked(func() {
		if !l.ctl["ctl"].runMarkers || l.ctl["svc/"].runMarkers {
			t.Fatalf("after reopening svc/ alone: ctl acknowledged %v, svc/ %v; want ctl kept, svc/ not",
				l.ctl["ctl"].runMarkers, l.ctl["svc/"].runMarkers)
		}
	})

	b.web.runMarkersUnsupported.Store(false)
	reopenSvc(8001, true)
	locked(func() {
		if l.sess != before {
			t.Fatalf("the link reconnected: session %d, was %d", l.sess, before)
		}
	})
}

// An older peer ignores the markers field: the link remains useful for ordinary envelopes, but
// status does not claim that absence of run markers means no run began or ended.
func TestFederationReportsRunMarkersNotAcknowledged(t *testing.T) {
	dirB := t.TempDir()
	b := newTestInstance(t, fakeSpec(t, "b", nil), dirB)
	b.web.runMarkersUnsupported.Store(true)
	l := newPeerLink(PeerSpec{Name: "b", StateFile: filepath.Join(dirB, "b.json")}, NewHub("a"))
	cur := l.hub.Subscribe("peer/b/", nil, "")
	cancel := startRunLink(t, l, false)
	defer cancel()

	start := b.plane.hub.PublishNewRun("svc/x", event("service.starting", map[string]any{}))
	b.plane.hub.PublishRun("svc/x", event(runBeginKind, map[string]any{}), *start.Run)
	b.plane.hub.PublishRun("svc/x", event("log", map[string]any{"line": "old peer"}), *start.Run)
	for {
		env := nextEnvWithin(t, cur, 10*time.Second)
		if env.Ch != "peer/b/svc/x" {
			continue
		}
		if isRunMarker(env) {
			t.Fatalf("older peer delivered an unacknowledged marker: %s", env.Raw())
		}
		if env.Ev.Kind == "log" {
			if env.Run == nil || *env.Run != *start.Run {
				t.Fatalf("ordinary mirrored run %+v, want %+v", env.Run, start.Run)
			}
			break
		}
	}
	if st := l.status(); st.RunMarkers {
		t.Fatalf("older peer reported run marker acknowledgement: %+v", st)
	}
}
