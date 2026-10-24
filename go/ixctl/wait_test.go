package main

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

const testWaitTimeoutMS int64 = 5000

func applyWS(t testing.TB, c *Client, desired *Desired) ApplyReply {
	t.Helper()
	var reply ApplyReply
	if err := json.Unmarshal(do(t, c, Request{Op: "apply", Desired: desired}), &reply); err != nil {
		t.Fatal(err)
	}
	return reply
}

func waitWS(t testing.TB, c *Client, req Request) WaitReply {
	t.Helper()
	var reply WaitReply
	if err := json.Unmarshal(do(t, c, req), &reply); err != nil {
		t.Fatal(err)
	}
	return reply
}

func waitPlane(t testing.TB, p *Plane, req Request) WaitReply {
	t.Helper()
	data, err := p.Request(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	reply, ok := data.(WaitReply)
	if !ok {
		t.Fatalf("wait reply has type %T", data)
	}
	return reply
}

func waitForWaiterCount(t testing.TB, p *Plane, want int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for len(p.waiters) != want {
		if time.Now().After(deadline) {
			t.Fatalf("waiters in flight %d, want %d", len(p.waiters), want)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestWaitGenerationLaunchAndRunEnd(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "pod", restart: restartNever}), "")
	c := dial(t, in)
	markers := subscribeMarkers(t, c, "svc/pod", runMarkers)

	first := applyWS(t, c, running("pod"))
	launch := waitWS(t, c, Request{Op: "wait", Wait: waitLaunch, Epoch: first.Epoch,
		Generation: first.Generation, Service: "pod", TimeoutMS: testWaitTimeoutMS})
	begin := waitEnv(t, markers, 5*time.Second, "run begin", func(e *Envelope) bool {
		return e.Ev.Kind == runBeginKind && genOf(e) == first.Generation
	})
	if launch.Outcome != waitLaunched || launch.Run == nil || begin.Run == nil || *launch.Run != *begin.Run {
		t.Fatalf("launch wait %+v, run.begin %s", launch, begin.Raw())
	}
	generation := waitWS(t, c, Request{Op: "wait", Wait: waitGeneration, Epoch: first.Epoch,
		Generation: first.Generation, TimeoutMS: testWaitTimeoutMS})
	if generation.Outcome != waitConverged {
		t.Fatalf("generation wait %+v", generation)
	}

	second := applyWS(t, c, running("pod"))
	body := map[string]any{"wait": waitLaunch, "epoch": second.Epoch, "generation": second.Generation,
		"service": "pod", "timeout_ms": 50}
	code, _, raw := httpDo(t, in, "POST", "/v1/wait", body, nil)
	var httpReply WaitReply
	if err := json.Unmarshal(raw, &httpReply); err != nil || code != 200 {
		t.Fatalf("HTTP wait: %d %s (%v)", code, raw, err)
	}
	wsReply := waitWS(t, c, Request{Op: "wait", Wait: waitLaunch, Epoch: second.Epoch,
		Generation: second.Generation, Service: "pod", TimeoutMS: 50})
	current := second.Results["pod"].CurrentRun
	if !reflect.DeepEqual(httpReply, wsReply) || wsReply.Outcome != waitNotLaunched ||
		current == nil || wsReply.CurrentRun == nil || *wsReply.CurrentRun != *current {
		t.Fatalf("HTTP %+v, WebSocket %+v, apply current_run %+v", httpReply, wsReply, current)
	}

	do(t, c, Request{Op: "stop", Service: "pod"})
	ended := waitWS(t, c, Request{Op: "wait", Wait: waitRun, Run: launch.Run,
		TimeoutMS: testWaitTimeoutMS})
	if ended.Outcome != waitEnded || ended.Run == nil || *ended.Run != *launch.Run ||
		ended.End == nil || ended.End.Code == nil || *ended.End.Code != 0 {
		t.Fatalf("run-end wait %+v", ended)
	}
}

func TestGenerationWaitDiverges(t *testing.T) {
	doc := []byte(`{"name":"local","services":[{"name":"ghost","command":"no-such-binary"}]}`)
	spec, err := ParseSpec(doc, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	in := newTestInstance(t, spec, "")
	c := dial(t, in)
	applied := applyWS(t, c, running("ghost"))
	reply := waitWS(t, c, Request{Op: "wait", Wait: waitGeneration, Epoch: applied.Epoch,
		Generation: applied.Generation, TimeoutMS: testWaitTimeoutMS})
	if reply.Outcome != waitDiverged || reply.Service != "ghost" || !strings.HasPrefix(reply.Reason, "launch: ") {
		t.Fatalf("diverged wait %+v", reply)
	}
}

func TestRunWaitNeverFollowsARestart(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "crashy", args: []string{"--crash-after-ms", "1200"}, restart: restartOnFailure, max: 3}), "")
	c := dial(t, in)
	applied := applyWS(t, c, running("crashy"))
	launch := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "crashy", TimeoutMS: testWaitTimeoutMS})
	if launch.Run == nil {
		t.Fatalf("launch wait %+v", launch)
	}
	type result struct {
		reply WaitReply
		err   error
	}
	done := make(chan result, 1)
	go func() {
		data, err := in.plane.Request(context.Background(), Request{
			Op: "wait", Wait: waitRun, Run: launch.Run, TimeoutMS: testWaitTimeoutMS})
		var reply WaitReply
		if err == nil {
			reply = data.(WaitReply)
		}
		done <- result{reply: reply, err: err}
	}()
	waitForWaiterCount(t, in.plane, 1)
	first := <-done
	if first.err != nil || first.reply.Outcome != waitEnded || first.reply.Run == nil ||
		*first.reply.Run != *launch.Run || first.reply.End == nil || first.reply.End.Code == nil ||
		*first.reply.End.Code != 3 {
		t.Fatalf("first run wait %+v, %v", first.reply, first.err)
	}
	status := waitStatus(t, in, 5*time.Second, "second run current", func(st StatusReply) bool {
		run := svcStatus(st, "crashy").Run
		return run != nil && *run != *launch.Run
	})
	run2 := svcStatus(status, "crashy").Run
	again := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitRun, Run: launch.Run,
		TimeoutMS: testWaitTimeoutMS})
	if again.Outcome != waitEnded || again.Run == nil || *again.Run != *launch.Run ||
		again.End == nil || again.End.Code == nil || *again.End.Code != 3 || *again.Run == *run2 {
		t.Fatalf("old run wait %+v, current run %+v", again, run2)
	}
	// The restart launched run 2 under the same generation; the run the generation
	// launched is still run 1.
	relaunch := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "crashy", TimeoutMS: testWaitTimeoutMS})
	if relaunch.Outcome != waitLaunched || relaunch.Run == nil || *relaunch.Run != *launch.Run {
		t.Fatalf("launch wait after a restart %+v, want run %+v (current %+v)", relaunch, *launch.Run, *run2)
	}
}

// A launch wait on a service the generation did not name is a refused question, not
// `unknown (evicted)`: nothing of it was ever recorded.
func TestLaunchWaitOnAServiceTheGenerationDidNotNameIsRefused(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "a", restart: restartNever}, fakeSvc{name: "b", restart: restartNever}), "")
	c := dial(t, in)
	applied := applyWS(t, c, running("a"))
	code, _, raw := httpDo(t, in, "POST", "/v1/wait", map[string]any{"wait": waitLaunch,
		"epoch": applied.Epoch, "generation": applied.Generation, "service": "b",
		"timeout_ms": testWaitTimeoutMS}, nil)
	if code != 400 {
		t.Fatalf("launch wait on a service generation %d did not name: %d %s", applied.Generation, code, raw)
	}
}

// A generation or launch wait on a generation this boot has not issued is refused at once:
// the plane never waits for a generation to come into being. Both #454 consumers read the
// README as if it would. Pins present behaviour.
func TestWaitOnAGenerationNotYetIssuedIsRefused(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil, fakeSvc{name: "pod", restart: restartNever}), "")
	c := dial(t, in)
	applied := applyWS(t, c, running("pod"))
	next := applied.Generation + 1
	for _, body := range []map[string]any{
		{"wait": waitGeneration, "epoch": applied.Epoch, "generation": next, "timeout_ms": testWaitTimeoutMS},
		{"wait": waitLaunch, "epoch": applied.Epoch, "generation": next, "service": "pod",
			"timeout_ms": testWaitTimeoutMS},
	} {
		start := time.Now()
		code, _, raw := httpDo(t, in, "POST", "/v1/wait", body, nil)
		if took := time.Since(start); code != 400 || took > time.Second {
			t.Fatalf("%s wait on generation %d, one past the current: %d %s after %v", body["wait"], next, code, raw, took)
		}
	}
}

// A launch wait on a noop result is refused while the plane keeps that admission result;
// once newer results evict it the plane can no longer tell what it was, and the wait is
// unknown (evicted) like any dropped result. Pins present behaviour.
func TestLaunchWaitOnANoopIsRefusedUntilEvicted(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil, fakeSvc{name: "pod", restart: restartNever}), "")
	c := dial(t, in)
	noop := applyWS(t, c, desiredOf(desiredStopped, "pod"))
	if noop.Results["pod"].Result != applyNoop {
		t.Fatalf("apply of a stopped service %+v", noop)
	}
	body := map[string]any{"wait": waitLaunch, "epoch": noop.Epoch, "generation": noop.Generation,
		"service": "pod", "timeout_ms": testWaitTimeoutMS}
	if code, _, raw := httpDo(t, in, "POST", "/v1/wait", body, nil); code != 400 {
		t.Fatalf("launch wait on a retained noop: %d %s", code, raw)
	}
	for i := 0; i < keptApplyResults; i++ {
		applyWS(t, c, desiredOf(desiredStopped, "pod"))
	}
	code, _, raw := httpDo(t, in, "POST", "/v1/wait", body, nil)
	var reply WaitReply
	if err := json.Unmarshal(raw, &reply); err != nil || code != 200 ||
		reply.Outcome != waitUnknown || reply.Reason != waitEvicted {
		t.Fatalf("launch wait on an evicted noop: %d %s (%v)", code, raw, err)
	}
}

func TestWaitsFromAnotherBootAreUnknown(t *testing.T) {
	stateDir := t.TempDir()
	spec := fakeSpec(t, "local", nil, fakeSvc{name: "pod", restart: restartNever})
	first := newTestInstance(t, spec, stateDir)
	c := dial(t, first)
	applied := applyWS(t, c, running("pod"))
	launch := waitPlane(t, first.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "pod", TimeoutMS: testWaitTimeoutMS})
	if launch.Run == nil {
		t.Fatalf("launch wait %+v", launch)
	}
	first.close()

	second := newTestInstance(t, spec, stateDir)
	for _, body := range []map[string]any{
		{"wait": waitGeneration, "epoch": applied.Epoch, "generation": applied.Generation,
			"timeout_ms": testWaitTimeoutMS},
		{"wait": waitRun, "run": launch.Run, "timeout_ms": testWaitTimeoutMS},
	} {
		code, _, raw := httpDo(t, second, "POST", "/v1/wait", body, nil)
		var reply WaitReply
		if err := json.Unmarshal(raw, &reply); err != nil || code != 200 ||
			reply.Outcome != waitUnknown || reply.Reason != waitOtherEpoch {
			t.Fatalf("old-epoch wait: %d %s (%v)", code, raw, err)
		}
	}
}

func TestRunWaitFallsBackToRecentExitsAfterADeterministicGap(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "gapped", restart: restartNever}), "")
	c := dial(t, in)
	applied := applyWS(t, c, running("gapped"))
	launch := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: applied.Epoch,
		Generation: applied.Generation, Service: "gapped", TimeoutMS: testWaitTimeoutMS})
	if launch.Run == nil {
		t.Fatalf("launch wait %+v", launch)
	}
	head := in.plane.hub.Head()
	cur := in.plane.hub.Subscribe("svc/gapped", &head, applied.Epoch)
	cur.exact = true
	do(t, c, Request{Op: "stop", Service: "gapped"})
	for i := 0; i < ringSize+1; i++ {
		in.plane.Publish("ctl", event("noise", map[string]any{"i": i}))
	}
	_, oldest := in.plane.hub.Span()
	if cur.next >= oldest {
		t.Fatalf("cursor next %d did not fall behind oldest %d", cur.next, oldest)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	reply, err := in.plane.waitRunEvents(ctx, *launch.Run, cur)
	if err != nil || reply.Outcome != waitEnded || reply.Run == nil || *reply.Run != *launch.Run ||
		reply.End == nil || reply.End.Code == nil || *reply.End.Code != 0 {
		t.Fatalf("gap fallback %+v, %v", reply, err)
	}
}

func TestLaunchWaitAdmissionHistoryIsBounded(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "pod", restart: restartNever}), "")
	c := dial(t, in)
	first := applyWS(t, c, running("pod"))
	waitStatus(t, in, 5*time.Second, "pod running", func(st StatusReply) bool {
		return svcStatus(st, "pod").State == stateRunning
	})
	var last ApplyReply
	for i := 0; i < keptFailures; i++ {
		last = applyWS(t, c, running("pod"))
	}
	evicted := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: first.Epoch,
		Generation: first.Generation, Service: "pod", TimeoutMS: testWaitTimeoutMS})
	current := waitPlane(t, in.plane, Request{Op: "wait", Wait: waitLaunch, Epoch: last.Epoch,
		Generation: last.Generation, Service: "pod", TimeoutMS: testWaitTimeoutMS})
	if evicted.Outcome != waitUnknown || evicted.Reason != waitEvicted ||
		current.Outcome != waitNotLaunched || current.CurrentRun == nil {
		t.Fatalf("evicted %+v, newest %+v", evicted, current)
	}
}
