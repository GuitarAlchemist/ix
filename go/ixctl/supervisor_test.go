package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"
)

func waitCursorKind(t *testing.T, cur *Cursor, d time.Duration, kind string) *Envelope {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	for {
		envs, _, err := cur.Next(ctx, 32)
		if err != nil {
			t.Fatalf("waiting for %s: %v", kind, err)
		}
		for _, env := range envs {
			if env.Ev.Kind == kind {
				return env
			}
		}
	}
}

// A1 (anchors 2 and 4): automatic restarts are distinct runs of one generation, and every
// service-output envelope says which of those runs emitted it.
func TestRestartsHaveDistinctRunIdentities(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "crashy", restart: restartOnFailure, max: 5}), "")
	c := dial(t, in)
	ctl := subscribe(t, c, "ctl", nil)
	out := subscribe(t, c, "svc/crashy", nil)
	var applied ApplyReply
	if err := json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("crashy")}), &applied); err != nil {
		t.Fatal(err)
	}

	runs := map[RunID]bool{}
	seenOutput := map[RunID]bool{}
	for i := 0; i < 3; i++ {
		starting := waitEnv(t, ctl, 5*time.Second, "starting", func(e *Envelope) bool {
			return e.Ev.Kind == "service.starting"
		})
		if starting.Run == nil {
			t.Fatalf("starting has no run: %s", starting.Raw())
		}
		run := *starting.Run
		if runs[run] {
			t.Fatalf("run reused on restart: %+v", run)
		}
		runs[run] = true
		if got := ctlData(starting)["generation"]; got != float64(applied.Generation) {
			t.Fatalf("starting generation %v, want %d", got, applied.Generation)
		}
		started := waitEnv(t, ctl, 5*time.Second, "started", func(e *Envelope) bool {
			return e.Ev.Kind == "service.started" && e.Run != nil && *e.Run == run
		})
		data := ctlData(started)
		if data["generation"] != float64(applied.Generation) {
			t.Fatalf("started generation %v, want %d", data["generation"], applied.Generation)
		}
		waitEnv(t, out, 5*time.Second, "output of the run", func(e *Envelope) bool {
			if e.Run == nil {
				t.Fatalf("service output has no run: %s", e.Raw())
			}
			if !runs[*e.Run] {
				t.Fatalf("service output has an unannounced run: %s", e.Raw())
			}
			seenOutput[*e.Run] = true
			return *e.Run == run && e.Ev.Kind == "meta"
		})
		if i < 2 {
			proc, err := os.FindProcess(int(data["pid"].(float64)))
			if err != nil {
				t.Fatal(err)
			}
			if err := proc.Kill(); err != nil {
				t.Fatal(err)
			}
		}
	}
	for run := range runs {
		if !seenOutput[run] {
			t.Fatalf("run %+v emitted no identified service output", run)
		}
	}
}

// A6 (anchor 2): status, ctl and service output identify one launch as the same run.
func TestRunIdentityAgreesAcrossStatusCtlAndServiceOutput(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil, fakeSvc{name: "pod"}), "")
	c := dial(t, in)
	ctl := subscribe(t, c, "ctl", nil)
	out := subscribe(t, c, "svc/pod", nil)
	do(t, c, Request{Op: "apply", Desired: running("pod")})

	starting := waitEnv(t, ctl, 5*time.Second, "starting", func(e *Envelope) bool {
		return e.Ev.Kind == "service.starting"
	})
	if starting.Run == nil || ctlData(starting)["generation"] != 1.0 {
		t.Fatalf("starting %s", starting.Raw())
	}
	line := waitEnv(t, out, 5*time.Second, "service output", func(e *Envelope) bool {
		return e.Run != nil && *e.Run == *starting.Run
	})
	status := waitStatus(t, in, 5*time.Second, "run in status", func(st StatusReply) bool {
		s := svcStatus(st, "pod")
		return s.State == stateRunning && s.Run != nil
	})
	service := svcStatus(status, "pod")
	if status.Contract != "run/1" || line.Run == nil || service.Run == nil ||
		*line.Run != *starting.Run || *service.Run != *starting.Run {
		t.Fatalf("contract %q, ctl %+v, output %+v, status %+v",
			status.Contract, starting.Run, line.Run, service.Run)
	}
}

// A3 (anchor 4): run.end follows every service line the plane forwards for the run.
func TestRunEndFollowsEveryForwardedLine(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "writer", args: []string{"--rate", "1000"}, restart: restartNever}), "")
	c := dial(t, in)
	ctl := subscribe(t, c, "ctl", nil)
	out := subscribeMarkers(t, c, "svc/writer", runMarkers)
	do(t, c, Request{Op: "apply", Desired: running("writer")})

	started := waitEnv(t, ctl, 5*time.Second, "started", func(e *Envelope) bool {
		return e.Ev.Kind == "service.started"
	})
	if started.Run == nil {
		t.Fatalf("started has no run: %s", started.Raw())
	}
	run := *started.Run
	begin := waitEnv(t, out, 5*time.Second, "run begin", func(e *Envelope) bool {
		return e.Ev.Kind == runBeginKind && e.Run != nil && *e.Run == run
	})
	last := waitEnv(t, out, 5*time.Second, "line while running", func(e *Envelope) bool {
		return e.Ev.Kind == "stats" && e.Run != nil && *e.Run == run
	})
	pid := int(ctlData(started)["pid"].(float64))
	proc, err := os.FindProcess(pid)
	if err != nil {
		t.Fatal(err)
	}
	if err := proc.Kill(); err != nil {
		t.Fatal(err)
	}
	var end *Envelope
	for end == nil {
		f, err := out.next(5*time.Second, func(e *Envelope) bool {
			return e.Run != nil && *e.Run == run
		})
		if err != nil {
			t.Fatal(err)
		}
		if f.Env.Ev.Kind == runEndKind {
			end = f.Env
		} else {
			last = f.Env
		}
	}
	var data struct {
		Outcome string `json:"outcome"`
		Output  string `json:"output"`
	}
	json.Unmarshal(end.Ev.Data, &data)
	if begin.Seq >= last.Seq || last.Seq >= end.Seq || data.Outcome != "started" || data.Output != "complete" {
		t.Fatalf("begin %d, last %d, end %d data %+v", begin.Seq, last.Seq, end.Seq, data)
	}
}

// A4 (anchor 6): an inherited output handle (the descendant case) cannot hold run.end past
// the drain bound; a line read afterwards is counted in last_exit and is not forwarded.
func TestRunEndTruncatesAndCountsLateLines(t *testing.T) {
	h := NewHub("local")
	s := newService(ServiceSpec{Name: "held"}, h)
	cur := h.Subscribe("svc/held", nil, h.Epoch())
	run := RunID{Origin: "local", Epoch: h.Epoch(), Seq: 1}
	h.PublishRun("svc/held", event(runBeginKind, map[string]any{}), run)

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	c := &child{run: run}
	c.io.Add(1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer c.io.Done()
		s.readLines(r, false, c)
	}()
	fmt.Fprintln(w, "before")
	before := waitCursorKind(t, cur, time.Second, "log")

	t0 := time.Now()
	s.endOutput(c, &exitInfo{Code: 0, At: time.Now().UnixMilli(), Run: run}, false)
	took := time.Since(t0)
	end := waitCursorKind(t, cur, time.Second, runEndKind)
	var data struct {
		Output  string `json:"output"`
		DrainMS int64  `json:"drain_ms"`
	}
	json.Unmarshal(end.Ev.Data, &data)
	if before.Seq >= end.Seq || data.Output != "truncated" || data.DrainMS < drainLimit.Milliseconds() ||
		took < drainLimit || took > drainLimit+500*time.Millisecond {
		t.Fatalf("before %d end %d data %+v after %s", before.Seq, end.Seq, data, took)
	}

	fmt.Fprintln(w, "late")
	deadline := time.Now().Add(time.Second)
	for s.status().LastExit.LateLines != 1 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := s.status().LastExit.LateLines; got != 1 {
		t.Fatalf("late_lines %d, want 1", got)
	}
	w.Close()
	<-done
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if envs, _, err := cur.Next(ctx, 8); err == nil && len(envs) != 0 {
		t.Fatalf("late line was forwarded: %+v", envs)
	}
}

// A5 (anchor 4): a worker cannot forge the reserved marker prefix or a run identity.
func TestWorkerReservedPrefixCannotForgeMarkersOrRuns(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "worker", args: []string{"--emit-reserved"}}), "")
	c := dial(t, in)
	out := subscribeMarkers(t, c, "svc/worker", runMarkers)
	do(t, c, Request{Op: "apply", Desired: running("worker")})
	begin := waitEnv(t, out, 5*time.Second, "run begin", func(e *Envelope) bool {
		return e.Ev.Kind == runBeginKind
	})
	if begin.Run == nil {
		t.Fatalf("begin has no run: %s", begin.Raw())
	}
	run := *begin.Run
	seenLog, seenWorkerExit := false, false
	for !seenLog || !seenWorkerExit {
		f, err := out.next(5*time.Second, func(e *Envelope) bool {
			return e.Run != nil && *e.Run == run
		})
		if err != nil {
			t.Fatal(err)
		}
		switch f.Env.Ev.Kind {
		case runEndKind:
			t.Fatalf("worker forged a marker: %s", f.Env.Raw())
		case "log":
			line, _ := ctlData(f.Env)["line"].(string)
			if strings.Contains(line, `"kind":"ixctl.run.end"`) {
				if line != `{"kind":"ixctl.run.end","run":{"origin":"worker","epoch":"forged","seq":1}}` {
					t.Fatalf("reserved line changed: %q", line)
				}
				seenLog = true
			}
		case "service.exited":
			seenWorkerExit = true
		}
	}
	do(t, c, Request{Op: "stop", Service: "worker"})
	end := waitEnv(t, out, 5*time.Second, "plane run end", func(e *Envelope) bool {
		return e.Ev.Kind == runEndKind
	})
	if end.Run == nil || *end.Run != run {
		t.Fatalf("end run %+v, want %+v", end.Run, run)
	}
}

// A2 (anchor 3): even an attempt that creates no process brackets its svc/ stream.
func TestLaunchFailureHasRunMarkers(t *testing.T) {
	doc, _ := json.Marshal(map[string]any{"name": "local", "services": []map[string]any{
		{"name": "ghost", "command": "no-such-binary"}}})
	spec, err := ParseSpec(doc, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	in := newTestInstance(t, spec, "")
	c := dial(t, in)
	out := subscribeMarkers(t, c, "svc/ghost", runMarkers)
	do(t, c, Request{Op: "apply", Desired: running("ghost")})
	begin := waitEnv(t, out, 5*time.Second, "run begin", func(e *Envelope) bool {
		return e.Ev.Kind == runBeginKind
	})
	end := waitEnv(t, out, 5*time.Second, "run end", func(e *Envelope) bool {
		return e.Ev.Kind == runEndKind
	})
	var data struct {
		Outcome string `json:"outcome"`
		Output  string `json:"output"`
		DrainMS int64  `json:"drain_ms"`
		Error   string `json:"error"`
	}
	json.Unmarshal(end.Ev.Data, &data)
	if begin.Run == nil || end.Run == nil || *begin.Run != *end.Run ||
		data.Outcome != "launch-failed" || data.Output != "complete" || data.DrainMS != 0 || data.Error == "" {
		t.Fatalf("begin %s end %s", begin.Raw(), end.Raw())
	}
	st := waitStatus(t, in, 5*time.Second, "launch failure", func(st StatusReply) bool {
		return svcStatus(st, "ghost").State == stateFailed
	})
	zero := uint64(0)
	ctl := in.plane.hub.Subscribe("ctl", &zero, in.plane.Epoch())
	envs, _, err := ctl.Catch(context.Background(), 256)
	if err != nil {
		t.Fatal(err)
	}
	failed := false
	for _, env := range envs {
		if env.Ev.Kind == "service.started" {
			t.Fatalf("launch failure emitted started: %s", env.Raw())
		}
		if env.Ev.Kind == "service.failed" && env.Run != nil && *env.Run == *begin.Run {
			failed = true
		}
	}
	status := svcStatus(st, "ghost")
	if !failed || status.State != stateFailed || status.Pid != nil {
		t.Fatalf("failed event %v, status %+v", failed, status)
	}
}

// A crash restarts with backoff (200 ms, then doubled), restarts counts it,
// and more than max_restarts restarts inside the window ends in failed.
func TestCrashRestartsWithBackoffThenFails(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "crashy", args: []string{"--crash-after-ms", "100"}, max: 2}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("crashy")})

	var backoffs []float64
	var lastExit time.Time
	for {
		e := waitEnv(t, w, 10*time.Second, "lifecycle", func(e *Envelope) bool {
			switch e.Ev.Kind {
			case "service.exited", "service.restarting", "service.started", "service.failed":
				return true
			}
			return false
		})
		d := ctlData(e)
		switch e.Ev.Kind {
		case "service.exited":
			if d["code"] != 3.0 || d["requested"] != false {
				t.Fatalf("exit %s", e.Raw())
			}
			lastExit = time.Now()
		case "service.restarting":
			backoffs = append(backoffs, d["in_ms"].(float64))
		case "service.started":
			if !lastExit.IsZero() {
				if gap := time.Since(lastExit); gap < 190*time.Millisecond {
					t.Fatalf("restarted %s after the crash: no backoff", gap)
				}
			}
		}
		if e.Ev.Kind == "service.failed" {
			break
		}
	}
	if len(backoffs) != 2 || backoffs[0] != 200 || backoffs[1] != 400 {
		t.Fatalf("backoffs %v", backoffs)
	}
	s := svcStatus(in.plane.Status(), "crashy")
	if s.State != stateFailed || s.Restarts != 2 || s.LastExit == nil || s.LastExit.Code != 3 {
		t.Fatalf("status %+v", s)
	}
	// Every restart belonged to the run generation 1 started.
	if s.Generation != 1 || s.LastExit.Generation != 1 {
		t.Fatalf("run generation %d, exit generation %d, want 1 and 1", s.Generation, s.LastExit.Generation)
	}
	// A failed service starts again only on a new apply.
	time.Sleep(300 * time.Millisecond)
	if s := svcStatus(in.plane.Status(), "crashy"); s.State != stateFailed {
		t.Fatalf("failed service restarted by itself: %+v", s)
	}
	do(t, c, Request{Op: "apply", Desired: running("crashy")})
	waitEnv(t, w, 5*time.Second, "fresh start", func(e *Envelope) bool { return e.Ev.Kind == "service.started" })
}

// A stop queued on a failed service, then a start: the start pre-marks the service
// `starting` before the stop is read, and the stop still ends as stopped (its apply
// converges) before the start runs; a launch failure is stamped with the start's generation.
func TestALaunchFailureIsKeptForItsGenerationWhileAQueuedStartMovesOn(t *testing.T) {
	doc, _ := json.Marshal(map[string]any{"name": "local", "services": []map[string]any{
		{"name": "ghost", "command": "no-such-binary"}}})
	spec, err := ParseSpec(doc, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	s := newService(spec.Services[0], NewHub("local"))
	s.want(desiredRunning, 1)
	s.want(desiredStopped, 2)
	s.want(desiredRunning, 3)
	s.command(<-s.cmds) // the first start, which cannot launch
	// The queued start owns the state again, so the failure is not visible there...
	if st := s.cur(); st != stateStarting || s.startGen != 3 {
		t.Fatalf("after the failed start: %s, generation %d", st, s.startGen)
	}
	// ...but generation 1's apply can still ask for it by generation, until it has settled.
	if r, ok := s.failure(1); !ok || r == "" {
		t.Fatalf("failure(1) = %q, %v", r, ok)
	}
	if r, ok := s.failure(3); ok {
		t.Fatalf("failure(3) = %q: generation 3 has not run", r)
	}
	s.forget(1)
	if _, ok := s.failure(1); ok {
		t.Fatal("failure(1) still there after forget")
	}
}

func TestAStopQueuedBeforeAPreMarkedStartStillStops(t *testing.T) {
	doc, _ := json.Marshal(map[string]any{"name": "local", "services": []map[string]any{
		{"name": "ghost", "command": "no-such-binary"}}})
	spec, err := ParseSpec(doc, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	h := NewHub("local")
	cur := h.Subscribe("ctl", nil, h.Epoch())
	s := newService(spec.Services[0], h)
	s.setState(stateFailed)
	s.want(desiredStopped, 2)
	s.want(desiredRunning, 3)
	s.want(desiredStopped, 4)
	if st := s.cur(); st != stateStarting || s.startGen != 3 {
		t.Fatalf("after the wants: %s, generation %d", st, s.startGen)
	}
	s.command(<-s.cmds) // the stop: the start is still queued, so the service stays on its way
	if st := s.cur(); st != stateStarting || s.startGen != 3 {
		t.Fatalf("after the stop: %s, generation %d", st, s.startGen)
	}
	s.command(<-s.cmds) // the start, which cannot launch
	if st := s.cur(); st != stateFailed {
		t.Fatalf("after the start: %s", st)
	}
	s.command(<-s.cmds) // the second stop, on a failed service
	if st := s.cur(); st != stateStopped {
		t.Fatalf("after the second stop: %s", st)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	var got []string // kind:generation, so each stop's apply has its own event to settle on
	for len(got) < 4 {
		envs, _, err := cur.Next(ctx, 8)
		if err != nil {
			t.Fatalf("events so far %v: %v", got, err)
		}
		for _, e := range envs {
			got = append(got, fmt.Sprintf("%s:%v", e.Ev.Kind, ctlData(e)["generation"]))
		}
	}
	want := []string{"service.stopped:2", "service.starting:3", "service.failed:3", "service.stopped:4"}
	if !slices.Equal(got, want) {
		t.Fatalf("events %v, want %v", got, want)
	}
}

// A stop between two starts, all three queued: once the stop is done the service shows
// `starting` for the queued start (not `stopped`), stamped with it, so no plan meanwhile
// announces a start of its own; the queued start then launches under its generation.
func TestAStopKeepsAQueuedStartVisible(t *testing.T) {
	spec := fakeSpec(t, "local", nil, fakeSvc{name: "pod", args: []string{"--nodes", "1"}})
	h := NewHub("local")
	s := newService(spec.Services[0], h)
	t.Cleanup(s.halt)
	s.want(desiredRunning, 1)
	s.want(desiredStopped, 2)
	s.want(desiredRunning, 3)
	s.command(<-s.cmds) // the first start
	if st := s.cur(); st != stateRunning || s.proc == nil || s.startGen != 1 {
		t.Fatalf("after the first start: %s, generation %d", st, s.startGen)
	}
	s.command(<-s.cmds) // the stop
	if st := s.cur(); st != stateStarting || s.proc != nil || s.startGen != 3 || s.lastExit == nil || s.lastExit.Generation != 1 {
		t.Fatalf("after the stop: %s, generation %d, exit %+v", st, s.startGen, s.lastExit)
	}
	if d, st := s.observed(); d != desiredRunning || st != stateStarting {
		t.Fatalf("a plan would see %s/%s and announce a start", d, st)
	}
	s.command(<-s.cmds) // the queued start
	if st := s.cur(); st != stateRunning || s.proc == nil || s.startGen != 3 || s.queued != 0 {
		t.Fatalf("after the queued start: %s, generation %d, queued %d", st, s.startGen, s.queued)
	}
}

func TestRestartNeverStops(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "once", args: []string{"--crash-after-ms", "50"}, restart: restartNever}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("once")})
	x := waitEnv(t, w, 5*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	if d := ctlData(x); d["generation"] != 1.0 || d["requested"] != false {
		t.Fatalf("exit %s", x.Raw())
	}
	e := waitEnv(t, w, 5*time.Second, "next", func(e *Envelope) bool {
		return e.Ev.Kind == "service.stopped" || e.Ev.Kind == "service.restarting"
	})
	if e.Ev.Kind != "service.stopped" {
		t.Fatalf("restart never: %s", e.Raw())
	}
	// A job: stopped by its own exit, desired still running, the exit naming its generation.
	s := svcStatus(in.plane.Status(), "once")
	if s.State != stateStopped || s.Desired != desiredRunning || s.LastExit == nil || s.LastExit.Generation != 1 || s.Generation != 1 {
		t.Fatalf("after the exit: %+v (exit %+v)", s, s.LastExit)
	}
}

// A run that lasts past limits.wall_ms is killed by the plane and its exit says so; a job so
// ended stays stopped, with the reason in its last_exit.
func TestARunPastItsWallLimitIsKilledAndSaysSo(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "sleepy", restart: restartNever, wallMs: 300}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	t0 := time.Now()
	do(t, c, Request{Op: "apply", Desired: running("sleepy")})
	x := waitEnv(t, w, 5*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	took := time.Since(t0)
	d := ctlData(x)
	if d["reason"] != reasonWall || d["wall_ms"] != 300.0 || d["requested"] != false || d["generation"] != 1.0 || d["code"] == 0.0 {
		t.Fatalf("exit %s", x.Raw())
	}
	if took < 300*time.Millisecond || took > 3*time.Second {
		t.Fatalf("killed after %v, limit 300 ms", took)
	}
	e := waitEnv(t, w, 5*time.Second, "next", func(e *Envelope) bool {
		return e.Ev.Kind == "service.stopped" || e.Ev.Kind == "service.restarting"
	})
	if e.Ev.Kind != "service.stopped" || ctlData(e)["reason"] != reasonWall {
		t.Fatalf("a job killed at its limit is not restarted: %s", e.Raw())
	}
	s := svcStatus(in.plane.Status(), "sleepy")
	if s.State != stateStopped || s.Desired != desiredRunning || s.LastExit == nil || s.LastExit.Reason != reasonWall || s.LastExit.Generation != 1 || s.Pid != nil {
		t.Fatalf("after the kill: %+v (exit %+v)", s, s.LastExit)
	}
}

// A run that holds more than limits.memory_mb is killed by the plane and its exit says so,
// with the peak the OS counted; a job so ended stays stopped.
func TestARunPastItsMemoryLimitIsKilledAndSaysSo(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "hog", args: []string{"--hog-mb", "256"}, restart: restartNever, memMB: 48}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("hog")})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	if d["reason"] != reasonMemory || d["memory_mb"] != 48.0 || d["requested"] != false || d["code"] == 0.0 {
		t.Fatalf("exit %s", x.Raw())
	}
	e := waitEnv(t, w, 5*time.Second, "next", func(e *Envelope) bool {
		return e.Ev.Kind == "service.stopped" || e.Ev.Kind == "service.restarting"
	})
	if e.Ev.Kind != "service.stopped" {
		t.Fatalf("a job killed at its limit is not restarted: %s", e.Raw())
	}
	s := svcStatus(in.plane.Status(), "hog")
	if s.State != stateStopped || s.LastExit == nil || s.LastExit.Reason != reasonMemory || s.LastExit.PeakMemoryBytes <= 0 {
		t.Fatalf("after the kill: %+v (exit %+v)", s, s.LastExit)
	}
}

// A run that uses more processor time than its limit is killed, its exit names the limit
// (with the limit under `cpu_limit_ms`, since `cpu_ms` is what it used, past the limit by
// construction) and a job so ended stays stopped.
func TestARunPastItsCpuLimitIsKilledAndSaysSo(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "spinner", args: []string{"--burn-ms", "20000"}, restart: restartNever, cpuMs: 200}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("spinner")})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	used, _ := d["cpu_ms"].(float64)
	if d["reason"] != reasonCPU || d["cpu_limit_ms"] != 200.0 || d["requested"] != false || d["code"] == 0.0 || used <= 200 {
		t.Fatalf("exit %s", x.Raw())
	}
	e := waitEnv(t, w, 5*time.Second, "next", func(e *Envelope) bool {
		return e.Ev.Kind == "service.stopped" || e.Ev.Kind == "service.restarting"
	})
	if e.Ev.Kind != "service.stopped" {
		t.Fatalf("a job killed at its limit is not restarted: %s", e.Raw())
	}
	s := svcStatus(in.plane.Status(), "spinner")
	if s.State != stateStopped || s.LastExit == nil || s.LastExit.Reason != reasonCPU || s.LastExit.CpuMs <= 200 {
		t.Fatalf("after the kill: %+v (exit %+v)", s, s.LastExit)
	}
}

// A run under a processor-time limit it never reaches ends as any other, its exit without a
// reason and with the time it did use.
func TestARunUnderItsCpuLimitEndsOnItsOwn(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "brief", args: []string{"--burn-ms", "150", "--crash-after-ms", "50", "--crash-code", "0"}, restart: restartNever, cpuMs: 5000}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("brief")})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	used, _ := d["cpu_ms"].(float64)
	if _, has := d["reason"]; has || d["code"] != 0.0 || used <= 0 {
		t.Fatalf("exit %s", x.Raw())
	}
}

// A run that goes past its processor-time limit and ends before a poll could kill it is past
// it all the same: the exit is classified from the figure it reports, so a clean exit code
// does not pass for a run within its limit (Codex round 18).
func TestARunPastItsCpuLimitThatEndsBeforeThePollSaysSoToo(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "flash", args: []string{"--burn-ms", "60", "--crash-after-ms", "1", "--crash-code", "0"}, restart: restartNever, cpuMs: 20}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("flash")})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	used, _ := d["cpu_ms"].(float64)
	if d["reason"] != reasonCPU || d["cpu_limit_ms"] != 20.0 || used <= 20 {
		t.Fatalf("exit %s", x.Raw())
	}
	s := svcStatus(in.plane.Status(), "flash")
	if s.State != stateStopped || s.LastExit == nil || s.LastExit.Reason != reasonCPU {
		t.Fatalf("after the exit: %+v (exit %+v)", s, s.LastExit)
	}
}

// A run that holds more processes at once than its limit is killed, its exit names the limit
// and the most processes it held; a job so ended stays stopped. On Windows the job's cap (an
// eighth above, at least one) refuses the last spawn: the fake goes on with four processes,
// past the limit of three, and the watch (or the kernel's message) ends it.
func TestARunPastItsProcessLimitIsKilledAndSaysSo(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "forky", args: []string{"--children", "4"}, restart: restartNever, procs: 3}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("forky")})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	peak, _ := d["peak_processes"].(float64)
	if d["reason"] != reasonProcesses || d["processes"] != 3.0 || d["requested"] != false || d["code"] == 0.0 || peak <= 3 {
		t.Fatalf("exit %s", x.Raw())
	}
	e := waitEnv(t, w, 5*time.Second, "next", func(e *Envelope) bool {
		return e.Ev.Kind == "service.stopped" || e.Ev.Kind == "service.restarting"
	})
	if e.Ev.Kind != "service.stopped" {
		t.Fatalf("a job killed at its limit is not restarted: %s", e.Raw())
	}
	s := svcStatus(in.plane.Status(), "forky")
	if s.State != stateStopped || s.LastExit == nil || s.LastExit.Reason != reasonProcesses || s.LastExit.PeakProcesses <= 3 {
		t.Fatalf("after the kill: %+v (exit %+v)", s, s.LastExit)
	}
}

// A run under a process limit it never reaches ends as any other, its exit without a reason
// and with the most processes it held (the leader and its two children).
func TestARunUnderItsProcessLimitEndsOnItsOwn(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "small", args: []string{"--children", "2", "--crash-after-ms", "300", "--crash-code", "0"}, restart: restartNever, procs: 5}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("small")})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	if _, has := d["reason"]; has || d["code"] != 0.0 || d["peak_processes"] != 3.0 {
		t.Fatalf("exit %s", x.Raw())
	}
}

// A requested stop reports what the run cost too.
func TestAStopReportsTheRunsUsage(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "pod", args: []string{"--hog-mb", "16", "--burn-ms", "200"}}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("pod")})
	waitEnv(t, w, 5*time.Second, "started", func(e *Envelope) bool { return e.Ev.Kind == "service.started" })
	do(t, c, Request{Op: "stop", Service: "pod"})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	peak, _ := d["peak_memory_bytes"].(float64)
	cpu, _ := d["cpu_ms"].(float64)
	if d["requested"] != true || peak < 16<<20 || cpu < 50 {
		t.Fatalf("exit %s", x.Raw())
	}
}

// A run that ends on its own reports what it used, limit or not: the memory it peaked at and
// its processor time.
func TestAnExitReportsItsUsage(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "once", args: []string{"--hog-mb", "32", "--burn-ms", "300", "--crash-after-ms", "200"}, restart: restartNever}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("once")})
	x := waitEnv(t, w, 10*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	d := ctlData(x)
	peak, _ := d["peak_memory_bytes"].(float64)
	cpu, _ := d["cpu_ms"].(float64)
	if peak < 32<<20 || cpu < 100 || d["reason"] != nil {
		t.Fatalf("exit %s", x.Raw())
	}
	s := svcStatus(in.plane.Status(), "once")
	if s.LastExit == nil || s.LastExit.CpuMs < 100 || s.LastExit.PeakMemoryBytes < 32<<20 {
		t.Fatalf("status exit %+v", s.LastExit)
	}
}

// With a restart policy, a run killed at its limit is a failure: the service restarts.
func TestARunPastItsWallLimitRestarts(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "sleepy", restart: restartOnFailure, wallMs: 100, max: 5}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	do(t, c, Request{Op: "apply", Desired: running("sleepy")})
	waitEnv(t, w, 5*time.Second, "exit", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	e := waitEnv(t, w, 5*time.Second, "next", func(e *Envelope) bool {
		return e.Ev.Kind == "service.stopped" || e.Ev.Kind == "service.restarting"
	})
	if e.Ev.Kind != "service.restarting" {
		t.Fatalf("on-failure after a wall kill: %s", e.Raw())
	}
}

func childPid(t *testing.T, w *frameWaiter) int {
	t.Helper()
	e := waitEnv(t, w, 10*time.Second, "child event", func(e *Envelope) bool { return e.Ev.Kind == "child" })
	var d struct {
		Pid int `json:"pid"`
	}
	json.Unmarshal(e.Ev.Data, &d)
	return d.Pid
}

// stop kills the whole tree: the service and the grandchild it spawned.
func TestStopKillsGrandchild(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "parent", args: []string{"--spawn-child"}}), "")
	c := dial(t, in)
	w := subscribe(t, c, "svc/parent", nil)
	do(t, c, Request{Op: "apply", Desired: running("parent")})
	gc := childPid(t, w)
	pid := *svcStatus(in.plane.Status(), "parent").Pid
	if !pidAlive(pid) || !pidAlive(gc) {
		t.Fatalf("not running: service %d %v, grandchild %d %v", pid, pidAlive(pid), gc, pidAlive(gc))
	}
	t0 := time.Now()
	do(t, c, Request{Op: "stop", Service: "parent"})
	t.Logf("stop with a grandchild: %s", time.Since(t0))
	if pidAlive(pid) || pidAlive(gc) {
		t.Fatalf("after stop: service %d alive %v, grandchild %d alive %v", pid, pidAlive(pid), gc, pidAlive(gc))
	}
}

// A root that ignores stdin gets the 2 s grace, then the tree is killed.
func TestStopKillsTreeIgnoringStdin(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "stubborn", args: []string{"--spawn-child", "--ignore-stdin"}}), "")
	c := dial(t, in)
	w := subscribe(t, c, "svc/stubborn", nil)
	do(t, c, Request{Op: "apply", Desired: running("stubborn")})
	gc := childPid(t, w)
	pid := *svcStatus(in.plane.Status(), "stubborn").Pid
	t0 := time.Now()
	do(t, c, Request{Op: "stop", Service: "stubborn"})
	if d := time.Since(t0); d < stopGrace {
		t.Fatalf("killed before the grace: %s", d)
	}
	if pidAlive(pid) || pidAlive(gc) {
		t.Fatalf("after stop: service alive %v, grandchild alive %v", pidAlive(pid), pidAlive(gc))
	}
}

// Shutting the plane down kills its own trees and nothing else.
func TestShutdownTouchesOnlyItsOwnProcesses(t *testing.T) {
	foreign := exec.Command(testExe(t))
	foreign.Env = append(os.Environ(), "IXCTL_TEST_ROLE=grandchild")
	if err := foreign.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { foreign.Process.Kill(); foreign.Wait() }()

	spec := fakeSpec(t, "local", nil, fakeSvc{name: "pod", args: []string{"--spawn-child"}})
	in, err := startInstance(spec, "", "127.0.0.1:0", t.TempDir(), "disk", nil)
	if err != nil {
		t.Fatal(err)
	}
	in.plane.Start()
	c := dial(t, in)
	w := subscribe(t, c, "svc/pod", nil)
	do(t, c, Request{Op: "apply", Desired: running("pod")})
	gc := childPid(t, w)
	pid := *svcStatus(in.plane.Status(), "pod").Pid
	in.close()
	if pidAlive(pid) || pidAlive(gc) {
		t.Fatalf("own tree survived shutdown: %v %v", pidAlive(pid), pidAlive(gc))
	}
	if !pidAlive(foreign.Process.Pid) {
		t.Fatal("shutdown touched a process the plane did not start")
	}
	if _, err := os.Stat(in.stateFile); !os.IsNotExist(err) {
		t.Fatalf("state file left behind: %v", err)
	}
}
