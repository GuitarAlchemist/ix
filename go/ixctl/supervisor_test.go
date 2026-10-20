package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"testing"
	"time"
)

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
	if st := s.cur(); st != stateStarting || s.startGen != 3 {
		t.Fatalf("after the wants: %s, generation %d", st, s.startGen)
	}
	s.command(<-s.cmds) // the stop
	if st := s.cur(); st != stateStopped {
		t.Fatalf("after the stop: %s", st)
	}
	s.command(<-s.cmds) // the start, which cannot launch
	if st := s.cur(); st != stateFailed {
		t.Fatalf("after the start: %s", st)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	var kinds []string
	var failedGen any
	for len(kinds) < 3 {
		envs, _, err := cur.Next(ctx, 8)
		if err != nil {
			t.Fatalf("events so far %v: %v", kinds, err)
		}
		for _, e := range envs {
			kinds = append(kinds, e.Ev.Kind)
			if e.Ev.Kind == "service.failed" {
				failedGen = ctlData(e)["generation"]
			}
		}
	}
	if kinds[0] != "service.stopped" || kinds[1] != "service.starting" || kinds[2] != "service.failed" || failedGen != 3.0 {
		t.Fatalf("events %v, failed generation %v", kinds, failedGen)
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
	in, err := startInstance(spec, "", "127.0.0.1:0", t.TempDir())
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
