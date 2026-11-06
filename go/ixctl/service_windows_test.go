//go:build windows

package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"golang.org/x/sys/windows/svc"
)

// The adapter hands the SCM's stop, shutdown and interrogate to the host, maps each report
// to the SCM's status, leaves the final stopped one to the svc package, and returns the
// host's code as service-specific.
func TestTheSCMAdapterCarriesControlsAndStates(t *testing.T) {
	got := make(chan hostControl, 4)
	h := &scmHandler{setup: okSetup, run: func(_ hostOptions, c <-chan hostControl, report func(hostStatus)) uint32 {
		report(hostStatus{State: hostStartPending, Checkpoint: 1, WaitHint: 10 * time.Second})
		report(hostStatus{State: hostRunning, Accepts: true})
		for i := 0; i < 3; i++ {
			got <- <-c
		}
		report(hostStatus{State: hostStopPending, Checkpoint: 2, WaitHint: 1500 * time.Millisecond})
		report(hostStatus{State: hostStopped, ExitCode: exitStopUnobserved})
		return exitStopUnobserved
	}}
	r := make(chan svc.ChangeRequest, 4)
	s := make(chan svc.Status, 8)
	r <- svc.ChangeRequest{Cmd: svc.Interrogate}
	r <- svc.ChangeRequest{Cmd: svc.ParamChange} // not ours: dropped
	r <- svc.ChangeRequest{Cmd: svc.Stop}
	r <- svc.ChangeRequest{Cmd: svc.Shutdown}
	specific, code := h.Execute(nil, r, s)
	if !specific || code != exitStopUnobserved {
		t.Fatalf("Execute returned %v %d", specific, code)
	}
	close(s)
	var states []string
	for st := range s {
		states = append(states, fmt.Sprintf("%d/%d/%d/%d", st.State, st.Accepts, st.CheckPoint, st.WaitHint))
	}
	want := []string{
		fmt.Sprintf("%d/0/1/10000", svc.StartPending), // the adapter's own, before setup
		fmt.Sprintf("%d/0/1/10000", svc.StartPending),
		fmt.Sprintf("%d/%d/0/0", svc.Running, svc.AcceptStop|svc.AcceptShutdown),
		fmt.Sprintf("%d/0/2/1500", svc.StopPending),
	}
	if fmt.Sprint(states) != fmt.Sprint(want) {
		t.Fatalf("statuses %v, want %v", states, want)
	}
	close(got)
	var controls []hostControl
	for c := range got {
		controls = append(controls, c)
	}
	wantC := []hostControl{{controlInterrogate, "scm"}, {controlStop, "scm"}, {controlShutdown, "scm"}}
	if fmt.Sprint(controls) != fmt.Sprint(wantC) {
		t.Fatalf("controls %v, want %v", controls, wantC)
	}
	ok := &scmHandler{setup: okSetup, run: func(hostOptions, <-chan hostControl, func(hostStatus)) uint32 { return exitOK }}
	if specific, code := ok.Execute(nil, r, make(chan svc.Status, 1)); specific || code != 0 {
		t.Fatalf("a clean stop returned %v %d", specific, code)
	}
}

// A killed host takes its trees with it (the job closes with the process); the next host
// starts a new epoch, and what the dead one was asked to run is not run again.
func TestAKilledHostTakesItsTreesAndTheNextOneReplaysNothing(t *testing.T) {
	dir := t.TempDir()
	specPath := filepath.Join(dir, "spec.json")
	specDoc, _ := json.Marshal(map[string]any{"name": "local", "services": []map[string]any{{
		"name": "task", "command": testExe(t), "args": []string{"--spawn-child"},
		"env": map[string]string{"IXCTL_TEST_ROLE": "fake"},
	}}})
	os.WriteFile(specPath, specDoc, 0o600)
	state := filepath.Join(dir, "state")
	host := func() *exec.Cmd {
		cmd := exec.Command(testExe(t), "service", "run", specPath, "--state", state)
		cmd.Env = append(os.Environ(), "IXCTL_TEST_ROLE=main")
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
		return cmd
	}
	waitClient := func(pid int) *instance {
		deadline := time.Now().Add(15 * time.Second)
		for {
			if st, err := readStateFile(filepath.Join(state, "local.json")); err == nil && st.Pid == pid {
				return &instance{addr: st.Addr, token: st.Token}
			}
			if time.Now().After(deadline) {
				t.Fatalf("no state file for host %d", pid)
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
	epochOf := func(in *instance) string {
		_, _, body := httpDo(t, in, "GET", "/health", nil, nil)
		var h struct{ Epoch string }
		json.Unmarshal(body, &h)
		return h.Epoch
	}

	first := host()
	in := waitClient(first.Process.Pid)
	c := dial(t, in)
	w := subscribe(t, c, "svc/task", nil)
	do(t, c, Request{Op: "apply", Desired: running("task")})
	gc := childPid(t, w)
	// Opened now, while it lives: the handle pins the process, so the cleanup cannot reach
	// another one that reused the pid.
	if p, err := os.FindProcess(gc); err == nil {
		t.Cleanup(func() { p.Kill(); p.Release() })
	}
	var st StatusReply
	json.Unmarshal(do(t, c, Request{Op: "status"}), &st)
	pid := *svcStatus(st, "task").Pid
	epoch := epochOf(in)
	c.Close()

	first.Process.Kill()
	first.Wait()
	if !waitPid(pid, false, 10*time.Second) || !waitPid(gc, false, 10*time.Second) {
		t.Fatalf("the killed host's tree survived: task %v, grandchild %v", pidAlive(pid), pidAlive(gc))
	}

	second := host()
	in2 := waitClient(second.Process.Pid)
	if e := epochOf(in2); e == "" || e == epoch {
		t.Fatalf("epoch %q after %q", e, epoch)
	}
	code, _, body := httpDo(t, in2, "GET", "/v1/status", nil, nil)
	json.Unmarshal(body, &st)
	if code != 200 || st.Generation != 0 || svcStatus(st, "task").State != stateStopped {
		t.Fatalf("after the restart: %d %s", code, body)
	}
	if code, _, _ := httpDo(t, in2, "POST", "/v1/shutdown", nil, nil); code != 200 {
		t.Fatalf("shutdown %d", code)
	}
	exited := make(chan error, 1)
	go func() { exited <- second.Wait() }()
	select {
	case <-exited:
	case <-time.After(15 * time.Second):
		t.Fatal("the second host did not exit")
	}
	logPath := filepath.Join(state, "local.service.log")
	f, err := os.Open(logPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var ready, observed int
	for sc := bufio.NewScanner(f); sc.Scan(); {
		var e map[string]any
		json.Unmarshal(sc.Bytes(), &e)
		if e["event"] == "host.ready" {
			ready++
		}
		if e["event"] == "host.stop" && e["phase"] == "observed" && e["source"] == "api" {
			observed++
		}
	}
	if ready != 2 || observed != 1 {
		t.Fatalf("log %s: %d ready, %d observed api stops", logPath, ready, observed)
	}
}

func okSetup() (hostOptions, func(), error) { return hostOptions{}, func() {}, nil }

// Under the SCM the setup (spec, state dir, log) runs after the dispatcher connected and
// start-pending was reported: a failed setup is a service-specific start failure the SCM sees.
func TestASetupFailureUnderTheSCMIsAStartFailure(t *testing.T) {
	h := &scmHandler{
		setup: func() (hostOptions, func(), error) { return hostOptions{}, nil, errors.New("no spec") },
		run: func(hostOptions, <-chan hostControl, func(hostStatus)) uint32 {
			t.Error("the host ran without its setup")
			return exitOK
		},
	}
	s := make(chan svc.Status, 4)
	specific, code := h.Execute(nil, make(chan svc.ChangeRequest), s)
	if !specific || code != exitStartFailed || h.code != exitStartFailed {
		t.Fatalf("Execute returned %v %d, kept %d for the process", specific, code, h.code)
	}
	close(s)
	first, ok := <-s
	if !ok || first.State != svc.StartPending {
		t.Fatalf("first status %+v %v", first, ok)
	}
}
