package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// #450 A4 with a real descendant. The worker starts a process that inherits its stdout and
// exits at once. Where that descendant can leave the tree (a setsid on Linux), the plane's kill
// misses it: run.end comes at the drain bound marked truncated, and the line the descendant
// writes afterwards (released by the test once it holds run.end) is counted in late_lines,
// never forwarded. Where it cannot (a job object on
// Windows), the descendant dies with the tree and the output is complete.
func TestADescendantHoldingStdoutEndsTheRunAtTheDrainBound(t *testing.T) {
	release := filepath.Join(t.TempDir(), "release")
	t.Setenv("IXCTL_TEST_RELEASE", release) // the service, and so the descendant, inherit it
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "holder", args: []string{"--hold-stdout"}, restart: restartNever}), "")
	c := dial(t, in)
	ctl := subscribe(t, c, "ctl", nil)
	out := subscribeMarkers(t, c, "svc/holder", runMarkers)
	do(t, c, Request{Op: "apply", Desired: running("holder")})

	started := waitEnv(t, ctl, 5*time.Second, "started", func(e *Envelope) bool {
		return e.Ev.Kind == "service.started"
	})
	if started.Run == nil {
		t.Fatalf("started has no run: %s", started.Raw())
	}
	run := *started.Run
	holder := childPid(t, out)
	t.Cleanup(func() {
		if p, err := os.FindProcess(holder); err == nil && pidAlive(holder) {
			p.Kill()
		}
	})
	var end *Envelope
	late := false // the descendant's line was forwarded before run.end
	for end == nil {
		f, err := out.next(10*time.Second, func(e *Envelope) bool { return e.Run != nil && *e.Run == run })
		if err != nil {
			t.Fatalf("waiting for run end: %v", err)
		}
		switch {
		case f.Env.Ev.Kind == runEndKind:
			end = f.Env
		case f.Env.Ev.Kind == "log" && strings.Contains(string(f.Env.Ev.Data), "late from holder"):
			late = true
			t.Logf("the descendant's line at +%d ms", f.Env.T-started.T)
		}
	}
	var data struct {
		Outcome string `json:"outcome"`
		Output  string `json:"output"`
		DrainMS int64  `json:"drain_ms"`
	}
	json.Unmarshal(end.Ev.Data, &data)
	t.Logf("run.end at +%d ms", end.T-started.T)
	if err := os.WriteFile(release, nil, 0o600); err != nil { // the descendant may write now
		t.Fatal(err)
	}
	t.Logf("run.end %s; descendant %d alive %v, its line forwarded %v", end.Ev.Data, holder, pidAlive(holder), late)

	if runtime.GOOS == "windows" {
		// The job kills the descendant with the tree: it never writes, the drain is immediate.
		if data.Outcome != "started" || data.Output != "complete" || late || pidAlive(holder) {
			t.Fatalf("windows: end %+v, descendant %d alive %v, its line forwarded %v",
				data, holder, pidAlive(holder), late)
		}
		return
	}
	if late {
		t.Fatal("the descendant's line was forwarded before run.end")
	}
	bound := drainLimit.Milliseconds()
	if data.Outcome != "started" || data.Output != "truncated" || data.DrainMS < bound ||
		data.DrainMS > bound+500 || !pidAlive(holder) {
		t.Fatalf("end %+v (bound %d ms), descendant %d alive %v", data, bound, holder, pidAlive(holder))
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if e := svcStatus(in.plane.Status(), "holder").LastExit; e != nil && e.LateLines == 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if e := svcStatus(in.plane.Status(), "holder").LastExit; e == nil || e.LateLines != 1 {
		t.Fatalf("last_exit %+v, want late_lines 1", e)
	}
	if f, err := out.next(300*time.Millisecond, func(e *Envelope) bool {
		return e.Run != nil && *e.Run == run && e.Ev.Kind != runEndKind
	}); err == nil {
		t.Fatalf("a line after run.end was forwarded: %s", f.Env.Raw())
	}
}

// upProcess is `ixctl up` as its own process, the way an operator runs it: the test binary in
// its main role, read through its ready line and state file.
func upProcess(t *testing.T, spec, state, name string) (*exec.Cmd, *instance) {
	t.Helper()
	cmd := exec.Command(testExe(t), "up", spec, "--state", state, "--name", name)
	cmd.Env = append(os.Environ(), "IXCTL_TEST_ROLE=main")
	stdout, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil {
		t.Fatalf("no ready line: %v", err)
	}
	var r readyLine
	if err := json.Unmarshal([]byte(line), &r); err != nil || r.Event != "ready" || r.Pid != cmd.Process.Pid {
		t.Fatalf("ready line %q", line)
	}
	st, err := readStateFile(r.StateFile)
	if err != nil {
		t.Fatal(err)
	}
	return cmd, &instance{addr: st.Addr, token: st.Token}
}

// #453 B4 after a real hard stop. A plane process runs a job; a client waits on that run; the
// process is killed outright (no shutdown, no close). Each step is kept apart:
//   - the apply answered start-pending, and the launch wait named the run;
//   - the wait in flight on the dead plane fails as a transport error, with no outcome;
//   - the job's tree died with the plane;
//   - a new plane on the same state dir boots a new epoch at generation 0, the job stopped;
//   - the same run and generation, asked of the new plane, end unknown (other-epoch).
func TestAWaitAfterAHardPlaneStopEndsUnknownOtherEpoch(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state")
	specPath := filepath.Join(dir, "spec.json")
	spec := fakeSpec(t, "hard", nil, fakeSvc{name: "job", restart: restartNever})
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(specPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}

	first, in := upProcess(t, specPath, state, "hard")
	code, _, body := httpDo(t, in, "POST", "/v1/apply", running("job"), nil)
	var applied ApplyReply
	if err := json.Unmarshal(body, &applied); err != nil || code != 200 ||
		applied.Results["job"].Result != "start-pending" {
		t.Fatalf("apply: %d %s", code, body)
	}
	code, _, body = httpDo(t, in, "POST", "/v1/wait", map[string]any{"wait": waitLaunch,
		"epoch": applied.Epoch, "generation": applied.Generation, "service": "job",
		"timeout_ms": testWaitTimeoutMS}, nil)
	var launch WaitReply
	if err := json.Unmarshal(body, &launch); err != nil || code != 200 || launch.Run == nil {
		t.Fatalf("launch wait: %d %s", code, body)
	}
	pid := 0
	for deadline := time.Now().Add(5 * time.Second); pid == 0 && time.Now().Before(deadline); {
		code, _, body = httpDo(t, in, "GET", "/v1/status", nil, nil)
		var st StatusReply
		json.Unmarshal(body, &st)
		if p := svcStatus(st, "job").Pid; p != nil {
			pid = *p
		}
	}
	if pid == 0 || !pidAlive(pid) {
		t.Fatalf("the job is not running: pid %d", pid)
	}

	inflight := make(chan error, 1)
	go func() {
		_, err := httpRaw(in, "POST", "/v1/wait",
			map[string]any{"wait": waitRun, "run": launch.Run, "timeout_ms": 30000})
		inflight <- err
	}()
	time.Sleep(200 * time.Millisecond) // the wait is admitted before the kill
	if err := first.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	first.Wait()
	select {
	case err := <-inflight:
		if err == nil {
			t.Fatal("the wait in flight on the killed plane returned an answer")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the wait in flight did not end with its plane")
	}
	if !waitPid(pid, false, 10*time.Second) {
		t.Fatalf("job %d outlived its killed plane", pid)
	}

	_, again := upProcess(t, specPath, state, "hard")
	code, _, body = httpDo(t, again, "GET", "/v1/status", nil, nil)
	var st StatusReply
	if err := json.Unmarshal(body, &st); err != nil || code != 200 {
		t.Fatalf("status: %d %s", code, body)
	}
	if st.Epoch == applied.Epoch || st.Generation != 0 || svcStatus(st, "job").State != "stopped" {
		t.Fatalf("after the hard stop: epoch %s (was %s), generation %d, job %s",
			st.Epoch, applied.Epoch, st.Generation, svcStatus(st, "job").State)
	}
	for _, req := range []map[string]any{
		{"wait": waitRun, "run": launch.Run, "timeout_ms": testWaitTimeoutMS},
		{"wait": waitGeneration, "epoch": applied.Epoch, "generation": applied.Generation,
			"timeout_ms": testWaitTimeoutMS},
	} {
		code, _, body := httpDo(t, again, "POST", "/v1/wait", req, nil)
		var reply WaitReply
		if err := json.Unmarshal(body, &reply); err != nil || code != 200 ||
			reply.Outcome != waitUnknown || reply.Reason != waitOtherEpoch {
			t.Fatalf("%v after the hard stop: %d %s", req["wait"], code, body)
		}
	}
}

// httpRaw is httpDo without the test's fatal on a transport error: the error is the result.
func httpRaw(in *instance, method, path string, body any) ([]byte, error) {
	b, _ := json.Marshal(body)
	req, err := http.NewRequest(method, "http://"+in.addr+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+in.token)
	resp, err := loopbackHTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}
