package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeSCM drives runHost as the service control manager would, and keeps what it reports.
type fakeSCM struct {
	controls chan hostControl
	mu       sync.Mutex
	reports  []hostStatus
}

func newFakeSCM() *fakeSCM { return &fakeSCM{controls: make(chan hostControl, 4)} }

func (f *fakeSCM) report(st hostStatus) {
	f.mu.Lock()
	f.reports = append(f.reports, st)
	f.mu.Unlock()
}

func (f *fakeSCM) all() []hostStatus {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]hostStatus(nil), f.reports...)
}

// states is the reported sequence of states, repeats folded.
func (f *fakeSCM) states() []hostState {
	var out []hostState
	for _, st := range f.all() {
		if len(out) == 0 || out[len(out)-1] != st.State {
			out = append(out, st.State)
		}
	}
	return out
}

func (f *fakeSCM) waitState(t testing.TB, s hostState, d time.Duration) hostStatus {
	t.Helper()
	deadline := time.Now().Add(d)
	for {
		for _, st := range f.all() {
			if st.State == s {
				return st
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("waiting for %s; reported %v", s, f.states())
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// syncBuffer is a log the host writes while the test reads it.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// logEvents parses the host's log lines.
func logEvents(t testing.TB, log string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(log), "\n") {
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("log line %q is not JSON: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

func hostOpts(spec *Spec, stateDir string, log *syncBuffer) hostOptions {
	return hostOptions{spec: spec, addr: "127.0.0.1:0", stateDir: stateDir, stopTimeout: 10 * time.Second, log: log}
}

// startHost runs a host in process; the channel yields its exit code.
func startHost(o hostOptions) (*fakeSCM, <-chan uint32) {
	scm := newFakeSCM()
	done := make(chan uint32, 1)
	go func() { done <- runHost(o, scm.controls, scm.report) }()
	return scm, done
}

func waitExit(t testing.TB, done <-chan uint32, d time.Duration) uint32 {
	t.Helper()
	select {
	case code := <-done:
		return code
	case <-time.After(d):
		t.Fatal("the host did not return")
		return 0
	}
}

// hostClient reads the running host's state file, as any local client does.
func hostClient(t testing.TB, stateDir, name string) *instance {
	t.Helper()
	st, err := readStateFile(filepath.Join(stateDir, name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	return &instance{addr: st.Addr, token: st.Token}
}

// The SCM sees start-pending, then running (stop and shutdown accepted), then on a stop
// stop-pending and stopped with code 0; the name's lock and state file are let go.
func TestAHostReportsStartRunningThenStoppedOnAStop(t *testing.T) {
	dir := t.TempDir()
	log := &syncBuffer{}
	spec := fakeSpec(t, "solo", nil)
	scm, done := startHost(hostOpts(spec, dir, log))
	run := scm.waitState(t, hostRunning, 10*time.Second)
	if !run.Accepts {
		t.Fatalf("running accepts no stop: %+v", run)
	}
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	if code := waitExit(t, done, 10*time.Second); code != exitOK {
		t.Fatalf("exit code %d", code)
	}
	got := scm.states()
	want := []hostState{hostStartPending, hostRunning, hostStopPending, hostStopped}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("states %v, want %v", got, want)
	}
	if last := scm.all()[len(scm.all())-1]; last.ExitCode != exitOK || last.Accepts {
		t.Fatalf("last report %+v", last)
	}
	if _, err := os.Stat(filepath.Join(dir, "solo.json")); !os.IsNotExist(err) {
		t.Fatalf("state file left: %v", err)
	}
	scm2, done2 := startHost(hostOpts(spec, dir, &syncBuffer{}))
	scm2.waitState(t, hostRunning, 10*time.Second)
	scm2.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	waitExit(t, done2, 10*time.Second)
}

// An interrogation gets the current status back, and changes nothing.
func TestAnInterrogationReportsTheCurrentStatus(t *testing.T) {
	scm, done := startHost(hostOpts(fakeSpec(t, "solo", nil), t.TempDir(), &syncBuffer{}))
	scm.waitState(t, hostRunning, 10*time.Second)
	n := len(scm.all())
	scm.controls <- hostControl{Cmd: controlInterrogate, Source: "scm"}
	deadline := time.Now().Add(5 * time.Second)
	for len(scm.all()) == n {
		if time.Now().After(deadline) {
			t.Fatal("no report after an interrogation")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if st := scm.all()[n]; st.State != hostRunning || !st.Accepts {
		t.Fatalf("interrogation answered %+v", st)
	}
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	waitExit(t, done, 10*time.Second)
}

// A stop while a task runs kills its whole tree within the bound, and the three moments are
// told apart: requested and received on ctl (before the task's own stop), observed in the log.
func TestAStopDuringARunningTaskKillsItsTreeAndTellsEachPhase(t *testing.T) {
	dir := t.TempDir()
	log := &syncBuffer{}
	spec := fakeSpec(t, "local", nil, fakeSvc{name: "task", args: []string{"--spawn-child", "--ignore-stdin"}})
	scm, done := startHost(hostOpts(spec, dir, log))
	scm.waitState(t, hostRunning, 10*time.Second)
	in := hostClient(t, dir, "local")
	c := dial(t, in)
	ctl := subscribe(t, c, "ctl", nil)
	w := subscribe(t, c, "svc/task", nil)
	do(t, c, Request{Op: "apply", Desired: running("task")})
	gc := childPid(t, w)
	var st StatusReply
	json.Unmarshal(do(t, c, Request{Op: "status"}), &st)
	pid := *svcStatus(st, "task").Pid

	t0 := time.Now()
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	var phases []string
	for len(phases) < 2 {
		e := waitEnv(t, ctl, 10*time.Second, "host.stop", func(e *Envelope) bool {
			return e.Ev.Kind == "host.stop" || e.Ev.Kind == "service.stopped"
		})
		if e.Ev.Kind == "service.stopped" {
			t.Fatalf("the task stopped before the host said %v", phases)
		}
		d := ctlData(e)
		if d["source"] != "scm" {
			t.Fatalf("host.stop %v", d)
		}
		phases = append(phases, fmt.Sprint(d["phase"]))
	}
	if fmt.Sprint(phases) != "[requested received]" {
		t.Fatalf("phases on ctl %v", phases)
	}
	if code := waitExit(t, done, 15*time.Second); code != exitOK {
		t.Fatalf("exit code %d", code)
	}
	if d := time.Since(t0); d < stopGrace {
		t.Fatalf("the stubborn task was killed before its grace: %s", d)
	}
	if pidAlive(pid) || pidAlive(gc) {
		t.Fatalf("after the stop: task alive %v, grandchild alive %v", pidAlive(pid), pidAlive(gc))
	}
	pending := 0
	var last uint32
	for _, r := range scm.all() {
		if r.State != hostStopPending {
			continue
		}
		pending++
		if r.WaitHint <= 0 || r.Checkpoint <= last {
			t.Fatalf("stop-pending report %+v after checkpoint %d", r, last)
		}
		last = r.Checkpoint
	}
	if pending < 2 {
		t.Fatalf("%d stop-pending reports over a %s stop; want progress", pending, time.Since(t0))
	}
	var observed map[string]any
	for _, e := range logEvents(t, log.String()) {
		if e["event"] == "host.stop" && e["phase"] == "observed" {
			observed = e
		}
	}
	if observed == nil || observed["ms"].(float64) < float64(stopGrace.Milliseconds()) {
		t.Fatalf("observed stop in the log: %v\n%s", observed, log.String())
	}
	if strings.Contains(log.String(), in.token) {
		t.Fatal("the token reached the log")
	}
}

// A second host for the same name is refused before it reports running, with its own exit
// code, and the first one goes on serving.
func TestASecondHostForTheSameNameIsRefusedAndTheFirstKeepsServing(t *testing.T) {
	dir := t.TempDir()
	spec := fakeSpec(t, "solo", nil)
	first, done := startHost(hostOpts(spec, dir, &syncBuffer{}))
	first.waitState(t, hostRunning, 10*time.Second)

	log := &syncBuffer{}
	second, done2 := startHost(hostOpts(spec, dir, log))
	if code := waitExit(t, done2, 10*time.Second); code != exitAlreadyRunning {
		t.Fatalf("second host exit %d, want %d", code, exitAlreadyRunning)
	}
	for _, s := range second.states() {
		if s == hostRunning {
			t.Fatalf("the second host reported running: %v", second.states())
		}
	}
	if !strings.Contains(log.String(), `"event":"host.refused"`) {
		t.Fatalf("log: %s", log.String())
	}
	in := hostClient(t, dir, "solo")
	if code, _, body := httpDo(t, in, "GET", "/v1/status", nil, nil); code != 200 {
		t.Fatalf("first host after the refusal: %d %s", code, body)
	}
	first.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	waitExit(t, done, 10*time.Second)
}

// A stop that outlives its bound is reported as not observed: the host returns with its own
// code and does not pretend the trees are gone.
func TestAStopPastItsBoundSaysItWasNotObserved(t *testing.T) {
	log := &syncBuffer{}
	o := hostOpts(fakeSpec(t, "solo", nil), t.TempDir(), log)
	o.stopTimeout = 200 * time.Millisecond
	closed := make(chan struct{})
	o.close = func(in *instance) {
		time.Sleep(time.Second)
		in.close()
		close(closed)
	}
	scm, done := startHost(o)
	scm.waitState(t, hostRunning, 10*time.Second)
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	if code := waitExit(t, done, 5*time.Second); code != exitStopUnobserved {
		t.Fatalf("exit %d, want %d", code, exitStopUnobserved)
	}
	if last := scm.all()[len(scm.all())-1]; last.State != hostStopped || last.ExitCode != exitStopUnobserved {
		t.Fatalf("last report %+v", last)
	}
	var unobserved bool
	for _, e := range logEvents(t, log.String()) {
		if e["event"] == "host.stop" && e["phase"] == "unobserved" {
			unobserved = true
		}
		if e["event"] == "host.stop" && e["phase"] == "observed" {
			t.Fatalf("observed before the close ended: %v", e)
		}
	}
	if !unobserved {
		t.Fatalf("log: %s", log.String())
	}
	<-closed
}

// The state file of a host that died is not a running instance: the next host starts.
func TestAStaleStateFileDoesNotBlockTheNextHost(t *testing.T) {
	dir := t.TempDir()
	stale, _ := json.Marshal(stateFile{Name: "solo", Addr: "127.0.0.1:1", Pid: 1, Token: strings.Repeat("0", 64)})
	if err := os.WriteFile(filepath.Join(dir, "solo.json"), stale, 0o600); err != nil {
		t.Fatal(err)
	}
	scm, done := startHost(hostOpts(fakeSpec(t, "solo", nil), dir, &syncBuffer{}))
	scm.waitState(t, hostRunning, 10*time.Second)
	if st := hostClient(t, dir, "solo"); st.addr == "127.0.0.1:1" {
		t.Fatal("the stale state file was kept")
	}
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	waitExit(t, done, 10*time.Second)
}

// The service log keeps two files at most, each within its bound.
func TestTheServiceLogRotatesWithinItsBound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.service.log")
	l, err := openRotatingLog(path, 100)
	if err != nil {
		t.Fatal(err)
	}
	line := []byte(strings.Repeat("a", 59) + "\n")
	for i := 0; i < 5; i++ {
		if _, err := l.Write(line); err != nil {
			t.Fatal(err)
		}
	}
	l.Close()
	matches, _ := filepath.Glob(path + "*")
	if len(matches) != 2 {
		t.Fatalf("files %v", matches)
	}
	for _, m := range matches {
		if fi, _ := os.Stat(m); fi.Size() > 100 || fi.Size() == 0 {
			t.Fatalf("%s holds %d bytes", m, fi.Size())
		}
	}
}

// The install plan is explicit: a virtual account with no password, a manual start, no
// automatic restart, loopback only, and a way back.
func TestTheInstallPlanIsExplicitAndStoresNoSecret(t *testing.T) {
	p := installPlan(planOptions{service: "ixctl", exe: `C:\Program Files\ixctl\ixctl.exe`,
		dir: `C:\ProgramData\ixctl`, clientUser: `HOST\stef`})
	for _, want := range []string{
		`obj= "NT SERVICE\ixctl"`, `start= demand`, `--addr 127.0.0.1:0`,
		`--state "C:\ProgramData\ixctl"`, `icacls "C:\ProgramData\ixctl"`, `"NT SERVICE\ixctl:(OI)(CI)M"`,
		`"HOST\stef:(OI)(CI)RX"`, `sc.exe start ixctl`, `sc.exe query ixctl`, `sc.exe stop ixctl`,
		`sc.exe delete ixctl`,
	} {
		if !strings.Contains(p, want) {
			t.Errorf("plan lacks %q", want)
		}
	}
	for _, bad := range []string{"password=", "0.0.0.0", "failure", "LocalSystem", "start= auto"} {
		if strings.Contains(p, bad) {
			t.Errorf("plan holds %q", bad)
		}
	}
	for _, section := range []string{"# install", "# start", "# status", "# stop", "# uninstall"} {
		if !strings.Contains(p, section) {
			t.Errorf("plan lacks section %q", section)
		}
	}
}

// The plane refuses what it is asked once a host stop was received.
func TestNoApplyIsAdmittedOnceTheStopIsReceived(t *testing.T) {
	dir := t.TempDir()
	o := hostOpts(fakeSpec(t, "local", nil, fakeSvc{name: "task"}), dir, &syncBuffer{})
	release := make(chan struct{})
	o.close = func(in *instance) { <-release; in.close() }
	scm, done := startHost(o)
	scm.waitState(t, hostRunning, 10*time.Second)
	in := hostClient(t, dir, "local")
	c := dial(t, in)
	ctl := subscribe(t, c, "ctl", nil)
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	waitEnv(t, ctl, 10*time.Second, "received", func(e *Envelope) bool {
		return e.Ev.Kind == "host.stop" && ctlData(e)["phase"] == "received"
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", "http://"+in.addr+"/v1/apply",
		strings.NewReader(`{"services":{"task":{"state":"running"}}}`))
	req.Header.Set("Authorization", "Bearer "+in.token)
	resp, err := loopbackHTTP.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("apply after the stop was received: %d", resp.StatusCode)
	}
	close(release)
	waitExit(t, done, 10*time.Second)
}
