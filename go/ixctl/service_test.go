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

// slowLog takes its time over the line holding slow, as a log on a busy disk would.
type slowLog struct {
	syncBuffer
	slow  string
	delay time.Duration
}

func (l *slowLog) Write(p []byte) (int, error) {
	if strings.Contains(string(p), l.slow) {
		time.Sleep(l.delay)
	}
	return l.syncBuffer.Write(p)
}

// A stop that outlives its bound is reported as not observed: the host returns with its own
// code, while the close is still held, and does not pretend the trees are gone. The bound
// counts from the request: time spent writing the log after it is inside the bound.
func TestAStopPastItsBoundSaysItWasNotObserved(t *testing.T) {
	log := &slowLog{slow: `"phase":"received"`, delay: 400 * time.Millisecond}
	o := hostOptions{spec: fakeSpec(t, "solo", nil), addr: "127.0.0.1:0", stateDir: t.TempDir(), log: log}
	o.stopTimeout = 200 * time.Millisecond
	release, closed := make(chan struct{}), make(chan struct{})
	o.close = func(in *instance) {
		<-release
		in.close()
		close(closed)
	}
	scm, done := startHost(o)
	scm.waitState(t, hostRunning, 10*time.Second)
	t0 := time.Now()
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	code := waitExit(t, done, 5*time.Second)
	d := time.Since(t0)
	close(release)
	if code != exitStopUnobserved {
		t.Fatalf("exit %d, want %d", code, exitStopUnobserved)
	}
	// 400 ms of log, then the bound already past: well under 400 + 200 ms, and under the tick.
	if d > 550*time.Millisecond {
		t.Fatalf("a 200 ms bound, 400 ms of it spent logging, answered after %s", d)
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
// A host told where Delta's journal is lets a delta@1 node read it, as `ixctl up
// --delta-journal` does; without it the kind stays refused.
func TestAHostGivenDeltasJournalAcceptsADeltaNode(t *testing.T) {
	for _, tc := range []struct {
		journal string
		want    int
	}{{fixtureJournal(t), http.StatusOK}, {"", http.StatusBadRequest}} {
		dir := t.TempDir()
		o := hostOpts(fakeSpec(t, "local", nil, fakeSvc{name: "task"}), dir, &syncBuffer{})
		o.deltaJournal = tc.journal
		scm, done := startHost(o)
		scm.waitState(t, hostRunning, 10*time.Second)
		in := hostClient(t, dir, "local")
		code, _, body := httpDo(t, in, "POST", "/v1/probe/create",
			map[string]any{"name": "d", "kind": "delta@1", "every_ms": 10}, nil)
		if code != tc.want {
			t.Errorf("journal %q: create answered %d, want %d: %s", tc.journal, code, tc.want, body)
		}
		scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
		waitExit(t, done, 20*time.Second)
	}
}

// The plan passes the journal to the service and lets its account read that one file.
func TestThePlanGivesTheServiceDeltasJournalToRead(t *testing.T) {
	journal := `C:\Users\stef\delta\delta-events.jsonl`
	p := installPlan(planOptions{service: "ixctl", exe: `C:\Program Files\ixctl\ixctl.exe`,
		dir: `C:\ProgramData\ixctl`, clientUser: `HOST\stef`, name: "solo", deltaJournal: journal})
	for _, want := range []string{
		`--delta-journal "` + journal + `"`,
		`icacls "` + journal + `" /grant "NT SERVICE\ixctl:R"`,
		`icacls "` + journal + `" /remove "NT SERVICE\ixctl"`,
	} {
		if !strings.Contains(p, want) {
			t.Errorf("plan lacks %q:\n%s", want, p)
		}
	}
	if strings.Index(p, `/remove "NT SERVICE\ixctl"`) < strings.Index(p, "# uninstall") {
		t.Errorf("the grant is taken back outside the uninstall section:\n%s", p)
	}
	if bare := installPlan(planOptions{service: "ixctl", exe: `C:\x\ixctl.exe`, dir: `C:\d`,
		clientUser: `HOST\stef`, name: "solo"}); strings.Contains(bare, "delta") {
		t.Errorf("a plan without a journal names one:\n%s", bare)
	}
}

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
	// Each section that chains steps stops at the first failure, a native one included.
	prefs := "$ErrorActionPreference = 'Stop'; $PSNativeCommandUseErrorActionPreference = $true"
	p := installPlan(planOptions{service: "ixctl", exe: `C:\Program Files\ixctl\ixctl.exe`,
		dir: `C:\ProgramData\ixctl`, clientUser: `HOST\stef`, name: "solo"})
	for _, want := range []string{
		`obj= "NT SERVICE\ixctl"`, `start= demand`, `--addr 127.0.0.1:0`,
		`--state "C:\ProgramData\ixctl"`, `icacls "C:\ProgramData\ixctl"`, `"NT SERVICE\ixctl:(OI)(CI)M"`,
		`"HOST\stef:(OI)(CI)RX"`, `sc.exe start ixctl`, `sc.exe query ixctl`, `Stop-Service ixctl`,
		`sc.exe delete ixctl`, `C:\ProgramData\ixctl\solo.json`, `C:\ProgramData\ixctl\solo.service.log`,
	} {
		if !strings.Contains(p, want) {
			t.Errorf("plan lacks %q", want)
		}
	}
	for _, bad := range []string{"password=", "0.0.0.0", "failure", "LocalSystem", "start= auto", "<name>"} {
		if strings.Contains(p, bad) {
			t.Errorf("plan holds %q", bad)
		}
	}
	for _, section := range []string{"# install", "# start", "# status", "# stop", "# uninstall"} {
		if !strings.Contains(p, section) {
			t.Errorf("plan lacks section %q", section)
		}
	}
	// sc.exe start and stop only ask: each is followed by a bounded wait for the state, whose
	// timeout ends the section.
	stop := "Stop-Service ixctl -NoWait\n(Get-Service ixctl).WaitForStatus('Stopped', [TimeSpan]::FromSeconds(60))\n"
	start := "sc.exe start ixctl\n(Get-Service ixctl).WaitForStatus('Running', [TimeSpan]::FromSeconds(30))\n"
	if strings.Contains(p, "sc.exe stop") || strings.Count(p, stop) != 2 || !strings.Contains(p, start) {
		t.Errorf("a start or a stop is not waited for:\n%s", p)
	}
	if !strings.Contains(p, "PowerShell 7.4 or later") {
		t.Errorf("the plan does not name PowerShell 7.4, where native errors can stop a script:\n%s", p)
	}
	for _, section := range []string{"# install", "# start", "# status", "# stop", "# uninstall"} {
		if !strings.HasPrefix(p[strings.Index(p, section):], section+"\n"+prefs+"\n") {
			t.Errorf("section %s does not stop at its first failure:\n%s", section, p)
		}
	}
	un := p[strings.Index(p, "# uninstall"):]
	if strings.Index(un, "WaitForStatus('Stopped'") > strings.Index(un, "sc.exe delete ixctl") {
		t.Errorf("uninstall deletes before it stops:\n%s", p)
	}
	// Status shows where the plane listens, never its bearer token on the operator's screen.
	status := p[strings.Index(p, "# status"):strings.Index(p, "# stop")]
	if !strings.Contains(status, `Get-Content -Raw "C:\ProgramData\ixctl\solo.json" | ConvertFrom-Json | Select-Object name, addr, pid`) ||
		strings.Contains(status, "token") {
		t.Errorf("status prints the state file's token:\n%s", status)
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

// A tree the plane did not see gone (its job still held a process at the bound) makes the
// stop unobserved, even though the close returned.
func TestATreeNotSeenGoneMakesTheStopUnobserved(t *testing.T) {
	old := treeGone
	treeGone = func(*procTree, time.Duration) bool { return false }
	t.Cleanup(func() { treeGone = old })
	dir := t.TempDir()
	log := &syncBuffer{}
	scm, done := startHost(hostOpts(fakeSpec(t, "local", nil, fakeSvc{name: "task"}), dir, log))
	scm.waitState(t, hostRunning, 10*time.Second)
	c := dial(t, hostClient(t, dir, "local"))
	w := subscribe(t, c, "svc/task", nil)
	do(t, c, Request{Op: "apply", Desired: running("task")})
	waitEnv(t, w, 10*time.Second, "a task line", func(*Envelope) bool { return true })
	scm.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	if code := waitExit(t, done, 15*time.Second); code != exitStopUnobserved {
		t.Fatalf("exit %d, want %d\n%s", code, exitStopUnobserved, log.String())
	}
	var e map[string]any
	for _, x := range logEvents(t, log.String()) {
		if x["event"] == "host.stop" && x["phase"] != "requested" && x["phase"] != "received" {
			e = x
		}
	}
	if e == nil || e["phase"] != "unobserved" || e["trees_not_seen_gone"] != float64(1) {
		t.Fatalf("last stop phase %v\n%s", e, log.String())
	}
}

// On the console, the process exits with the host's own code, not a generic failure.
func TestTheConsoleHostExitsWithTheHostCode(t *testing.T) {
	dir := t.TempDir()
	specPath := filepath.Join(dir, "spec.json")
	os.WriteFile(specPath, []byte(`{"name":"solo","services":[]}`), 0o600)
	spec, err := LoadSpec(specPath)
	if err != nil {
		t.Fatal(err)
	}
	first, done := startHost(hostOpts(spec, dir, &syncBuffer{}))
	first.waitState(t, hostRunning, 10*time.Second)
	var stderr bytes.Buffer
	if code := cli([]string{"service", "run", specPath, "--state", dir}, &bytes.Buffer{}, &stderr); code != int(exitAlreadyRunning) {
		t.Fatalf("console exit %d, want %d: %s", code, exitAlreadyRunning, stderr.String())
	}
	first.controls <- hostControl{Cmd: controlStop, Source: "scm"}
	waitExit(t, done, 10*time.Second)
}

// The plan names the spec and the binary by absolute paths: it runs later, from another shell.
func TestThePlanNamesTheSpecByItsAbsolutePath(t *testing.T) {
	dir := t.TempDir()
	abs := filepath.Join(dir, "spec.json")
	os.WriteFile(abs, []byte(`{"name":"solo","services":[]}`), 0o600)
	wd, _ := os.Getwd()
	rel, err := filepath.Rel(wd, abs)
	if err != nil || filepath.IsAbs(rel) {
		t.Skipf("no relative path from %s to %s", wd, abs)
	}
	var out bytes.Buffer
	if err := cmdServicePlan([]string{"--spec", rel, "--client-user", `HOST\stef`}, &out); err != nil {
		t.Fatal(err)
	}
	exe, _ := os.Executable()
	if !strings.Contains(out.String(), `Copy-Item "`+abs+`"`) || !strings.Contains(out.String(), `Copy-Item "`+exe+`"`) {
		t.Fatalf("plan:\n%s", out.String())
	}
}
