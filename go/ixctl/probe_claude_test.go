package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// The claude-sessions@1 kind reads `claude agents --json` (Claude Code's own list of its live
// sessions) and publishes, per session, its identity, a verified incarnation and its declared
// state. A session is `idle` only when its declared status says so AND its process is the one
// that started the session; anything else (no pid, a pid now held by a younger process, an
// unreadable process, a session gone from the list, a failed read) is never `idle`.

// fakeAgents is a sampler over a scripted list: each read returns the next answer.
type fakeAgents struct {
	answers []string
	errs    []error
	starts  map[int]int64 // pid -> process creation, Unix ms; absent: the process is gone
}

func (f *fakeAgents) sampler() *claudeSessions {
	s := newClaudeSessions("claude")
	s.read = func(context.Context) ([]byte, error) {
		a, err := f.answers[0], f.errs[0]
		f.answers, f.errs = f.answers[1:], f.errs[1:]
		return []byte(a), err
	}
	s.procStart = func(pid int) (int64, error) {
		if ms, ok := f.starts[pid]; ok {
			return ms, nil
		}
		return 0, errors.New("no such process")
	}
	return s
}

func (f *fakeAgents) then(answer string, err error) *fakeAgents {
	f.answers, f.errs = append(f.answers, answer), append(f.errs, err)
	return f
}

func sessionsOf(t *testing.T, rec map[string]any, key string) map[string]map[string]any {
	t.Helper()
	out := map[string]map[string]any{}
	list, _ := rec[key].([]map[string]any)
	for _, s := range list {
		out[s["session_id"].(string)] = s
	}
	return out
}

const (
	idleSession  = `{"name":"a","sessionId":"s-a","pid":100,"kind":"interactive","status":"idle","startedAt":1000000}`
	busySession  = `{"name":"b","sessionId":"s-b","pid":200,"kind":"interactive","status":"busy","startedAt":2000000}`
	bgNoPid      = `{"name":"bg","sessionId":"s-bg","kind":"background","state":"blocked","startedAt":3000000}`
	resumedA     = `{"name":"a","sessionId":"s-a","pid":101,"kind":"interactive","status":"idle","startedAt":5000000}`
	startedAfter = 2500 // a process is created a little before its session says it started
)

func TestDeclaredIdleAndBusyWithAVerifiedIncarnation(t *testing.T) {
	f := (&fakeAgents{starts: map[int]int64{100: 1000000 - startedAfter, 200: 2000000 - startedAfter}}).
		then(`[`+idleSession+`,`+busySession+`]`, nil)
	rec, err := f.sampler().sample()
	if err != nil {
		t.Fatal(err)
	}
	got := sessionsOf(t, rec, "sessions")
	a, b := got["s-a"], got["s-b"]
	if a["availability"] != "idle" || a["identity"] != "verified" || a["incarnation"] != "100:997500" ||
		a["declared"] != "idle" || a["name"] != "a" || a["started_at_ms"] != int64(1000000) {
		t.Fatalf("a: %v", a)
	}
	if b["availability"] != "busy" || b["identity"] != "verified" {
		t.Fatalf("b: %v", b)
	}
	if rec["count"] != 2 || rec["read_at_ms"] == nil || rec["read_ms"] == nil {
		t.Fatalf("record: %v", rec)
	}
}

func TestAPidHeldByAYoungerProcessIsNotTheSession(t *testing.T) {
	// The session's process ended and the system gave its pid to a process created later.
	f := (&fakeAgents{starts: map[int]int64{100: 1000000 + 60000}}).then(`[`+idleSession+`]`, nil)
	rec, _ := f.sampler().sample()
	a := sessionsOf(t, rec, "sessions")["s-a"]
	if a["availability"] != "unknown" || a["identity"] != "unknown" || a["reason"] != "pid-reused" ||
		a["incarnation"] != nil {
		t.Fatalf("a: %v", a)
	}
}

func TestAProcessFarOlderThanItsSessionIsNotVerified(t *testing.T) {
	f := (&fakeAgents{starts: map[int]int64{100: 1000000 - 10*60000}}).then(`[`+idleSession+`]`, nil)
	rec, _ := f.sampler().sample()
	if a := sessionsOf(t, rec, "sessions")["s-a"]; a["availability"] != "unknown" || a["reason"] != "start-mismatch" {
		t.Fatalf("a: %v", a)
	}
}

func TestAGoneProcessOrAMissingPidIsUnknown(t *testing.T) {
	f := (&fakeAgents{starts: map[int]int64{}}).then(`[`+idleSession+`,`+bgNoPid+`]`, nil)
	rec, _ := f.sampler().sample()
	got := sessionsOf(t, rec, "sessions")
	if a := got["s-a"]; a["availability"] != "unknown" || a["reason"] != "process-unreadable" {
		t.Fatalf("a: %v", a)
	}
	if bg := got["s-bg"]; bg["availability"] != "unknown" || bg["reason"] != "no-pid" || bg["declared"] != "blocked" {
		t.Fatalf("bg: %v", bg)
	}
}

func TestASessionGoneFromTheListIsSaidOnceAndNeverIdle(t *testing.T) {
	f := (&fakeAgents{starts: map[int]int64{100: 1000000 - startedAfter, 200: 2000000 - startedAfter}}).
		then(`[`+idleSession+`,`+busySession+`]`, nil).
		then(`[`+busySession+`]`, nil).
		then(`[`+busySession+`]`, nil)
	s := f.sampler()
	s.sample()
	rec, _ := s.sample()
	gone := sessionsOf(t, rec, "gone")["s-a"]
	if gone["availability"] != "unknown" || gone["reason"] != "absent" || gone["last_incarnation"] != "100:997500" {
		t.Fatalf("gone: %v (record %v)", gone, rec)
	}
	if _, listed := sessionsOf(t, rec, "sessions")["s-a"]; listed {
		t.Fatal("a session gone from the list is still listed")
	}
	rec, _ = s.sample()
	if len(sessionsOf(t, rec, "gone")) != 0 {
		t.Fatalf("absence said twice: %v", rec)
	}
}

func TestAResumedSessionIsANewIncarnation(t *testing.T) {
	f := (&fakeAgents{starts: map[int]int64{100: 1000000 - startedAfter, 101: 5000000 - startedAfter}}).
		then(`[`+idleSession+`]`, nil).
		then(`[`+resumedA+`]`, nil)
	s := f.sampler()
	first, _ := s.sample()
	if a := sessionsOf(t, first, "sessions")["s-a"]; a["incarnation_changed"] != nil {
		t.Fatalf("first sight marked as a change: %v", a)
	}
	rec, _ := s.sample()
	a := sessionsOf(t, rec, "sessions")["s-a"]
	if a["incarnation"] != "101:4997500" || a["incarnation_changed"] != true || a["availability"] != "idle" {
		t.Fatalf("a after resume: %v", a)
	}
}

func TestAFailedOrSlowReadPublishesNoAvailability(t *testing.T) {
	f := (&fakeAgents{starts: map[int]int64{100: 1000000 - startedAfter}}).
		then(`[`+idleSession+`]`, nil).
		then(``, errors.New("exit status 1")).
		then(`not json`, nil)
	s := f.sampler()
	s.sample()
	if rec, err := s.sample(); err == nil || rec != nil {
		t.Fatalf("failed read: %v %v", rec, err)
	}
	if rec, err := s.sample(); err == nil || rec != nil {
		t.Fatalf("unreadable answer: %v %v", rec, err)
	}

	slow := newClaudeSessions("claude")
	slow.timeout = 50 * time.Millisecond
	slow.read = func(ctx context.Context) ([]byte, error) { <-ctx.Done(); return nil, ctx.Err() }
	t0 := time.Now()
	if rec, err := slow.sample(); err == nil || rec != nil || !strings.Contains(err.Error(), "deadline") ||
		time.Since(t0) > 2*time.Second {
		t.Fatalf("slow read: %v %v after %s", rec, err, time.Since(t0))
	}
}

func TestTheClaudeSessionsKindNeedsTheCLIAndASlowInterval(t *testing.T) {
	p := probePlane(t)
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("c", "claude-sessions@1", 10000, nil)}); err == nil ||
		!strings.Contains(err.Error(), "--claude-bin") {
		t.Fatalf("without --claude-bin: %v", err)
	}
	p.SetClaudeBin("claude")
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("c", "claude-sessions@1", 1000, nil)}); err == nil ||
		!strings.Contains(err.Error(), "5000") {
		t.Fatalf("every_ms 1000: %v", err)
	}
	mustCreate(t, p, node("c", "claude-sessions@1", 5000, nil))
}

func TestProcStartReadsThisProcessAndRefusesAMissingOne(t *testing.T) {
	ms, err := procStartMs(os.Getpid())
	now := time.Now().UnixMilli()
	if err != nil || ms > now || ms < now-24*3600*1000 {
		t.Fatalf("own start %d (now %d): %v", ms, now, err)
	}
	if _, err := procStartMs(0x7ffffff0); err == nil {
		t.Fatal("a missing process has a start time")
	}
}

// agentsFixture runs the test binary as `claude agents --json` in the given role, and returns
// a function that reads the pid of the descendant it started.
func agentsFixture(t *testing.T, role string) func() int {
	t.Helper()
	pidfile := filepath.Join(t.TempDir(), "descendant.pid")
	t.Setenv("IXCTL_TEST_ROLE", role) // the CLI the sampler starts inherits it
	t.Setenv("IXCTL_TEST_PIDFILE", pidfile)
	pid := func() int {
		b, _ := os.ReadFile(pidfile)
		n, _ := strconv.Atoi(strings.TrimSpace(string(b)))
		return n
	}
	t.Cleanup(func() {
		if n := pid(); n > 0 && pidAlive(n) {
			if p, err := os.FindProcess(n); err == nil {
				p.Kill()
			}
		}
	})
	return pid
}

// sampleWithin runs one sample and fails the test if it has not returned within d.
func sampleWithin(t *testing.T, s *claudeSessions, d time.Duration) (map[string]any, error, time.Duration) {
	t.Helper()
	type result struct {
		rec map[string]any
		err error
	}
	done := make(chan result, 1)
	t0 := time.Now()
	go func() { rec, err := s.sample(); done <- result{rec, err} }()
	select {
	case r := <-done:
		return r.rec, r.err, time.Since(t0)
	case <-time.After(d):
		t.Fatalf("the read had not returned after %s", d)
		return nil, nil, 0
	}
}

// The CLI wrote its list and exited, but a descendant keeps its stdout open. The read ends at
// its bound, refuses an answer it could not see closed, and leaves no process behind.
func TestAReadWhoseDescendantHoldsTheOutputEndsBoundedAndLeavesNoProcess(t *testing.T) {
	descendant := agentsFixture(t, "agents-holder")
	s := newClaudeSessions(testExe(t))
	s.timeout = 10 * time.Second
	rec, err, took := sampleWithin(t, s, 6*time.Second)
	if err == nil || rec != nil {
		t.Fatalf("held output: %v %v", rec, err)
	}
	if n := descendant(); n == 0 || pidAlive(n) {
		t.Fatalf("descendant %d alive after the read (%s): %v", n, took, err)
	}
	t.Logf("held output: %v after %s", err, took)
}

// A CLI that never answers is killed with its whole tree at the read's bound.
func TestASlowReadIsKilledWithItsTreeAtItsBound(t *testing.T) {
	descendant := agentsFixture(t, "agents-hang")
	s := newClaudeSessions(testExe(t))
	s.timeout = 8 * time.Second // past the CLI's start and its descendant's (seconds on Windows)
	rec, err, took := sampleWithin(t, s, 15*time.Second)
	if err == nil || rec != nil || !strings.Contains(err.Error(), "deadline") || took > s.timeout+3*time.Second {
		t.Fatalf("hung read: %v %v after %s", rec, err, took)
	}
	if n := descendant(); n == 0 || pidAlive(n) {
		t.Fatalf("descendant %d alive after the read (%s)", n, took)
	}
	t.Logf("hung read ended after %s", took)
}

// A probe stop during a read cancels it: the stop returns at once and the CLI's tree is gone.
func TestAProbeStopCancelsTheReadInFlight(t *testing.T) {
	descendant := agentsFixture(t, "agents-hang")
	p := probePlane(t)
	p.SetClaudeBin(testExe(t))
	mustCreate(t, p, node("c", "claude-sessions@1", claudeSessionsMinEveryMs, nil))
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: "c"}); err != nil {
		t.Fatal(err)
	}
	for deadline := time.Now().Add(15 * time.Second); descendant() == 0; time.Sleep(20 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the first read never started")
		}
	}
	t0 := time.Now()
	if _, err := probeOp(t, p, Request{Op: "probe.stop", Node: "c"}); err != nil {
		t.Fatal(err)
	}
	if took := time.Since(t0); took > 3*time.Second {
		t.Fatalf("stop took %s with a read in flight", took)
	}
	if n := descendant(); pidAlive(n) {
		t.Fatalf("descendant %d alive after the stop", n)
	}
}

// `null`, or a list naming one session twice, is not a list of sessions: an error record, and
// nothing said absent because of it.
func TestANullOrDuplicatedListIsAnErrorAndFabricatesNoAbsence(t *testing.T) {
	f := (&fakeAgents{starts: map[int]int64{100: 1000000 - startedAfter, 200: 2000000 - startedAfter}}).
		then(`[`+idleSession+`,`+busySession+`]`, nil).
		then(`null`, nil).
		then(`[`+idleSession+`,`+idleSession+`]`, nil).
		then(`[`+idleSession+`,`+busySession+`]`, nil)
	s := f.sampler()
	s.sample()
	if rec, err := s.sample(); err == nil || rec != nil {
		t.Fatalf("null: %v %v", rec, err)
	}
	if rec, err := s.sample(); err == nil || rec != nil || !strings.Contains(err.Error(), "s-a") {
		t.Fatalf("duplicate: %v %v", rec, err)
	}
	rec, err := s.sample()
	if err != nil || len(sessionsOf(t, rec, "gone")) != 0 || len(sessionsOf(t, rec, "sessions")) != 2 {
		t.Fatalf("after the bad reads: %v %v", rec, err)
	}
}

// `ixctl up --claude-bin` is what gives the plane its CLI: the real cmdUp, a claude-sessions@1
// node created and started over HTTP, and the read a consumer does (the SSE stream of
// probe/<node> from since=0, with the bearer token of the plane's state file) sees one sample.
func TestUpWithClaudeBinServesTheDocumentedRead(t *testing.T) {
	t.Setenv("IXCTL_TEST_ROLE", "agents-list") // the CLI the sampler starts inherits it
	dir := t.TempDir()
	specPath := filepath.Join(dir, "spec.json")
	if err := os.WriteFile(specPath, []byte(`{"name":"claude-up","services":[]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	pr, pw := io.Pipe()
	done := make(chan error, 1)
	go func() {
		err := cmdUp([]string{"--state", filepath.Join(dir, "state"), "--claude-bin", testExe(t), specPath}, pw)
		pw.Close()
		done <- err
	}()
	line, err := bufio.NewReader(pr).ReadString('\n')
	if err != nil {
		t.Fatalf("ready line: %v", err)
	}
	go io.Copy(io.Discard, pr)
	var ready readyLine
	if err := json.Unmarshal([]byte(line), &ready); err != nil || ready.Addr == "" || ready.StateFile == "" {
		t.Fatalf("ready line %q: %v", line, err)
	}
	b, err := os.ReadFile(ready.StateFile)
	if err != nil {
		t.Fatal(err)
	}
	var st stateFile
	if err := json.Unmarshal(b, &st); err != nil || st.Token == "" {
		t.Fatalf("state file: %v", err)
	}
	base := "http://" + ready.Addr
	post := func(path, body string) {
		t.Helper()
		req, _ := http.NewRequest("POST", base+path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+st.Token)
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		msg, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode/100 != 2 {
			t.Fatalf("%s refused: %d %s", path, resp.StatusCode, msg)
		}
	}
	stopped := false
	t.Cleanup(func() { // a failed test still stops the plane before its state dir is removed
		if !stopped {
			req, _ := http.NewRequest("POST", base+"/v1/shutdown", strings.NewReader(`{}`))
			req.Header.Set("Authorization", "Bearer "+st.Token)
			if resp, err := http.DefaultClient.Do(req); err == nil {
				resp.Body.Close()
				select {
				case <-done:
				case <-time.After(15 * time.Second):
				}
			}
		}
	})
	shutdown := func() {
		post("/v1/shutdown", `{}`)
		stopped = true
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("cmdUp: %v", err)
			}
		case <-time.After(15 * time.Second):
			t.Fatal("cmdUp had not returned 15 s after /v1/shutdown")
		}
	}
	post("/v1/probe/create", `{"name":"c","kind":"claude-sessions@1","every_ms":5000}`)
	post("/v1/probe/start", `{"node":"c"}`)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "GET", base+"/v1/events?ch=probe/c&since=0", nil)
	req.Header.Set("Authorization", "Bearer "+st.Token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("events: %d", resp.StatusCode)
	}
	var env struct {
		Ch     string `json:"ch"`
		Seq    uint64 `json:"seq"`
		Epoch  string `json:"epoch"`
		Origin string `json:"origin"`
		Ev     struct {
			Kind string         `json:"kind"`
			Data map[string]any `json:"data"`
		} `json:"ev"`
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	found := false
	for !found && sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &env); err != nil {
			t.Fatalf("envelope %q: %v", data, err)
		}
		found = env.Ch == "probe/c" && env.Ev.Kind == "sample"
	}
	resp.Body.Close()
	if !found {
		t.Fatalf("no sample on probe/c: %v", sc.Err())
	}
	if env.Origin != "claude-up" || env.Epoch == "" || env.Seq == 0 {
		t.Fatalf("envelope origin %q epoch %q seq %d", env.Origin, env.Epoch, env.Seq)
	}
	d := env.Ev.Data
	sessions, _ := d["sessions"].([]any)
	if d["kind"] != "claude-sessions@1" || d["count"] != float64(1) || len(sessions) != 1 {
		t.Fatalf("sample data: %v", d)
	}
	if s, _ := sessions[0].(map[string]any); s["session_id"] != "s1" {
		t.Fatalf("session: %v", sessions[0])
	}
	shutdown()
}
