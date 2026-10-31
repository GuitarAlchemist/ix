package main

import (
	"context"
	"errors"
	"os"
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
