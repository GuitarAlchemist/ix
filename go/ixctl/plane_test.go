package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func twoPods(t *testing.T) *instance {
	return newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "pod-a", args: []string{"--nodes", "3"}},
		fakeSvc{name: "pod-b", args: []string{"--nodes", "2"}}), "")
}

// plan is pure: any number of calls change no generation, no state, no
// event and no journal line.
func TestPlanIsPure(t *testing.T) {
	in := twoPods(t)
	head := in.plane.hub.Head()
	for i := 0; i < 1000; i++ {
		r, err := in.plane.Plan(running("pod-a", "pod-b"))
		if err != nil {
			t.Fatal(err)
		}
		if r.Generation != 0 || len(r.Actions) != 2 || len(r.Noop) != 0 {
			t.Fatalf("plan %+v", r)
		}
	}
	code, _, body := httpDo(t, in, "POST", "/v1/plan", running("pod-a"), nil)
	if code != 200 || !strings.Contains(string(body), `"op":"start"`) {
		t.Fatalf("HTTP plan %d %s", code, body)
	}
	st := in.plane.Status()
	if st.Generation != 0 || in.plane.hub.Head() != head {
		t.Fatalf("plan changed something: generation %d head %d -> %d", st.Generation, head, in.plane.hub.Head())
	}
	for _, s := range st.Services {
		if s.State != stateStopped || s.Desired != desiredStopped {
			t.Fatalf("plan changed %s: %+v", s.Name, s)
		}
	}
	j, _ := os.ReadFile(filepath.Join(filepath.Dir(in.stateFile), "local.journal.jsonl"))
	if len(j) != 0 {
		t.Fatalf("plan wrote the journal: %s", j)
	}
}

// apply bumps the generation, answers at once, and ctl says converged once
// both services are observed started; status and the journal follow.
func TestApplyConverges(t *testing.T) {
	in := twoPods(t)
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	var r ApplyReply
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("pod-a", "pod-b")}), &r)
	if r.Generation != 1 || len(r.Actions) != 2 {
		t.Fatalf("apply %+v", r)
	}
	started := map[string]bool{}
	for len(started) < 2 {
		e := waitEnv(t, w, 10*time.Second, "service.started", func(e *Envelope) bool { return e.Ev.Kind == "service.started" })
		started[ctlData(e)["service"].(string)] = true
	}
	e := waitEnv(t, w, 10*time.Second, "converged", func(e *Envelope) bool { return e.Ev.Kind == "converged" })
	if ctlData(e)["generation"] != 1.0 {
		t.Fatalf("converged %s", e.Raw())
	}
	st := in.plane.Status()
	if st.Generation != 1 || st.ObservedGeneration != 1 || st.Diverged != nil {
		t.Fatalf("generations %d/%d, diverged %+v", st.Generation, st.ObservedGeneration, st.Diverged)
	}
	p, _ := in.plane.Plan(running("pod-a", "pod-b"))
	if len(p.Actions) != 0 || len(p.Noop) != 2 {
		t.Fatalf("plan after convergence: %+v", p)
	}

	var sr StopReply
	json.Unmarshal(do(t, c, Request{Op: "stop", Service: "pod-a"}), &sr)
	if sr.Stopped != "pod-a" || sr.Generation != 2 || sr.Exit == nil {
		t.Fatalf("stop %+v", sr)
	}
	if s := svcStatus(in.plane.Status(), "pod-a"); s.State != stateStopped || s.Pid != nil {
		t.Fatalf("after stop: %+v", s)
	}

	j, err := os.ReadFile(filepath.Join(filepath.Dir(in.stateFile), "local.journal.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(j)
	for _, want := range []string{`"kind":"apply"`, `"generation":2`, `"kind":"service.started"`, `"kind":"converged"`, `"kind":"service.stopped"`} {
		if !strings.Contains(text, want) {
			t.Errorf("journal lacks %s", want)
		}
	}
}

// Two declared services, each one process running several nodes: status
// shows each one's nodes from its last stats event.
func TestStatusShowsNodes(t *testing.T) {
	in := twoPods(t)
	if _, _, err := in.plane.Apply(running("pod-a", "pod-b")); err != nil {
		t.Fatal(err)
	}
	st := waitStatus(t, in, 10*time.Second, "nodes", func(st StatusReply) bool {
		var a, b []map[string]any
		json.Unmarshal(svcStatus(st, "pod-a").Nodes, &a)
		json.Unmarshal(svcStatus(st, "pod-b").Nodes, &b)
		return len(a) == 3 && len(b) == 2
	})
	code, _, body := httpDo(t, in, "GET", "/v1/status", nil, nil)
	if code != 200 || !strings.Contains(string(body), `"name":"n2"`) {
		t.Fatalf("HTTP status %d %s", code, body)
	}
	for _, s := range st.Services {
		if s.State != stateRunning || s.Pid == nil || s.StartedAt == nil {
			t.Fatalf("%+v", s)
		}
	}
}

// Unknown names are refused and nothing is planned or applied.
func TestUnknownNamesRefused(t *testing.T) {
	in := twoPods(t)
	bad := &Desired{Services: map[string]DesiredService{"pod-a": {State: "running"}, "nope": {State: "running"}}}
	for _, path := range []string{"/v1/plan", "/v1/apply"} {
		code, _, body := httpDo(t, in, "POST", path, bad, nil)
		if code != 400 || !strings.Contains(string(body), "nope") {
			t.Errorf("%s: %d %s", path, code, body)
		}
	}
	if code, _, _ := httpDo(t, in, "POST", "/v1/stop", map[string]string{"service": "nope"}, nil); code != 400 {
		t.Errorf("stop unknown: %d", code)
	}
	if code, _, _ := httpDo(t, in, "POST", "/v1/apply", desiredOf("paused", "pod-a"), nil); code != 400 {
		t.Errorf("bad state: %d", code)
	}
	if code, _, _ := httpDo(t, in, "POST", "/v1/apply", map[string]any{"services": map[string]any{}, "command": "x"}, nil); code != 400 {
		t.Errorf("unknown field: %d", code)
	}
	c := dial(t, in)
	if _, err := c.Do(t.Context(), Request{Op: "apply", Desired: bad}); err == nil || !strings.Contains(err.Error(), "nope") {
		t.Errorf("WS apply unknown: %v", err)
	}
	if _, err := c.Do(t.Context(), Request{Op: "frobnicate"}); err == nil {
		t.Errorf("unknown op accepted")
	}
	st := in.plane.Status()
	if st.Generation != 0 || svcStatus(st, "pod-a").State != stateStopped {
		t.Fatalf("a refused document changed state: %+v", st)
	}
}

// A declared service whose command does not exist: the launch publishes
// service.failed with a reason, the apply ends diverged, and status names
// that generation while observed_generation stays behind it.
func TestLaunchFailureDiverges(t *testing.T) {
	spec, err := ParseSpec([]byte(`{"name":"local","services":[{"name":"ghost","command":"no-such-binary"}]}`), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	in := newTestInstance(t, spec, "")
	if st := in.plane.Status(); st.Diverged != nil {
		t.Fatalf("diverged before any apply: %+v", st.Diverged)
	}
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	var r ApplyReply
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("ghost")}), &r)

	e := waitEnv(t, w, 5*time.Second, "service.failed", func(e *Envelope) bool { return e.Ev.Kind == "service.failed" })
	if d := ctlData(e); d["service"] != "ghost" || !strings.HasPrefix(fmt.Sprint(d["reason"]), "launch: ") {
		t.Fatalf("service.failed %s", e.Raw())
	}
	e = waitEnv(t, w, 5*time.Second, "diverged", func(e *Envelope) bool { return e.Ev.Kind == "diverged" })
	if d := ctlData(e); d["generation"] != float64(r.Generation) || d["service"] != "ghost" || d["reason"] == "" {
		t.Fatalf("diverged %s", e.Raw())
	}
	st := in.plane.Status()
	if st.Diverged == nil || st.Diverged.Generation != r.Generation || st.Diverged.Service != "ghost" ||
		st.Diverged.Reason == "" || st.ObservedGeneration >= r.Generation {
		t.Fatalf("status after a failed launch: observed %d diverged %+v", st.ObservedGeneration, st.Diverged)
	}
	if s := svcStatus(st, "ghost"); s.State != stateFailed {
		t.Fatalf("ghost %+v", s)
	}
	code, _, body := httpDo(t, in, "GET", "/v1/status", nil, nil)
	if code != 200 || !strings.Contains(string(body), `"diverged":{"generation":1,"service":"ghost"`) {
		t.Fatalf("HTTP status %d %s", code, body)
	}

	// A later generation that converges supersedes the failed one.
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: desiredOf(desiredStopped, "ghost")}), &r)
	waitEnv(t, w, 5*time.Second, "converged", func(e *Envelope) bool { return e.Ev.Kind == "converged" && genOf(e) == r.Generation })
	if st := in.plane.Status(); st.ObservedGeneration != r.Generation || st.Diverged == nil || st.Diverged.Generation != 1 {
		t.Fatalf("after a later converged apply: observed %d diverged %+v", st.ObservedGeneration, st.Diverged)
	}
}
