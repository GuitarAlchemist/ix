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
	if lines := journalLines(t, j); len(lines) != 1 || lines[0]["kind"] != "boot" {
		t.Fatalf("plan wrote the journal (only the boot line belongs there): %s", j)
	}
}

// journalLines parses a journal file, one JSON object per line.
func journalLines(t *testing.T, b []byte) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("journal line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

// F3.1: generations and seqs restart at 1 at each boot while the journal file
// keeps growing. Each boot now opens with a header line carrying the instance
// epoch, and every ctl envelope after it carries the same epoch, so two boots'
// `generation 1` read apart.
func TestJournalBootHeaderTellsBootsApart(t *testing.T) {
	stateDir := t.TempDir()
	var epochs []string
	for boot := 0; boot < 2; boot++ {
		in := newTestInstance(t, fakeSpec(t, "local", nil, fakeSvc{name: "pod-a", args: []string{"--nodes", "2"}}), stateDir)
		c := dial(t, in)
		w := subscribe(t, c, "ctl", nil)
		var r ApplyReply
		json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("pod-a")}), &r)
		if r.Generation != 1 {
			t.Fatalf("boot %d: apply generation %d, want 1 (generations restart per boot)", boot, r.Generation)
		}
		waitEnv(t, w, 10*time.Second, "converged", func(e *Envelope) bool { return e.Ev.Kind == "converged" })
		epochs = append(epochs, in.plane.hub.Epoch())
		in.close()
	}
	if epochs[0] == epochs[1] {
		t.Fatalf("two boots drew the same epoch %q", epochs[0])
	}

	j, err := os.ReadFile(filepath.Join(stateDir, "local.journal.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	lines := journalLines(t, j)
	boot, applies, ctl := -1, 0, make([]int, 2)
	for i, m := range lines {
		switch {
		case m["kind"] == "boot":
			boot++
			if boot > 1 {
				t.Fatalf("line %d: a third boot line", i)
			}
			if m["schema"] != journalSchema || m["name"] != "local" || m["epoch"] != epochs[boot] || m["t"] == nil {
				t.Fatalf("line %d: boot header %v, want schema %q name local epoch %q", i, m, journalSchema, epochs[boot])
			}
		case boot < 0:
			t.Fatalf("line %d: %v before any boot line", i, m)
		case m["kind"] == "apply":
			applies++
			if m["generation"] != 1.0 {
				t.Fatalf("line %d: apply generation %v, want 1", i, m["generation"])
			}
		case m["ch"] == "ctl":
			ctl[boot]++
			if m["epoch"] != epochs[boot] {
				t.Fatalf("line %d: ctl envelope epoch %v under boot %q", i, m["epoch"], epochs[boot])
			}
		default:
			t.Fatalf("line %d: unknown line %v", i, m)
		}
	}
	if boot != 1 || applies != 2 {
		t.Fatalf("saw %d boot lines and %d apply records, want 2 and 2", boot+1, applies)
	}
	// The epoch check above is vacuous unless each boot's section holds ctl records.
	if ctl[0] == 0 || ctl[1] == 0 {
		t.Fatalf("ctl envelopes per boot %v, want at least one under each header", ctl)
	}
}

func applyConverged(t testing.TB, p *Plane, d *Desired) ApplyReply {
	t.Helper()
	reply, done, err := p.Apply(d)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case result := <-done:
		if !result.Converged {
			t.Fatalf("apply %d did not converge: %+v", reply.Generation, result)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("apply %d did not settle", reply.Generation)
	}
	return reply
}

func assertFreshStoppedBoot(t testing.TB, in *instance, oldEpoch string, names ...string) {
	t.Helper()
	st := in.plane.Status()
	if st.Epoch == oldEpoch {
		t.Fatalf("restart kept epoch %q", st.Epoch)
	}
	if st.Generation != 0 || st.ObservedGeneration != 0 || st.SettledGeneration != 0 ||
		st.Diverged != nil || len(st.Failed) != 0 {
		t.Fatalf("fresh boot generations: %+v", st)
	}
	if len(st.Services) != len(names) {
		t.Fatalf("fresh boot services %+v, want %v", st.Services, names)
	}
	for i, name := range names {
		s := st.Services[i]
		if s.Name != name || s.Desired != desiredStopped || s.State != stateStopped ||
			s.Pid != nil || s.Restarts != 0 || s.StartedAt != nil || s.Generation != 0 ||
			s.Run != nil || s.LastExit != nil {
			t.Fatalf("fresh boot service %q: %+v", name, s)
		}
	}
}

// D14-T1: applied desired state is scoped to one boot, even when successive
// documents each name only one service.
func TestDesiredStateDoesNotSurviveAPlaneRestart(t *testing.T) {
	stateDir := t.TempDir()
	spec := fakeSpec(t, "local", nil,
		fakeSvc{name: "a", args: []string{"--nodes", "1"}},
		fakeSvc{name: "b", args: []string{"--nodes", "1"}})
	first := newTestInstance(t, spec, stateDir)
	oldEpoch := first.plane.Epoch()
	r1 := applyConverged(t, first.plane, running("a"))
	r2 := applyConverged(t, first.plane, running("b"))
	if r1.Generation != 1 || r2.Generation != 2 {
		t.Fatalf("partial apply generations %d, %d, want 1, 2", r1.Generation, r2.Generation)
	}
	first.close()

	second := newTestInstance(t, spec, stateDir)
	assertFreshStoppedBoot(t, second, oldEpoch, "a", "b")
}

// D14-T2: autostart is a new boot's generation 1 and launches a new run.
func TestAutostartAfterRestartBelongsToTheNewBoot(t *testing.T) {
	stateDir := t.TempDir()
	spec := fakeSpec(t, "local", nil, fakeSvc{name: "a", args: []string{"--nodes", "1"}})
	spec.Services[0].Autostart = true
	first := newTestInstance(t, spec, stateDir)
	old := waitStatus(t, first, 10*time.Second, "first boot's autostart", func(st StatusReply) bool {
		s := svcStatus(st, "a")
		return st.SettledGeneration == 1 && s.State == stateRunning && s.Run != nil
	})
	oldRun := *svcStatus(old, "a").Run
	first.close()

	second := newTestInstance(t, spec, stateDir)
	fresh := waitStatus(t, second, 10*time.Second, "second boot's autostart", func(st StatusReply) bool {
		s := svcStatus(st, "a")
		return st.SettledGeneration == 1 && s.State == stateRunning && s.Run != nil
	})
	run := svcStatus(fresh, "a").Run
	if fresh.Epoch == old.Epoch || fresh.Generation != 1 || fresh.ObservedGeneration != 1 ||
		run.Epoch != fresh.Epoch || run.Origin != "local" || *run == oldRun {
		t.Fatalf("old boot %+v run %+v; new boot %+v run %+v", old, oldRun, fresh, *run)
	}
}

// D14-T4: a cut audit record is neither desired state nor a place to append
// the next boot header.
func TestATruncatedJournalTailCannotAffectTheNextBoot(t *testing.T) {
	stateDir := t.TempDir()
	spec := fakeSpec(t, "local", nil,
		fakeSvc{name: "a", args: []string{"--nodes", "1"}},
		fakeSvc{name: "b", args: []string{"--nodes", "1"}})
	first := newTestInstance(t, spec, stateDir)
	oldEpoch := first.plane.Epoch()
	applyConverged(t, first.plane, running("a"))
	applyConverged(t, first.plane, running("b"))
	first.close()

	journalPath := filepath.Join(stateDir, "local.journal.jsonl")
	f, err := os.OpenFile(journalPath, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	const cut = `{"kind":"apply","generation":999,"desired":{"services":{"a":{"state":"running"}}}`
	if _, err := f.WriteString(cut); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	second := newTestInstance(t, spec, stateDir)
	assertFreshStoppedBoot(t, second, oldEpoch, "a", "b")
	b, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	if len(lines) < 2 || lines[len(lines)-2] != cut {
		t.Fatalf("cut tail is not its own physical line: %q", lines)
	}
	var boot map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &boot); err != nil {
		t.Fatalf("new boot header after cut tail: %v; line %q", err, lines[len(lines)-1])
	}
	if boot["kind"] != "boot" || boot["schema"] != journalSchema ||
		boot["name"] != "local" || boot["epoch"] != second.plane.Epoch() {
		t.Fatalf("new boot header after cut tail: %v", boot)
	}
}

// D14-T5: the current spec, not names found in the audit journal, declares
// the services of a new boot.
func TestRestartUsesOnlyTheNewSpecServices(t *testing.T) {
	stateDir := t.TempDir()
	firstSpec := fakeSpec(t, "local", nil,
		fakeSvc{name: "a", args: []string{"--nodes", "1"}},
		fakeSvc{name: "b", args: []string{"--nodes", "1"}})
	first := newTestInstance(t, firstSpec, stateDir)
	oldEpoch := first.plane.Epoch()
	applyConverged(t, first.plane, running("b"))
	first.close()

	secondSpec := fakeSpec(t, "local", nil, fakeSvc{name: "a", args: []string{"--nodes", "1"}})
	second := newTestInstance(t, secondSpec, stateDir)
	assertFreshStoppedBoot(t, second, oldEpoch, "a")
}

// apply bumps the generation, answers at once, and ctl says converged once
// both services are observed started; status and the journal follow.
func TestApplyConverges(t *testing.T) {
	in := twoPods(t)
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	var r ApplyReply
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("pod-a", "pod-b")}), &r)
	if r.Generation != 1 || len(r.Actions) != 2 || r.Epoch != in.plane.Epoch() {
		t.Fatalf("apply %+v (epoch %q)", r, in.plane.Epoch())
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
	if st.Generation != 1 || st.ObservedGeneration != 1 || st.Diverged != nil || st.Epoch != in.plane.Epoch() {
		t.Fatalf("generations %d/%d, diverged %+v, epoch %q", st.Generation, st.ObservedGeneration, st.Diverged, st.Epoch)
	}
	p, _ := in.plane.Plan(running("pod-a", "pod-b"))
	if len(p.Actions) != 0 || len(p.Noop) != 2 {
		t.Fatalf("plan after convergence: %+v", p)
	}

	var sr StopReply
	json.Unmarshal(do(t, c, Request{Op: "stop", Service: "pod-a"}), &sr)
	// The exit names the generation that launched the run it ends, not the stop's.
	if sr.Stopped != "pod-a" || sr.Generation != 2 || sr.Exit == nil || sr.Exit.Generation != 1 {
		t.Fatalf("stop %+v (exit %+v)", sr, sr.Exit)
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

// F2.1 of the one-way-doors doc: generation 1 fails on one service, generation 2 starts
// another and converges. observed_generation alone then reads 1 as done; settled_generation
// and failed say what happened to each generation.
// An apply that finds a service on its way (a start the supervisor has not made yet, a restart
// in its backoff) says noop and launches nothing; the run keeps the generation that launched
// it, and only that generation's service.started settles a start.
func TestAnApplyDuringAStartOrABackoffIsANoop(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "crashy", args: []string{"--crash-after-ms", "300"}, max: 3}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	var r1, r2, r3, r4 ApplyReply
	// Two applies in a row: the second finds the first's start under way.
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("crashy")}), &r1)
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("crashy")}), &r2)
	if len(r1.Actions) != 1 || len(r2.Actions) != 0 {
		t.Fatalf("actions %v then %v", r1.Actions, r2.Actions)
	}
	// The noop apply converges at once, maybe before the first's service.started: take the
	// three events in whatever order they come.
	seen := map[string]bool{}
	for len(seen) < 3 {
		e := waitEnv(t, w, 5*time.Second, "started or converged", func(e *Envelope) bool {
			return e.Ev.Kind == "service.started" || e.Ev.Kind == "converged"
		})
		switch {
		case e.Ev.Kind == "service.started":
			if g := ctlData(e)["generation"]; g != float64(r1.Generation) {
				t.Fatalf("the run is stamped %v, want %d", g, r1.Generation)
			}
			seen["started"] = true
		case genOf(e) == r1.Generation:
			seen["converged 1"] = true
		case genOf(e) == r2.Generation:
			seen["converged 2"] = true
		default:
			t.Fatalf("unexpected %s", e.Raw())
		}
	}
	if s := svcStatus(in.plane.Status(), "crashy"); s.Generation != r1.Generation {
		t.Fatalf("status %+v", s)
	}
	// In the backoff after the crash: an apply is a noop too, and the restart stays the run's.
	waitEnv(t, w, 5*time.Second, "restarting", func(e *Envelope) bool { return e.Ev.Kind == "service.restarting" })
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("crashy")}), &r3)
	if len(r3.Actions) != 0 {
		t.Fatalf("apply in the backoff: %v", r3.Actions)
	}
	waitEnv(t, w, 5*time.Second, "converged", func(e *Envelope) bool { return e.Ev.Kind == "converged" && genOf(e) == r3.Generation })
	restarted := waitEnv(t, w, 5*time.Second, "restarted", func(e *Envelope) bool { return e.Ev.Kind == "service.started" })
	if g := ctlData(restarted)["generation"]; g != float64(r1.Generation) || ctlData(restarted)["restarts"] != 1.0 {
		t.Fatalf("the restart is stamped %v, want %d: %s", g, r1.Generation, restarted.Raw())
	}
	// Stopped, a new apply is a real start, stamped with its own generation.
	do(t, c, Request{Op: "stop", Service: "crashy"})
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("crashy")}), &r4)
	if len(r4.Actions) != 1 {
		t.Fatalf("apply after a stop: %v", r4.Actions)
	}
	fresh := waitEnv(t, w, 5*time.Second, "fresh start", func(e *Envelope) bool { return e.Ev.Kind == "service.started" })
	if g := ctlData(fresh)["generation"]; g != float64(r4.Generation) {
		t.Fatalf("the new run is stamped %v, want %d", g, r4.Generation)
	}
	waitEnv(t, w, 5*time.Second, "converged", func(e *Envelope) bool { return e.Ev.Kind == "converged" && genOf(e) == r4.Generation })
}

// A `running` the plan found running asks nothing of the supervisor: when the job ends before
// the supervisor would have read that want, nothing relaunches it under the noop generation.
func TestANoopApplyLaunchesNothingWhenTheJobEnds(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil,
		fakeSvc{name: "once", args: []string{"--crash-after-ms", "400"}, restart: restartNever}), "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)
	var r1, r2 ApplyReply
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("once")}), &r1)
	waitEnv(t, w, 5*time.Second, "started", func(e *Envelope) bool { return e.Ev.Kind == "service.started" })
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("once")}), &r2)
	if len(r2.Actions) != 0 {
		t.Fatalf("apply on a running job: %+v", r2)
	}
	waitEnv(t, w, 5*time.Second, "converged", func(e *Envelope) bool { return e.Ev.Kind == "converged" && genOf(e) == r2.Generation })
	waitEnv(t, w, 5*time.Second, "exited", func(e *Envelope) bool { return e.Ev.Kind == "service.exited" })
	time.Sleep(400 * time.Millisecond)
	s := svcStatus(in.plane.Status(), "once")
	if s.State != stateStopped || s.Pid != nil || s.Generation != r1.Generation || s.Desired != desiredRunning {
		t.Fatalf("after the job ended: %+v", s)
	}
	j, err := os.ReadFile(filepath.Join(filepath.Dir(in.stateFile), "local.journal.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(string(j), `"kind":"service.started"`); n != 1 {
		t.Fatalf("%d starts in the journal, want 1 (the noop generation relaunched the job)", n)
	}
	// The ended job takes a start again, from an apply that plans one.
	if p, _ := in.plane.Plan(running("once")); len(p.Actions) != 1 {
		t.Fatalf("plan on the ended job: %+v", p)
	}
}

func TestALaterGenerationDoesNotHideAFailedOne(t *testing.T) {
	doc, _ := json.Marshal(map[string]any{"name": "local", "services": []map[string]any{
		{"name": "ghost", "command": "no-such-binary"},
		{"name": "ok", "command": testExe(t), "env": map[string]string{"IXCTL_TEST_ROLE": "fake"}},
	}})
	spec, err := ParseSpec(doc, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	in := newTestInstance(t, spec, "")
	c := dial(t, in)
	w := subscribe(t, c, "ctl", nil)

	var failed, later ApplyReply
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("ghost")}), &failed)
	waitEnv(t, w, 5*time.Second, "diverged", func(e *Envelope) bool { return e.Ev.Kind == "diverged" && genOf(e) == failed.Generation })
	st := in.plane.Status()
	if st.SettledGeneration != failed.Generation || len(st.Failed) != 1 || st.Failed[0].Generation != failed.Generation ||
		st.Failed[0].Service != "ghost" || !strings.HasPrefix(st.Failed[0].Reason, "launch: ") {
		t.Fatalf("after the failed apply: settled %d failed %+v", st.SettledGeneration, st.Failed)
	}

	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("ok")}), &later)
	waitEnv(t, w, 5*time.Second, "converged", func(e *Envelope) bool { return e.Ev.Kind == "converged" && genOf(e) == later.Generation })
	st = in.plane.Status()
	// The old summary reads generation 1 as done...
	if st.ObservedGeneration != later.Generation || st.Diverged == nil || st.Diverged.Generation != failed.Generation {
		t.Fatalf("old summary after the later apply: observed %d diverged %+v", st.ObservedGeneration, st.Diverged)
	}
	// ...the new one still names it failed, and 2 as the only settled, converged one.
	if st.SettledGeneration != later.Generation || len(st.Failed) != 1 || st.Failed[0].Generation != failed.Generation {
		t.Fatalf("new summary after the later apply: settled %d failed %+v", st.SettledGeneration, st.Failed)
	}
	code, _, body := httpDo(t, in, "GET", "/v1/status", nil, nil)
	if code != 200 || !strings.Contains(string(body), `"settled_generation":2`) ||
		!strings.Contains(string(body), `"failed":[{"generation":1,"service":"ghost"`) {
		t.Fatalf("HTTP status %d %s", code, body)
	}
}
