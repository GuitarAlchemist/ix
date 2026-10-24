package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	stateStopped    = "stopped"
	stateStarting   = "starting"
	stateRunning    = "running"
	stateRestarting = "restarting"
	stateFailed     = "failed"

	backoffMin    = 200 * time.Millisecond
	backoffMax    = 10 * time.Second
	restartWindow = 60 * time.Second
	stopGrace     = 2 * time.Second
	treeGoneLimit = 5 * time.Second
	drainLimit    = 200 * time.Millisecond
	maxLine       = 1 << 20
)

const recentExitsCapacity = 16

type exitInfo struct {
	Code      int    `json:"code"`
	At        int64  `json:"at"`
	Run       RunID  `json:"run"`
	LateLines uint64 `json:"late_lines,omitempty"`
	// Generation is the apply that launched the run this exit ends (a restart keeps its
	// run's generation), so a waiter can tell this run's exit from an earlier run's.
	Generation uint64 `json:"generation,omitempty"`
	// Reason names the limit the plane killed the run for (reasonWall, reasonMemory,
	// reasonCPU, reasonProcesses); absent when the process ended on its own or on a
	// requested stop.
	Reason string `json:"reason,omitempty"`
	// PeakMemoryBytes is the most memory the run held at once, as the OS accounted it: the
	// job's peak commit charge on Windows; on Linux the leader's peak resident set (rusage),
	// or the group's largest polled resident set when a memory limit was watched. Absent
	// when unknown.
	PeakMemoryBytes int64 `json:"peak_memory_bytes,omitempty"`
	// CpuMs is the processor time the run used, user and kernel, in milliseconds, as the OS
	// accounted it: the job's total on Windows (every process of the tree); on Linux the
	// leader's rusage with the children it waited for. Absent when unknown or under a
	// millisecond.
	CpuMs int64 `json:"cpu_ms,omitempty"`
	// PeakProcesses is the most processes the tree held at once, as the watch counted them
	// every 100 ms (only under a processor-time or process limit). Absent when not watched.
	PeakProcesses int64 `json:"peak_processes,omitempty"`
}

// runEnd is both the data of ixctl.run.end and, with Run set, one terminal record in status.
// Pointer terminal fields preserve false and zero for a started run while leaving them absent
// for an attempt that never launched.
type runEnd struct {
	Run       *RunID `json:"run,omitempty"`
	Outcome   string `json:"outcome"`
	Code      *int   `json:"code,omitempty"`
	Requested *bool  `json:"requested,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Error     string `json:"error,omitempty"`
	Output    string `json:"output"`
	DrainMS   int64  `json:"drain_ms"`
}

// Exit reasons: the limit a run was killed for.
const (
	reasonWall      = "wall_limit"    // lasted past limits.wall_ms
	reasonMemory    = "memory_limit"  // held more than limits.memory_mb
	reasonCPU       = "cpu_limit"     // used more processor time than limits.cpu_ms
	reasonProcesses = "process_limit" // held more processes at once than limits.processes
)

// command is one desired state handed to the supervisor, with the generation that asked.
type command struct {
	state      string
	generation uint64
}

// service supervises one declared service: one process tree at a time.
// Everything that changes the tree happens on its run goroutine.
type service struct {
	spec ServiceSpec
	hub  *Hub
	ch   string

	mu        sync.Mutex
	desired   string
	state     string
	pid       int
	restarts  int
	startedAt int64
	startGen  uint64 // the generation that launched the current or last run; 0 before any
	curRun    RunID  // the current or last run, valid once hasRun
	hasRun    bool
	// `running` wants queued and not yet read, and the generation of the last of them: while
	// one is queued the service shows `starting` whenever its process is gone, so a plan sees
	// the start on its way and does not announce another.
	queued  int
	tailGen uint64
	// Failure reasons by the generation of the run that failed, kept until that generation's
	// apply has read them (`forget`): a start whose service.failed was lost from the ring still
	// settles on its own failure, even after a queued start has moved the service on to
	// `starting` again. Bounded by keptFailures: a failure after its apply settled (a run out
	// of restarts) is never read.
	failures    map[uint64]string
	lastExit    *exitInfo
	recentExits []runEnd
	nodes       json.RawMessage

	cmds chan command // desired states, in apply order

	// owned by run
	proc   *child
	recent []time.Time // restarts inside the window
	timer  *time.Timer
	wall   *time.Timer // the current run's wall limit, when the spec sets one
}

// child is one launched process tree.
type child struct {
	tree        *procTree
	run         RunID
	stdin       *os.File
	pid         int
	exited      chan exitStat // how the main process ended, once
	io          sync.WaitGroup
	outputMu    sync.Mutex
	outputEnded bool
	lateLines   uint64
}

// exitStat is how a process ended, with what the OS accounted for it at that moment
// (leaderUsage); a tree that had to be killed without its exit being read reports code -1
// and nothing else.
type exitStat struct {
	code     int
	rss, cpu int64
}

// monitorLimit bounds the wait for a tree's limit monitor to finish after its processes are
// gone, before the exit is classified.
const monitorLimit = time.Second

func newService(spec ServiceSpec, hub *Hub) *service {
	return &service{
		spec: spec, hub: hub, ch: "svc/" + spec.Name,
		desired: desiredStopped, state: stateStopped,
		cmds: make(chan command, 1024),
	}
}

// want records a desired state and hands it to the supervisor, with the generation of
// the apply that asked for it. A start the supervisor has not made yet shows at once: the
// service is `starting`, stamped with this generation, so a plan that follows sees it on its
// way and says noop instead of claiming a start this generation would not make.
func (s *service) want(state string, generation uint64) {
	s.mu.Lock()
	s.desired = state
	if state == desiredRunning {
		s.queued++
		s.tailGen = generation
		if s.state == stateStopped || s.state == stateFailed {
			s.state = stateStarting
			s.startGen = generation
		}
	}
	s.mu.Unlock()
	s.cmds <- command{state: state, generation: generation}
}

// setDesired records the document's state for a service the plan found already there.
func (s *service) setDesired(state string) {
	s.mu.Lock()
	s.desired = state
	s.mu.Unlock()
}

// settle moves a service whose process is gone to state, unless a `running` want is still
// queued behind this command: then it shows `starting`, stamped with that want's generation,
// so no plan in between announces a start the queued one will make.
func (s *service) settle(state string) {
	s.mu.Lock()
	s.settleLocked(state)
	s.mu.Unlock()
}

func (s *service) settleLocked(state string) {
	if s.queued > 0 {
		s.state, s.startGen = stateStarting, s.tailGen
	} else {
		s.state = state
	}
}

// keptFailures bounds the failures a service remembers by generation.
const keptFailures = 16

// failure is the reason the run generation g asked for failed, if it did and no apply has
// forgotten it yet; whatever state the service shows now.
func (s *service) failure(g uint64) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.failures[g]
	return r, ok
}

// forget drops generation g's failure once its apply has settled.
func (s *service) forget(g uint64) {
	s.mu.Lock()
	delete(s.failures, g)
	s.mu.Unlock()
}

func (s *service) observed() (desired, state string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.desired, s.state
}

func (s *service) status() ServiceStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := ServiceStatus{Name: s.spec.Name, Desired: s.desired, State: s.state,
		Restarts: s.restarts, Generation: s.startGen, RecentExitsCapacity: recentExitsCapacity,
		RecentExits: append([]runEnd{}, s.recentExits...), Nodes: s.nodes}
	if s.hasRun {
		run := s.curRun
		st.Run = &run
	}
	if s.pid != 0 {
		pid := s.pid
		st.Pid = &pid
	}
	if s.startedAt != 0 {
		at := s.startedAt
		st.StartedAt = &at
	}
	if s.lastExit != nil {
		e := *s.lastExit
		st.LastExit = &e
	}
	if st.Nodes == nil {
		st.Nodes = json.RawMessage("[]")
	}
	return st
}

func (s *service) appendRecentExitLocked(e runEnd) {
	if len(s.recentExits) == recentExitsCapacity {
		copy(s.recentExits, s.recentExits[1:])
		s.recentExits[len(s.recentExits)-1] = e
		return
	}
	s.recentExits = append(s.recentExits, e)
}

func (s *service) setState(state string) {
	s.mu.Lock()
	s.state = state
	s.mu.Unlock()
}

// cur is the state under mu: want writes it from the apply goroutine, so the supervisor
// goroutine reads it here rather than bare.
func (s *service) cur() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

func (s *service) ctl(kind string, data map[string]any) {
	data["service"] = s.spec.Name
	s.hub.Publish("ctl", event(kind, data))
}

func (s *service) ctlRun(run RunID, kind string, data map[string]any) {
	data["service"] = s.spec.Name
	s.hub.PublishRun("ctl", event(kind, data), run)
}

func (s *service) run(ctx context.Context) {
	for {
		var exited <-chan exitStat
		if s.proc != nil {
			exited = s.proc.exited
		}
		var backoff <-chan time.Time
		if s.timer != nil {
			backoff = s.timer.C
		}
		var wall <-chan time.Time
		if s.wall != nil {
			wall = s.wall.C
		}
		var over <-chan string
		if s.proc != nil {
			over = s.proc.tree.exceeded()
		}
		select {
		case <-ctx.Done():
			s.halt()
			return
		case want := <-s.cmds:
			s.command(want)
		case st := <-exited:
			s.ended(st, "")
		case <-wall:
			s.wall = nil
			s.onLimit(reasonWall)
		case reason := <-over:
			s.onLimit(reason)
		case <-backoff:
			s.timer = nil
			if s.cur() == stateRestarting {
				s.launch(true)
			}
		}
	}
}

func (s *service) command(want command) {
	switch want.state {
	case desiredRunning:
		s.mu.Lock()
		s.queued--
		s.mu.Unlock()
		if s.proc != nil || s.cur() == stateRestarting {
			// Already on its way: the run keeps the generation that launched it.
			return
		}
		s.recent = nil
		// The run about to start, and every restart of it, belongs to this generation (want
		// stamped it already; a stop queued between the two could not have changed it).
		s.mu.Lock()
		s.startGen = want.generation
		s.mu.Unlock()
		s.launch(false)
	case desiredStopped:
		if s.proc != nil {
			s.stopTree(want.generation)
			return
		}
		// No process: restarting, failed, a start a later want pre-marked (see want), or
		// stopped already (the run ended between the plan and this command). This stop ends
		// as stopped all the same and says so, stamped, so its apply settles on its own
		// event and on no other stop's.
		s.cancelTimer()
		s.settle(stateStopped)
		s.ctl("service.stopped", map[string]any{"generation": want.generation})
	}
}

// fail moves the service to `failed` and publishes service.failed with data, stamped with the
// generation of the run that failed. A launch failure also publishes and records its run end
// under the same lock as the transition, so a status after the marker cannot miss the record.
// A nil end means the process exit was already recorded by endOutput.
func (s *service) fail(run RunID, data map[string]any, end *runEnd) {
	s.mu.Lock()
	if s.failures == nil {
		s.failures = map[uint64]string{}
	}
	s.failures[s.startGen], _ = data["reason"].(string)
	for len(s.failures) > keptFailures {
		oldest := s.startGen
		for g := range s.failures {
			if g < oldest {
				oldest = g
			}
		}
		delete(s.failures, oldest)
	}
	data["generation"] = s.startGen
	s.settleLocked(stateFailed)
	if end != nil {
		s.hub.PublishRun(s.ch, event(runEndKind, *end), run)
		r := run
		end.Run = &r
		s.appendRecentExitLocked(*end)
	}
	s.mu.Unlock()
	s.ctlRun(run, "service.failed", data)
}

func (s *service) cancelTimer() {
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
}

func (s *service) launch(restart bool) {
	s.mu.Lock()
	s.state = stateStarting
	gen := s.startGen
	env := s.hub.PublishNewRun("ctl", event("service.starting",
		map[string]any{"service": s.spec.Name, "restart": restart, "generation": gen}))
	run := *env.Run
	s.curRun, s.hasRun = run, true
	s.mu.Unlock()
	s.hub.PublishRun(s.ch, event(runBeginKind, map[string]any{"generation": gen}), run)
	c, err := s.start(run)
	if err != nil {
		reason := "launch: " + err.Error()
		end := &runEnd{Outcome: "launch-failed", Error: reason, Output: "complete"}
		s.fail(run, map[string]any{"reason": reason}, end)
		return
	}
	s.proc = c
	if l := s.spec.Limits; l != nil && l.WallMs > 0 {
		s.wall = time.NewTimer(time.Duration(l.WallMs) * time.Millisecond)
	}
	now := time.Now().UnixMilli()
	s.mu.Lock()
	if restart {
		s.restarts++
	}
	s.state, s.pid, s.startedAt = stateRunning, c.pid, now
	restarts, gen := s.restarts, s.startGen
	s.mu.Unlock()
	s.ctlRun(run, "service.started", map[string]any{"pid": c.pid, "restarts": restarts, "generation": gen})
}

func (s *service) start(run RunID) (*child, error) {
	cmd := exec.Command(resolveCommand(s.spec.Command), s.spec.Args...)
	cmd.Dir = s.spec.Dir
	cmd.Env = os.Environ()
	keys := make([]string, 0, len(s.spec.Env))
	for k := range s.spec.Env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		cmd.Env = append(cmd.Env, k+"="+s.spec.Env[k])
	}
	inR, inW, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	outR, outW, err := os.Pipe()
	if err != nil {
		inR.Close()
		inW.Close()
		return nil, err
	}
	errR, errW, err := os.Pipe()
	if err != nil {
		inR.Close()
		inW.Close()
		outR.Close()
		outW.Close()
		return nil, err
	}
	cmd.Stdin, cmd.Stdout, cmd.Stderr = inR, outW, errW
	tree, err := startTree(cmd, s.spec.Limits)
	inR.Close()
	outW.Close()
	errW.Close()
	if err != nil {
		inW.Close()
		outR.Close()
		errR.Close()
		return nil, err
	}
	c := &child{tree: tree, run: run, stdin: inW, pid: cmd.Process.Pid, exited: make(chan exitStat, 1)}
	c.io.Add(2)
	go func() { defer c.io.Done(); s.readLines(outR, false, c) }()
	go func() { defer c.io.Done(); s.readLines(errR, true, c) }()
	go func() {
		code := -1
		if err := cmd.Wait(); err == nil {
			code = 0
		} else if ee := (*exec.ExitError)(nil); errors.As(err, &ee) {
			code = exitCode(ee)
		}
		st := exitStat{code: code}
		st.rss, st.cpu = leaderUsage(cmd.ProcessState)
		c.exited <- st
	}()
	return c, nil
}

// reap finishes a tree whose main process has exited: whatever it left
// behind is killed, and its output is drained. reason names the limit the plane killed the
// run for; empty, the run ended on its own, unless the tree's monitor, read once every
// process is gone and its messages are in, says it went past a limit (a commit refused
// past the memory limit, which the process died of): then that is the reason. A requested
// stop (requested) keeps its reason empty whatever the monitor holds: the plane ended the run,
// not a limit.
func (s *service) reap(st exitStat, reason string, requested bool) *exitInfo {
	if s.wall != nil {
		s.wall.Stop()
		s.wall = nil
	}
	c := s.proc
	c.stdin.Close()
	c.tree.kill()
	c.tree.waitGone(treeGoneLimit)
	c.tree.drain(monitorLimit)
	if reason == "" && !requested {
		select {
		case reason = <-c.tree.exceeded():
		default:
		}
	}
	peak, cpu, procs := c.tree.usage() // before close: the job is gone after
	if st.rss > peak {
		peak = st.rss
	}
	if st.cpu > cpu {
		cpu = st.cpu
	}
	// The processor-time and process limits are read from these very figures: a run that
	// went past one and ended before a poll could kill it (or between two) is past it all
	// the same.
	if l := s.spec.Limits; reason == "" && !requested && l != nil {
		switch {
		case l.CpuMs > 0 && cpu > l.CpuMs:
			reason = reasonCPU
		case l.Processes > 0 && procs > l.Processes:
			reason = reasonProcesses
		}
	}
	c.tree.close()
	s.proc = nil
	e := &exitInfo{Code: st.code, At: time.Now().UnixMilli(), Run: c.run, Generation: s.startGen,
		Reason: reason, PeakMemoryBytes: peak, CpuMs: cpu, PeakProcesses: procs}
	s.endOutput(c, e, requested)
	return e
}

// endOutput closes the run's source stream after its readers drain, or at the drain bound.
// A reader and this marker hold outputMu while publishing, so no forwarded line can cross it.
func (s *service) endOutput(c *child, e *exitInfo, requested bool) {
	t0 := time.Now()
	drained := make(chan struct{})
	go func() { c.io.Wait(); close(drained) }()
	output := "complete"
	select {
	case <-drained:
	case <-time.After(drainLimit):
		output = "truncated"
	}
	drainMS := time.Since(t0).Milliseconds()

	c.outputMu.Lock()
	c.outputEnded = true
	code, wasRequested := e.Code, requested
	end := runEnd{Outcome: "started", Code: &code, Requested: &wasRequested,
		Output: output, DrainMS: drainMS}
	if e.Reason != "" {
		end.Reason = e.Reason
	}
	s.mu.Lock()
	// Keep the marker and its status fallback indivisible to a status reader. Lock order is
	// outputMu -> s.mu -> hub; no path takes the hub lock and then s.mu.
	s.hub.PublishRun(s.ch, event(runEndKind, end), c.run)
	run := c.run
	end.Run = &run
	s.appendRecentExitLocked(end)
	e.LateLines = c.lateLines
	s.pid, s.lastExit = 0, e
	s.mu.Unlock()
	c.outputMu.Unlock()
}

// exitData is the service.exited event for e: how the run ended, the limit it was killed for
// if any (with the limit's value), and what it cost.
func (s *service) exitData(pid int, requested bool, e *exitInfo) map[string]any {
	d := map[string]any{"pid": pid, "code": e.Code, "requested": requested, "generation": e.Generation}
	switch e.Reason {
	case reasonWall:
		d["reason"], d["wall_ms"] = e.Reason, s.spec.Limits.WallMs
	case reasonMemory:
		d["reason"], d["memory_mb"] = e.Reason, s.spec.Limits.MemoryMB
	case reasonCPU:
		// `cpu_ms` is what the run used, so the limit travels as `cpu_limit_ms`.
		d["reason"], d["cpu_limit_ms"] = e.Reason, s.spec.Limits.CpuMs
	case reasonProcesses:
		d["reason"], d["processes"] = e.Reason, s.spec.Limits.Processes
	}
	if e.PeakMemoryBytes > 0 {
		d["peak_memory_bytes"] = e.PeakMemoryBytes
	}
	if e.CpuMs > 0 {
		d["cpu_ms"] = e.CpuMs
	}
	if e.PeakProcesses > 0 {
		d["peak_processes"] = e.PeakProcesses
	}
	return d
}

// stopTree is a requested stop: close stdin, give the tree stopGrace to
// leave, then kill all of it. generation is the stop's apply's (0 for a shutdown).
func (s *service) stopTree(generation uint64) {
	t0 := time.Now()
	pid := s.proc.pid
	s.proc.stdin.Close()
	var st exitStat
	select {
	case st = <-s.proc.exited:
	case <-time.After(stopGrace):
		s.proc.tree.kill()
		select {
		case st = <-s.proc.exited:
		case <-time.After(treeGoneLimit):
			st = exitStat{code: -1}
		}
	}
	e := s.reap(st, "", true)
	s.settle(stateStopped)
	ms := time.Since(t0).Milliseconds()
	s.ctlRun(e.Run, "service.exited", s.exitData(pid, true, e))
	stopped := map[string]any{"pid": pid, "code": e.Code, "ms": ms}
	if generation != 0 {
		stopped["generation"] = generation
	}
	s.ctlRun(e.Run, "service.stopped", stopped)
}

// halt stops the service for good: the plane is shutting down.
func (s *service) halt() {
	s.cancelTimer()
	if s.proc != nil {
		s.stopTree(0)
		return
	}
	if s.cur() != stateStopped {
		s.setState(stateStopped)
		s.ctl("service.stopped", map[string]any{})
	}
}

// onLimit ends a run that went past a limit of its spec (reason): the tree is killed, and
// the exit, marked with the reason, goes through the restart policy as a failure does. An
// exit already waiting when the limit fired (the select picks at random between the two) is
// the run's own end: in time for the wall limit, so not marked with it; past the memory or
// processor-time limit all the same, since the accounting said so.
func (s *service) onLimit(reason string) {
	select {
	case st := <-s.proc.exited:
		if reason == reasonWall {
			reason = ""
		}
		s.ended(st, reason)
		return
	default:
	}
	s.proc.tree.kill()
	var st exitStat
	select {
	case st = <-s.proc.exited:
	case <-time.After(treeGoneLimit):
		st = exitStat{code: -1}
	}
	s.ended(st, reason)
}

// ended handles an exit nobody asked for, by the restart policy. reason names the limit the
// plane killed the run for, or is empty when the run ended on its own (reap then asks the
// tree's monitor whether a limit was gone past all the same).
func (s *service) ended(st exitStat, reason string) {
	pid := s.proc.pid
	e := s.reap(st, reason, false)
	s.ctlRun(e.Run, "service.exited", s.exitData(pid, false, e))
	if s.spec.Restart == restartNever || (s.spec.Restart == restartOnFailure && e.Code == 0 && e.Reason == "") {
		// The run's own end: stamped with the run's generation, which no stop's apply
		// carries, so a stop queued meanwhile settles on its own event only.
		s.settle(stateStopped)
		stopped := map[string]any{"pid": pid, "code": e.Code, "generation": e.Generation}
		if e.Reason != "" {
			stopped["reason"] = e.Reason
		}
		s.ctlRun(e.Run, "service.stopped", stopped)
		return
	}
	now := time.Now()
	kept := s.recent[:0]
	for _, t := range s.recent {
		if now.Sub(t) < restartWindow {
			kept = append(kept, t)
		}
	}
	s.recent = kept
	if len(s.recent) >= *s.spec.MaxRestarts {
		s.fail(e.Run, map[string]any{"reason": "too many restarts", "restarts_in_window": len(s.recent), "code": e.Code}, nil)
		return
	}
	delay := backoffMin << len(s.recent)
	if delay > backoffMax || delay <= 0 {
		delay = backoffMax
	}
	s.recent = append(s.recent, now)
	s.setState(stateRestarting)
	s.timer = time.NewTimer(delay)
	s.ctlRun(e.Run, "service.restarting", map[string]any{"attempt": len(s.recent), "in_ms": delay.Milliseconds()})
}

// readLines turns a service's output into events on svc/<name>.
func (s *service) readLines(r *os.File, stderr bool, c *child) {
	defer r.Close()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), maxLine)
	for sc.Scan() {
		s.onLine(sc.Bytes(), stderr, c)
	}
	if sc.Err() != nil {
		c.outputMu.Lock()
		if !c.outputEnded {
			s.hub.PublishRun(s.ch, event("log", map[string]any{"error": sc.Err().Error()}), c.run)
		}
		c.outputMu.Unlock()
		io.Copy(io.Discard, r)
	}
}

func (s *service) onLine(line []byte, stderr bool, c *child) {
	c.outputMu.Lock()
	defer c.outputMu.Unlock()
	if c.outputEnded {
		if c.lateLines != ^uint64(0) {
			c.lateLines++
		}
		s.mu.Lock()
		if s.lastExit != nil && s.lastExit.Run == c.run {
			s.lastExit.LateLines = c.lateLines
		}
		s.mu.Unlock()
		return
	}
	if stderr {
		s.hub.PublishRun(s.ch, event("stderr", map[string]any{"line": string(line)}), c.run)
		return
	}
	ev, ok := parseEventLine(line)
	if !ok {
		s.hub.PublishRun(s.ch, event("log", map[string]any{"line": string(line)}), c.run)
		return
	}
	if ev.Kind == "stats" {
		var st struct {
			Nodes json.RawMessage `json:"nodes"`
		}
		if json.Unmarshal(ev.Data, &st) == nil && len(st.Nodes) > 0 && st.Nodes[0] == '[' {
			s.mu.Lock()
			s.nodes = st.Nodes
			s.mu.Unlock()
		}
	}
	s.hub.PublishRun(s.ch, ev, c.run)
}

// parseEventLine accepts a line that is a JSON object with a string kind.
func parseEventLine(line []byte) (Event, bool) {
	rawLine := string(line)
	line = bytes.TrimSpace(line)
	if len(line) == 0 || line[0] != '{' {
		return Event{}, false
	}
	var raw struct {
		Kind json.RawMessage `json:"kind"`
		T    json.RawMessage `json:"t"`
		Data json.RawMessage `json:"data"`
	}
	if json.Unmarshal(line, &raw) != nil || len(raw.Kind) == 0 || raw.Kind[0] != '"' {
		return Event{}, false
	}
	var ev Event
	if json.Unmarshal(raw.Kind, &ev.Kind) != nil {
		return Event{}, false
	}
	if strings.HasPrefix(ev.Kind, "ixctl.") {
		return event("log", map[string]any{"line": rawLine}), true
	}
	// t is Unix milliseconds on the service's clock: rounded, never truncated,
	// and dropped when it is not a finite number.
	var t float64
	if len(raw.T) > 0 && json.Unmarshal(raw.T, &t) == nil && !math.IsNaN(t) && !math.IsInf(t, 0) {
		ev.T = int64(math.Round(t))
	}
	if len(raw.Data) > 0 && !bytes.Equal(raw.Data, []byte("null")) {
		ev.Data = raw.Data
	}
	return ev, true
}
