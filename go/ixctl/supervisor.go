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

type exitInfo struct {
	Code int   `json:"code"`
	At   int64 `json:"at"`
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
	lastExit  *exitInfo
	nodes     json.RawMessage

	cmds chan string // desired states, in apply order

	// owned by run
	proc   *child
	recent []time.Time // restarts inside the window
	timer  *time.Timer
}

// child is one launched process tree.
type child struct {
	tree   *procTree
	stdin  *os.File
	pid    int
	exited chan int // the main process's exit code
	io     sync.WaitGroup
}

func newService(spec ServiceSpec, hub *Hub) *service {
	return &service{
		spec: spec, hub: hub, ch: "svc/" + spec.Name,
		desired: desiredStopped, state: stateStopped,
		cmds: make(chan string, 1024),
	}
}

// want records a desired state and hands it to the supervisor.
func (s *service) want(state string) {
	s.mu.Lock()
	s.desired = state
	s.mu.Unlock()
	s.cmds <- state
}

func (s *service) observed() (desired, state string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.desired, s.state
}

func (s *service) status() ServiceStatus {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := ServiceStatus{Name: s.spec.Name, Desired: s.desired, State: s.state, Restarts: s.restarts, Nodes: s.nodes}
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

func (s *service) setState(state string) {
	s.mu.Lock()
	s.state = state
	s.mu.Unlock()
}

func (s *service) ctl(kind string, data map[string]any) {
	data["service"] = s.spec.Name
	s.hub.Publish("ctl", event(kind, data))
}

func (s *service) run(ctx context.Context) {
	for {
		var exited <-chan int
		if s.proc != nil {
			exited = s.proc.exited
		}
		var backoff <-chan time.Time
		if s.timer != nil {
			backoff = s.timer.C
		}
		select {
		case <-ctx.Done():
			s.halt()
			return
		case want := <-s.cmds:
			s.command(want)
		case code := <-exited:
			s.onExit(code)
		case <-backoff:
			s.timer = nil
			if s.state == stateRestarting {
				s.launch(true)
			}
		}
	}
}

func (s *service) command(want string) {
	switch want {
	case desiredRunning:
		if s.proc != nil || s.state == stateRestarting {
			return
		}
		s.recent = nil
		s.launch(false)
	case desiredStopped:
		switch {
		case s.proc != nil:
			s.stopTree()
		case s.state == stateRestarting || s.state == stateFailed:
			s.cancelTimer()
			s.setState(stateStopped)
			s.ctl("service.stopped", map[string]any{})
		}
	}
}

func (s *service) cancelTimer() {
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
}

func (s *service) launch(restart bool) {
	s.setState(stateStarting)
	s.ctl("service.starting", map[string]any{"restart": restart})
	c, err := s.start()
	if err != nil {
		s.setState(stateFailed)
		s.ctl("service.failed", map[string]any{"reason": "launch: " + err.Error()})
		return
	}
	s.proc = c
	now := time.Now().UnixMilli()
	s.mu.Lock()
	if restart {
		s.restarts++
	}
	s.state, s.pid, s.startedAt = stateRunning, c.pid, now
	restarts := s.restarts
	s.mu.Unlock()
	s.ctl("service.started", map[string]any{"pid": c.pid, "restarts": restarts})
}

func (s *service) start() (*child, error) {
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
	tree, err := startTree(cmd)
	inR.Close()
	outW.Close()
	errW.Close()
	if err != nil {
		inW.Close()
		outR.Close()
		errR.Close()
		return nil, err
	}
	c := &child{tree: tree, stdin: inW, pid: cmd.Process.Pid, exited: make(chan int, 1)}
	c.io.Add(2)
	go func() { defer c.io.Done(); s.readLines(outR, false) }()
	go func() { defer c.io.Done(); s.readLines(errR, true) }()
	go func() {
		code := -1
		if err := cmd.Wait(); err == nil {
			code = 0
		} else if ee := (*exec.ExitError)(nil); errors.As(err, &ee) {
			code = exitCode(ee)
		}
		c.exited <- code
	}()
	return c, nil
}

// reap finishes a tree whose main process has exited: whatever it left
// behind is killed, and its output is drained.
func (s *service) reap(code int) *exitInfo {
	c := s.proc
	c.stdin.Close()
	c.tree.kill()
	c.tree.waitGone(treeGoneLimit)
	c.tree.close()
	drained := make(chan struct{})
	go func() { c.io.Wait(); close(drained) }()
	select {
	case <-drained:
	case <-time.After(drainLimit):
	}
	s.proc = nil
	e := &exitInfo{Code: code, At: time.Now().UnixMilli()}
	s.mu.Lock()
	s.pid, s.lastExit = 0, e
	s.mu.Unlock()
	return e
}

// stopTree is a requested stop: close stdin, give the tree stopGrace to
// leave, then kill all of it.
func (s *service) stopTree() {
	t0 := time.Now()
	pid := s.proc.pid
	s.proc.stdin.Close()
	var code int
	select {
	case code = <-s.proc.exited:
	case <-time.After(stopGrace):
		s.proc.tree.kill()
		select {
		case code = <-s.proc.exited:
		case <-time.After(treeGoneLimit):
			code = -1
		}
	}
	s.reap(code)
	s.setState(stateStopped)
	ms := time.Since(t0).Milliseconds()
	s.ctl("service.exited", map[string]any{"pid": pid, "code": code, "requested": true})
	s.ctl("service.stopped", map[string]any{"pid": pid, "code": code, "ms": ms})
}

// halt stops the service for good: the plane is shutting down.
func (s *service) halt() {
	s.cancelTimer()
	if s.proc != nil {
		s.stopTree()
		return
	}
	if s.state != stateStopped {
		s.setState(stateStopped)
		s.ctl("service.stopped", map[string]any{})
	}
}

// onExit handles an exit nobody asked for, by the restart policy.
func (s *service) onExit(code int) {
	pid := s.proc.pid
	s.reap(code)
	s.ctl("service.exited", map[string]any{"pid": pid, "code": code, "requested": false})
	if s.spec.Restart == restartNever || (s.spec.Restart == restartOnFailure && code == 0) {
		s.setState(stateStopped)
		s.ctl("service.stopped", map[string]any{"pid": pid, "code": code})
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
		s.setState(stateFailed)
		s.ctl("service.failed", map[string]any{
			"reason": "too many restarts", "restarts_in_window": len(s.recent), "code": code})
		return
	}
	delay := backoffMin << len(s.recent)
	if delay > backoffMax || delay <= 0 {
		delay = backoffMax
	}
	s.recent = append(s.recent, now)
	s.setState(stateRestarting)
	s.timer = time.NewTimer(delay)
	s.ctl("service.restarting", map[string]any{"attempt": len(s.recent), "in_ms": delay.Milliseconds()})
}

// readLines turns a service's output into events on svc/<name>.
func (s *service) readLines(r *os.File, stderr bool) {
	defer r.Close()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), maxLine)
	for sc.Scan() {
		s.onLine(sc.Bytes(), stderr)
	}
	if sc.Err() != nil {
		s.hub.Publish(s.ch, event("log", map[string]any{"error": sc.Err().Error()}))
		io.Copy(io.Discard, r)
	}
}

func (s *service) onLine(line []byte, stderr bool) {
	if stderr {
		s.hub.Publish(s.ch, event("stderr", map[string]any{"line": string(line)}))
		return
	}
	ev, ok := parseEventLine(line)
	if !ok {
		s.hub.Publish(s.ch, event("log", map[string]any{"line": string(line)}))
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
	s.hub.Publish(s.ch, ev)
}

// parseEventLine accepts a line that is a JSON object with a string kind.
func parseEventLine(line []byte) (Event, bool) {
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
