package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"time"
)

// benchConfig: services[0] of the spec is the subject of every timed step;
// services[1], if declared, is expected to exit on its own and measures the
// restart. The bench runs its own instances (bench-b, and bench-a with b as
// peer) in a temporary state dir.
type benchConfig struct {
	spec   string
	exe    string   // the ixctl binary to run as "up"
	env    []string // extra environment for it
	n      int      // iterations of each timed step
	rtt    int      // WebSocket status round trips
	fanout int      // fan-out samples
	quiet  io.Writer
}

type dist struct {
	N      int     `json:"n"`
	Median float64 `json:"median"`
	P99    float64 `json:"p99"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

func summarize(xs []float64) *dist {
	if len(xs) == 0 {
		return nil
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	p99 := int(float64(len(s))*0.99+0.5) - 1
	if p99 < 0 {
		p99 = 0
	}
	if p99 >= len(s) {
		p99 = len(s) - 1
	}
	med := s[len(s)/2]
	if len(s)%2 == 0 {
		med = (s[len(s)/2-1] + s[len(s)/2]) / 2
	}
	return &dist{N: len(s), Median: round3(med), P99: round3(s[p99]), Min: round3(s[0]), Max: round3(s[len(s)-1])}
}

func round3(x float64) float64 { return float64(int64(x*1000+0.5)) / 1000 }

type benchResult struct {
	Subject              string   `json:"subject"`
	CrashSubject         string   `json:"crash_subject,omitempty"`
	PlanUs               *dist    `json:"plan_us"`
	ReadyMs              *dist    `json:"ready_ms"`
	ReadyInternalMs      *dist    `json:"ready_internal_ms"`
	ApplyReplyMs         *dist    `json:"apply_reply_ms"`
	ApplyStartedMs       *dist    `json:"apply_to_started_ms"`
	ApplyFirstEventMs    *dist    `json:"apply_to_first_event_ms"`
	ApplyConvergedMs     *dist    `json:"apply_to_converged_ms"`
	StopMs               *dist    `json:"stop_tree_ms"`
	RestartMs            *dist    `json:"restart_ms,omitempty"`
	RestartBackoffMs     *dist    `json:"restart_backoff_ms,omitempty"`
	WSStatusRTTUs        *dist    `json:"ws_status_rtt_us"`
	FanoutUs             *dist    `json:"fanout_us"`
	FanoutClock          string   `json:"fanout_clock"`
	PeerApplyReplyMs     *dist    `json:"peer_apply_reply_ms"`
	PeerApplyConvergedMs *dist    `json:"peer_apply_converged_ms"`
	PeerStopMs           *dist    `json:"peer_stop_ms"`
	Notes                []string `json:"notes,omitempty"`
}

func cmdBench(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	n := fs.Int("n", 10, "iterations of each timed step")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("bench needs exactly one spec file")
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	res, err := runBench(benchConfig{spec: pos[0], exe: exe, n: *n, rtt: 1000, fanout: 1000, quiet: io.Discard})
	if err != nil {
		return err
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(res)
}

func ms(ns int64) float64 { return float64(ns) / 1e6 }
func us(ns int64) float64 { return float64(ns) / 1e3 }

func runBench(cfg benchConfig) (*benchResult, error) {
	spec, err := LoadSpec(cfg.spec)
	if err != nil {
		return nil, err
	}
	if len(spec.Services) == 0 {
		return nil, errors.New("bench needs at least one service")
	}
	tmp, err := os.MkdirTemp("", "ixctl-bench-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	res := &benchResult{Subject: spec.Services[0].Name}
	subject := res.Subject

	// Derived specs, with every path already absolute.
	specB := *spec
	specB.Name, specB.Peers = "bench", nil
	pathB := filepath.Join(tmp, "spec-b.json")
	specA := specB
	specA.Peers = []PeerSpec{{Name: "b", StateFile: filepath.Join(tmp, "bench-b.json")}}
	pathA := filepath.Join(tmp, "spec-a.json")
	for p, s := range map[string]*Spec{pathB: &specB, pathA: &specA} {
		b, _ := json.Marshal(s)
		if err := os.WriteFile(p, b, 0o600); err != nil {
			return nil, err
		}
	}

	res.PlanUs = benchPlan(&specB)

	var ready, internal []float64
	for i := 0; i < cfg.n; i++ {
		c, err := spawnUp(cfg, pathB, fmt.Sprintf("bench-r%d", i), tmp)
		if err != nil {
			return nil, err
		}
		ready = append(ready, ms(c.tReady-c.t0))
		internal = append(internal, c.ready.MsToReady)
		if err := c.shutdown(); err != nil {
			return nil, err
		}
	}
	res.ReadyMs, res.ReadyInternalMs = summarize(ready), summarize(internal)

	b, err := spawnUp(cfg, pathB, "bench-b", tmp)
	if err != nil {
		return nil, err
	}
	defer b.shutdown()
	a, err := spawnUp(cfg, pathA, "bench-a", tmp)
	if err != nil {
		return nil, err
	}
	defer a.shutdown()

	ctx := context.Background()
	ev, err := DialWS(ctx, b.ready.Addr, b.token)
	if err != nil {
		return nil, err
	}
	defer ev.Close()
	_, frames, err := ev.Subscribe(ctx, "", nil, 1<<16)
	if err != nil {
		return nil, err
	}
	w := &frameWaiter{frames: frames}
	run := func(st string) Request {
		return Request{Op: "apply", Desired: &Desired{Services: map[string]DesiredService{subject: {State: st}}}}
	}

	var applyReply, started, first, converged, stops []float64
	for i := 0; i < cfg.n; i++ {
		t0 := hrnow()
		data, err := ev.Do(ctx, run(desiredRunning))
		if err != nil {
			return nil, err
		}
		applyReply = append(applyReply, ms(hrnow()-t0))
		var ar ApplyReply
		json.Unmarshal(data, &ar)
		// A service event counts as the first one only after this launch's
		// service.starting: anything earlier belongs to the previous process.
		var tStarted, tFirst, tConv int64
		starting := false
		for tStarted == 0 || tFirst == 0 || tConv == 0 {
			f, err := w.next(10*time.Second, func(e *Envelope) bool {
				if e.Ch == "svc/"+subject {
					return starting && tFirst == 0
				}
				return isCtl(e, "service.starting", subject) || isCtl(e, "service.started", subject) ||
					(e.Ch == "ctl" && e.Ev.Kind == "converged" && genOf(e) == ar.Generation)
			})
			if err != nil {
				return nil, fmt.Errorf("apply %d: %w", i, err)
			}
			switch {
			case f.Env.Ch != "ctl":
				tFirst = f.At
			case f.Env.Ev.Kind == "service.starting":
				starting = true
			case f.Env.Ev.Kind == "service.started":
				tStarted = f.At
			default:
				tConv = f.At
			}
		}
		started = append(started, ms(tStarted-t0))
		first = append(first, ms(tFirst-t0))
		converged = append(converged, ms(tConv-t0))
		if i == cfg.n-1 {
			break // keep the last one running for fan-out and round trips
		}
		t0 = hrnow()
		if _, err := ev.Do(ctx, Request{Op: "stop", Service: subject}); err != nil {
			return nil, err
		}
		stops = append(stops, ms(hrnow()-t0))
	}
	res.ApplyReplyMs, res.ApplyStartedMs = summarize(applyReply), summarize(started)
	res.ApplyFirstEventMs, res.ApplyConvergedMs = summarize(first), summarize(converged)

	// Fan-out: service stdout line -> WebSocket client, by the line's own
	// timestamp: data.hr_ns (a machine-wide clock) when the service gives
	// it, else t in ms.
	var fanHr, fanMs []float64
	deadline := time.Now().Add(10 * time.Second)
	for len(fanHr) < cfg.fanout && len(fanMs) < cfg.fanout && time.Now().Before(deadline) {
		f, err := w.next(5*time.Second, func(e *Envelope) bool { return e.Ch == "svc/"+subject })
		if err != nil {
			break
		}
		var d struct {
			HrNs int64 `json:"hr_ns"`
		}
		json.Unmarshal(f.Env.Ev.Data, &d)
		switch {
		case d.HrNs != 0:
			fanHr = append(fanHr, us(f.At-d.HrNs))
		case f.Env.Ev.T != 0:
			fanMs = append(fanMs, float64(time.Now().UnixMilli()-f.Env.Ev.T)*1000)
		}
	}
	res.FanoutUs, res.FanoutClock = summarize(fanHr), "hr_ns"
	if len(fanHr) == 0 {
		res.FanoutUs, res.FanoutClock = summarize(fanMs), "t_ms"
	}

	rc, err := DialWS(ctx, b.ready.Addr, b.token)
	if err != nil {
		return nil, err
	}
	var rtt []float64
	for i := 0; i < cfg.rtt; i++ {
		t0 := hrnow()
		if _, err := rc.Do(ctx, Request{Op: "status"}); err != nil {
			rc.Close()
			return nil, err
		}
		rtt = append(rtt, us(hrnow()-t0))
	}
	rc.Close()
	res.WSStatusRTTUs = summarize(rtt)

	t0 := hrnow()
	if _, err := ev.Do(ctx, Request{Op: "stop", Service: subject}); err != nil {
		return nil, err
	}
	stops = append(stops, ms(hrnow()-t0))
	res.StopMs = summarize(stops)

	if len(spec.Services) > 1 {
		res.CrashSubject = spec.Services[1].Name
		if err := benchRestart(ctx, ev, w, res, cfg.n); err != nil {
			res.Notes = append(res.Notes, "restart: "+err.Error())
		}
	} else {
		res.Notes = append(res.Notes, "restart: no second service declared")
	}

	if err := benchPeer(ctx, cfg, a, subject, res); err != nil {
		return nil, fmt.Errorf("peer: %w", err)
	}
	return res, nil
}

// benchPlan times the pure plan in process, in batches of 100 calls.
func benchPlan(spec *Spec) *dist {
	p, err := NewPlane(spec, "bench-plan", "")
	if err != nil {
		return nil
	}
	defer p.Shutdown()
	d := &Desired{Services: map[string]DesiredService{}}
	for _, s := range spec.Services {
		d.Services[s.Name] = DesiredService{State: desiredRunning}
	}
	var per []float64
	for b := 0; b < 100; b++ {
		t0 := hrnow()
		for i := 0; i < 100; i++ {
			p.Plan(d)
		}
		per = append(per, us(hrnow()-t0)/100)
	}
	return summarize(per)
}

func benchRestart(ctx context.Context, ev *Client, w *frameWaiter, res *benchResult, n int) error {
	name := res.CrashSubject
	var restart, backoff []float64
	if n > 5 {
		n = 5
	}
	for i := 0; i < n; i++ {
		if _, err := ev.Do(ctx, Request{Op: "apply", Desired: &Desired{
			Services: map[string]DesiredService{name: {State: desiredRunning}}}}); err != nil {
			return err
		}
		if _, err := w.next(10*time.Second, func(e *Envelope) bool { return isCtl(e, "service.started", name) }); err != nil {
			return err
		}
		ex, err := w.next(15*time.Second, func(e *Envelope) bool {
			return isCtl(e, "service.exited", name) || isCtl(e, "service.failed", name)
		})
		if err != nil {
			return fmt.Errorf("%s never exited on its own: %w", name, err)
		}
		if ex.Env.Ev.Kind == "service.failed" {
			return fmt.Errorf("%s failed: %s", name, ex.Env.Ev.Data)
		}
		rs, err := w.next(5*time.Second, func(e *Envelope) bool {
			return isCtl(e, "service.restarting", name) || isCtl(e, "service.stopped", name)
		})
		if err != nil {
			return err
		}
		if rs.Env.Ev.Kind != "service.restarting" {
			return fmt.Errorf("%s is not restarted by its policy", name)
		}
		var d struct {
			InMs float64 `json:"in_ms"`
		}
		json.Unmarshal(rs.Env.Ev.Data, &d)
		st, err := w.next(15*time.Second, func(e *Envelope) bool { return isCtl(e, "service.started", name) })
		if err != nil {
			return err
		}
		restart = append(restart, ms(st.At-ex.At))
		backoff = append(backoff, d.InMs)
		if _, err := ev.Do(ctx, Request{Op: "stop", Service: name}); err != nil {
			return err
		}
	}
	res.RestartMs, res.RestartBackoffMs = summarize(restart), summarize(backoff)
	return nil
}

func benchPeer(ctx context.Context, cfg benchConfig, a *benchChild, subject string, res *benchResult) error {
	c, err := DialWS(ctx, a.ready.Addr, a.token)
	if err != nil {
		return err
	}
	defer c.Close()
	if err := waitPeerUp(ctx, c, "b", 10*time.Second); err != nil {
		return err
	}
	_, frames, err := c.Subscribe(ctx, "peer/b/", nil, 1<<16)
	if err != nil {
		return err
	}
	w := &frameWaiter{frames: frames}
	var reply, conv, stop []float64
	for i := 0; i < cfg.n; i++ {
		t0 := hrnow()
		data, err := c.Do(ctx, Request{Op: "apply", Peer: "b", Desired: &Desired{
			Services: map[string]DesiredService{subject: {State: desiredRunning}}}})
		if err != nil {
			return err
		}
		reply = append(reply, ms(hrnow()-t0))
		var ar ApplyReply
		json.Unmarshal(data, &ar)
		// As in the local loop, stop only a service that has spoken: the
		// first mirrored event after this launch's service.starting.
		var tConv int64
		starting, spoke := false, false
		for tConv == 0 || !spoke {
			f, err := w.next(10*time.Second, func(e *Envelope) bool {
				if e.Ch == "peer/b/svc/"+subject {
					return starting
				}
				return e.Ch == "peer/b/ctl" && ((e.Ev.Kind == "converged" && genOf(e) == ar.Generation) ||
					(e.Ev.Kind == "service.starting" && !starting))
			})
			if err != nil {
				return err
			}
			switch {
			case f.Env.Ev.Kind == "converged":
				tConv = f.At
			case f.Env.Ev.Kind == "service.starting":
				starting = true
			default:
				spoke = true
			}
		}
		conv = append(conv, ms(tConv-t0))
		t0 = hrnow()
		if _, err := c.Do(ctx, Request{Op: "stop", Peer: "b", Service: subject}); err != nil {
			return err
		}
		stop = append(stop, ms(hrnow()-t0))
	}
	res.PeerApplyReplyMs, res.PeerApplyConvergedMs, res.PeerStopMs = summarize(reply), summarize(conv), summarize(stop)
	return nil
}

func waitPeerUp(ctx context.Context, c *Client, peer string, d time.Duration) error {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		data, err := c.Do(ctx, Request{Op: "status"})
		if err != nil {
			return err
		}
		var st StatusReply
		json.Unmarshal(data, &st)
		for _, p := range st.Peers {
			if p.Name == peer && p.State == "up" {
				return nil
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	return fmt.Errorf("peer %s not up after %s", peer, d)
}

func isCtl(e *Envelope, kind, service string) bool {
	if e.Ch != "ctl" || e.Ev.Kind != kind {
		return false
	}
	var d struct {
		Service string `json:"service"`
	}
	json.Unmarshal(e.Ev.Data, &d)
	return d.Service == service
}

func genOf(e *Envelope) uint64 {
	var d struct {
		Generation uint64 `json:"generation"`
	}
	json.Unmarshal(e.Ev.Data, &d)
	return d.Generation
}

// frameWaiter reads a subscription until a matching envelope arrives.
type frameWaiter struct {
	frames <-chan Frame
}

func (w *frameWaiter) next(d time.Duration, match func(*Envelope) bool) (Frame, error) {
	t := time.NewTimer(d)
	defer t.Stop()
	for {
		select {
		case f, ok := <-w.frames:
			if !ok {
				return Frame{}, errors.New("subscription closed")
			}
			if f.Gap != nil {
				return Frame{}, fmt.Errorf("gap: %s from %d", f.Gap.Reason, f.Gap.From)
			}
			if f.Env != nil && match(f.Env) {
				return f, nil
			}
		case <-t.C:
			return Frame{}, fmt.Errorf("timeout after %s", d)
		}
	}
}

// benchChild is an "ixctl up" the bench started, and only it stops.
type benchChild struct {
	cmd    *exec.Cmd
	ready  readyLine
	token  string
	t0     int64
	tReady int64
	exited chan struct{}
}

func spawnUp(cfg benchConfig, spec, name, stateDir string) (*benchChild, error) {
	cmd := exec.Command(cfg.exe, "up", spec, "--name", name, "--state", stateDir)
	cmd.Env = append(os.Environ(), cfg.env...)
	cmd.Stderr = cfg.quiet
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	c := &benchChild{cmd: cmd, exited: make(chan struct{})}
	c.t0 = hrnow()
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	line := make(chan string, 1)
	go func() {
		s, _ := bufio.NewReader(out).ReadString('\n')
		line <- s
		io.Copy(io.Discard, out)
		cmd.Wait()
		close(c.exited)
	}()
	select {
	case s := <-line:
		c.tReady = hrnow()
		if err := json.Unmarshal([]byte(s), &c.ready); err != nil || c.ready.Event != "ready" {
			cmd.Process.Kill()
			return nil, fmt.Errorf("%s: no ready line (%q)", name, s)
		}
		if err := checkLoopback(c.ready.Addr); err != nil {
			cmd.Process.Kill()
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	case <-time.After(10 * time.Second):
		cmd.Process.Kill()
		return nil, fmt.Errorf("%s: not ready after 10s", name)
	}
	st, err := readStateFile(c.ready.StateFile)
	if err != nil {
		cmd.Process.Kill()
		return nil, err
	}
	c.token = st.Token
	return c, nil
}

func (c *benchChild) shutdown() error {
	select {
	case <-c.exited:
		return nil
	default:
	}
	if checkLoopback(c.ready.Addr) == nil { // the token goes to loopback only
		req, _ := http.NewRequest("POST", "http://"+c.ready.Addr+"/v1/shutdown", nil)
		req.Header.Set("Authorization", "Bearer "+c.token)
		if resp, err := loopbackHTTP.Do(req); err == nil {
			resp.Body.Close()
		}
	}
	select {
	case <-c.exited:
		return nil
	case <-time.After(15 * time.Second):
		c.cmd.Process.Kill()
		<-c.exited
		return fmt.Errorf("%s did not shut down; killed", c.ready.Name)
	}
}
