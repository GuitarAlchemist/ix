package main

import (
	"context"
	"encoding/json"
	"os"
	"sync"
	"time"
)

// Plane is one ixctl instance: declared services, their supervisors, the hub
// and the peers. It reconciles desired state with observed state.
type Plane struct {
	name    string
	spec    *Spec
	hub     *Hub
	svcs    map[string]*service
	order   []string
	peers   map[string]*peerLink
	journal *journal
	started time.Time

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	applyMu  sync.Mutex // one apply at a time: generations are a total order
	mu       sync.Mutex
	gen      uint64
	observed uint64          // the highest generation that converged
	settled  uint64          // every generation <= settled has ended, converged or diverged
	ended    map[uint64]bool // generations above settled that have ended
	diverged *Diverged       // the diverged apply of highest generation, if any
	failed   []Diverged      // every diverged apply since boot, by generation

	quit     chan struct{} // closed when a shutdown is requested
	quitOnce sync.Once
	downOnce sync.Once
}

// awaitTimeout bounds how long an apply waits for its actions to be observed.
const awaitTimeout = 60 * time.Second

func NewPlane(spec *Spec, name, journalPath string) (*Plane, error) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &Plane{
		name: name, spec: spec, hub: NewHub(name),
		svcs: map[string]*service{}, peers: map[string]*peerLink{}, ended: map[uint64]bool{},
		started: time.Now(), ctx: ctx, cancel: cancel, quit: make(chan struct{}),
	}
	if journalPath != "" {
		j, err := openJournal(journalPath)
		if err != nil {
			cancel()
			return nil, err
		}
		p.journal = j
		p.hub.tap = func(env *Envelope) {
			if env.Ch == "ctl" {
				j.write(env.Raw())
			}
		}
	}
	for _, ss := range spec.Services {
		p.svcs[ss.Name] = newService(ss, p.hub)
		p.order = append(p.order, ss.Name)
	}
	for _, ps := range spec.Peers {
		p.peers[ps.Name] = newPeerLink(ps, p.hub)
	}
	return p, nil
}

// Start runs the supervisors and the peer links, then applies the autostart
// services as the first generation.
func (p *Plane) Start() {
	for _, n := range p.order {
		s := p.svcs[n]
		p.wg.Add(1)
		go func() { defer p.wg.Done(); s.run(p.ctx) }()
	}
	for _, l := range p.peers {
		p.wg.Add(1)
		go func() { defer p.wg.Done(); l.run(p.ctx) }()
	}
	auto := &Desired{Services: map[string]DesiredService{}}
	for _, ss := range p.spec.Services {
		if ss.Autostart {
			auto.Services[ss.Name] = DesiredService{State: desiredRunning}
		}
	}
	if len(auto.Services) > 0 {
		p.Apply(auto)
	}
}

func (p *Plane) Publish(ch string, ev Event) *Envelope { return p.hub.Publish(ch, ev) }

func (p *Plane) Subscribe(prefix string, since *uint64, epoch string) *Cursor {
	return p.hub.Subscribe(prefix, since, epoch)
}

// Epoch is this boot's id, from the hub.
func (p *Plane) Epoch() string { return p.hub.Epoch() }

// Request is the one entry point every transport uses.
func (p *Plane) Request(ctx context.Context, req Request) (any, error) {
	peer := req.Peer
	if peer == "" && req.Desired != nil {
		peer = req.Desired.Peer
	}
	if peer != "" {
		return p.forward(ctx, peer, req)
	}
	switch req.Op {
	case "status":
		return p.Status(), nil
	case "ping":
		return map[string]any{"pong": true, "name": p.name}, nil
	case "plan":
		if req.Desired == nil {
			return nil, badRequest("plan needs a desired-state document")
		}
		return p.Plan(req.Desired)
	case "apply":
		if req.Desired == nil {
			return nil, badRequest("apply needs a desired-state document")
		}
		r, _, err := p.Apply(req.Desired)
		return r, err
	case "stop":
		return p.Stop(ctx, req.Service)
	case "shutdown":
		p.RequestShutdown()
		return map[string]any{"shutdown": true}, nil
	case "":
		return nil, badRequest("missing op")
	}
	return nil, badRequest("unknown op %q", req.Op)
}

// forward hands an op to a peer, which decides by its own rules. The peer
// field is stripped so an op crosses at most one hop.
func (p *Plane) forward(ctx context.Context, peer string, req Request) (any, error) {
	l := p.peers[peer]
	if l == nil {
		return nil, badRequest("unknown peer %q", peer)
	}
	switch req.Op {
	case "status", "ping", "plan", "apply", "stop":
	default:
		return nil, badRequest("op %q cannot be forwarded", req.Op)
	}
	fwd := Request{Op: req.Op, Service: req.Service}
	if req.Desired != nil {
		d := *req.Desired
		d.Peer = ""
		fwd.Desired = &d
	}
	return l.forward(ctx, fwd)
}

type ServiceStatus struct {
	Name      string          `json:"name"`
	Desired   string          `json:"desired"`
	State     string          `json:"state"`
	Pid       *int            `json:"pid"`
	Restarts  int             `json:"restarts"`
	StartedAt *int64          `json:"started_at"`
	LastExit  *exitInfo       `json:"last_exit"`
	Nodes     json.RawMessage `json:"nodes"`
}

type PeerStatus struct {
	Name     string          `json:"name"`
	State    string          `json:"state"`
	Services json.RawMessage `json:"services"`
}

// StatusReply summarises the generations. Every generation <= SettledGeneration has
// ended, converged or diverged, and Failed lists the diverged ones since boot, lowest
// generation first: g has failed iff Failed names it, and is done iff g <=
// SettledGeneration and Failed does not name it. ObservedGeneration (the highest that
// converged) and Diverged (the highest that failed) stay for older readers; on their own
// they let a later converged generation hide an earlier pending or failed one.
type StatusReply struct {
	Name               string          `json:"name"`
	Generation         uint64          `json:"generation"`
	ObservedGeneration uint64          `json:"observed_generation"`
	SettledGeneration  uint64          `json:"settled_generation"`
	Diverged           *Diverged       `json:"diverged"`
	Failed             []Diverged      `json:"failed"`
	Services           []ServiceStatus `json:"services"`
	Peers              []PeerStatus    `json:"peers"`
}

func (p *Plane) Status() StatusReply {
	p.mu.Lock()
	r := StatusReply{Name: p.name, Generation: p.gen, ObservedGeneration: p.observed, SettledGeneration: p.settled}
	if p.diverged != nil {
		d := *p.diverged
		r.Diverged = &d
	}
	r.Failed = append([]Diverged{}, p.failed...)
	p.mu.Unlock()
	r.Services = make([]ServiceStatus, 0, len(p.order))
	for _, n := range p.order {
		r.Services = append(r.Services, p.svcs[n].status())
	}
	r.Peers = make([]PeerStatus, 0, len(p.spec.Peers))
	for _, ps := range p.spec.Peers {
		r.Peers = append(r.Peers, p.peers[ps.Name].status())
	}
	return r
}

type Action struct {
	Op      string `json:"op"`
	Service string `json:"service"`
}

type PlanReply struct {
	Generation uint64   `json:"generation"`
	Actions    []Action `json:"actions"`
	Noop       []string `json:"noop"`
}

type ApplyReply struct {
	Generation uint64   `json:"generation"`
	Actions    []Action `json:"actions"`
}

// Diverged names an apply that will not converge. Status lists them all in
// Failed, so a later generation that converges never hides an earlier failure.
type Diverged struct {
	Generation uint64 `json:"generation"`
	Service    string `json:"service"`
	Reason     string `json:"reason"`
}

// applyResult is how an apply ended: converged, or diverged on one service.
type applyResult struct {
	Generation uint64
	Converged  bool
	Service    string
	Reason     string
}

func (p *Plane) validate(d *Desired) error {
	if len(d.Services) == 0 {
		return badRequest("desired-state document names no service")
	}
	for _, n := range d.names() {
		if p.svcs[n] == nil {
			return badRequest("unknown service %q", n)
		}
		switch d.Services[n].State {
		case desiredRunning, desiredStopped:
		default:
			return badRequest("service %q: state %q is not running|stopped", n, d.Services[n].State)
		}
	}
	return nil
}

// Plan is pure: it compares the document with the observed state and
// changes nothing.
func (p *Plane) Plan(d *Desired) (PlanReply, error) {
	if err := p.validate(d); err != nil {
		return PlanReply{}, err
	}
	p.mu.Lock()
	g := p.gen
	p.mu.Unlock()
	actions, noop := p.plan(d)
	return PlanReply{Generation: g, Actions: actions, Noop: noop}, nil
}

func (p *Plane) plan(d *Desired) ([]Action, []string) {
	actions, noop := []Action{}, []string{}
	for _, n := range d.names() {
		desired, state := p.svcs[n].observed()
		switch d.Services[n].State {
		case desiredRunning:
			if desired == desiredRunning && state == stateRunning {
				noop = append(noop, n)
			} else {
				actions = append(actions, Action{Op: "start", Service: n})
			}
		case desiredStopped:
			if state == stateStopped {
				noop = append(noop, n)
			} else {
				actions = append(actions, Action{Op: "stop", Service: n})
			}
		}
	}
	return actions, noop
}

// Apply bumps the generation, records the document, hands the actions to the
// supervisors and answers at once. ctl says converged (or diverged) later;
// the returned channel says it too, for in-process callers.
func (p *Plane) Apply(d *Desired) (ApplyReply, <-chan applyResult, error) {
	if err := p.validate(d); err != nil {
		return ApplyReply{}, nil, err
	}
	select {
	case <-p.quit:
		return ApplyReply{}, nil, &opError{status: 503, msg: "shutting down"}
	default:
	}
	p.applyMu.Lock()
	defer p.applyMu.Unlock()
	head := p.hub.Head()
	cur := p.hub.Subscribe("ctl", &head, p.hub.Epoch())
	actions, _ := p.plan(d)
	p.mu.Lock()
	p.gen++
	g := p.gen
	p.mu.Unlock()
	p.journal.record(g, d, actions)
	for _, n := range d.names() {
		p.svcs[n].want(d.Services[n].State)
	}
	done := make(chan applyResult, 1)
	go func() { done <- p.await(g, actions, cur) }()
	return ApplyReply{Generation: g, Actions: actions}, done, nil
}

// await watches ctl until every action is observed, then publishes the
// outcome. A start is observed by service.started, a stop by service.stopped
// (the tree is gone by then).
func (p *Plane) await(g uint64, actions []Action, cur *Cursor) applyResult {
	pending := map[string]string{}
	for _, a := range actions {
		pending[a.Service] = a.Op
	}
	ctx, cancel := context.WithTimeout(p.ctx, awaitTimeout)
	defer cancel()
	res := applyResult{Generation: g, Converged: true}
	for len(pending) > 0 {
		envs, gap, err := cur.Next(ctx, 256)
		if err != nil {
			res = applyResult{Generation: g, Reason: "timeout"}
			if p.ctx.Err() != nil {
				res.Reason = "shutdown"
			}
			for n := range pending {
				res.Service = n
				break
			}
			break
		}
		if gap != nil {
			// Events were lost: fall back on the observed state.
			for n, op := range pending {
				_, state := p.svcs[n].observed()
				if (op == "start" && state == stateRunning) || (op == "stop" && state == stateStopped) {
					delete(pending, n)
				}
			}
			continue
		}
		for _, env := range envs {
			var data struct {
				Service string `json:"service"`
				Reason  string `json:"reason"`
			}
			json.Unmarshal(env.Ev.Data, &data)
			op, ok := pending[data.Service]
			if !ok {
				continue
			}
			switch {
			case env.Ev.Kind == "service.started" && op == "start",
				env.Ev.Kind == "service.stopped" && op == "stop":
				delete(pending, data.Service)
			case env.Ev.Kind == "service.failed" && op == "start":
				res = applyResult{Generation: g, Service: data.Service, Reason: data.Reason}
				pending = nil
			}
			if res.Reason != "" {
				break
			}
		}
	}
	p.mu.Lock()
	if res.Converged {
		if g > p.observed {
			p.observed = g
		}
	} else {
		d := Diverged{Generation: g, Service: res.Service, Reason: res.Reason}
		p.failed = append(p.failed, d)
		if p.diverged == nil || g > p.diverged.Generation {
			p.diverged = &d
		}
	}
	// Generations end in any order; settled only moves past an unbroken run of ended ones,
	// so a later generation never counts an earlier, still pending one as done.
	p.ended[g] = true
	for p.ended[p.settled+1] {
		delete(p.ended, p.settled+1)
		p.settled++
	}
	p.mu.Unlock()
	if res.Converged {
		p.hub.Publish("ctl", event("converged", map[string]any{"generation": g}))
	} else {
		p.hub.Publish("ctl", event("diverged", Diverged{Generation: g, Service: res.Service, Reason: res.Reason}))
	}
	return res
}

type StopReply struct {
	Stopped    string    `json:"stopped"`
	Generation uint64    `json:"generation"`
	Exit       *exitInfo `json:"exit"`
}

// Stop applies {name: stopped} and answers once the tree is gone.
func (p *Plane) Stop(ctx context.Context, name string) (StopReply, error) {
	if name == "" {
		return StopReply{}, badRequest("stop needs a service")
	}
	r, done, err := p.Apply(&Desired{Services: map[string]DesiredService{name: {State: desiredStopped}}})
	if err != nil {
		return StopReply{}, err
	}
	select {
	case res := <-done:
		if !res.Converged {
			return StopReply{}, &opError{status: 500, msg: "stop " + name + ": " + res.Reason}
		}
	case <-ctx.Done():
		return StopReply{}, ctx.Err()
	}
	return StopReply{Stopped: name, Generation: r.Generation, Exit: p.svcs[name].status().LastExit}, nil
}

// RequestShutdown asks the process hosting the plane to shut it down.
func (p *Plane) RequestShutdown() { p.quitOnce.Do(func() { close(p.quit) }) }

// Quit is closed once a shutdown has been requested.
func (p *Plane) Quit() <-chan struct{} { return p.quit }

// Shutdown stops every service (whole trees) and every peer link, then
// closes the journal. It returns once all of it is done.
func (p *Plane) Shutdown() {
	p.downOnce.Do(func() {
		p.RequestShutdown()
		p.cancel()
		p.wg.Wait()
		p.journal.close()
	})
}

// event is one of the plane's own events; its t is Unix milliseconds now.
func event(kind string, data any) Event {
	b, _ := json.Marshal(data)
	return Event{Kind: kind, T: time.Now().UnixMilli(), Data: b}
}

// journal appends every applied document and every ctl event, as JSON lines.
type journal struct {
	mu sync.Mutex
	f  *os.File
}

func openJournal(path string) (*journal, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	return &journal{f: f}, nil
}

func (j *journal) write(b []byte) {
	if j == nil {
		return
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.f != nil {
		j.f.Write(append(b[:len(b):len(b)], '\n'))
	}
}

func (j *journal) record(g uint64, d *Desired, actions []Action) {
	if j == nil {
		return
	}
	b, _ := json.Marshal(map[string]any{
		"t": time.Now().UnixMilli(), "kind": "apply", "generation": g, "desired": d, "actions": actions})
	j.write(b)
}

func (j *journal) close() {
	if j == nil {
		return
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.f != nil {
		j.f.Close()
		j.f = nil
	}
}
