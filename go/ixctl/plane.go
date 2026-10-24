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
	name      string
	spec      *Spec
	hub       *Hub
	svcs      map[string]*service
	order     []string
	peers     map[string]*peerLink
	probes    *probeTree
	bookmarks *bookmarkStore
	journal   *journal
	started   time.Time

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

// NewPlane builds a plane: journalPath is its audit journal (none when empty) and
// bookmarksPath its bookmark file (an in-memory store when empty).
func NewPlane(spec *Spec, name, journalPath, bookmarksPath string) (*Plane, error) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &Plane{
		name: name, spec: spec, hub: NewHub(name),
		svcs: map[string]*service{}, peers: map[string]*peerLink{}, ended: map[uint64]bool{},
		started: time.Now(), ctx: ctx, cancel: cancel, quit: make(chan struct{}),
	}
	p.probes = newProbeTree(p.hub)
	bms, err := newBookmarkStore(p.hub, bookmarksPath)
	if err != nil {
		cancel()
		return nil, err
	}
	p.bookmarks = bms
	if journalPath != "" {
		j, err := openJournal(journalPath)
		if err != nil {
			cancel()
			return nil, err
		}
		p.journal = j
		j.boot(name, p.hub.Epoch())
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
	if req.Peer != "" {
		return p.forward(ctx, req.Peer, req)
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
	case "probe.kinds":
		return kindsReply(p.probes.budgetNow()), nil
	case "probe.create":
		if req.Probe == nil {
			return nil, badRequest("probe.create needs a node under `probe`")
		}
		return p.probes.create(req.Probe)
	case "probe.list":
		return p.probes.list(req.WithCursor), nil
	case "probe.start":
		return p.probes.start(req.Node)
	case "probe.stop":
		return p.probes.stop(req.Node, "requested")
	case "probe.touch":
		return p.probes.touch(req.Node)
	case "probe.remove":
		return p.probes.remove(req.Node, "removed")
	case "bookmark.set":
		return p.bookmarks.set(req)
	case "bookmark.get":
		return p.bookmarks.get(req.Bookmark)
	case "bookmark.list":
		return p.bookmarks.list(), nil
	case "bookmark.remove":
		return p.bookmarks.remove(req.Bookmark)
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
	case "status", "ping", "plan", "apply", "stop",
		"probe.kinds", "probe.list", "probe.create", "probe.start", "probe.stop", "probe.touch", "probe.remove",
		"bookmark.set", "bookmark.get", "bookmark.list", "bookmark.remove":
	default:
		return nil, badRequest("op %q cannot be forwarded", req.Op)
	}
	return l.forward(ctx, Request{Op: req.Op, Service: req.Service, Desired: req.Desired, Probe: req.Probe, Node: req.Node,
		Bookmark: req.Bookmark, Since: req.Since, Epoch: req.Epoch, Note: req.Note, WithCursor: req.WithCursor})
}

type ServiceStatus struct {
	Name      string `json:"name"`
	Desired   string `json:"desired"`
	State     string `json:"state"`
	Pid       *int   `json:"pid"`
	Restarts  int    `json:"restarts"`
	StartedAt *int64 `json:"started_at"`
	// Generation is the apply that launched the current or last run; absent before any.
	Generation uint64          `json:"generation,omitempty"`
	Run        *RunID          `json:"run,omitempty"`
	LastExit   *exitInfo       `json:"last_exit"`
	Nodes      json.RawMessage `json:"nodes"`
}

type PeerStatus struct {
	Name     string          `json:"name"`
	State    string          `json:"state"`
	Services json.RawMessage `json:"services"`
	// Probes: one entry per node named in the peer spec's `probes`, in that order, with what
	// it is right now: `mirrored` (exportable and wanted, its pipe crosses), `not-exportable`
	// (the peer holds it without export), `opening` (subscribing to its pipe), `absent` (the
	// peer holds no such node, or the link is down).
	Probes []PeerProbeStatus `json:"probes,omitempty"`
	// Live: the link is up and both its control subscriptions (the peer's ctl and svc/) are past
	// their replay, so what the peer publishes now crosses as it happens. Reconciled: the link is
	// up, the peer's probe list was applied this session and no wanted pipe is still opening, so
	// the mirrored lifecycle and pipes stand for the peer's state. Degraded is their conjunction
	// denied: lifted only when the flow is live and the state reconciled (criterion 7 of ix#408).
	// Jumps counts the catch-ups the link gave up on since it was made, each marked `jumped` on
	// the peer's mirrored ctl; absent while zero.
	Live       bool   `json:"live"`
	Reconciled bool   `json:"reconciled"`
	Degraded   bool   `json:"degraded"`
	Jumps      uint64 `json:"jumps,omitempty"`
}

type PeerProbeStatus struct {
	Name  string `json:"name"`
	State string `json:"state"`
	// Dropped: records of this pipe the link discarded for want of room in its pipe queue, since
	// the link was made (across sessions); each run of drops is marked on the mirrored pipe
	// before the record that followed (`link.gap`, reason `dropped`). Absent while zero.
	Dropped uint64 `json:"dropped,omitempty"`
	// CreatedSeq and SrcEpoch: the identity of the incarnation the link mirrors or declined under
	// this name, the seq and boot of its probe.created on the peer (from the peer's cursor-bound
	// list or the followed probe.created; S2 slice 4): a consumer that reads this, then the pipe,
	// keys what it reads from then on `origin:src_epoch:node:created_seq` without having read
	// that event (a reading, not a point of the stream: a frame read before it is told apart by
	// the mirrored probe.created or the list with its cursor). Absent while the name is absent.
	CreatedSeq uint64 `json:"created_seq,omitempty"`
	SrcEpoch   string `json:"src_epoch,omitempty"`
}

// StatusReply summarises the generations. Every generation <= SettledGeneration has
// ended, converged or diverged, and Failed lists the diverged ones since boot, lowest
// generation first: g has failed iff Failed names it, and is done iff g <=
// SettledGeneration and Failed does not name it. ObservedGeneration (the highest that
// converged) and Diverged (the highest that failed) stay for older readers; on their own
// they let a later converged generation hide an earlier pending or failed one.
type StatusReply struct {
	Name               string          `json:"name"`
	Epoch              string          `json:"epoch"` // the boot these generations belong to
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
	r := StatusReply{Name: p.name, Epoch: p.hub.Epoch(), Generation: p.gen,
		ObservedGeneration: p.observed, SettledGeneration: p.settled}
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

// ApplyReply names the new generation and the boot it belongs to: a wait
// that carries both can tell a restart from a slow apply.
type ApplyReply struct {
	Generation uint64   `json:"generation"`
	Epoch      string   `json:"epoch"`
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
	if d.Schema != "" && d.Schema != desiredSchema {
		return badRequest("desired-state schema %q is not %q", d.Schema, desiredSchema)
	}
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
			// Running, or on its way (a start the supervisor has not made yet, a restart in
			// its backoff): nothing to start. Only a stopped or failed service, or one being
			// stopped, takes a start.
			if desired == desiredRunning && (state == stateRunning || state == stateStarting || state == stateRestarting) {
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
	p.applyMu.Lock()
	defer p.applyMu.Unlock()
	// Under applyMu, as Shutdown closes quit under it: no apply pre-marks a start or queues
	// a want on a supervisor that is being reaped.
	select {
	case <-p.quit:
		return ApplyReply{}, nil, &opError{status: 503, msg: "shutting down"}
	default:
	}
	head := p.hub.Head()
	cur := p.hub.Subscribe("ctl", &head, p.hub.Epoch())
	actions, noop := p.plan(d)
	p.mu.Lock()
	p.gen++
	g := p.gen
	p.mu.Unlock()
	p.journal.record(g, d, actions)
	// A noop asks nothing of the supervisor: a `running` want on a service already on its
	// way would launch it under this generation if it exited before the want was read, a run
	// no action announced; a `stopped` want on a stopped service would answer a
	// service.stopped nobody waits for. Only `desired` follows the document.
	skip := map[string]bool{}
	for _, n := range noop {
		skip[n] = true
		p.svcs[n].setDesired(d.Services[n].State)
	}
	for _, n := range d.names() {
		if !skip[n] {
			p.svcs[n].want(d.Services[n].State, g)
		}
	}
	done := make(chan applyResult, 1)
	go func() { done <- p.await(g, actions, cur) }()
	return ApplyReply{Generation: g, Epoch: p.hub.Epoch(), Actions: actions}, done, nil
}

// await watches ctl until every action is observed, then publishes the
// outcome. A start is observed by the service.started or service.failed
// stamped with this generation (an earlier run's settles nothing), a stop by
// the service.stopped stamped with it (the tree is gone by then; another
// stop's, or a run's own end, settles nothing).
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
			// Events were lost: fall back on the observed state. A start's failure is asked
			// of the service by generation, not read off its state: a start queued behind
			// may own the state again by now (`settle`).
			for n, op := range pending {
				st := p.svcs[n].status()
				reason, failed := p.svcs[n].failure(g)
				switch {
				case op == "start" && failed:
					res = applyResult{Generation: g, Service: n, Reason: reason}
					pending = nil
				case op == "start" && st.State == stateRunning && st.Generation == g,
					op == "stop" && st.State == stateStopped:
					delete(pending, n)
				}
				if res.Reason != "" {
					break
				}
			}
			continue
		}
		for _, env := range envs {
			var data struct {
				Service    string `json:"service"`
				Reason     string `json:"reason"`
				Generation uint64 `json:"generation"`
			}
			json.Unmarshal(env.Ev.Data, &data)
			op, ok := pending[data.Service]
			if !ok {
				continue
			}
			switch {
			case env.Ev.Kind == "service.started" && op == "start" && data.Generation == g,
				env.Ev.Kind == "service.stopped" && op == "stop" && data.Generation == g:
				delete(pending, data.Service)
			case env.Ev.Kind == "service.failed" && op == "start" && data.Generation == g:
				res = applyResult{Generation: g, Service: data.Service, Reason: data.Reason}
				pending = nil
			}
			if res.Reason != "" {
				break
			}
		}
	}
	// This generation has settled: its starts' failures, read or not, are of no use now.
	for _, a := range actions {
		if a.Op == "start" {
			p.svcs[a.Service].forget(g)
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
		// Under applyMu: an apply that passed its quit check has queued its wants before
		// the supervisors are cancelled, and none passes it afterwards.
		p.applyMu.Lock()
		p.RequestShutdown()
		p.applyMu.Unlock()
		p.probes.shutdown()
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

// journalSchema names the journal's line format; a reader checks it on the
// boot line before trusting the lines that follow.
const journalSchema = "ixctl-journal/0"

// journal appends every applied document and every ctl event, as JSON lines.
// Each boot opens with a header line carrying the instance epoch, so the
// generations and seqs that restart at 1 read as (epoch, generation) and
// (epoch, seq) across the file.
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

// boot writes the header line that opens this boot's section of the file.
func (j *journal) boot(name, epoch string) {
	if j == nil {
		return
	}
	b, _ := json.Marshal(map[string]any{
		"t": time.Now().UnixMilli(), "kind": "boot", "schema": journalSchema, "name": name, "epoch": epoch})
	j.write(b)
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

// SetProbeBudget replaces the probe tree's budget (D10, ix#416) before any node exists: what a
// subtree may declare to emit and what the whole tree may, in records per second. Refused when
// either is not positive, the tree's ceiling is under the subtree's, or a node already exists.
func (p *Plane) SetProbeBudget(maxRatePerS, treeRatePerS int64) error {
	return p.probes.setBudget(probeBudget{maxRate: maxRatePerS, treeRate: treeRatePerS})
}

// SetDeltaJournal names the JSONL journal the `delta@1` kind tails (`ixctl up --delta-journal`).
// Without it the kind is listed by `probe.kinds` and refused at create, naming the flag.
func (p *Plane) SetDeltaJournal(path string) {
	p.probes.setDeltaJournal(path)
}
