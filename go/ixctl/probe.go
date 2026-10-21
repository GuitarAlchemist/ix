package main

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"sync"
	"time"
)

// Probes are the plane's own sensors: versioned kinds compiled into the binary, run as
// nodes inside the plane's process and composed into a tree whose pipes are hub channels.
// A node of kind `mux@1` has no sampler; it multiplexes its children. Every record a
// node emits goes on `probe/<node>` as a `sample` event, after the node's filter.

// probeKind is one compiled, versioned probe: what it measures and how.
type probeKind struct {
	Name    string   // the kind's name, `cpu`
	Version int      // its version, `1`: together `cpu@1`
	Fields  []string // the fields of a sample, in order; nil for a mux
	Doc     string   // one line on what it measures and where it reads it
	// newSampler makes the kind's sampler for one node; nil for a mux, whose input is its
	// children. A sampler is the port an OS reading sits behind: the tree only calls sample.
	newSampler func() sampler
}

// sampler is one node's source of records: called once per interval, on the node's own
// goroutine, never concurrently with itself.
type sampler interface {
	sample() (map[string]any, error)
}

// heartbeat counts its own ticks: the kind tests and demonstrations use.
type heartbeat struct{ n int64 }

func (h *heartbeat) sample() (map[string]any, error) {
	h.n++
	return map[string]any{"n": h.n}, nil
}

// errNoDelta is the first cpu sample: a busy share needs two readings, so nothing is emitted.
var errNoDelta = errors.New("no interval yet")

// delta turns two cumulative readings (idle and total, in any unit) into the busy share of
// the interval between them. The first call primes and emits nothing.
func (c *cpuProbe) delta(idle, total uint64) (map[string]any, error) {
	if !c.primed {
		c.idle, c.total, c.primed = idle, total, true
		return nil, errNoDelta
	}
	di, dt := float64(idle-c.idle), float64(total-c.total)
	c.idle, c.total = idle, total
	busy := 0.0
	if dt > 0 {
		busy = 100 * (dt - di) / dt
	}
	if busy < 0 {
		busy = 0
	}
	if busy > 100 {
		busy = 100
	}
	return map[string]any{"busy_pct": busy, "cores": cpuCores()}, nil
}

// memFields is a memory sample from its two readings, in bytes.
func memFields(total, avail uint64) map[string]any {
	used := 0.0
	if total > 0 {
		used = 100 * float64(total-avail) / float64(total)
	}
	return map[string]any{"total_bytes": total, "avail_bytes": avail, "used_pct": used}
}

// String is the kind as the protocol names it, `name@version`.
func (k probeKind) String() string { return k.Name + "@" + itoa(k.Version) }

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// probeKinds is the registry: the only kinds this binary can run. A kind is added here
// with its sampler; a release ships this list as its manifest (`ixctl probes`).
var probeKinds = []probeKind{
	{Name: "heartbeat", Version: 1, Fields: []string{"n"},
		Doc:        "counts its own ticks: n, from 1; the kind tests and demonstrations use",
		newSampler: func() sampler { return &heartbeat{} }},
	{Name: "cpu", Version: 1, Fields: []string{"busy_pct", "cores"},
		Doc:        "the host's processor busy share over the last interval, 0..100, and its logical cores (GetSystemTimes on Windows, /proc/stat on Linux); the first interval emits nothing",
		newSampler: func() sampler { return &cpuProbe{} }},
	{Name: "mem", Version: 1, Fields: []string{"total_bytes", "avail_bytes", "used_pct"},
		Doc:        "the host's physical memory, total and available, and the used share 0..100 (GlobalMemoryStatusEx on Windows, /proc/meminfo on Linux)",
		newSampler: func() sampler { return memProbe{} }},
	{Name: "mux", Version: 1, Fields: nil,
		Doc: "no sampler: multiplexes its children's records, each stamped with its source node as src"},
}

// lookupKind finds a kind by `name@version`.
func lookupKind(s string) (probeKind, bool) {
	name, ver, ok := strings.Cut(s, "@")
	if !ok {
		return probeKind{}, false
	}
	for _, k := range probeKinds {
		if k.Name == name && itoa(k.Version) == ver {
			return k, true
		}
	}
	return probeKind{}, false
}

// The bounds the probe tree applies. Every one of them is reported by `probe.kinds`
// under `limits`, by these names, and every refusal names the bound it hit: the system
// knows its limits and says them.
const (
	probeMaxNodes   = 256       // nodes in one plane's tree, every kind counted
	probeMaxDepth   = 8         // a leaf under at most this many muxes
	probeMinEveryMs = 10        // the shortest sampling interval
	probeMaxEveryMs = 3_600_000 // the longest, an hour
	probeMaxTTLMs   = maxWallMs // the longest lease, what a time.Duration holds
	probeMaxWhere   = 16        // terms in a filter's where
	probeMaxKeep    = 32        // fields in a filter's keep
)

// probeLimits is the `limits` member of `probe.kinds`.
func probeLimits() map[string]int64 {
	return map[string]int64{
		"max_nodes": probeMaxNodes, "max_depth": probeMaxDepth,
		"min_every_ms": probeMinEveryMs, "max_every_ms": probeMaxEveryMs, "max_ttl_ms": probeMaxTTLMs,
		"ring": ringSize, "max_where": probeMaxWhere, "max_keep": probeMaxKeep,
	}
}

// NodeSpec is one node of the tree as a client declares it.
type NodeSpec struct {
	Name    string  `json:"name"`
	Kind    string  `json:"kind"`               // `name@version`, one of the registry
	EveryMs int64   `json:"every_ms,omitempty"` // the sampling interval; ignored by a mux
	Parent  string  `json:"parent,omitempty"`   // the mux this node feeds, or none
	Filter  *Filter `json:"filter,omitempty"`   // what the node lets through, of what it emits
	TTLMs   int64   `json:"ttl_ms,omitempty"`   // the lease from a start, renewed by probe.touch; 0 is none
}

// Filter is the smallest filter that composes: every `where` term must hold, only the
// `keep` fields pass (all when empty), and one record in `every` goes through.
type Filter struct {
	Where []Term   `json:"where,omitempty"`
	Keep  []string `json:"keep,omitempty"`
	Every int      `json:"every,omitempty"`
}

// Term compares one field of a record with a value.
type Term struct {
	Field string `json:"field"`
	Op    string `json:"op"` // == != < <= > >=
	Value any    `json:"value"`
}

var termOps = map[string]bool{"==": true, "!=": true, "<": true, "<=": true, ">": true, ">=": true}

// validate checks a filter against the bounds, naming the one it hits.
func (f *Filter) validate() error {
	if f == nil {
		return nil
	}
	if len(f.Where) > probeMaxWhere {
		return badRequest("filter has %d where terms; max_where is %d", len(f.Where), probeMaxWhere)
	}
	if len(f.Keep) > probeMaxKeep {
		return badRequest("filter keeps %d fields; max_keep is %d", len(f.Keep), probeMaxKeep)
	}
	for i, t := range f.Where {
		if t.Field == "" {
			return badRequest("where[%d] names no field", i)
		}
		if !termOps[t.Op] {
			return badRequest("where[%d]: unknown op %q (== != < <= > >=)", i, t.Op)
		}
	}
	if f.Every < 0 {
		return badRequest("filter every %d must be positive (one record in every)", f.Every)
	}
	if f.Every == 0 && f.Keep == nil && f.Where == nil {
		return badRequest("filter is empty: every must be at least 1, or name where or keep")
	}
	return nil
}

// matches reports whether every where term holds on rec. A field the record lacks, or
// whose type the value's does not match, fails the term: nothing is coerced.
func (f *Filter) matches(rec map[string]any) bool {
	if f == nil {
		return true
	}
	for _, t := range f.Where {
		v, ok := rec[t.Field]
		if !ok || !t.holds(v) {
			return false
		}
	}
	return true
}

func (t Term) holds(v any) bool {
	if a, ok := num(v); ok {
		b, ok := num(t.Value)
		return ok && cmp(a-b, t.Op)
	}
	switch a := v.(type) {
	case string:
		b, ok := t.Value.(string)
		return ok && cmp(float64(strings.Compare(a, b)), t.Op)
	case bool:
		b, ok := t.Value.(bool)
		switch t.Op {
		case "==":
			return ok && a == b
		case "!=":
			return ok && a != b
		}
	}
	return false
}

// cmp reads the sign of a difference against an op.
func cmp(d float64, op string) bool {
	switch op {
	case "==":
		return d == 0
	case "!=":
		return d != 0
	case "<":
		return d < 0
	case "<=":
		return d <= 0
	case ">":
		return d > 0
	case ">=":
		return d >= 0
	}
	return false
}

// num reads any numeric value (a sampler's int64, JSON's float64, a test's int) as float64.
func num(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case uint64:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	}
	return 0, false
}

// apply runs the whole filter on a record: where, then every (passed counts the records
// where let through), then keep. It returns the fields to emit, or false.
func (f *Filter) apply(fields map[string]any, passed *uint64) (map[string]any, bool) {
	if f == nil {
		return fields, true
	}
	if !f.matches(fields) {
		return nil, false
	}
	*passed++
	if f.Every > 1 && *passed%uint64(f.Every) != 0 {
		return nil, false
	}
	if len(f.Keep) == 0 {
		return fields, true
	}
	kept := make(map[string]any, len(f.Keep))
	for _, k := range f.Keep {
		if v, ok := fields[k]; ok {
			kept[k] = v
		}
	}
	return kept, true
}

// probeNode is one node of the tree, live.
type probeNode struct {
	spec     NodeSpec
	kind     probeKind
	children []string
	state    string // stopped | running
	seq      uint64 // records emitted, after the filter: a reader counts its losses by it
	passed   uint64 // records the filter's where let through, for every
	errors   uint64 // samples the sampler failed, emitted with an `error` field

	cancel context.CancelFunc // ends the run; nil when stopped
	done   chan struct{}      // closed when the run's goroutine has returned

	expires time.Time   // when the running lease ends; zero without one
	lease   *time.Timer // fires expire at `expires`; nil without a running lease
}

// leaseLeft is what is left of a running lease, in milliseconds, or nil.
func (n *probeNode) leaseLeft(now time.Time) *int64 {
	if n.state != probeRunning || n.spec.TTLMs == 0 {
		return nil
	}
	left := n.expires.Sub(now).Milliseconds()
	if left < 0 {
		left = 0
	}
	return &left
}

// renew starts or restarts the node's lease from now; the caller holds the lock.
func (t *probeTree) renew(n *probeNode, now time.Time) {
	if n.spec.TTLMs == 0 {
		return
	}
	if n.lease != nil {
		n.lease.Stop()
	}
	ttl := time.Duration(n.spec.TTLMs) * time.Millisecond
	n.expires = now.Add(ttl)
	name := n.spec.Name
	n.lease = time.AfterFunc(ttl, func() { t.expire(name) })
}

// expire ends a node whose lease ran out: stopped with reason `expired`, then removed with
// its subtree. A timer that fires after a touch moved the lease, or after a stop or remove,
// finds nothing due and does nothing.
func (t *probeTree) expire(name string) {
	t.mu.Lock()
	n := t.nodes[name]
	due := n != nil && n.state == probeRunning && n.spec.TTLMs > 0 && !time.Now().Before(n.expires)
	t.mu.Unlock()
	if !due {
		return
	}
	t.remove(name, "expired")
}

// touch renews a node's lease from now and answers what is left of it.
func (t *probeTree) touch(name string) (any, error) {
	t.mu.Lock()
	n := t.nodes[name]
	if n == nil {
		t.mu.Unlock()
		return nil, badRequest("node %q does not exist", name)
	}
	if n.spec.TTLMs == 0 {
		t.mu.Unlock()
		return nil, badRequest("node %q has no lease (ttl_ms 0)", name)
	}
	now := time.Now()
	if n.state == probeRunning {
		t.renew(n, now)
	}
	reply := map[string]any{"node": name, "state": n.state, "ttl_ms": n.spec.TTLMs}
	if left := n.leaseLeft(now); left != nil {
		reply["expires_in_ms"] = *left
	}
	t.mu.Unlock()
	return reply, nil
}

// remove ends a node and its whole subtree, descendants first: each running one is stopped
// (the node with `reason`, a descendant with `parent-removed`), then forgotten, with a
// `probe.removed` event each.
func (t *probeTree) remove(name, reason string) (any, error) {
	t.mu.Lock()
	if t.nodes[name] == nil {
		t.mu.Unlock()
		return nil, badRequest("node %q does not exist", name)
	}
	var order []string // post-order: a child before its parent
	var walk func(string)
	walk = func(nm string) {
		for _, c := range t.nodes[nm].children {
			walk(c)
		}
		order = append(order, nm)
	}
	walk(name)
	t.mu.Unlock()
	for _, nm := range order {
		why := reason
		if nm != name {
			why = "parent-removed"
		}
		t.stop(nm, why)
		t.mu.Lock()
		n := t.nodes[nm]
		if n == nil { // removed meanwhile by another op
			t.mu.Unlock()
			continue
		}
		delete(t.nodes, nm)
		for i, o := range t.order {
			if o == nm {
				t.order = append(t.order[:i], t.order[i+1:]...)
				break
			}
		}
		if parent := t.nodes[n.spec.Parent]; parent != nil {
			for i, c := range parent.children {
				if c == nm {
					parent.children = append(parent.children[:i], parent.children[i+1:]...)
					break
				}
			}
		}
		t.mu.Unlock()
		t.event("probe.removed", n, map[string]any{"reason": why})
	}
	return map[string]any{"node": name, "removed": len(order)}, nil
}

const (
	probeStopped = "stopped"
	probeRunning = "running"
)

// event publishes a node's lifecycle on ctl.
func (t *probeTree) event(kind string, n *probeNode, extra map[string]any) {
	d := map[string]any{"node": n.spec.Name, "kind": n.kind.String()}
	for k, v := range extra {
		d[k] = v
	}
	t.hub.Publish("ctl", event(kind, d))
}

// start runs a node; a second start is a noop. Only a node with a sampler runs on its own:
// a mux runs as its children feed it.
func (t *probeTree) start(name string) (any, error) {
	t.mu.Lock()
	n := t.nodes[name]
	if n == nil {
		t.mu.Unlock()
		return nil, badRequest("node %q does not exist", name)
	}
	if n.state == probeRunning {
		t.mu.Unlock()
		return map[string]any{"node": name, "state": probeRunning, "noop": true}, nil
	}
	if n.kind.newSampler == nil && n.kind.Name != "mux" {
		t.mu.Unlock()
		return nil, badRequest("kind %s has no sampler on this platform", n.kind)
	}
	n.state = probeRunning
	if n.kind.newSampler != nil {
		ctx, cancel := context.WithCancel(context.Background())
		n.cancel, n.done = cancel, make(chan struct{})
		go t.run(ctx, n, n.kind.newSampler(), n.done)
	}
	now := time.Now()
	t.renew(n, now)
	reply := map[string]any{"node": name, "state": probeRunning}
	if left := n.leaseLeft(now); left != nil {
		reply["expires_in_ms"] = *left
	}
	t.mu.Unlock()
	t.event("probe.started", n, nil)
	return reply, nil
}

// run is a leaf's loop: one sample per interval until the context ends.
func (t *probeTree) run(ctx context.Context, n *probeNode, s sampler, done chan struct{}) {
	defer close(done)
	tick := time.NewTicker(time.Duration(n.spec.EveryMs) * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		fields, err := s.sample()
		if errors.Is(err, errNoDelta) {
			continue
		}
		if err != nil {
			fields = map[string]any{"error": err.Error()}
		}
		t.emit(n, fields, err != nil)
	}
}

// emit publishes one record of a node on its pipe (the fields that pass its filter, stamped
// with the node, its kind and the record's seq), then hands it to the node's parent when
// that mux is running, stamped `src` with this node: the multiplexing, synchronous and
// bounded by the children's own rates (Publish never blocks; the ring is the one place a
// slow reader loses records, and it says so with a gap).
func (t *probeTree) emit(n *probeNode, fields map[string]any, failed bool) {
	t.mu.Lock()
	if failed {
		n.errors++
	}
	fields, ok := n.spec.Filter.apply(fields, &n.passed)
	if !ok {
		t.mu.Unlock()
		return
	}
	n.seq++
	rec := make(map[string]any, len(fields)+4)
	for k, v := range fields {
		rec[k] = v
	}
	rec["node"], rec["kind"], rec["seq"] = n.spec.Name, n.kind.String(), n.seq
	parent := t.nodes[n.spec.Parent]
	if parent != nil && parent.state != probeRunning {
		parent = nil
	}
	t.mu.Unlock()
	t.hub.Publish("probe/"+n.spec.Name, event("sample", rec))
	if parent != nil {
		up := make(map[string]any, len(fields)+1)
		for k, v := range fields {
			up[k] = v
		}
		up["src"] = n.spec.Name
		t.emit(parent, up, false)
	}
}

// stop ends a node's run with a reason (`requested` from the op); a stopped node is a noop.
func (t *probeTree) stop(name, reason string) (any, error) {
	t.mu.Lock()
	n := t.nodes[name]
	if n == nil {
		t.mu.Unlock()
		return nil, badRequest("node %q does not exist", name)
	}
	if n.state != probeRunning {
		t.mu.Unlock()
		return map[string]any{"node": name, "state": n.state, "noop": true}, nil
	}
	cancel, done := n.cancel, n.done
	n.cancel, n.done, n.state = nil, nil, probeStopped
	if n.lease != nil {
		n.lease.Stop()
		n.lease, n.expires = nil, time.Time{}
	}
	t.mu.Unlock()
	if cancel != nil { // a mux has no run of its own
		cancel()
		<-done
	}
	t.event("probe.stopped", n, map[string]any{"reason": reason})
	return map[string]any{"node": name, "state": probeStopped}, nil
}

// shutdown stops every running node with the reason `shutdown`, children before
// parents (a child is always created after its parent, so reverse creation order
// does it), so no run outlives the plane and every pipe ends with its stop on ctl.
func (t *probeTree) shutdown() {
	t.mu.Lock()
	var running []string
	for i := len(t.order) - 1; i >= 0; i-- {
		if t.nodes[t.order[i]].state == probeRunning {
			running = append(running, t.order[i])
		}
	}
	t.mu.Unlock()
	for _, name := range running {
		t.stop(name, "shutdown")
	}
}

// probeTree is the plane's tree of nodes: one per plane, every op under one lock.
type probeTree struct {
	hub   *Hub
	mu    sync.Mutex
	nodes map[string]*probeNode
	order []string // creation order, for list
}

func newProbeTree(hub *Hub) *probeTree {
	return &probeTree{hub: hub, nodes: map[string]*probeNode{}}
}

// create admits a node, stopped. Every refusal names the bound or the rule it hit.
func (t *probeTree) create(spec *NodeSpec) (any, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !nameRE.MatchString(spec.Name) {
		return nil, badRequest("node name %q must match %s", spec.Name, nameRE)
	}
	if _, dup := t.nodes[spec.Name]; dup {
		return nil, badRequest("node %q already exists", spec.Name)
	}
	if len(t.nodes) >= probeMaxNodes {
		return nil, badRequest("the tree holds %d nodes; max_nodes is %d", len(t.nodes), probeMaxNodes)
	}
	kind, ok := lookupKind(spec.Kind)
	if !ok {
		return nil, badRequest("unknown kind %q (probe.kinds lists them)", spec.Kind)
	}
	if kind.Name == "mux" {
		spec.EveryMs = 0
	} else {
		switch {
		case spec.EveryMs < probeMinEveryMs:
			return nil, badRequest("every_ms %d is under min_every_ms %d", spec.EveryMs, probeMinEveryMs)
		case spec.EveryMs > probeMaxEveryMs:
			return nil, badRequest("every_ms %d is over max_every_ms %d", spec.EveryMs, probeMaxEveryMs)
		}
	}
	if spec.TTLMs < 0 || spec.TTLMs > probeMaxTTLMs {
		return nil, badRequest("ttl_ms %d must be 0..max_ttl_ms %d", spec.TTLMs, probeMaxTTLMs)
	}
	if err := spec.Filter.validate(); err != nil {
		return nil, err
	}
	depth := 0
	if spec.Parent != "" {
		parent := t.nodes[spec.Parent]
		if parent == nil {
			return nil, badRequest("parent %q does not exist", spec.Parent)
		}
		if parent.kind.Name != "mux" {
			return nil, badRequest("parent %q is a %s, not a mux", spec.Parent, parent.kind)
		}
		for p := parent; p != nil; p = t.nodes[p.spec.Parent] {
			depth++
		}
		if depth >= probeMaxDepth {
			return nil, badRequest("node would sit at depth %d under %q; max_depth is %d", depth, spec.Parent, probeMaxDepth)
		}
	}
	n := &probeNode{spec: *spec, kind: kind, state: probeStopped}
	t.nodes[spec.Name] = n
	t.order = append(t.order, spec.Name)
	extra := map[string]any{}
	if spec.Parent != "" {
		t.nodes[spec.Parent].children = append(t.nodes[spec.Parent].children, spec.Name)
		extra["parent"] = spec.Parent
	}
	t.event("probe.created", n, extra)
	return map[string]any{"node": spec.Name, "kind": kind.String(), "state": n.state, "depth": depth}, nil
}

// list answers `probe.list`: every node in creation order, with its usage against the bounds.
func (t *probeTree) list() any {
	t.mu.Lock()
	defer t.mu.Unlock()
	nodes := make([]map[string]any, 0, len(t.order))
	now := time.Now()
	for _, name := range t.order {
		n := t.nodes[name]
		m := map[string]any{
			"name": name, "kind": n.kind.String(), "state": n.state, "every_ms": n.spec.EveryMs,
			"ttl_ms": n.spec.TTLMs, "seq": n.seq, "errors": n.errors,
			"children": len(n.children),
		}
		if left := n.leaseLeft(now); left != nil {
			m["expires_in_ms"] = *left
		}
		if n.spec.Parent != "" {
			m["parent"] = n.spec.Parent
		}
		if n.spec.Filter != nil {
			m["filter"] = n.spec.Filter
		}
		nodes = append(nodes, m)
	}
	return map[string]any{"nodes": nodes, "count": len(nodes), "max_nodes": probeMaxNodes}
}

// kindsReply answers `probe.kinds`: the registry, sorted by name, and the limits.
func kindsReply() map[string]any {
	kinds := make([]map[string]any, 0, len(probeKinds))
	sorted := append([]probeKind(nil), probeKinds...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].String() < sorted[j].String() })
	for _, k := range sorted {
		fields := k.Fields
		if fields == nil {
			fields = []string{}
		}
		kinds = append(kinds, map[string]any{"kind": k.String(), "version": k.Version, "fields": fields, "doc": k.Doc})
	}
	return map[string]any{"kinds": kinds, "limits": probeLimits(), "version": version}
}
