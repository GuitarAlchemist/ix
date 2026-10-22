package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
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
// goroutine, never concurrently with itself. A sample must return: a stop waits for the read
// in flight so that probe.stopped is the pipe's last word, and the plane's shutdown waits for
// that stop. A kind whose source can hang (an RPC, WMI) bounds its own read.
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
	if dt <= 0 { // the counters did not advance (a tick shorter than the clock's grain): no interval to report
		return nil, errNoDelta
	}
	busy := 100 * (dt - di) / dt
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

// probeBudget is D10 (ix#416): what a subtree may declare to emit, in records per second, the
// worst case summed over its leaves at `create` (1000 / every_ms each; the filter does not lower
// it, a failing sampler's error records bypassing it), whatever the nodes' state; and the tree's
// own ceiling, both compared as exact rationals (a float sum carries 1000.0000000000001 at
// twelve leaves of 12 ms; a floored integer admits 32 leaves of 31 999 ms on a budget of 1). Set at
// `ixctl up`, said in `limits`; a mux may narrow its own with `max_rate_per_s`, never widen it.
// A create that would cross a bound is refused in advance, like `max_nodes`. CPU and RSS are
// measured on the process, never attributed to probes: one process cannot separate them.
type probeBudget struct {
	maxRate  int64 // `max_rate_per_s`: per subtree (a mux and its descendants; a root leaf alone)
	treeRate int64 // `tree_rate_per_s`: the whole tree
}

// defaultProbeBudget is about a quarter of a core, extrapolating gate M2 (1 600 records/s cost
// ~17 % of one on the measured desktop); the gate measures it again.
var defaultProbeBudget = probeBudget{maxRate: 1000, treeRate: 2500}

// probeMaxRatePerS caps a budget: a billion records a second is beyond any plane, and keeps the
// arithmetic in range.
const probeMaxRatePerS = 1_000_000_000

func (b probeBudget) validate() error {
	if b.maxRate <= 0 {
		return fmt.Errorf("max_rate_per_s %d must be positive", b.maxRate)
	}
	if b.treeRate > probeMaxRatePerS {
		return fmt.Errorf("tree_rate_per_s %d is over %d, more than any plane carries", b.treeRate, probeMaxRatePerS)
	}
	if b.treeRate < b.maxRate {
		return fmt.Errorf("tree_rate_per_s %d is under max_rate_per_s %d", b.treeRate, b.maxRate)
	}
	return nil
}

// probeLimits is the `limits` member of `probe.kinds`.
func probeLimits(b probeBudget) map[string]int64 {
	return map[string]int64{
		"max_nodes": probeMaxNodes, "max_depth": probeMaxDepth,
		"min_every_ms": probeMinEveryMs, "max_every_ms": probeMaxEveryMs, "max_ttl_ms": probeMaxTTLMs,
		"ring": ringSize, "max_where": probeMaxWhere, "max_keep": probeMaxKeep,
		"max_rate_per_s": b.maxRate, "tree_rate_per_s": b.treeRate,
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
	// MaxRatePerS narrows the plane's budget for this mux's subtree (D10); a leaf carries none.
	MaxRatePerS int64 `json:"max_rate_per_s,omitempty"`
	// Export marks the node's pipe as available to federation (D4, ix#402): a peer that names
	// this node in its spec mirrors it. An availability, not a request; immutable; false by
	// default. Direct subscription to the pipe is unchanged.
	Export bool `json:"export,omitempty"`
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
		// A value a term can never hold against is refused here, not silently false forever.
		switch v := t.Value.(type) {
		case string:
		case bool:
			if t.Op != "==" && t.Op != "!=" {
				return badRequest("where[%d]: a bool takes == or !=, not %q", i, t.Op)
			}
		default:
			if _, ok := num(v); !ok {
				return badRequest("where[%d]: value must be a number, a string or a bool", i)
			}
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
	kept := make(map[string]any, len(f.Keep)+2)
	for _, k := range f.Keep {
		if v, ok := fields[k]; ok {
			kept[k] = v
		}
	}
	// What the protocol promises on every record survives any projection: the source a mux
	// stamped, and the error a failed sample carries.
	for _, k := range []string{"src", "error"} {
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
	bytes    uint64 // bytes of the records emitted, encoded (D10: measured, not bounded)
	created  uint64 // the seq of its probe.created: the incarnation's identity under its name (S2 slice 4)

	cancel context.CancelFunc // ends the run; nil when stopped
	done   chan struct{}      // closed when the run's goroutine has returned

	stopping bool // a stop is waiting for the run to return: no start until it has
	removing bool // a remove owns this node: no start, touch or child until it is gone

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
// finds nothing due and does nothing: the check and the claim on the subtree happen under
// one lock, so a touch that lands first keeps its node.
func (t *probeTree) expire(name string) {
	t.removeIf(name, "expired", func(n *probeNode) bool {
		return n.state == probeRunning && n.spec.TTLMs > 0 && !time.Now().Before(n.expires)
	})
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
	if n.removing {
		t.mu.Unlock()
		return nil, badRequest("node %q is being removed", name)
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
	return t.removeIf(name, reason, nil)
}

// removeIf is remove when `due` (if given) holds for the node, decided under the lock that
// also claims the whole subtree (`removing`): from then on no start, touch or new child can
// reach those nodes, so stopping and forgetting them outside the lock races nothing.
func (t *probeTree) removeIf(name, reason string, due func(*probeNode) bool) (any, error) {
	t.mu.Lock()
	n := t.nodes[name]
	if n == nil {
		t.mu.Unlock()
		return nil, badRequest("node %q does not exist", name)
	}
	if t.closed { // nothing of the tree is said after shutdown has returned
		t.mu.Unlock()
		return nil, badRequest("the plane is shutting down")
	}
	if n.removing {
		t.mu.Unlock()
		return nil, badRequest("node %q is being removed", name)
	}
	if due != nil && !due(n) {
		t.mu.Unlock()
		return nil, nil
	}
	// The claim is on the nodes themselves, not their names: an overlapping removal (an
	// ancestor's, a descendant's, a lease's) may forget one of them meanwhile and a create may
	// put a new node under the same name; that one is nobody's to stop or forget here.
	var claimed []*probeNode // post-order: a child before its parent
	var walk func(string)
	walk = func(nm string) {
		t.nodes[nm].removing = true
		for _, c := range t.nodes[nm].children {
			walk(c)
		}
		claimed = append(claimed, t.nodes[nm])
	}
	walk(name)
	t.removals++ // shutdown waits for every removal in flight (see finishRemoval)
	t.mu.Unlock()
	defer t.finishRemoval()
	for _, n := range claimed {
		nm := n.spec.Name
		why := reason
		if nm != name {
			why = "parent-removed"
		}
		t.stopIf(nm, why, n)
		t.mu.Lock()
		if t.nodes[nm] != n { // forgotten meanwhile by another op (a replacement is not ours)
			t.mu.Unlock()
			continue
		}
		// A stop by another op may still be in progress (its loop ending, its probe.stopped
		// not out yet): probe.removed waits for it, so ctl says stopped, then removed.
		for n.stopping {
			t.done.Wait()
		}
		if t.nodes[nm] != n { // forgotten meanwhile by an overlapping removal that waited for the same stop
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
		// Under the lock: a create of the same name that follows cannot say probe.created
		// before this probe.removed (which would close the series it opened).
		t.event("probe.removed", n, map[string]any{"reason": why})
		t.mu.Unlock()
	}
	return map[string]any{"node": name, "removed": len(claimed)}, nil
}

// finishRemoval ends a removal once its last probe.removed is out: a shutdown waiting for the
// tree to fall silent may go on.
func (t *probeTree) finishRemoval() {
	t.mu.Lock()
	t.removals--
	t.done.Broadcast()
	t.mu.Unlock()
}

const (
	probeStopped  = "stopped"
	probeRunning  = "running"
	probeStopping = "stopping" // on the list only: a leaf's stop in progress, its probe.stopped not out yet
)

// event publishes a node's lifecycle on ctl and returns the envelope (its seq names the change).
func (t *probeTree) event(kind string, n *probeNode, extra map[string]any) *Envelope {
	d := map[string]any{"node": n.spec.Name, "kind": n.kind.String()}
	for k, v := range extra {
		d[k] = v
	}
	return t.hub.Publish("ctl", event(kind, d))
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
	switch {
	case t.closed:
		t.mu.Unlock()
		return nil, badRequest("the plane is shutting down")
	case n.removing:
		t.mu.Unlock()
		return nil, badRequest("node %q is being removed", name)
	case n.stopping:
		t.mu.Unlock()
		return nil, badRequest("node %q is stopping; start it again once probe.stopped is out", name)
	case n.kind.newSampler == nil && n.kind.Name != "mux":
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
	extra := map[string]any{}
	if left := n.leaseLeft(now); left != nil {
		reply["expires_in_ms"], extra["expires_in_ms"] = *left, *left
	}
	t.event("probe.started", n, extra) // under the lock: ctl says the changes in their order
	t.mu.Unlock()
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
// slow reader loses records, and it says so with a gap). The record is published under the
// tree's lock, with the state check and the seq it was given: a node's records reach its
// pipe in seq order even when several children feed a mux at the same instant, and no
// record of a node follows its probe.stopped (the stop changes the state under the same
// lock). A failed read bypasses the filter: an error is never kept off the pipe by a
// `where` on a measurement or by `every`, and it does not count toward `every`.
func (t *probeTree) emit(n *probeNode, fields map[string]any, failed bool) {
	t.mu.Lock()
	// A record that reaches a node after its stop or removal (a leaf's last tick, a child's
	// sample in flight to a mux) is not emitted: probe.stopped is the last thing a pipe says.
	if t.nodes[n.spec.Name] != n || n.state != probeRunning {
		t.mu.Unlock()
		return
	}
	if failed {
		n.errors++
	} else {
		var ok bool
		if fields, ok = n.spec.Filter.apply(fields, &n.passed); !ok {
			t.mu.Unlock()
			return
		}
	}
	n.seq++
	rec := make(map[string]any, len(fields)+4)
	for k, v := range fields {
		rec[k] = v
	}
	rec["node"], rec["kind"], rec["seq"] = n.spec.Name, n.kind.String(), n.seq
	if data, err := json.Marshal(rec); err == nil {
		n.bytes += uint64(len(data))
	}
	parent := t.nodes[n.spec.Parent]
	if parent != nil && parent.state != probeRunning {
		parent = nil
	}
	t.hub.Publish("probe/"+n.spec.Name, event("sample", rec))
	t.mu.Unlock()
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
	return t.stopIf(name, reason, nil)
}

// stopIf is stop when the node under `name` is still `want` (a removal stops the node it
// claimed, never a replacement created under the same name meanwhile); nil wants any.
func (t *probeTree) stopIf(name, reason string, want *probeNode) (any, error) {
	t.mu.Lock()
	n := t.nodes[name]
	if n == nil || (want != nil && n != want) {
		t.mu.Unlock()
		return nil, badRequest("node %q does not exist", name)
	}
	if n.state != probeRunning {
		state := n.state // read under the lock: a concurrent start writes it (CI's -race on the order test)
		t.mu.Unlock()
		return map[string]any{"node": name, "state": state, "noop": true}, nil
	}
	cancel, done := n.cancel, n.done
	n.cancel, n.done, n.state = nil, nil, probeStopped
	if n.lease != nil {
		n.lease.Stop()
		n.lease, n.expires = nil, time.Time{}
	}
	reply := map[string]any{"node": name, "state": probeStopped}
	if cancel == nil {
		// A mux has no run of its own to wait for: its probe.stopped is said under the lock,
		// after its last record and in order with the other changes of its state.
		t.event("probe.stopped", n, map[string]any{"reason": reason})
		t.mu.Unlock()
		return reply, nil
	}
	// A leaf's probe.stopped waits for its loop's last tick, outside the lock; until it is out
	// the node is `stopping` (so says the list): a start is refused and a remove waits
	// (finishStop).
	n.stopping = true
	t.mu.Unlock()
	cancel()
	<-done
	t.finishStop(n, reason)
	return reply, nil
}

// finishStop ends a leaf's stop once its run is over: its probe.stopped is said and the stop
// ended in one step under the lock, so a list never says `stopping` once ctl said stopped (S2
// slice 4: the list is exactly what ctl says up to its cursor); the node may start again, and a
// remove that met the stop in progress may go on (it waits on t.done).
func (t *probeTree) finishStop(n *probeNode, reason string) {
	t.mu.Lock()
	t.event("probe.stopped", n, map[string]any{"reason": reason})
	n.stopping = false
	t.done.Broadcast()
	t.mu.Unlock()
}

// shutdown closes admission (no create or start from here on), then stops every running
// node with the reason `shutdown`, children before parents (a child is always created after
// its parent, so reverse creation order does it), so no run outlives the plane and every
// pipe ends with its stop on ctl.
func (t *probeTree) shutdown() {
	t.mu.Lock()
	t.closed = true
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
	// A leaf another op was already stopping is not in the snapshot (its state is stopped,
	// its probe.stopped not out yet), and a removal in flight has its probe.removed to say:
	// wait for both, so nothing of the tree is said after shutdown has returned (a new remove
	// is refused once closed, like a create or a start).
	t.mu.Lock()
	for t.anyStopping() || t.removals > 0 {
		t.done.Wait()
	}
	t.mu.Unlock()
}

// anyStopping says whether a leaf's stop is still in progress; under t.mu.
func (t *probeTree) anyStopping() bool {
	for _, n := range t.nodes {
		if n.stopping {
			return true
		}
	}
	return false
}

// probeTree is the plane's tree of nodes: one per plane, every op under one lock.
type probeTree struct {
	hub   *Hub
	mu    sync.Mutex
	done  *sync.Cond // signalled when a stop or a removal in progress has said its last word
	nodes map[string]*probeNode
	order []string // creation order, for list

	removals int  // removals in flight (claimed, their probe.removed not all out)
	closed   bool // the plane is shutting down: nothing is created, started or removed any more

	budget probeBudget // D10: what a subtree, and the tree, may declare to emit
}

func newProbeTree(hub *Hub) *probeTree {
	t := &probeTree{hub: hub, nodes: map[string]*probeNode{}, budget: defaultProbeBudget}
	t.done = sync.NewCond(&t.mu)
	return t
}

// setBudget replaces the plane's budget before any node exists (that is the gate, not the
// plane's start): nothing declared can end up over it. The caller holds no lock.
func (t *probeTree) setBudget(b probeBudget) error {
	if err := b.validate(); err != nil {
		return err
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.nodes) > 0 {
		return fmt.Errorf("the tree holds %d nodes: the budget is set before any", len(t.nodes))
	}
	t.budget = b
	return nil
}

// budgetNow is the budget under the lock, for `probe.kinds`.
func (t *probeTree) budgetNow() probeBudget {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.budget
}

// leafRate is what a leaf declares to emit, in records per second, as an exact rational: the
// worst case, 1000 / every_ms. The filter does not lower it: a failing sampler emits an error
// record every interval, and errors bypass `where` and `every` (they must reach the pipe).
// Admission compares exact sums: a float sum carries 1000.0000000000001 at twelve leaves of
// 12 ms and refuses the twelfth; a floored integer admits 32 leaves of 31 999 ms on a budget of 1.
func leafRate(spec *NodeSpec) *big.Rat {
	if spec.EveryMs <= 0 {
		return new(big.Rat)
	}
	return big.NewRat(1000, spec.EveryMs)
}

// over says whether have + add exceeds a budget, exactly.
func over(have, add *big.Rat, budget int64) bool {
	return new(big.Rat).Add(have, add).Cmp(big.NewRat(budget, 1)) > 0
}

// subtreeRate sums the declared rates of a node and its descendants; the caller holds the lock.
func (t *probeTree) subtreeRate(name string) *big.Rat {
	r := new(big.Rat)
	n := t.nodes[name]
	if n == nil {
		return r
	}
	if n.kind.Name != "mux" {
		r.Set(leafRate(&n.spec))
	}
	for _, c := range n.children {
		r.Add(r, t.subtreeRate(c))
	}
	return r
}

// treeRate sums every root's subtree; the caller holds the lock.
func (t *probeTree) treeRate() *big.Rat {
	r := new(big.Rat)
	for _, n := range t.nodes {
		if n.spec.Parent == "" {
			r.Add(r, t.subtreeRate(n.spec.Name))
		}
	}
	return r
}

// budgetOf is the budget a node's subtree is held to: the nearest `max_rate_per_s` on it or
// above it, else the plane's; the caller holds the lock.
func (t *probeTree) budgetOf(n *probeNode) int64 {
	for a := n; a != nil; a = t.nodes[a.spec.Parent] {
		if a.spec.MaxRatePerS > 0 {
			return a.spec.MaxRatePerS
		}
	}
	return t.budget.maxRate
}

// ratePerS is a declared rate as a reply says it: records per second, to the thousandth.
func ratePerS(r *big.Rat) float64 {
	f, _ := r.Float64()
	return math.Round(f*1000) / 1000
}

// fmtRate is a declared rate as a refusal says it, to the millionth: a bound crossed by a hair
// is still named.
func fmtRate(r *big.Rat) string {
	f, _ := r.Float64()
	return strconv.FormatFloat(math.Round(f*1_000_000)/1_000_000, 'f', -1, 64)
}

// create admits a node, stopped. Every refusal names the bound or the rule it hit.
func (t *probeTree) create(spec *NodeSpec) (any, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil, badRequest("the plane is shutting down")
	}
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
		if parent.removing {
			return nil, badRequest("parent %q is being removed", spec.Parent)
		}
		for p := parent; p != nil; p = t.nodes[p.spec.Parent] {
			depth++
		}
		if depth > probeMaxDepth { // depth counts the muxes above: max_depth of them is allowed
			return nil, badRequest("node would sit at depth %d under %q; max_depth is %d", depth, spec.Parent, probeMaxDepth)
		}
	}
	if kind.Name == "mux" {
		if spec.MaxRatePerS < 0 {
			return nil, badRequest("max_rate_per_s %d must be positive", spec.MaxRatePerS)
		}
		if spec.MaxRatePerS > 0 {
			ceiling, holder := t.budget.maxRate, "the plane"
			if spec.Parent != "" {
				ceiling, holder = t.budgetOf(t.nodes[spec.Parent]), fmt.Sprintf("mux %q", spec.Parent)
			}
			if spec.MaxRatePerS > ceiling {
				return nil, badRequest("max_rate_per_s %d on %q would widen %s's max_rate_per_s %d", spec.MaxRatePerS, spec.Name, holder, ceiling)
			}
		}
	} else {
		if spec.MaxRatePerS != 0 {
			return nil, badRequest("max_rate_per_s is a mux's: leaf %q carries no budget", spec.Name)
		}
		r := leafRate(spec)
		for a := t.nodes[spec.Parent]; a != nil; a = t.nodes[a.spec.Parent] {
			if have, budget := t.subtreeRate(a.spec.Name), t.budgetOf(a); over(have, r, budget) {
				return nil, badRequest("subtree %q would declare %s records/s with %q; its max_rate_per_s is %d", a.spec.Name, fmtRate(have.Add(have, r)), spec.Name, budget)
			}
		}
		if spec.Parent == "" && over(new(big.Rat), r, t.budget.maxRate) {
			return nil, badRequest("node %q would declare %s records/s alone; max_rate_per_s is %d", spec.Name, fmtRate(r), t.budget.maxRate)
		}
		if have := t.treeRate(); over(have, r, t.budget.treeRate) {
			return nil, badRequest("the tree would declare %s records/s with %q; tree_rate_per_s is %d", fmtRate(have.Add(have, r)), spec.Name, t.budget.treeRate)
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
	if spec.Export {
		extra["export"] = true
	}
	n.created = t.event("probe.created", n, extra).Seq
	return map[string]any{"node": spec.Name, "kind": kind.String(), "state": n.state, "depth": depth}, nil
}

// list answers `probe.list`: every node in creation order, with its usage against the bounds and
// the seq of its probe.created (`created_seq`, the incarnation's identity). With the cursor, the
// reply says the head and the epoch the list stands at, read under the lock that every lifecycle
// event is published under: the list is exactly what ctl says up to that seq, nothing of what
// follows (S2 slice 4). A leaf whose stop is in progress says `stopping`: its probe.stopped is
// not out yet, so ctl still says running and a start is refused until it is.
func (t *probeTree) list(withCursor bool) any {
	t.mu.Lock()
	defer t.mu.Unlock()
	nodes := make([]map[string]any, 0, len(t.order))
	now := time.Now()
	for _, name := range t.order {
		n := t.nodes[name]
		state := n.state
		if n.stopping {
			state = probeStopping
		}
		m := map[string]any{
			"name": name, "kind": n.kind.String(), "state": state, "every_ms": n.spec.EveryMs,
			"ttl_ms": n.spec.TTLMs, "seq": n.seq, "errors": n.errors, "bytes": n.bytes, "created_seq": n.created,
			"children": len(n.children), "rate_per_s": ratePerS(t.subtreeRate(name)),
		}
		if n.kind.Name == "mux" {
			m["max_rate_per_s"] = t.budgetOf(n)
		}
		if n.spec.Export {
			m["export"] = true
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
	reply := map[string]any{
		"nodes": nodes, "count": len(nodes), "max_nodes": probeMaxNodes,
		"rate_per_s": ratePerS(t.treeRate()), "max_rate_per_s": t.budget.maxRate, "tree_rate_per_s": t.budget.treeRate,
	}
	if withCursor {
		reply["cursor"] = map[string]any{"seq": t.hub.Head(), "epoch": t.hub.Epoch()}
	}
	return reply
}

// kindsReply answers `probe.kinds`: the registry, sorted by name, and the limits.
func kindsReply(b probeBudget) map[string]any {
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
	return map[string]any{"kinds": kinds, "limits": probeLimits(b), "version": version}
}
