package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// probePlane is a plane with no services, for the probe ops: the tree lives in the plane.
func probePlane(t *testing.T) *Plane {
	t.Helper()
	p, err := NewPlane(fakeSpec(t, "a", nil), "a", "", "")
	if err != nil {
		t.Fatal(err)
	}
	p.Start()
	t.Cleanup(p.Shutdown)
	return p
}

// probeOp runs one probe op through the plane's one entry point and returns its reply as JSON.
func probeOp(t *testing.T, p *Plane, req Request) (json.RawMessage, error) {
	t.Helper()
	data, err := p.Request(context.Background(), req)
	if err != nil {
		return nil, err
	}
	b, merr := json.Marshal(data)
	if merr != nil {
		t.Fatal(merr)
	}
	return b, nil
}

// node is a test's shorthand for a node spec as the protocol carries it.
func node(name, kind string, every int64, extra map[string]any) *NodeSpec {
	b, _ := json.Marshal(map[string]any{"name": name, "kind": kind, "every_ms": every})
	var n NodeSpec
	json.Unmarshal(b, &n)
	if extra != nil {
		m := map[string]any{"name": name, "kind": kind, "every_ms": every}
		for k, v := range extra {
			m[k] = v
		}
		b, _ = json.Marshal(m)
		json.Unmarshal(b, &n)
	}
	return &n
}

type listedNode struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"`
	State    string `json:"state"`
	Parent   string `json:"parent"`
	EveryMs  int64  `json:"every_ms"`
	TTLMs    int64  `json:"ttl_ms"`
	Seq      uint64 `json:"seq"`
	Errors   uint64 `json:"errors"`
	LeftMs   *int64 `json:"expires_in_ms"`
	Children int    `json:"children"`
}

func listNodes(t *testing.T, p *Plane) []listedNode {
	t.Helper()
	raw, err := probeOp(t, p, Request{Op: "probe.list"})
	if err != nil {
		t.Fatal(err)
	}
	var reply struct {
		Nodes []listedNode `json:"nodes"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		t.Fatal(err)
	}
	return reply.Nodes
}

func TestANodeIsCreatedStoppedAndEveryRefusalNamesItsBound(t *testing.T) {
	p := probePlane(t)
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("agg", "mux@1", 0, nil)}); err != nil {
		t.Fatal(err)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("hb", "heartbeat@1", 50, map[string]any{"parent": "agg", "ttl_ms": 5000})}); err != nil {
		t.Fatal(err)
	}
	nodes := listNodes(t, p)
	if len(nodes) != 2 || nodes[0].Name != "agg" || nodes[1].Name != "hb" {
		t.Fatalf("list %+v, want agg then hb", nodes)
	}
	if nodes[1].State != "stopped" || nodes[1].Parent != "agg" || nodes[1].Kind != "heartbeat@1" || nodes[1].EveryMs != 50 || nodes[1].TTLMs != 5000 || nodes[0].Children != 1 {
		t.Errorf("hb %+v / agg %+v", nodes[1], nodes[0])
	}
	refused := []struct {
		spec *NodeSpec
		word string // what the refusal must name
	}{
		{node("hb", "heartbeat@1", 50, nil), "exists"},
		{node("Bad Name", "heartbeat@1", 50, nil), "name"},
		{node("x", "cpu@9", 50, nil), "cpu@9"},
		{node("x", "heartbeat@1", 5, nil), "min_every_ms"},
		{node("x", "heartbeat@1", 4_000_000, nil), "max_every_ms"},
		{node("x", "heartbeat@1", 50, map[string]any{"ttl_ms": -1}), "ttl_ms"},
		{node("x", "heartbeat@1", 50, map[string]any{"parent": "nope"}), "nope"},
		{node("x", "heartbeat@1", 50, map[string]any{"parent": "hb"}), "mux"},
		{node("x", "heartbeat@1", 50, map[string]any{"filter": map[string]any{"where": make([]map[string]any, probeMaxWhere+1)}}), "max_where"},
		{node("x", "heartbeat@1", 50, map[string]any{"filter": map[string]any{"keep": make([]string, probeMaxKeep+1)}}), "max_keep"},
		{node("x", "heartbeat@1", 50, map[string]any{"filter": map[string]any{"where": []map[string]any{{"field": "n", "op": "~", "value": 1}}}}), "op"},
		{node("x", "heartbeat@1", 50, map[string]any{"filter": map[string]any{"every": 0}}), "every"},
	}
	for _, r := range refused {
		_, err := probeOp(t, p, Request{Op: "probe.create", Probe: r.spec})
		if err == nil || !strings.Contains(err.Error(), r.word) || statusOf(err) != 400 {
			t.Errorf("create %+v: err %v, want a 400 naming %q", r.spec, err, r.word)
		}
	}
	// Depth: a chain of muxes past probeMaxDepth is refused, naming the bound.
	parent := "agg"
	var err error
	for i := 0; i < probeMaxDepth+1; i++ {
		name := "m" + itoa(i)
		_, err = probeOp(t, p, Request{Op: "probe.create", Probe: node(name, "mux@1", 0, map[string]any{"parent": parent})})
		if err != nil {
			break
		}
		parent = name
	}
	if err == nil || !strings.Contains(err.Error(), "max_depth") {
		t.Errorf("deep chain: err %v, want max_depth", err)
	}
	// A leaf may sit under max_depth muxes: agg + m0..m(max_depth-2) are max_depth muxes, so
	// m(max_depth-1) (a mux under max_depth muxes) is admitted and the next one is refused.
	if len(listNodes(t, p)) != 2+probeMaxDepth {
		t.Errorf("%d nodes after the chain, want %d", len(listNodes(t, p)), 2+probeMaxDepth)
	}
	// Filter values are typed: an object, an array or null never matches, so they are refused
	// at creation, and a bool only takes == or !=.
	for _, c := range []struct {
		value any
		op    string
	}{{map[string]any{"a": 1}, "=="}, {[]any{1}, "=="}, {nil, "=="}, {true, "<"}} {
		f := map[string]any{"where": []map[string]any{{"field": "n", "op": c.op, "value": c.value}}}
		_, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("x", "heartbeat@1", 50, map[string]any{"filter": f})})
		if err == nil || !strings.Contains(err.Error(), "where[0]") {
			t.Errorf("value %v with %s: err %v, want a refusal naming where[0]", c.value, c.op, err)
		}
	}
}

// Round-20 findings: a mux's keep never drops src; a started lease says when it expires;
// a cpu interval with no advance emits nothing; nothing is admitted once the plane shuts down.
func TestKeepKeepsSrcAndNothingIsAdmittedAfterShutdown(t *testing.T) {
	p := probePlane(t)
	for _, spec := range []*NodeSpec{
		node("agg", "mux@1", 0, map[string]any{"filter": map[string]any{"keep": []string{"n"}}}),
		node("hb", "heartbeat@1", 10, map[string]any{"parent": "agg", "ttl_ms": 60000}),
	} {
		if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: spec}); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{"agg", "hb"} {
		if _, err := probeOp(t, p, Request{Op: "probe.start", Node: name}); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range readSamples(t, p, "probe/agg", 2, 2*time.Second) {
		if s["src"] != "hb" || s["n"] == nil || len(s) != 5 {
			t.Errorf("mux sample under keep [n]: %v, want n, src and the three stamps", s)
		}
	}
	started := ctlEvents(t, p, "probe.started")
	if len(started) != 2 || started[1]["node"] != "hb" || started[1]["expires_in_ms"] == nil || started[0]["expires_in_ms"] != nil {
		t.Errorf("probe.started events %v: the leased one carries expires_in_ms, the other does not", started)
	}
	var c cpuProbe
	if _, err := c.delta(10, 100); !errors.Is(err, errNoDelta) {
		t.Errorf("first reading: %v, want errNoDelta", err)
	}
	if _, err := c.delta(10, 100); !errors.Is(err, errNoDelta) {
		t.Errorf("no advance: %v, want errNoDelta, not a fabricated idle interval", err)
	}
	if f, err := c.delta(20, 200); err != nil || f["busy_pct"] != 90.0 {
		t.Errorf("an interval of 100 with 10 idle: %v %v, want busy_pct 90", f, err)
	}
	p.Shutdown()
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: "hb"}); err == nil || !strings.Contains(err.Error(), "shutting down") {
		t.Errorf("start after shutdown: %v", err)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("late", "heartbeat@1", 10, nil)}); err == nil || !strings.Contains(err.Error(), "shutting down") {
		t.Errorf("create after shutdown: %v", err)
	}
	for _, n := range listNodes(t, p) {
		if n.State != "stopped" {
			t.Errorf("%s is %s after shutdown", n.Name, n.State)
		}
	}
}

// readSamples reads up to max `sample` events on prefix within d, as their data objects.
func readSamples(t *testing.T, p *Plane, prefix string, max int, d time.Duration) []map[string]any {
	t.Helper()
	cur := p.Subscribe(prefix, uint64p(0), "")
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	var out []map[string]any
	for len(out) < max {
		envs, gap, err := cur.Next(ctx, max-len(out))
		if err != nil {
			break
		}
		if gap != nil {
			t.Fatalf("gap on %s: %+v", prefix, gap)
		}
		for _, env := range envs {
			if env.Ev.Kind != "sample" {
				continue
			}
			var m map[string]any
			if err := json.Unmarshal(env.Ev.Data, &m); err != nil {
				t.Fatal(err)
			}
			out = append(out, m)
		}
	}
	return out
}

func uint64p(n uint64) *uint64 { return &n }

// ctlEvents reads the ctl events of the given kinds published so far (replayed from seq 0).
func ctlEvents(t *testing.T, p *Plane, kinds ...string) []map[string]any {
	t.Helper()
	cur := p.Subscribe("ctl", uint64p(0), "")
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	want := map[string]bool{}
	for _, k := range kinds {
		want[k] = true
	}
	var out []map[string]any
	for {
		envs, _, err := cur.Next(ctx, 256)
		if err != nil {
			return out
		}
		for _, env := range envs {
			if !want[env.Ev.Kind] {
				continue
			}
			var m map[string]any
			json.Unmarshal(env.Ev.Data, &m)
			m["kind"] = env.Ev.Kind
			out = append(out, m)
		}
	}
}

func TestAStartedHeartbeatSamplesOnItsPipeUntilStopped(t *testing.T) {
	p := probePlane(t)
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("hb", "heartbeat@1", 20, nil)}); err != nil {
		t.Fatal(err)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: "nope"}); err == nil || !strings.Contains(err.Error(), "nope") {
		t.Errorf("start of an unknown node: %v", err)
	}
	raw, err := probeOp(t, p, Request{Op: "probe.start", Node: "hb"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"state":"running"`) {
		t.Errorf("start reply %s", raw)
	}
	samples := readSamples(t, p, "probe/hb", 3, 2*time.Second)
	if len(samples) != 3 {
		t.Fatalf("%d samples in 2 s at 20 ms, want 3", len(samples))
	}
	for i, s := range samples {
		if s["node"] != "hb" || s["kind"] != "heartbeat@1" || s["seq"] != float64(i+1) || s["n"] != float64(i+1) {
			t.Errorf("sample %d: %v", i, s)
		}
	}
	if n := listNodes(t, p)[0]; n.State != "running" || n.Seq < 3 {
		t.Errorf("listed %+v, want running with seq >= 3", n)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: "hb"}); err != nil {
		t.Errorf("a second start is a noop, got %v", err)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.stop", Node: "hb"}); err != nil {
		t.Fatal(err)
	}
	n := listNodes(t, p)[0]
	if n.State != "stopped" {
		t.Errorf("after stop: %+v", n)
	}
	after := n.Seq
	time.Sleep(80 * time.Millisecond)
	if again := listNodes(t, p)[0].Seq; again != after {
		t.Errorf("seq moved from %d to %d after the stop", after, again)
	}
	evs := ctlEvents(t, p, "probe.created", "probe.started", "probe.stopped")
	if len(evs) != 3 || evs[0]["kind"] != "probe.created" || evs[1]["kind"] != "probe.started" || evs[2]["kind"] != "probe.stopped" ||
		evs[2]["node"] != "hb" || evs[2]["reason"] != "requested" || evs[1]["kind_name"] != nil {
		t.Errorf("ctl events %v", evs)
	}
}

func TestANodesFilterShapesWhatItEmits(t *testing.T) {
	p := probePlane(t)
	filter := map[string]any{
		"where": []map[string]any{{"field": "n", "op": ">", "value": 2}},
		"keep":  []string{"n"},
		"every": 2,
	}
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("hb", "heartbeat@1", 10, map[string]any{"filter": filter})}); err != nil {
		t.Fatal(err)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: "hb"}); err != nil {
		t.Fatal(err)
	}
	samples := readSamples(t, p, "probe/hb", 3, 2*time.Second)
	if len(samples) != 3 {
		t.Fatalf("%d samples, want 3", len(samples))
	}
	// where keeps 3, 4, 5, 6, 7, 8...; every 2 keeps the 2nd, 4th, 6th of those: 4, 6, 8.
	for i, want := range []float64{4, 6, 8} {
		s := samples[i]
		if s["n"] != want || s["seq"] != float64(i+1) || s["node"] != "hb" || s["kind"] != "heartbeat@1" || len(s) != 4 {
			t.Errorf("sample %d: %v, want n %v, seq %d and the three stamps only", i, s, want, i+1)
		}
	}
	// The pure part, on records of every value kind the samples carry.
	cases := []struct {
		f    Filter
		rec  map[string]any
		pass bool
	}{
		{Filter{Where: []Term{{"busy_pct", ">=", 50.0}}}, map[string]any{"busy_pct": 50.0}, true},
		{Filter{Where: []Term{{"busy_pct", "<", 50}}}, map[string]any{"busy_pct": 50.0}, false},
		{Filter{Where: []Term{{"n", "==", 3}}}, map[string]any{"n": int64(3)}, true},
		{Filter{Where: []Term{{"n", "!=", 3}}}, map[string]any{"n": int64(3)}, false},
		{Filter{Where: []Term{{"state", "==", "ok"}}}, map[string]any{"state": "ok"}, true},
		{Filter{Where: []Term{{"state", "<", "p"}}}, map[string]any{"state": "ok"}, true},
		{Filter{Where: []Term{{"missing", "==", 1}}}, map[string]any{"n": 1}, false},
		{Filter{Where: []Term{{"n", "==", "1"}}}, map[string]any{"n": 1}, false},
		{Filter{Where: []Term{{"up", "==", true}}}, map[string]any{"up": true}, true},
		{Filter{Where: []Term{{"n", ">", 1}, {"n", "<", 3}}}, map[string]any{"n": 2}, true},
		{Filter{Where: []Term{{"n", ">", 1}, {"n", "<", 3}}}, map[string]any{"n": 3}, false},
	}
	for i, c := range cases {
		if got := c.f.matches(c.rec); got != c.pass {
			t.Errorf("case %d: %+v on %v: %v, want %v", i, c.f.Where, c.rec, got, c.pass)
		}
	}
}

func TestAMuxMultiplexesItsChildrenStampedWithTheirSource(t *testing.T) {
	p := probePlane(t)
	create := func(spec *NodeSpec) {
		t.Helper()
		if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: spec}); err != nil {
			t.Fatal(err)
		}
	}
	op := func(op, name string) {
		t.Helper()
		if _, err := probeOp(t, p, Request{Op: op, Node: name}); err != nil {
			t.Fatal(err)
		}
	}
	create(node("agg", "mux@1", 0, map[string]any{"filter": map[string]any{"where": []map[string]any{{"field": "n", "op": ">", "value": 1}}}}))
	create(node("a", "heartbeat@1", 10, map[string]any{"parent": "agg"}))
	create(node("b", "heartbeat@1", 10, map[string]any{"parent": "agg", "filter": map[string]any{"every": 2}}))
	// Children of a stopped mux keep their own pipes; nothing reaches the mux's.
	op("probe.start", "a")
	if got := readSamples(t, p, "probe/a", 2, time.Second); len(got) != 2 {
		t.Fatalf("a alone: %d samples on its pipe, want 2", len(got))
	}
	if got := readSamples(t, p, "probe/agg", 1, 100*time.Millisecond); len(got) != 0 {
		t.Fatalf("stopped mux got %v", got)
	}
	op("probe.start", "agg")
	op("probe.start", "b")
	samples := readSamples(t, p, "probe/agg", 8, 2*time.Second)
	if len(samples) != 8 {
		t.Fatalf("%d samples on the mux, want 8", len(samples))
	}
	src := map[string]int{}
	for i, s := range samples {
		src[s["src"].(string)]++
		n, _ := s["n"].(float64)
		if s["node"] != "agg" || s["kind"] != "mux@1" || s["seq"] != float64(i+1) || n <= 1 {
			t.Errorf("mux sample %d: %v", i, s)
		}
		if s["src"] == "b" && int(n)%2 != 0 {
			t.Errorf("b's filter (every 2) did not shape the edge: %v", s)
		}
	}
	if src["a"] == 0 || src["b"] == 0 {
		t.Errorf("sources %v, want both children", src)
	}
	agg := listNodes(t, p)[0]
	if agg.State != "running" || agg.Seq < 8 || agg.Children != 2 {
		t.Errorf("mux listed %+v", agg)
	}
	// A stopped mux again ignores its running children; its seq stands still.
	op("probe.stop", "agg")
	seq := listNodes(t, p)[0].Seq
	time.Sleep(50 * time.Millisecond)
	if again := listNodes(t, p)[0].Seq; again != seq {
		t.Errorf("stopped mux's seq moved %d -> %d", seq, again)
	}
}

func TestALeaseExpiresUnlessTouchedAndARemoveTakesTheSubtree(t *testing.T) {
	p := probePlane(t)
	create := func(spec *NodeSpec) {
		t.Helper()
		if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: spec}); err != nil {
			t.Fatal(err)
		}
	}
	op := func(op, name string) json.RawMessage {
		t.Helper()
		raw, err := probeOp(t, p, Request{Op: op, Node: name})
		if err != nil {
			t.Fatal(op, name, err)
		}
		return raw
	}
	names := func() []string {
		var out []string
		for _, n := range listNodes(t, p) {
			out = append(out, n.Name)
		}
		return out
	}
	create(node("agg", "mux@1", 0, nil))
	create(node("hb", "heartbeat@1", 10, map[string]any{"parent": "agg", "ttl_ms": 150}))
	if _, err := probeOp(t, p, Request{Op: "probe.touch", Node: "agg"}); err == nil || !strings.Contains(err.Error(), "lease") {
		t.Errorf("touch of a node without a lease: %v", err)
	}
	if left := listNodes(t, p)[1].LeftMs; left != nil {
		t.Errorf("a stopped node has no running lease, got expires_in_ms %d", *left)
	}
	op("probe.start", "agg")
	op("probe.start", "hb")
	if left := listNodes(t, p)[1].LeftMs; left == nil || *left <= 0 || *left > 150 {
		t.Errorf("expires_in_ms after start: %v", left)
	}
	time.Sleep(100 * time.Millisecond)
	raw := op("probe.touch", "hb")
	if !strings.Contains(string(raw), `"expires_in_ms"`) {
		t.Errorf("touch reply %s", raw)
	}
	time.Sleep(100 * time.Millisecond) // 200 ms after the start: alive only because of the touch
	if got := names(); len(got) != 2 {
		t.Fatalf("after a touch at 100 ms, nodes %v at 200 ms, want both", got)
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(names()) == 2 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if got := names(); len(got) != 1 || got[0] != "agg" {
		t.Fatalf("after the lease, nodes %v, want agg alone", got)
	}
	evs := ctlEvents(t, p, "probe.stopped", "probe.removed")
	if len(evs) != 2 || evs[0]["kind"] != "probe.stopped" || evs[0]["node"] != "hb" || evs[0]["reason"] != "expired" ||
		evs[1]["kind"] != "probe.removed" || evs[1]["node"] != "hb" || evs[1]["reason"] != "expired" {
		t.Errorf("expiry events %v", evs)
	}
	// Remove takes the subtree: a running child under a running mux, both stopped and gone.
	create(node("c", "heartbeat@1", 10, map[string]any{"parent": "agg"}))
	op("probe.start", "c")
	if _, err := probeOp(t, p, Request{Op: "probe.remove", Node: "nope"}); err == nil {
		t.Error("remove of an unknown node succeeded")
	}
	op("probe.remove", "agg")
	if got := names(); len(got) != 0 {
		t.Fatalf("after removing agg, nodes %v", got)
	}
	evs = ctlEvents(t, p, "probe.stopped", "probe.removed")[2:]
	reasons := map[string]string{}
	for _, e := range evs {
		reasons[e["kind"].(string)+":"+e["node"].(string)] = e["reason"].(string)
	}
	if len(evs) != 4 || reasons["probe.stopped:c"] != "parent-removed" || reasons["probe.stopped:agg"] != "removed" ||
		reasons["probe.removed:c"] != "parent-removed" || reasons["probe.removed:agg"] != "removed" {
		t.Errorf("remove events %v", evs)
	}
	// The names are free again, and a mux's own lease takes its children with it.
	create(node("agg", "mux@1", 0, map[string]any{"ttl_ms": 60}))
	create(node("c", "heartbeat@1", 10, map[string]any{"parent": "agg"}))
	op("probe.start", "c")
	op("probe.start", "agg")
	deadline = time.Now().Add(2 * time.Second)
	for len(names()) > 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if got := names(); len(got) != 0 {
		t.Errorf("after the mux's lease, nodes %v, want none", got)
	}
}

func TestCpuAndMemKindsReadTheHostWithinRange(t *testing.T) {
	p := probePlane(t)
	for _, spec := range []*NodeSpec{node("cpu", "cpu@1", 50, nil), node("mem", "mem@1", 50, nil)} {
		if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: spec}); err != nil {
			t.Fatal(err)
		}
		if _, err := probeOp(t, p, Request{Op: "probe.start", Node: spec.Name}); err != nil {
			t.Fatal(err)
		}
	}
	cpu := readSamples(t, p, "probe/cpu", 2, 3*time.Second)
	if len(cpu) != 2 {
		t.Fatalf("cpu: %d samples, want 2 (the first interval has no delta and emits nothing)", len(cpu))
	}
	for _, s := range cpu {
		busy, _ := s["busy_pct"].(float64)
		cores, _ := s["cores"].(float64)
		if _, failed := s["error"]; failed || busy < 0 || busy > 100 || cores < 1 {
			t.Errorf("cpu sample %v", s)
		}
	}
	mem := readSamples(t, p, "probe/mem", 1, 3*time.Second)
	if len(mem) != 1 {
		t.Fatalf("mem: %d samples, want 1", len(mem))
	}
	total, _ := mem[0]["total_bytes"].(float64)
	avail, _ := mem[0]["avail_bytes"].(float64)
	used, _ := mem[0]["used_pct"].(float64)
	if _, failed := mem[0]["error"]; failed || total < 1<<30 || avail <= 0 || avail >= total || used <= 0 || used >= 100 {
		t.Errorf("mem sample %v", mem[0])
	}
	for _, n := range listNodes(t, p) {
		if n.Errors != 0 {
			t.Errorf("%s: %d errors", n.Name, n.Errors)
		}
	}
}

// httpJSON calls an instance over HTTP with its token and decodes the JSON answer.
func httpJSON(t *testing.T, in *instance, method, path string, body any) (int, map[string]any) {
	t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, "http://"+in.addr+path, rd)
	req.Header.Set("Authorization", "Bearer "+in.token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var m map[string]any
	json.NewDecoder(resp.Body).Decode(&m)
	return resp.StatusCode, m
}

func TestProbeOpsAreServedOverHTTPAndTheWebSocketAndEndAtShutdown(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "a", nil), "")
	code, kinds := httpJSON(t, in, "GET", "/v1/probes", nil)
	if code != 200 || kinds["limits"] == nil || len(kinds["kinds"].([]any)) < 4 {
		t.Fatalf("GET /v1/probes: %d %v", code, kinds)
	}
	code, r := httpJSON(t, in, "POST", "/v1/probe/create", map[string]any{"name": "agg", "kind": "mux@1"})
	if code != 200 || r["state"] != "stopped" {
		t.Fatalf("create agg: %d %v", code, r)
	}
	code, r = httpJSON(t, in, "POST", "/v1/probe/create", map[string]any{"name": "hb", "kind": "heartbeat@1", "every_ms": 5, "parent": "agg"})
	if code != 400 || !strings.Contains(r["error"].(string), "min_every_ms") {
		t.Fatalf("create under min_every_ms: %d %v", code, r)
	}
	code, r = httpJSON(t, in, "POST", "/v1/probe/create", map[string]any{"name": "hb", "kind": "heartbeat@1", "every_ms": 10, "parent": "agg", "ttl_ms": 5000})
	if code != 200 {
		t.Fatalf("create hb: %d %v", code, r)
	}
	for _, op := range []string{"start", "touch"} {
		if code, r = httpJSON(t, in, "POST", "/v1/probe/"+op, map[string]any{"node": "hb"}); code != 200 {
			t.Fatalf("%s hb: %d %v", op, code, r)
		}
	}
	code, r = httpJSON(t, in, "POST", "/v1/probe/start", map[string]any{"node": "agg"})
	if code != 200 {
		t.Fatalf("start agg: %d %v", code, r)
	}
	// The same ops over ixctl.v1: the frame carries `probe` or `node`.
	c := dial(t, in)
	w := subscribe(t, c, "probe/agg", nil)
	raw := do(t, c, Request{Op: "probe.create", Probe: node("hb2", "heartbeat@1", 10, map[string]any{"parent": "agg"})})
	if !strings.Contains(string(raw), `"node":"hb2"`) {
		t.Fatalf("ws create: %s", raw)
	}
	do(t, c, Request{Op: "probe.start", Node: "hb2"})
	waitEnv(t, w, 2*time.Second, "a sample from hb2 on agg", func(e *Envelope) bool {
		return e.Ev.Kind == "sample" && bytes.Contains(e.Ev.Data, []byte(`"src":"hb2"`))
	})
	raw = do(t, c, Request{Op: "probe.list"})
	if !strings.Contains(string(raw), `"count":3`) {
		t.Fatalf("ws list: %s", raw)
	}
	code, r = httpJSON(t, in, "GET", "/v1/probe", nil)
	if code != 200 || r["count"] != 3.0 {
		t.Fatalf("GET /v1/probe: %d %v", code, r)
	}
	code, r = httpJSON(t, in, "POST", "/v1/probe/stop", map[string]any{"node": "hb2"})
	if code != 200 || r["state"] != "stopped" {
		t.Fatalf("stop hb2: %d %v", code, r)
	}
	code, r = httpJSON(t, in, "POST", "/v1/probe/remove", map[string]any{"node": "hb2"})
	if code != 200 || r["removed"] != 1.0 {
		t.Fatalf("remove hb2: %d %v", code, r)
	}
	code, r = httpJSON(t, in, "POST", "/v1/probe/remove", map[string]any{"node": "hb2"})
	if code != 400 {
		t.Fatalf("remove twice: %d %v", code, r)
	}
	// Shutdown ends every running node and says so; nothing samples after it.
	in.close()
	evs := ctlEvents(t, in.plane, "probe.stopped")
	reasons := map[string]string{}
	for _, e := range evs {
		reasons[e["node"].(string)] = e["reason"].(string)
	}
	if reasons["hb"] != "shutdown" || reasons["agg"] != "shutdown" {
		t.Errorf("stop reasons at shutdown %v", reasons)
	}
	seq := in.plane.probes.nodes["hb"].seq
	time.Sleep(50 * time.Millisecond)
	if again := in.plane.probes.nodes["hb"].seq; again != seq {
		t.Errorf("hb sampled after shutdown: %d -> %d", seq, again)
	}
}

// A probe op named for a peer runs on that peer: the node lives there, its lifecycle
// events come back mirrored on peer/b/ctl, and the sample pipe stays local to b
// (cross-host pipes are decision D4 / story S2, not this slice).
func TestProbeOpsAreForwardedToAPeerAndItsLifecycleIsMirrored(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	specB := fakeSpec(t, "b", []PeerSpec{{Name: "a", StateFile: filepath.Join(dirA, "a.json")}})
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json")}})
	b := newTestInstance(t, specB, dirB)
	a := newTestInstance(t, specA, dirA)
	waitStatus(t, a, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })

	c := dial(t, a)
	w := subscribe(t, c, "peer/b/", nil)
	raw := do(t, c, Request{Op: "probe.create", Peer: "b", Probe: node("hb", "heartbeat@1", 10, nil)})
	if !strings.Contains(string(raw), `"node":"hb"`) {
		t.Fatalf("forwarded create: %s", raw)
	}
	if len(listNodes(t, a.plane)) != 0 || len(listNodes(t, b.plane)) != 1 {
		t.Fatal("the node was created on a, not b")
	}
	code, r := httpJSON(t, a, "POST", "/v1/probe/start", map[string]any{"node": "hb", "peer": "b"})
	if code != 200 || r["state"] != "running" {
		t.Fatalf("forwarded start over HTTP: %d %v", code, r)
	}
	e := waitEnv(t, w, 10*time.Second, "mirrored probe.started", func(e *Envelope) bool {
		return e.Ch == "peer/b/ctl" && e.Ev.Kind == "probe.started"
	})
	if e.Origin != "b" {
		t.Fatalf("mirrored origin %q", e.Origin)
	}
	code, r = httpJSON(t, a, "GET", "/v1/probe?peer=b", nil)
	if code != 200 || r["count"] != 1.0 {
		t.Fatalf("GET /v1/probe?peer=b: %d %v", code, r)
	}
	code, r = httpJSON(t, a, "GET", "/v1/probes?peer=b", nil)
	if code != 200 || r["limits"] == nil {
		t.Fatalf("GET /v1/probes?peer=b: %d %v", code, r)
	}
	if samples := readSamples(t, b.plane, "probe/hb", 1, 2*time.Second); len(samples) == 0 {
		t.Fatal("hb samples on b")
	}
	if n := len(readSamples(t, a.plane, "probe/", 1, 200*time.Millisecond)); n != 0 {
		t.Fatalf("%d samples reached a: cross-host pipes are S2", n)
	}
	if code, r = httpJSON(t, a, "POST", "/v1/probe/remove", map[string]any{"node": "hb", "peer": "b"}); code != 200 || r["removed"] != 1.0 {
		t.Fatalf("forwarded remove: %d %v", code, r)
	}
	waitEnv(t, w, 10*time.Second, "mirrored probe.removed", func(e *Envelope) bool {
		return e.Ch == "peer/b/ctl" && e.Ev.Kind == "probe.removed"
	})
}

func TestTheCLIPrintsTheKindsOffline(t *testing.T) {
	var out, errb bytes.Buffer
	if code := cli([]string{"probes"}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	var m map[string]any
	if err := json.Unmarshal(out.Bytes(), &m); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, out.String())
	}
	if len(m["kinds"].([]any)) < 4 || m["limits"] == nil || m["version"] != version {
		t.Errorf("probes printed %v", m)
	}
}

func TestTheKindsAreListedWithTheSystemsLimits(t *testing.T) {
	p := probePlane(t)
	raw, err := probeOp(t, p, Request{Op: "probe.kinds"})
	if err != nil {
		t.Fatal(err)
	}
	var reply struct {
		Kinds []struct {
			Kind    string   `json:"kind"`
			Version int      `json:"version"`
			Fields  []string `json:"fields"`
			Doc     string   `json:"doc"`
		} `json:"kinds"`
		Limits map[string]int64 `json:"limits"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		t.Fatal(err)
	}
	want := map[string][]string{
		"heartbeat@1": {"n"},
		"cpu@1":       {"busy_pct", "cores"},
		"mem@1":       {"total_bytes", "avail_bytes", "used_pct"},
		"mux@1":       nil,
	}
	got := map[string][]string{}
	for _, k := range reply.Kinds {
		got[k.Kind] = k.Fields
		if k.Version != 1 || k.Doc == "" {
			t.Errorf("%s: version %d doc %q", k.Kind, k.Version, k.Doc)
		}
	}
	for kind, fields := range want {
		if _, ok := got[kind]; !ok {
			t.Errorf("kind %s missing from %v", kind, got)
			continue
		}
		if len(got[kind]) != len(fields) {
			t.Errorf("%s fields %v, want %v", kind, got[kind], fields)
		}
	}
	// Every bound the plane applies is reported, by name, with a positive value.
	for _, name := range []string{"max_nodes", "max_depth", "min_every_ms", "max_every_ms", "max_ttl_ms", "ring", "max_where", "max_keep"} {
		if reply.Limits[name] <= 0 {
			t.Errorf("limit %s = %d, want > 0 (limits: %v)", name, reply.Limits[name], reply.Limits)
		}
	}
	if reply.Limits["ring"] != ringSize {
		t.Errorf("ring %d, want the hub's %d", reply.Limits["ring"], ringSize)
	}
}
