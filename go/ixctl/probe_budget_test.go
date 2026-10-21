package main

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

// budgetTree is `probe.list` read with the budget fields D10 (ix#416) added.
type budgetTree struct {
	Nodes []struct {
		Name        string  `json:"name"`
		Seq         uint64  `json:"seq"`
		Bytes       uint64  `json:"bytes"`
		RatePerS    float64 `json:"rate_per_s"`
		MaxRatePerS *int64  `json:"max_rate_per_s"`
	} `json:"nodes"`
	Count        int     `json:"count"`
	RatePerS     float64 `json:"rate_per_s"`
	MaxRatePerS  int64   `json:"max_rate_per_s"`
	TreeRatePerS int64   `json:"tree_rate_per_s"`
}

func listTree(t *testing.T, p *Plane) budgetTree {
	t.Helper()
	raw, err := probeOp(t, p, Request{Op: "probe.list"})
	if err != nil {
		t.Fatal(err)
	}
	var tree budgetTree
	if err := json.Unmarshal(raw, &tree); err != nil {
		t.Fatal(err)
	}
	return tree
}

func (b budgetTree) find(t *testing.T, name string) (rate float64, budget *int64, bytes uint64) {
	t.Helper()
	for _, n := range b.Nodes {
		if n.Name == name {
			return n.RatePerS, n.MaxRatePerS, n.Bytes
		}
	}
	t.Fatalf("node %q not listed", name)
	return 0, nil, 0
}

func create(t *testing.T, p *Plane, spec *NodeSpec) error {
	t.Helper()
	_, err := probeOp(t, p, Request{Op: "probe.create", Probe: spec})
	return err
}

// mustCreate admits a node or fails the test naming it.
func mustCreate(t *testing.T, p *Plane, spec *NodeSpec) {
	t.Helper()
	if err := create(t, p, spec); err != nil {
		t.Fatalf("create %s: %v", spec.Name, err)
	}
}

// refused fails the test unless the create is refused with every one of the words.
func refused(t *testing.T, err error, words ...string) {
	t.Helper()
	if err == nil {
		t.Fatalf("admitted; wanted a refusal naming %q", words)
	}
	for _, w := range words {
		if !strings.Contains(err.Error(), w) {
			t.Fatalf("refusal %q does not name %q", err.Error(), w)
		}
	}
}

// leaves declares n heartbeats at 10 ms (100 records/s each) under a mux, named prefix-i.
func leaves(t *testing.T, p *Plane, parent, prefix string, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		mustCreate(t, p, node(fmt.Sprintf("%s-%d", prefix, i), "heartbeat@1", 10, map[string]any{"parent": parent}))
	}
}

// D10 (ix#416): a budget bounds the declared record rate of every subtree, the worst case
// (1000 / every_ms per leaf, `every: N` dividing, `where` not counted), summed at `create`
// whatever the node's state. The plane's defaults are in `limits`; a mux may narrow its own,
// never widen; a leaf carries none; the tree has its own ceiling. A create that would cross a
// bound is refused with the bound's name and the subtree's.
func TestABudgetBoundsTheDeclaredRateOfEverySubtreeAndRefusalsNameIt(t *testing.T) {
	p := probePlane(t)
	raw, err := probeOp(t, p, Request{Op: "probe.kinds"})
	if err != nil {
		t.Fatal(err)
	}
	var kinds struct {
		Limits map[string]int64 `json:"limits"`
	}
	json.Unmarshal(raw, &kinds)
	if kinds.Limits["max_rate_per_s"] != 1000 || kinds.Limits["tree_rate_per_s"] != 2500 {
		t.Fatalf("default budget not in limits: %v", kinds.Limits)
	}

	// The default per subtree: ten leaves at 10 ms fill it, the eleventh is refused.
	mustCreate(t, p, node("agg", "mux@1", 0, nil))
	leaves(t, p, "agg", "a", 10)
	refused(t, create(t, p, node("a-10", "heartbeat@1", 10, map[string]any{"parent": "agg"})), "agg", "max_rate_per_s", "1000", "1100")

	// The filter does not lower the worst case: a failing sampler emits an error record every
	// interval, and errors bypass `where` and `every` (they must reach the pipe).
	mustCreate(t, p, node("half", "mux@1", 0, nil))
	mustCreate(t, p, node("h-0", "heartbeat@1", 10, map[string]any{"parent": "half",
		"filter": map[string]any{"every": 2, "where": []map[string]any{{"field": "n", "op": ">", "value": 1000}}}}))
	if rate, _, _ := listTree(t, p).find(t, "half"); rate != 100 {
		t.Fatalf("half declares %v/s, want 100: `every: 2` does not lower the worst case", rate)
	}

	// A leaf carries no budget.
	refused(t, create(t, p, node("solo", "heartbeat@1", 10, map[string]any{"max_rate_per_s": 10})), "leaf", "max_rate_per_s")

	// A mux narrows its budget; its children are held to it, and a mux under it cannot widen it.
	mustCreate(t, p, node("small", "mux@1", 0, map[string]any{"max_rate_per_s": 150}))
	leaves(t, p, "small", "s", 1)
	refused(t, create(t, p, node("s-1", "heartbeat@1", 10, map[string]any{"parent": "small"})), "small", "150", "200")
	refused(t, create(t, p, node("wide", "mux@1", 0, map[string]any{"max_rate_per_s": 3000})), "widen", "1000")
	refused(t, create(t, p, node("inner", "mux@1", 0, map[string]any{"parent": "small", "max_rate_per_s": 200})), "widen", "small", "150")

	// The tree's ceiling: 1000 + 100 + 100 declared so far; two more muxes and a root leaf take
	// it to 2 450, and the leaf that would make 2 550 is refused naming the tree, not a subtree.
	mustCreate(t, p, node("b", "mux@1", 0, nil))
	leaves(t, p, "b", "b", 10)
	mustCreate(t, p, node("c", "mux@1", 0, nil))
	leaves(t, p, "c", "c", 2)
	mustCreate(t, p, node("root-leaf", "heartbeat@1", 20, nil)) // 50/s, a subtree of one
	refused(t, create(t, p, node("c-2", "heartbeat@1", 10, map[string]any{"parent": "c"})), "tree_rate_per_s", "2500", "2550")

	// The list says the usage: each subtree's declared rate and effective budget, the tree's.
	tree := listTree(t, p)
	if tree.RatePerS != 2450 || tree.MaxRatePerS != 1000 || tree.TreeRatePerS != 2500 {
		t.Fatalf("tree rate %v, max_rate_per_s %d, tree_rate_per_s %d", tree.RatePerS, tree.MaxRatePerS, tree.TreeRatePerS)
	}
	for _, want := range []struct {
		name   string
		rate   float64
		budget int64
	}{{"agg", 1000, 1000}, {"half", 100, 1000}, {"small", 100, 150}, {"c", 200, 1000}} {
		rate, budget, _ := tree.find(t, want.name)
		if rate != want.rate || budget == nil || *budget != want.budget {
			t.Errorf("%s: rate %v budget %v, want %v / %d", want.name, rate, budget, want.rate, want.budget)
		}
	}
	rate, budget, _ := tree.find(t, "root-leaf")
	if rate != 50 || budget != nil {
		t.Errorf("a leaf declares its own rate (%v, want 50) and carries no budget (%v)", rate, budget)
	}

	// Removing a subtree frees its share: the refused leaf fits once `agg` is gone.
	if _, err := probeOp(t, p, Request{Op: "probe.remove", Node: "agg"}); err != nil {
		t.Fatal(err)
	}
	mustCreate(t, p, node("c-2", "heartbeat@1", 10, map[string]any{"parent": "c"}))
}

// D10: the operator sets the plane's budget at startup, narrower or wider than the compiled
// defaults, and `limits` says it; a tree ceiling under the subtree budget is a mistake named.
func TestTheOperatorSetsThePlanesBudgetAndTheLimitsSayIt(t *testing.T) {
	p, err := NewPlane(fakeSpec(t, "a", nil), "a", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetProbeBudget(100, 50); err == nil || !strings.Contains(err.Error(), "tree_rate_per_s 50") {
		t.Fatalf("a tree ceiling under the subtree budget must be refused: %v", err)
	}
	if err := p.SetProbeBudget(0, 50); err == nil || !strings.Contains(err.Error(), "max_rate_per_s") {
		t.Fatalf("a zero budget must be refused: %v", err)
	}
	if err := p.SetProbeBudget(200, 400); err != nil {
		t.Fatal(err)
	}
	p.Start()
	t.Cleanup(p.Shutdown)
	raw, err := probeOp(t, p, Request{Op: "probe.kinds"})
	if err != nil {
		t.Fatal(err)
	}
	var kinds struct {
		Limits map[string]int64 `json:"limits"`
	}
	json.Unmarshal(raw, &kinds)
	if kinds.Limits["max_rate_per_s"] != 200 || kinds.Limits["tree_rate_per_s"] != 400 {
		t.Fatalf("limits %v, want the operator's 200 / 400", kinds.Limits)
	}
	mustCreate(t, p, node("m", "mux@1", 0, nil))
	leaves(t, p, "m", "m", 2)
	refused(t, create(t, p, node("m-2", "heartbeat@1", 10, map[string]any{"parent": "m"})), "m", "200", "300")
	refused(t, create(t, p, node("w", "mux@1", 0, map[string]any{"max_rate_per_s": 250})), "widen", "200")
	// The budget is frozen once a node exists: nothing declared can end up over it.
	if err := p.SetProbeBudget(500, 1000); err == nil || !strings.Contains(err.Error(), "nodes") {
		t.Fatalf("a budget set after a create must be refused: %v", err)
	}

	// The same budget through the instance's startup options, as `ixctl up` passes them.
	in, err := startInstance(fakeSpec(t, "b", nil), "", "127.0.0.1:0", t.TempDir(), "disk", &probeBudget{maxRate: 300, treeRate: 600})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(in.close)
	raw, err = probeOp(t, in.plane, Request{Op: "probe.kinds"})
	if err != nil {
		t.Fatal(err)
	}
	json.Unmarshal(raw, &kinds)
	if kinds.Limits["max_rate_per_s"] != 300 || kinds.Limits["tree_rate_per_s"] != 600 {
		t.Fatalf("instance limits %v, want 300 / 600", kinds.Limits)
	}
	if _, err := startInstance(fakeSpec(t, "c", nil), "", "127.0.0.1:0", t.TempDir(), "disk", &probeBudget{maxRate: 300, treeRate: 100}); err == nil || !strings.Contains(err.Error(), "tree_rate_per_s") {
		t.Fatalf("a bad budget must refuse the startup: %v", err)
	}
}

// D10: what is only measured. Each node counts the bytes of the records it emitted (the
// encoded sample), beside `seq` and `errors`; a mux counts what it republished, `src` included.
func TestBytesEmittedAreCountedPerNode(t *testing.T) {
	p := probePlane(t)
	mustCreate(t, p, node("agg", "mux@1", 0, nil))
	mustCreate(t, p, node("hb", "heartbeat@1", 20, map[string]any{"parent": "agg"}))
	for _, name := range []string{"agg", "hb"} {
		if _, err := probeOp(t, p, Request{Op: "probe.start", Node: name}); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		var seq uint64
		for _, n := range listTree(t, p).Nodes {
			if n.Name == "hb" {
				seq = n.Seq
			}
		}
		if seq >= 3 || time.Now().After(deadline) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.stop", Node: "hb"}); err != nil {
		t.Fatal(err)
	}
	tree := listTree(t, p)
	var hbSeq uint64
	for _, n := range tree.Nodes {
		if n.Name == "hb" {
			hbSeq = n.Seq
		}
	}
	_, _, hbBytes := tree.find(t, "hb")
	_, _, aggBytes := tree.find(t, "agg")
	if hbSeq < 3 || hbBytes < hbSeq*30 {
		t.Fatalf("hb: seq %d, bytes %d; want a few records of at least 30 bytes each", hbSeq, hbBytes)
	}
	if aggBytes < hbBytes {
		t.Fatalf("agg republished hb's records with src: %d bytes, hb %d", aggBytes, hbBytes)
	}
}

// D10: the bound is exact. Twelve leaves at 12 ms declare 1000/s to the unit and are admitted;
// the thirteenth is refused. A float sum would carry 1000.0000000000001 and refuse the twelfth:
// the plane counts in micro-records per second, in integers.
func TestTheBudgetBoundIsExactNotAFloatSum(t *testing.T) {
	p := probePlane(t)
	mustCreate(t, p, node("twelve", "mux@1", 0, map[string]any{"max_rate_per_s": 1000}))
	for i := 0; i < 12; i++ {
		mustCreate(t, p, node(fmt.Sprintf("t-%d", i), "heartbeat@1", 12, map[string]any{"parent": "twelve"}))
	}
	refused(t, create(t, p, node("t-12", "heartbeat@1", 12, map[string]any{"parent": "twelve"})), "twelve", "1000", "1083.333")
	if rate, _, _ := listTree(t, p).find(t, "twelve"); rate != 1000 {
		t.Fatalf("twelve declares %v/s, want 1000 to the thousandth", rate)
	}
}
