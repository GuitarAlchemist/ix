package main

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

// What a pipe says, and in what order, is the consumer's ground (the observation contract,
// S7). Three guarantees the tree makes under its lock, each tested where the race lived:
// a node's records reach its pipe in the order of their seq even when several children feed
// a mux at once; a remove that meets a stop in progress waits for that stop's probe.stopped
// before it says probe.removed; a failed read reaches the pipe whatever the node's filter,
// so ERROR can be read from the stream.

func nodeOf(t *testing.T, p *Plane, name string) *probeNode {
	t.Helper()
	p.probes.mu.Lock()
	defer p.probes.mu.Unlock()
	n := p.probes.nodes[name]
	if n == nil {
		t.Fatalf("no node %q", name)
	}
	return n
}

func TestAMuxPublishesItsRecordsInSeqOrderUnderConcurrentChildren(t *testing.T) {
	p := probePlane(t)
	create := func(spec *NodeSpec) {
		t.Helper()
		if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: spec}); err != nil {
			t.Fatal(err)
		}
	}
	create(node("agg", "mux@1", 0, nil))
	children := []string{"a", "b", "c", "d"}
	for _, c := range children {
		create(node(c, "heartbeat@1", 1000, map[string]any{"parent": "agg"}))
	}
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: "agg"}); err != nil {
		t.Fatal(err)
	}
	// The children are created, not started: the test feeds them itself, from four
	// goroutines at once, the way four samplers would at the same instant.
	cur := p.Subscribe("probe/agg", uint64p(0), "")
	const each = 250
	var wg sync.WaitGroup
	for _, c := range children {
		n := nodeOf(t, p, c)
		p.probes.mu.Lock()
		n.state = probeRunning // a child feeds its parent only while running
		p.probes.mu.Unlock()
		wg.Add(1)
		go func(n *probeNode) {
			defer wg.Done()
			for i := 1; i <= each; i++ {
				p.probes.emit(n, map[string]any{"n": i}, false)
			}
		}(n)
	}
	wg.Wait()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	want := uint64(len(children) * each)
	var got uint64
	for got < want {
		envs, gap, err := cur.Next(ctx, 256)
		if err != nil {
			t.Fatalf("after %d records: %v", got, err)
		}
		if gap != nil {
			t.Fatalf("gap %+v", gap)
		}
		for _, env := range envs {
			var d struct {
				Seq uint64 `json:"seq"`
				Src string `json:"src"`
			}
			json.Unmarshal(env.Ev.Data, &d)
			got++
			if d.Seq != got {
				t.Fatalf("the mux's record %d came with seq %d (src %s): records of one node must reach its pipe in seq order", got, d.Seq, d.Src)
			}
		}
	}
}

func TestARemoveWaitsForAStopInProgress(t *testing.T) {
	p := probePlane(t)
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("hb", "heartbeat@1", 10, nil)}); err != nil {
		t.Fatal(err)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: "hb"}); err != nil {
		t.Fatal(err)
	}
	// Hold the stop open the way a slow last tick would: stopped by another op, whose
	// probe.stopped is not out yet.
	n := nodeOf(t, p, "hb")
	p.probes.mu.Lock()
	cancel, done := n.cancel, n.done
	n.cancel, n.done, n.state, n.stopping = nil, nil, probeStopped, true
	p.probes.mu.Unlock()
	cancel()
	<-done
	cur := p.Subscribe("ctl", uint64p(0), "")
	removed := make(chan error, 1)
	go func() {
		_, err := probeOp(t, p, Request{Op: "probe.remove", Node: "hb"})
		removed <- err
	}()
	select {
	case err := <-removed:
		t.Fatalf("remove returned while the stop was in progress (err %v): it must wait for probe.stopped", err)
	case <-time.After(150 * time.Millisecond):
	}
	// The stop finishes: its probe.stopped goes out, then the remove may proceed.
	p.probes.hub.Publish("ctl", event("probe.stopped", map[string]any{"node": "hb", "kind": "heartbeat@1", "reason": "requested"}))
	p.probes.finishStop(n)
	select {
	case err := <-removed:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("remove did not return once the stop had finished")
	}
	ctx, cancel2 := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel2()
	var kinds []string
	for len(kinds) < 2 {
		envs, _, err := cur.Next(ctx, 8)
		if err != nil {
			t.Fatal(err)
		}
		for _, env := range envs {
			if strings.HasPrefix(env.Ev.Kind, "probe.stopped") || env.Ev.Kind == "probe.removed" {
				kinds = append(kinds, env.Ev.Kind)
			}
		}
	}
	if kinds[0] != "probe.stopped" || kinds[1] != "probe.removed" {
		t.Fatalf("ctl said %v, want probe.stopped then probe.removed", kinds)
	}
	if nodes := listNodes(t, p); len(nodes) != 0 {
		t.Fatalf("after the remove: %+v", nodes)
	}
}

// ctl's lifecycle events come in the order of the state changes they report, whatever the
// interleaving of the ops: after any number of concurrent start/stop pairs on a mux, the last
// event of the node names its final state; after concurrent remove/create pairs under one
// name, no probe.removed follows the probe.created that opened the series it would close.
func TestLifecycleEventsFollowTheStateChangesUnderConcurrentOps(t *testing.T) {
	p := probePlane(t)
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("agg", "mux@1", 0, nil)}); err != nil {
		t.Fatal(err)
	}
	cur := p.Subscribe("ctl", uint64p(0), "")
	const rounds = 500 // start/stop rounds of four ops, then remove/create rounds: under the ring's 4096
	for i := 0; i < rounds; i++ {
		var wg sync.WaitGroup
		for _, op := range []string{"probe.start", "probe.stop", "probe.start", "probe.stop"} {
			wg.Add(1)
			go func(op string) {
				defer wg.Done()
				probeOp(t, p, Request{Op: op, Node: "agg"})
			}(op)
		}
		wg.Wait()
	}
	final := listNodes(t, p)[0].State
	mark := p.probes.hub.Head() // the removes below stop a running node first: judge up to here
	// Now remove and create under the same name, concurrently, until a create landed after a
	// remove in the same round.
	var wg sync.WaitGroup
	for i := 0; i < rounds/2; i++ {
		wg.Add(2)
		go func() { defer wg.Done(); probeOp(t, p, Request{Op: "probe.remove", Node: "agg"}) }()
		go func() {
			defer wg.Done()
			probeOp(t, p, Request{Op: "probe.create", Probe: node("agg", "mux@1", 0, nil)})
		}()
		wg.Wait()
	}
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node("agg", "mux@1", 0, nil)}); err != nil && !strings.Contains(err.Error(), "exists") {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	last, exists := "", false
	for {
		envs, gap, err := cur.Next(ctx, 256)
		if err != nil {
			t.Fatal(err)
		}
		if gap != nil {
			t.Fatalf("gap %+v", gap)
		}
		for _, env := range envs {
			switch env.Ev.Kind {
			case "probe.started", "probe.stopped":
				if env.Seq <= mark {
					last = env.Ev.Kind
				}
			case "probe.created":
				if exists {
					t.Fatalf("probe.created while the node existed, at ctl seq %d", env.Seq)
				}
				exists = true
			case "probe.removed":
				if !exists {
					t.Fatalf("probe.removed while no node existed, at ctl seq %d: it would close the next series", env.Seq)
				}
				exists = false
			}
		}
		if exists && p.probes.hub.Head() == envs[len(envs)-1].Seq {
			break
		}
	}
	if want := map[string]string{probeRunning: "probe.started", probeStopped: "probe.stopped"}[final]; last != want {
		t.Fatalf("the node ended %s but ctl's last lifecycle event before the removes was %s", final, last)
	}
}

func TestAFailedReadReachesThePipeWhateverTheFilter(t *testing.T) {
	p := probePlane(t)
	spec := node("hb", "heartbeat@1", 1000, map[string]any{"filter": map[string]any{
		"where": []map[string]any{{"field": "n", "op": ">", "value": 1}}, "every": 2, "keep": []string{"n"},
	}})
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: spec}); err != nil {
		t.Fatal(err)
	}
	n := nodeOf(t, p, "hb")
	p.probes.mu.Lock()
	n.state = probeRunning
	p.probes.mu.Unlock()
	p.probes.emit(n, map[string]any{"n": 1}, false)           // where: n > 1 fails, kept off the pipe
	p.probes.emit(n, map[string]any{"error": "boom"}, true)   // a failed read: on the pipe whatever the filter
	p.probes.emit(n, map[string]any{"n": 2}, false)           // every 2 keeps the 2nd, 4th... passing record: skipped
	p.probes.emit(n, map[string]any{"error": "boom 2"}, true) // again, and not counted by `every`
	p.probes.emit(n, map[string]any{"n": 3}, false)           // the 2nd passing record: kept
	p.probes.emit(n, map[string]any{"n": 4, "x": 1}, false)   // the 3rd: skipped
	p.probes.emit(n, map[string]any{"n": 5, "x": 1}, false)   // the 4th: kept, keep drops x
	samples := readSamples(t, p, "probe/hb", 4, time.Second)
	if len(samples) != 4 {
		t.Fatalf("%d samples, want 4: %v", len(samples), samples)
	}
	if samples[0]["error"] != "boom" || samples[1]["error"] != "boom 2" || samples[2]["n"] != float64(3) ||
		samples[3]["n"] != float64(5) || samples[3]["x"] != nil {
		t.Fatalf("pipe said %v", samples)
	}
	if hb := listNodes(t, p)[0]; hb.Errors != 2 || hb.Seq != 4 {
		t.Fatalf("listed %+v, want errors 2 and seq 4", hb)
	}
}
