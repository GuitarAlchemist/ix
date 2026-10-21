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

// blockSampler is a test kind's sampler that blocks inside sample() until the test releases
// it: a leaf's loop cannot end while a read is in flight, so a real probe.stop of the node
// stays in progress (state stopped, probe.stopped not out) for as long as the test wants,
// without touching the tree's fields.
type blockSampler struct{ entered, release chan struct{} }

func (b *blockSampler) sample() (map[string]any, error) {
	select {
	case b.entered <- struct{}{}:
	default:
	}
	<-b.release
	return map[string]any{"n": 1}, nil
}

// blockKind registers `block@1` in the kinds registry for one test (the suite is sequential:
// no test runs in parallel) and returns its sampler.
func blockKind(t *testing.T) *blockSampler {
	t.Helper()
	b := &blockSampler{entered: make(chan struct{}, 1), release: make(chan struct{})}
	saved := probeKinds
	probeKinds = append(append([]probeKind(nil), saved...), probeKind{
		Name: "block", Version: 1, Fields: []string{"n"},
		Doc:        "test only: blocks in sample until released",
		newSampler: func() sampler { return b }})
	t.Cleanup(func() { probeKinds = saved })
	return b
}

// stopInProgress starts a `block@1` leaf, lets its read block, and issues a real probe.stop
// that cannot finish until the sampler is released. It returns the stop's result channel.
func stopInProgress(t *testing.T, p *Plane, b *blockSampler, name string) <-chan error {
	t.Helper()
	if _, err := probeOp(t, p, Request{Op: "probe.create", Probe: node(name, "block@1", 10, nil)}); err != nil {
		t.Fatal(err)
	}
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: name}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-b.entered:
	case <-time.After(2 * time.Second):
		t.Fatal("the read never started")
	}
	stopped := make(chan error, 1)
	go func() {
		_, err := probeOp(t, p, Request{Op: "probe.stop", Node: name})
		stopped <- err
	}()
	deadline := time.Now().Add(2 * time.Second)
	for listNodes(t, p)[0].State != probeStopped { // the stop has set the state: it is in progress
		if time.Now().After(deadline) {
			t.Fatal("the stop never set the state")
		}
		time.Sleep(time.Millisecond)
	}
	return stopped
}

func TestARemoveWaitsForAStopInProgress(t *testing.T) {
	p := probePlane(t)
	b := blockKind(t)
	cur := p.Subscribe("ctl", uint64p(0), "")
	stopped := stopInProgress(t, p, b, "hb")
	removed := make(chan error, 1)
	go func() {
		_, err := probeOp(t, p, Request{Op: "probe.remove", Node: "hb"})
		removed <- err
	}()
	select {
	case err := <-removed:
		t.Fatalf("remove returned while the stop was in progress (err %v): it must wait for probe.stopped", err)
	case err := <-stopped:
		t.Fatalf("stop returned while its read was still in flight (err %v)", err)
	case <-time.After(150 * time.Millisecond):
	}
	// The read ends: the loop returns, the stop says probe.stopped, then the remove may go on.
	close(b.release)
	for _, c := range []<-chan error{stopped, removed} {
		select {
		case err := <-c:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("an op did not return once the read had ended")
		}
	}
	if kinds := lifecycleOf(t, cur, p.probes.hub.Head(), "probe.stopped", "probe.removed"); len(kinds) != 2 || kinds[0] != "probe.stopped" || kinds[1] != "probe.removed" {
		t.Fatalf("ctl said %v, want probe.stopped then probe.removed", kinds)
	}
	if nodes := listNodes(t, p); len(nodes) != 0 {
		t.Fatalf("after the remove: %+v", nodes)
	}
}

func TestShutdownWaitsForAStopInProgress(t *testing.T) {
	p := probePlane(t)
	b := blockKind(t)
	cur := p.Subscribe("ctl", uint64p(0), "")
	stopped := stopInProgress(t, p, b, "hb")
	down := make(chan struct{})
	go func() {
		p.probes.shutdown()
		close(down)
	}()
	select {
	case <-down:
		t.Fatal("shutdown returned while a stop was in progress: its probe.stopped would follow the shutdown")
	case <-time.After(150 * time.Millisecond):
	}
	close(b.release)
	select {
	case <-down:
	case <-time.After(2 * time.Second):
		t.Fatal("shutdown did not return once the read had ended")
	}
	if err := <-stopped; err != nil {
		t.Fatal(err)
	}
	if kinds := lifecycleOf(t, cur, p.probes.hub.Head(), "probe.stopped"); len(kinds) != 1 {
		t.Fatalf("ctl said %v, want one probe.stopped (the op's; shutdown had nothing running)", kinds)
	}
}

// lifecycleOf reads cur up to the hub seq `head` and returns the kinds among `wanted`, in order.
func lifecycleOf(t *testing.T, cur *Cursor, head uint64, wanted ...string) []string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	var kinds []string
	for {
		envs, gap, err := cur.Next(ctx, 64)
		if err != nil {
			t.Fatal(err)
		}
		if gap != nil {
			t.Fatalf("gap %+v", gap)
		}
		for _, env := range envs {
			for _, w := range wanted {
				if env.Ev.Kind == w {
					kinds = append(kinds, w)
				}
			}
		}
		if len(envs) > 0 && envs[len(envs)-1].Seq >= head {
			return kinds
		}
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
	wantLast := map[string]string{probeRunning: "probe.started", probeStopped: "probe.stopped"}
	var head uint64
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
		// After every round, ctl's last lifecycle word (up to the head read with the state)
		// names the state the round left: an inversion in any round is caught, not only the last.
		state := listNodes(t, p)[0].State
		if h := p.probes.hub.Head(); h > head {
			head = h
			if kinds := lifecycleOf(t, cur, head, "probe.started", "probe.stopped"); len(kinds) > 0 && kinds[len(kinds)-1] != wantLast[state] {
				t.Fatalf("round %d left the node %s but ctl's last lifecycle event was %s", i, state, kinds[len(kinds)-1])
			}
		}
	}
	// Now remove and create under the same name, concurrently: ctl must never say
	// probe.removed while no node exists, which would close the series the next
	// probe.created opened.
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
	exists := true // the node existed at `head`
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
