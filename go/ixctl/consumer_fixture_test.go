package main

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// A consumer's fixtures (story S7, ix#413, for the observation envelope of gaia#245): what a
// reader of the plane sees across a reconnect, a same-name recreation of a node and a restart
// of the plane, written by the real plane (IXCTL_WRITE_FIXTURES=1) into e2e/fixtures/ and
// checked by shape on every run, so the files a consumer develops against never drift from
// the plane. The shape of a frame is its channel, event kind, node and source (or its gap
// reason); counts and values (seq, t, epoch) vary with timing and are checked as invariants
// instead: no duplicate and no hole across a reconnect (the dedup key is origin:epoch:seq),
// a recreated node restarting its own seq under a new probe.created while the envelope seq
// goes on, and a restart answering epoch-changed first with an empty tree behind it.
type fixtureFrame struct {
	Step string    `json:"step"`
	Env  *Envelope `json:"env,omitempty"`
	Gap  *Gap      `json:"gap,omitempty"`
}

func fixtureShape(f fixtureFrame) string {
	if f.Gap != nil {
		return "gap " + f.Gap.Reason
	}
	var d struct {
		Node string `json:"node"`
		Src  string `json:"src"`
	}
	json.Unmarshal(f.Env.Ev.Data, &d)
	return "env " + f.Env.Ch + " " + f.Env.Ev.Kind + " node=" + d.Node + " src=" + d.Src
}

// shapesByStep is the sorted set of shapes each step produced: what a consumer can rely on
// seeing, independent of how many ticks a heartbeat managed between two ops.
func shapesByStep(frames []fixtureFrame) map[string][]string {
	seen := map[string]map[string]bool{}
	for _, f := range frames {
		if seen[f.Step] == nil {
			seen[f.Step] = map[string]bool{}
		}
		seen[f.Step][fixtureShape(f)] = true
	}
	out := map[string][]string{}
	for step, set := range seen {
		for s := range set {
			out[step] = append(out[step], s)
		}
		sort.Strings(out[step])
	}
	return out
}

func sampleSeq(e *Envelope) uint64 {
	var d struct {
		Seq uint64 `json:"seq"`
	}
	json.Unmarshal(e.Ev.Data, &d)
	return d.Seq
}

// collect reads frames off a subscription, tagging them with step, until done says so or the
// deadline passes (a failure: a consumer never waits forever either).
func collect(t *testing.T, frames <-chan Frame, step string, out *[]fixtureFrame, done func(fixtureFrame) bool) {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		select {
		case f, ok := <-frames:
			if !ok {
				t.Fatalf("%s: subscription closed", step)
			}
			if f.Env == nil && f.Gap == nil {
				continue
			}
			ff := fixtureFrame{Step: step, Env: f.Env, Gap: f.Gap}
			*out = append(*out, ff)
			if done(ff) {
				return
			}
		case <-deadline:
			t.Fatalf("%s: no end after %d frames", step, len(*out))
		}
	}
}

func nthSampleOf(node string, n int) func(fixtureFrame) bool {
	count := 0
	return func(f fixtureFrame) bool {
		if f.Env != nil && f.Env.Ch == "probe/"+node && f.Env.Ev.Kind == "sample" {
			count++
		}
		return count >= n
	}
}

func checkOrWriteFixture(t *testing.T, name string, frames []fixtureFrame) {
	t.Helper()
	path := filepath.Join("e2e", "fixtures", "consumer-"+name+".jsonl")
	if os.Getenv("IXCTL_WRITE_FIXTURES") != "" {
		os.MkdirAll(filepath.Dir(path), 0o755)
		var b strings.Builder
		for _, f := range frames {
			line, _ := json.Marshal(f)
			b.Write(line)
			b.WriteByte('\n')
		}
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s (%d frames)", path, len(frames))
		return
	}
	fh, err := os.Open(path)
	if err != nil {
		t.Fatalf("%v (write the fixtures with IXCTL_WRITE_FIXTURES=1 go test -run Fixture)", err)
	}
	defer fh.Close()
	var kept []fixtureFrame
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		var f fixtureFrame
		if err := json.Unmarshal(sc.Bytes(), &f); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		kept = append(kept, f)
	}
	want, got := shapesByStep(kept), shapesByStep(frames)
	for step := range want {
		if strings.Join(want[step], "|") != strings.Join(got[step], "|") {
			t.Errorf("%s, step %s: the plane now produces %v, the fixture holds %v (regenerate with IXCTL_WRITE_FIXTURES=1 if the change is meant)", path, step, got[step], want[step])
		}
	}
	for step := range got {
		if _, ok := want[step]; !ok {
			t.Errorf("%s: step %s missing from the fixture", path, step)
		}
	}
}

func TestAConsumerSeesReconnectRecreationAndRestartInItsFixtures(t *testing.T) {
	dir := t.TempDir()
	first := newTestInstance(t, fakeSpec(t, "a", nil), dir)
	c := dial(t, first)
	hb := node("hb", "heartbeat@1", 10, nil)

	// Reconnect: a live reader of everything sees the node created and started and its
	// samples; it drops the connection and resumes from the last seq it kept, with the epoch:
	// the next envelope is seq + 1, nothing twice, no gap.
	var reconnect []fixtureFrame
	_, frames, err := c.Subscribe(context.Background(), "", nil, 4096)
	if err != nil {
		t.Fatal(err)
	}
	do(t, c, Request{Op: "probe.create", Probe: hb})
	do(t, c, Request{Op: "probe.start", Node: "hb"})
	collect(t, frames, "live", &reconnect, nthSampleOf("hb", 3))
	last := reconnect[len(reconnect)-1].Env
	epoch := last.Epoch
	c.Close()
	c = dial(t, first)
	_, frames, err = c.SubscribeFrom(context.Background(), "", uint64p(last.Seq), epoch, 4096)
	if err != nil {
		t.Fatal(err)
	}
	collect(t, frames, "after-reconnect", &reconnect, nthSampleOf("hb", 3))
	var resumed fixtureFrame
	for _, f := range reconnect {
		if f.Step == "after-reconnect" {
			resumed = f
			break
		}
	}
	for i, f := range reconnect {
		if f.Gap != nil {
			t.Fatalf("reconnect: gap %+v at %d", f.Gap, i)
		}
		if i > 0 && f.Env.Seq != reconnect[i-1].Env.Seq+1 {
			t.Fatalf("reconnect: seq %d after %d (frame %d)", f.Env.Seq, reconnect[i-1].Env.Seq, i)
		}
		if f.Env.Epoch != epoch || f.Env.Origin != "a" {
			t.Fatalf("reconnect: epoch %s origin %s", f.Env.Epoch, f.Env.Origin)
		}
	}
	if resumed.Step != "after-reconnect" || resumed.Env.Seq != last.Seq+1 {
		t.Fatalf("resumed at %+v, want seq %d", resumed, last.Seq+1)
	}
	checkOrWriteFixture(t, "reconnect", reconnect)

	// Same-name recreation, on the live stream: hb is stopped (probe.stopped is the last thing
	// said about it until it starts again: a consumer marks it stale from there), removed, and
	// created again under the same name. The new node's own seq restarts at 1 under a new
	// probe.created while the envelope seq goes on: a series is keyed by the probe.created it
	// follows, not by the name alone.
	var recreation []fixtureFrame
	do(t, c, Request{Op: "probe.stop", Node: "hb"})
	collect(t, frames, "stopped", &recreation, func(f fixtureFrame) bool {
		return f.Env != nil && f.Env.Ch == "ctl" && f.Env.Ev.Kind == "probe.stopped"
	})
	do(t, c, Request{Op: "probe.remove", Node: "hb"})
	collect(t, frames, "removed", &recreation, func(f fixtureFrame) bool {
		return f.Env != nil && f.Env.Ch == "ctl" && f.Env.Ev.Kind == "probe.removed"
	})
	do(t, c, Request{Op: "probe.create", Probe: hb})
	do(t, c, Request{Op: "probe.start", Node: "hb"})
	collect(t, frames, "recreated", &recreation, nthSampleOf("hb", 2))
	var firstNew *Envelope
	for _, f := range recreation {
		if f.Step == "recreated" && f.Env.Ch == "probe/hb" && f.Env.Ev.Kind == "sample" {
			firstNew = f.Env
			break
		}
	}
	if firstNew == nil || sampleSeq(firstNew) != 1 || firstNew.Seq <= last.Seq || firstNew.Epoch != epoch {
		t.Fatalf("recreated hb's first sample: %+v (node seq %d), want node seq 1 past envelope seq %d", firstNew, sampleSeq(firstNew), last.Seq)
	}
	for _, f := range recreation {
		if f.Gap != nil {
			t.Fatalf("recreation: gap %+v", f.Gap)
		}
	}
	checkOrWriteFixture(t, "recreation", recreation)

	// Restart: the plane comes back under its name in the same state dir with another epoch and
	// an empty tree. A consumer resuming from the seq and epoch it kept is told epoch-changed
	// first; what follows is the new boot from its seq 1, so its own series end there: it marks
	// every node unknown until the new boot's probe.created, and must not merge the new boot's
	// seq 1 with the old boot's.
	var restart []fixtureFrame
	kept := recreation[len(recreation)-1].Env
	c.Close()
	first.close()
	second := newTestInstance(t, fakeSpec(t, "a", nil), dir)
	c = dial(t, second)
	var list struct {
		Count int `json:"count"`
	}
	json.Unmarshal(do(t, c, Request{Op: "probe.list"}), &list)
	if list.Count != 0 {
		t.Fatalf("after the restart the tree has %d nodes, want none (not persisted)", list.Count)
	}
	_, frames, err = c.SubscribeFrom(context.Background(), "", uint64p(kept.Seq), kept.Epoch, 4096)
	if err != nil {
		t.Fatal(err)
	}
	collect(t, frames, "resumed-after-restart", &restart, func(f fixtureFrame) bool { return true })
	if restart[0].Gap == nil || restart[0].Gap.Reason != "epoch-changed" || restart[0].Gap.Epoch != second.plane.Epoch() || restart[0].Gap.Oldest != 1 {
		t.Fatalf("after the restart: %+v, want epoch-changed naming the new boot, oldest 1", restart[0])
	}
	do(t, c, Request{Op: "probe.create", Probe: hb})
	do(t, c, Request{Op: "probe.start", Node: "hb"})
	collect(t, frames, "recreated-after-restart", &restart, nthSampleOf("hb", 2))
	for _, f := range restart[1:] {
		if f.Gap != nil || f.Env.Epoch != second.plane.Epoch() || f.Env.Epoch == epoch || f.Env.Origin != "a" {
			t.Fatalf("after the restart: %+v, want the new boot's envelopes only", f)
		}
		if f.Env.Ch == "probe/hb" && f.Env.Ev.Kind == "sample" && sampleSeq(f.Env) == 1 && f.Env.Seq >= kept.Seq {
			t.Fatalf("after the restart hb's seq 1 came at envelope seq %d, past the old boot's %d: a consumer keyed on seq alone would merge the two", f.Env.Seq, kept.Seq)
		}
	}
	checkOrWriteFixture(t, "restart", restart)
}
