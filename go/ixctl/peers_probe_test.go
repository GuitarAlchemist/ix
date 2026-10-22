package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// A tree spanning two planes (D4, ix#402, slice 1 of S2 #408): what crosses the link is the
// intersection of what B marks exportable at create and what A's peer spec asks for, by exact
// name. Aggregation lives with the reader; in V1 the reader is the consumer of
// peer/<peer>/probe/<node> on A.
func TestOnlyExportedAndWantedProbePipesCrossTheLink(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	specB := fakeSpec(t, "b", nil)
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json"), Probes: []string{"cpu", "mem", "late"}}})
	b := newTestInstance(t, specB, dirB)
	a := newTestInstance(t, specA, dirA)
	waitStatus(t, a, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })

	c := dial(t, a)
	pipes := subscribe(t, c, "peer/b/probe/", nil)
	ctl := subscribe(t, c, "peer/b/ctl", nil)
	hb := func(name string, export bool) { probeOn(t, c, name, "heartbeat@1", export) }
	hb("cpu", true)       // exportable and wanted: crosses
	hb("cpu-extra", true) // exportable, not wanted: names are exact, never a prefix
	hb("mem", false)      // wanted, not exportable: stays on B, and the status says so

	// B's own declaration says export, in the lifecycle event and in the list.
	e := waitEnv(t, ctl, 5*time.Second, "mirrored probe.created cpu", func(e *Envelope) bool {
		return e.Ev.Kind == "probe.created" && dataOf(e)["node"] == "cpu"
	})
	if dataOf(e)["export"] != true {
		t.Fatalf("probe.created of an exported node says %v", e.Ev.Data)
	}
	var list struct {
		Nodes []map[string]any `json:"nodes"`
	}
	json.Unmarshal(do(t, c, Request{Op: "probe.list", Peer: "b"}), &list)
	exported := map[string]bool{}
	for _, n := range list.Nodes {
		exported[n["name"].(string)] = n["export"] == true
	}
	if !exported["cpu"] || !exported["cpu-extra"] || exported["mem"] {
		t.Fatalf("probe.list export flags: %v", exported)
	}

	// The wanted, exported pipe crosses, with B's origin.
	s := waitEnv(t, pipes, 5*time.Second, "mirrored cpu sample", func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu" && e.Ev.Kind == "sample"
	})
	if s.Origin != "b" || dataOf(s)["node"] != "cpu" {
		t.Fatalf("mirrored sample %+v", s)
	}
	// Provenance (slice 2): the mirrored envelope says which of B's envelopes it is, so a
	// consumer on A keys the series (origin:src_epoch:node:<src_seq of probe.created>) and
	// dedups (origin:src_epoch:src_seq) as a consumer on B would.
	if s.SrcSeq == 0 || s.SrcEpoch == "" {
		t.Fatalf("mirrored sample carries no source provenance: src_seq %d, src_epoch %q", s.SrcSeq, s.SrcEpoch)
	}
	if e.SrcSeq == 0 || e.SrcEpoch != s.SrcEpoch {
		t.Fatalf("mirrored probe.created carries src_seq %d, src_epoch %q (sample's %q): the series cannot be keyed", e.SrcSeq, e.SrcEpoch, s.SrcEpoch)
	}
	var zero uint64
	direct := subscribe(t, dial(t, b), "probe/cpu", &zero)
	src := waitEnv(t, direct, 5*time.Second, "the source envelope of the mirrored sample", func(x *Envelope) bool {
		return x.Seq == s.SrcSeq
	})
	if src.Epoch != s.SrcEpoch || dataOf(src)["seq"] != dataOf(s)["seq"] || src.SrcSeq != 0 {
		t.Fatalf("source %d/%s data seq %v (src_seq %d), mirrored says %d/%s data seq %v",
			src.Seq, src.Epoch, dataOf(src)["seq"], src.SrcSeq, s.SrcSeq, s.SrcEpoch, dataOf(s)["seq"])
	}
	// The mirrored probe.created names B's own probe.created of this node (the series' anchor).
	ctlB := subscribe(t, dial(t, b), "ctl", &zero)
	created := waitEnv(t, ctlB, 5*time.Second, "the source probe.created of cpu", func(x *Envelope) bool {
		return x.Seq == e.SrcSeq
	})
	if created.Ev.Kind != "probe.created" || dataOf(created)["node"] != "cpu" || created.Epoch != e.SrcEpoch {
		t.Fatalf("the mirrored probe.created names b's envelope %d/%s, which is %s %s", e.SrcSeq, e.SrcEpoch, created.Ev.Kind, created.Ev.Data)
	}
	// The status says what each wanted name is: mirrored, not exportable, or absent.
	wantProbes := func(want ...string) func(StatusReply) bool {
		return func(st StatusReply) bool {
			got := make([]string, 0, len(st.Peers[0].Probes))
			for _, p := range st.Peers[0].Probes {
				got = append(got, p.Name+":"+p.State)
			}
			return strings.Join(got, " ") == strings.Join(want, " ")
		}
	}
	waitStatus(t, a, 5*time.Second, "probe states", wantProbes("cpu:mirrored", "mem:not-exportable", "late:absent"))

	// A node created after the link is up is picked up from the mirrored lifecycle.
	hb("late", true)
	waitEnv(t, pipes, 5*time.Second, "mirrored late sample", func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/late" && e.Ev.Kind == "sample"
	})
	waitStatus(t, a, 5*time.Second, "late mirrored", wantProbes("cpu:mirrored", "mem:not-exportable", "late:mirrored"))

	// Nothing else ever crosses: not the exportable-but-unwanted cpu-extra (a prefix match on
	// probe/cpu would carry it), not the wanted-but-not-exportable mem.
	if f, err := pipes.next(300*time.Millisecond, func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu-extra" || e.Ch == "peer/b/probe/mem"
	}); err == nil {
		t.Fatalf("crossed the link: %s", f.Env.Ch)
	} else if !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("waiting for nothing: %v", err)
	}

	// A removed node is absent again; its pipe is no longer mirrored, so a recreation under the
	// same name without export stays on B.
	do(t, c, Request{Op: "probe.remove", Peer: "b", Node: "cpu"})
	waitStatus(t, a, 5*time.Second, "cpu absent", wantProbes("cpu:absent", "mem:not-exportable", "late:mirrored"))
	// The unexported incarnation is a mem@1 so its samples can be told from a heartbeat's.
	probeOn(t, c, "cpu", "mem@1", false)
	waitStatus(t, a, 5*time.Second, "cpu not exportable", wantProbes("cpu:not-exportable", "mem:not-exportable", "late:mirrored"))
	// Matched by kind: a heartbeat of the removed incarnation may still sit in this reader's
	// queue, mirrored before the removal; only a mem@1 would be the new incarnation crossing.
	if f, err := pipes.next(300*time.Millisecond, func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu" && dataOf(e)["kind"] == "mem@1"
	}); err == nil {
		t.Fatalf("a recreated, unexported cpu crossed the link: seq %d", f.Env.Seq)
	} else if !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("waiting for nothing: %v", err)
	}

	// A third incarnation, exported again: the link resumes from nothing, not from the seq it
	// last mirrored, so the unexported incarnation's samples (still in B's ring on the same
	// channel name) are never replayed (Codex round 35).
	do(t, c, Request{Op: "probe.remove", Peer: "b", Node: "cpu"})
	waitStatus(t, a, 5*time.Second, "cpu absent again", wantProbes("cpu:absent", "mem:not-exportable", "late:mirrored"))
	hb("cpu", true)
	first := waitEnv(t, pipes, 5*time.Second, "the re-exported cpu's first mirrored sample", func(e *Envelope) bool {
		return e.Ch == "peer/b/probe/cpu" && e.Ev.Kind == "sample" // past the live start's marker
	})
	if k := dataOf(first)["kind"]; k != "heartbeat@1" {
		t.Fatalf("the re-exported cpu's first mirrored sample is a %v: the unexported incarnation was replayed", k)
	}
	waitStatus(t, a, 5*time.Second, "cpu mirrored again", wantProbes("cpu:mirrored", "mem:not-exportable", "late:mirrored"))
	_ = b
}

// probeOn creates a 10 ms node of the kind on B through A and starts it.
func probeOn(t *testing.T, c *Client, name, kind string, export bool) {
	t.Helper()
	extra := map[string]any{}
	if export {
		extra["export"] = true
	}
	do(t, c, Request{Op: "probe.create", Peer: "b", Probe: node(name, kind, 10, extra)})
	do(t, c, Request{Op: "probe.start", Peer: "b", Node: name})
}

func dataOf(e *Envelope) map[string]any {
	var d map[string]any
	json.Unmarshal(e.Ev.Data, &d)
	return d
}

// The parent's request is validated with the spec: names, no duplicates.
func TestPeerProbesAreValidatedInTheSpec(t *testing.T) {
	for _, tc := range []struct{ probes, want string }{
		{`["cpu","cpu"]`, "duplicate"},
		{`["Bad Name"]`, "must match"},
	} {
		path := filepath.Join(t.TempDir(), "a.json")
		spec := `{"name":"a","services":[],"peers":[{"name":"b","state_file":"b.json","probes":` + tc.probes + `}]}`
		if err := os.WriteFile(path, []byte(spec), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadSpec(path); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("probes %s: %v, want %q", tc.probes, err, tc.want)
		}
	}
}

// A dropped pipe leaves nothing behind: its forwarder ends with the subscription (Codex round
// 36: Unsubscribe removed the channel the client's reader would have closed with the
// connection, so every drop left a goroutine ranging over it for the life of the process).
func TestADroppedPipeLeavesNoGoroutineBehind(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	specB := fakeSpec(t, "b", nil)
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json"), Probes: []string{"cpu"}}})
	newTestInstance(t, specB, dirB)
	a := newTestInstance(t, specA, dirA)
	waitStatus(t, a, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })
	c := dial(t, a)
	state := func(want string) func(StatusReply) bool {
		return func(st StatusReply) bool {
			return len(st.Peers[0].Probes) == 1 && st.Peers[0].Probes[0].State == want
		}
	}
	cycle := func() {
		probeOn(t, c, "cpu", "heartbeat@1", true)
		waitStatus(t, a, 5*time.Second, "cpu mirrored", state(pipeMirrored))
		do(t, c, Request{Op: "probe.remove", Peer: "b", Node: "cpu"})
		waitStatus(t, a, 5*time.Second, "cpu absent", state(pipeAbsent))
	}
	cycle()
	time.Sleep(300 * time.Millisecond) // the first drop's unsubscribe and the node's end settle
	base := runtime.NumGoroutine()
	const drops = 10
	for i := 0; i < drops; i++ {
		cycle()
	}
	// +2 absorbs a timer or an unsubscribe still ending; a leak per drop shows as +10.
	var got int
	for d := time.Now().Add(3 * time.Second); time.Now().Before(d); time.Sleep(20 * time.Millisecond) {
		if got = runtime.NumGoroutine(); got <= base+2 {
			return
		}
	}
	t.Fatalf("%d goroutines after %d more drops, %d before: a dropped pipe's forwarder lives on", got, drops, base)
}

// A peer's frame without a seq or an epoch has no provenance to mirror: the hub assigns seqs
// from 1 and every boot has an epoch, so such a frame is malformed, and it ends the session
// rather than becoming a mirror the contract forbids (Codex round 39).
func TestAPeerFrameWithoutProvenanceEndsTheSession(t *testing.T) {
	l := &peerLink{spec: PeerSpec{Name: "b"}, hub: NewHub("a"), since: map[string]uint64{}}
	refresh := make(chan struct{}, 1)
	drain := func() {
		select {
		case <-refresh:
		default:
		}
	}
	for _, env := range []*Envelope{
		{Ch: "svc/x", Seq: 0, Epoch: "b0b0b0b0b0b0", Origin: "b", Ev: Event{Kind: "stats"}},
		{Ch: "svc/x", Seq: 7, Epoch: "", Origin: "b", Ev: Event{Kind: "stats"}},
	} {
		err := l.apply(context.Background(), nil, nil, nil, refresh, 1, "b", peerFrame{prefix: "svc/x", f: Frame{Env: env}})
		drain()
		if err == nil || !strings.Contains(err.Error(), "provenance") {
			t.Fatalf("frame seq %d epoch %q: %v, want a provenance error", env.Seq, env.Epoch, err)
		}
	}
	if l.hub.Head() != 0 {
		t.Fatalf("%d envelopes mirrored from malformed frames", l.hub.Head())
	}
	good := &Envelope{Ch: "svc/x", Seq: 7, Epoch: "b0b0b0b0b0b0", Origin: "b", Ev: Event{Kind: "stats"}}
	err := l.apply(context.Background(), nil, nil, nil, refresh, 1, "b", peerFrame{prefix: "svc/x", f: Frame{Env: good}})
	drain()
	if err != nil || l.hub.Head() != 1 {
		t.Fatalf("a well-formed frame: %v, head %d", err, l.hub.Head())
	}
}

// The hub writes provenance as a pair or not at all, in the struct as in the bytes.
func TestMirrorProvenanceIsAPairInStructAndBytes(t *testing.T) {
	h := NewHub("a")
	for _, tc := range []struct {
		seq   uint64
		epoch string
		want  bool
	}{{0, "b0b0b0b0b0b0", false}, {7, "", false}, {7, "b0b0b0b0b0b0", true}} {
		env := h.Mirror("peer/b/svc/x", "b", 0, Event{Kind: "stats"}, tc.seq, tc.epoch)
		raw := string(env.Raw())
		inRaw := strings.Contains(raw, `"src_seq":`) && strings.Contains(raw, `"src_epoch":`)
		halfRaw := strings.Contains(raw, `"src_seq":`) != strings.Contains(raw, `"src_epoch":`)
		inStruct := env.SrcSeq != 0 && env.SrcEpoch != ""
		halfStruct := (env.SrcSeq != 0) != (env.SrcEpoch != "")
		if inRaw != tc.want || inStruct != tc.want || halfRaw || halfStruct {
			t.Fatalf("Mirror(%d, %q): struct %d/%q, raw %s", tc.seq, tc.epoch, env.SrcSeq, env.SrcEpoch, raw)
		}
	}
}
