package main

import (
	"encoding/json"
	"os"
	"path/filepath"
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
	hb := func(name string, export bool) {
		extra := map[string]any{}
		if export {
			extra["export"] = true
		}
		do(t, c, Request{Op: "probe.create", Peer: "b", Probe: node(name, "heartbeat@1", 10, extra)})
		do(t, c, Request{Op: "probe.start", Peer: "b", Node: name})
	}
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
	hb("cpu", false)
	waitStatus(t, a, 5*time.Second, "cpu not exportable", wantProbes("cpu:not-exportable", "mem:not-exportable", "late:mirrored"))
	if f, err := pipes.next(300*time.Millisecond, func(e *Envelope) bool { return e.Ch == "peer/b/probe/cpu" }); err == nil {
		t.Fatalf("a recreated, unexported cpu crossed the link: seq %d", f.Env.Seq)
	}
	_ = b
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
