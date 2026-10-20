package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Two instances federated both ways: A starts a service on B through A's
// WebSocket, sees B's events mirrored as peer/b/..., and nothing loops.
func TestFederationForwardsAndMirrors(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	specB := fakeSpec(t, "b", []PeerSpec{{Name: "a", StateFile: filepath.Join(dirA, "a.json")}},
		fakeSvc{name: "pod-x", args: []string{"--nodes", "2", "--rate", "50"}})
	specA := fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dirB, "b.json")}})
	b := newTestInstance(t, specB, dirB)
	a := newTestInstance(t, specA, dirA)

	waitStatus(t, a, 10*time.Second, "peer b up", func(st StatusReply) bool { return st.Peers[0].State == "up" })
	waitStatus(t, b, 10*time.Second, "peer a up", func(st StatusReply) bool { return st.Peers[0].State == "up" })

	c := dial(t, a)
	w := subscribe(t, c, "peer/b/", nil)
	var r ApplyReply
	json.Unmarshal(do(t, c, Request{Op: "apply", Peer: "b", Desired: running("pod-x")}), &r)
	if r.Generation != 1 || len(r.Actions) != 1 {
		t.Fatalf("forwarded apply %+v", r)
	}
	if b.plane.Status().Generation != 1 || a.plane.Status().Generation != 0 {
		t.Fatal("the apply ran on the wrong instance")
	}
	e := waitEnv(t, w, 10*time.Second, "mirrored service.started", func(e *Envelope) bool {
		return e.Ch == "peer/b/ctl" && e.Ev.Kind == "service.started"
	})
	if e.Origin != "b" {
		t.Fatalf("mirrored origin %q", e.Origin)
	}
	waitEnv(t, w, 10*time.Second, "mirrored converged", func(e *Envelope) bool {
		return e.Ch == "peer/b/ctl" && e.Ev.Kind == "converged" && genOf(e) == 1
	})
	waitEnv(t, w, 10*time.Second, "mirrored stats", func(e *Envelope) bool {
		return e.Ch == "peer/b/svc/pod-x" && e.Ev.Kind == "stats"
	})
	waitStatus(t, a, 5*time.Second, "peer services", func(st StatusReply) bool {
		return strings.Contains(string(st.Peers[0].Services), `"state":"running"`)
	})

	// A plan forwarded to b is checked by b's rules.
	if _, err := c.Do(t.Context(), Request{Op: "plan", Peer: "b", Desired: running("nope")}); err == nil || !strings.Contains(err.Error(), "nope") {
		t.Fatalf("b's refusal relayed: %v", err)
	}
	// On HTTP the peer is named beside the document, as ?peer=; the body stays
	// the document alone (F2.3).
	if code, _, body := httpDo(t, a, "POST", "/v1/plan?peer=b", running("nope"), nil); code != 400 || !strings.Contains(string(body), "nope") {
		t.Fatalf("HTTP plan?peer=b refusal: %d %s", code, body)
	}
	code, _, body := httpDo(t, a, "POST", "/v1/plan?peer=b", running("pod-x"), nil)
	var pr PlanReply
	json.Unmarshal(body, &pr)
	if code != 200 || len(pr.Actions) != 0 || len(pr.Noop) != 1 || pr.Noop[0] != "pod-x" {
		t.Fatalf("HTTP plan?peer=b on b's running service: %d %s", code, body)
	}

	var sr StopReply
	json.Unmarshal(do(t, c, Request{Op: "stop", Peer: "b", Service: "pod-x"}), &sr)
	if sr.Stopped != "pod-x" || sr.Exit == nil {
		t.Fatalf("forwarded stop %+v", sr)
	}
	if s := svcStatus(b.plane.Status(), "pod-x"); s.State != stateStopped {
		t.Fatalf("on b after stop: %+v", s)
	}

	// Neither journal ever holds a token, its own or its peer's.
	for _, in := range []*instance{a, b} {
		j, err := os.ReadFile(filepath.Join(filepath.Dir(in.stateFile), in.plane.name+".journal.jsonl"))
		if err != nil || len(j) == 0 {
			t.Fatalf("journal of %s: %v", in.plane.name, err)
		}
		if strings.Contains(string(j), a.token) || strings.Contains(string(j), b.token) {
			t.Fatalf("the journal of %s holds a token", in.plane.name)
		}
	}

	// No loops: neither ring holds a mirror of a mirror.
	time.Sleep(100 * time.Millisecond)
	for _, in := range []*instance{a, b} {
		zero := uint64(0)
		cur := in.plane.Subscribe("peer/", &zero, "")
		for {
			envs, gap, err := cur.Next(shortCtx(t), 4096)
			if err != nil || gap != nil {
				break
			}
			for _, env := range envs {
				if strings.Count(env.Ch, "peer/") > 1 {
					t.Fatalf("%s mirrored a mirror: %s", in.plane.name, env.Ch)
				}
			}
		}
	}
}

func TestPeerDownIsReported(t *testing.T) {
	dir := t.TempDir()
	a := newTestInstance(t, fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(dir, "missing.json")}}), "")
	c := dial(t, a)
	_, err := c.Do(t.Context(), Request{Op: "status", Peer: "b"})
	if err == nil || !strings.Contains(err.Error(), "down") {
		t.Fatalf("op to a down peer: %v", err)
	}
	if code, _, _ := httpDo(t, a, "POST", "/v1/stop", map[string]string{"service": "x", "peer": "b"}, nil); code != 502 {
		t.Fatalf("HTTP stop to a down peer: %d", code)
	}
}
