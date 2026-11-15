package main

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

// The in-memory frame transport: two planes federated with no socket, no state file and no HTTP
// server. Only the link's connection changes; what flows over it is the same ixctl.v1 session
// a WebSocket carries (TestFederationForwardsAndMirrors keeps the WebSocket adapter covered).

// memPlane is a plane alone: no HTTP server, no state file, no journal.
func memPlane(t *testing.T, spec *Spec, name string) *Plane {
	t.Helper()
	p, err := NewPlane(spec, name, "", "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(p.Shutdown)
	return p
}

// connectMem is a peer link's connection to b in memory. The session lives as long as b: b's
// shutdown ends it, as a closed socket would. b names itself by its epoch, as a state file's
// token names an instance.
func connectMem(b *Plane) func(context.Context) (*Client, stateFile, error) {
	return func(context.Context) (*Client, stateFile, error) {
		st := stateFile{Name: b.name, Addr: "mem:" + b.name, Token: "mem:" + b.Epoch()}
		if b.ctx.Err() != nil {
			return nil, st, errors.New("peer plane " + b.name + " is shut down")
		}
		return dialMem(b.ctx, b), st, nil
	}
}

func waitPeerState(t *testing.T, p *Plane, want string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		st := p.Status()
		if len(st.Peers) == 1 && st.Peers[0].State == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("peer state %+v, want %s", st.Peers, want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// The in-memory adapter keeps the WebSocket's read limit: a client frame over wsReadLimit ends
// the connection instead of reaching the session.
func TestMemFramesRefusesAFrameOverTheReadLimit(t *testing.T) {
	p := memPlane(t, fakeSpec(t, "p", nil), "p")
	p.Start()
	c := dialMem(t.Context(), p)
	t.Cleanup(func() { c.Close() })
	if err := c.conn.Write(t.Context(), make([]byte, wsReadLimit+1)); err == nil {
		t.Fatal("a frame over the read limit was taken")
	}
	select {
	case <-c.Done():
	case <-time.After(5 * time.Second):
		t.Fatal("the connection outlived a frame over the read limit")
	}
	if err := c.conn.Write(t.Context(), []byte(`{"id":1,"op":"status"}`)); err == nil {
		t.Fatal("a closed in-memory connection took a frame")
	}
}

// A write returns once the frame is queued, as a socket write returns once the frame is in its
// buffer: a session whose output is full (its client not reading yet) must not hold the writer.
// The session then takes the frames one at a time, in order, as ws.go's read loop does.
func TestMemFramesWriteDoesNotWaitForTheSession(t *testing.T) {
	p := memPlane(t, fakeSpec(t, "p", nil), "p")
	p.Start()
	s := newSession(t.Context(), p, 1)
	s.out <- []byte(`{"filler":true}`) // the output is full: a subscribe's reply cannot be queued
	m := newMemFrames(s)
	t.Cleanup(func() { m.Close() })
	done := make(chan error, 1)
	go func() { done <- m.Write(t.Context(), []byte(`{"id":1,"op":"subscribe","ch":"ctl"}`)) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a write waited for the session to have room for its reply")
	}
}

// A closed session delivers nothing more, as ws.go's writer, whose write context the session's end
// cancels, sends nothing: a frame still queued when the session ends is dropped, not read. The
// select between the queue and the end picks at random when both are ready, so the test closes many
// sessions with a frame waiting.
func TestMemFramesReadsNothingAfterTheSessionEnds(t *testing.T) {
	p := memPlane(t, fakeSpec(t, "p", nil), "p")
	for i := range 200 {
		s := newSession(t.Context(), p, 1)
		s.out <- []byte(`{"queued":true}`)
		m := newMemFrames(s)
		m.Close()
		if f, err := m.Read(t.Context()); !errors.Is(err, errMemClosed) {
			t.Fatalf("round %d: read %q, %v after the session ended, want errMemClosed", i, f, err)
		}
	}
}

func TestFederationOverTheInMemoryTransport(t *testing.T) {
	b := memPlane(t, fakeSpec(t, "b", nil), "b")
	a := memPlane(t, fakeSpec(t, "a", []PeerSpec{{Name: "b", StateFile: filepath.Join(t.TempDir(), "never-read.json")}}), "a")
	a.peers["b"].connect = connectMem(b)
	b.Start()
	a.Start()

	c := dialMem(t.Context(), a) // the test's own client of a, in memory as well
	t.Cleanup(func() { c.Close() })
	ctl := subscribe(t, c, "ctl", nil)
	mirror := subscribe(t, c, "peer/b/", nil)
	waitPeerState(t, a, "up")
	waitEnv(t, ctl, 5*time.Second, "peer.up", func(e *Envelope) bool { return e.Ev.Kind == "peer.up" })

	// An op forwarded to b runs on b.
	var st StatusReply
	if err := json.Unmarshal(do(t, c, Request{Op: "status", Peer: "b"}), &st); err != nil || st.Name != "b" || st.Epoch != b.Epoch() {
		t.Fatalf("forwarded status: %+v, %v (b's epoch %s)", st, err, b.Epoch())
	}

	// b's ctl is mirrored on a as peer/b/ctl, naming b's point.
	sent := b.Publish("ctl", event("note.mem", map[string]any{"n": 1}))
	e := waitEnv(t, mirror, 5*time.Second, "mirrored note.mem", func(e *Envelope) bool {
		return e.Ch == "peer/b/ctl" && e.Ev.Kind == "note.mem"
	})
	if e.Origin != "b" || e.SrcSeq != sent.Seq || e.SrcEpoch != b.Epoch() {
		t.Fatalf("mirrored envelope origin %q src %d/%q, want b %d/%q", e.Origin, e.SrcSeq, e.SrcEpoch, sent.Seq, b.Epoch())
	}

	// b going away ends the session: a reports the peer down.
	b.Shutdown()
	waitEnv(t, ctl, 5*time.Second, "peer.down", func(e *Envelope) bool { return e.Ev.Kind == "peer.down" })
	waitPeerState(t, a, "down")
}
