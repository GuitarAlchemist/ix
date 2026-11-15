package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sync/atomic"
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

// requestCounter is a broker that counts the ops it is asked to run and the subscriptions it is
// asked for (a session asks for one while it handles the subscribe, before it answers).
type requestCounter struct {
	*Hub
	requests, subscribes atomic.Int64
}

func (b *requestCounter) Request(context.Context, Request) (any, error) {
	b.requests.Add(1)
	return nil, nil
}

func (b *requestCounter) Subscribe(prefix string, since *uint64, epoch string) *Cursor {
	b.subscribes.Add(1)
	return b.Hub.Subscribe(prefix, since, epoch)
}

// blockedMem is a connection whose session is held handling a frame: its output is full, and the
// first frame written (a malformed one, answered at once) waits there for room. Frames written
// after it wait in the connection's queue.
func blockedMem(t *testing.T, b Broker) *memFrames {
	t.Helper()
	s := newSession(t.Context(), b, 1)
	s.out <- []byte(`{"filler":true}`)
	m := newMemFrames(s)
	if err := m.Write(t.Context(), []byte(`{"id":1,"op":"status","peer":""}`)); err != nil {
		t.Fatal(err)
	}
	return m
}

// Nothing reaches the session once it has ended, as ws.go's read loop, whose reads the end
// cancels, hands nothing more: frames still queued when the connection closes are dropped. The
// select between the queue and the end picks at random when both are ready, so the test closes
// many connections with frames waiting; each queued frame is a subscribe, which the session would
// ask the broker for while it handles it, and the count is read once serve has returned.
func TestMemFramesHandsNothingAfterTheSessionEnds(t *testing.T) {
	for i := range 50 {
		b := &requestCounter{Hub: NewHub("p")}
		m := blockedMem(t, b)
		for id := 2; id < 6; id++ {
			if err := m.Write(t.Context(), []byte(fmt.Sprintf(`{"id":%d,"op":"subscribe","ch":"ctl"}`, id))); err != nil {
				t.Fatal(err)
			}
		}
		m.Close()
		<-m.served
		if n := b.subscribes.Load(); n != 0 {
			t.Fatalf("round %d: the session handled %d subscribes after it ended", i, n)
		}
	}
}

// A write that commits once the connection is closed fails, as a WebSocket write does: a nil
// return means the frame was queued while the session ran. The writer is held once its frame
// passed the checks, the connection closes, and serve, leaving, may take a queued frame and so
// free the slot the writer then takes; the test closes many connections so that it does.
func TestMemFramesWriteFailsOnceClosed(t *testing.T) {
	for i := range 100 {
		m := blockedMem(t, &requestCounter{Hub: NewHub("p")})
		for range memInQueue {
			if err := m.Write(t.Context(), []byte(`{"id":2,"op":"status"}`)); err != nil {
				t.Fatal(err)
			}
		}
		passed, release := make(chan struct{}), make(chan struct{})
		m.beforeSend = func() { close(passed); <-release }
		done := make(chan error, 1)
		go func() { done <- m.Write(t.Context(), []byte(`{"id":3,"op":"status"}`)) }()
		<-passed
		m.Close()
		<-m.served // serve has left, having taken a queued frame or not
		close(release)
		if err := <-done; err == nil {
			t.Fatalf("round %d: a write queued once the connection was closed returned nil", i)
		}
	}
}

// The in-memory adapter keeps the client's read limit as well: a server frame over clientReadLimit
// ends the connection instead of reaching the client, as the client's WebSocket refuses it.
func TestMemFramesRefusesAServerFrameOverTheClientsLimit(t *testing.T) {
	s := newSession(t.Context(), &requestCounter{Hub: NewHub("p")}, 1)
	s.out <- make([]byte, clientReadLimit+1)
	m := newMemFrames(s)
	t.Cleanup(func() { m.Close() })
	if f, err := m.Read(t.Context()); err == nil {
		t.Fatalf("a frame of %d bytes was read, over the client's limit", len(f))
	}
	if _, err := m.Read(t.Context()); !errors.Is(err, errMemClosed) {
		t.Fatalf("the connection outlived a frame over the client's limit: %v", err)
	}
}

// The session takes the frames one at a time, in the order they were written, as ws.go's read
// loop does: answers it gives as it takes each come out in that order. Each frame names an empty
// peer, which Handle refuses itself before it returns (no op runs on a goroutine of its own), so
// one-at-a-time handling answers them in order, always.
func TestMemFramesHandsFramesInOrder(t *testing.T) {
	s := newSession(t.Context(), &requestCounter{Hub: NewHub("p")}, 256)
	m := newMemFrames(s)
	t.Cleanup(func() { m.Close() })
	const n = 200
	for id := 1; id <= n; id++ {
		if err := m.Write(t.Context(), []byte(fmt.Sprintf(`{"id":%d,"op":"status","peer":""}`, id))); err != nil {
			t.Fatal(err)
		}
	}
	for want := int64(1); want <= n; want++ {
		f, err := m.Read(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var r replyFrame
		if err := json.Unmarshal(f, &r); err != nil || r.Re != want || r.OK {
			t.Fatalf("answer %s, want the refusal of request %d", f, want)
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
