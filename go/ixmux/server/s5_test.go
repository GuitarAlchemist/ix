package server

import (
	"context"
	"encoding/json"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// S5-2 (CI Linux, run 42326800534): a slow client gets what its queue held, a gap of reason slow,
// then close 1008, and publishing never waits for it. The lever holds the connection's writer
// before the socket, so its queue overflows whatever the socket buffers absorb.
func TestSlowClientGetsGapThenClose(t *testing.T) {
	dir := t.TempDir()
	const queue = 2
	s, err := Start(Config{Addr: "127.0.0.1:0", State: filepath.Join(dir, "state"), Queue: queue,
		Targets: []Named{{Name: "s", Path: filepath.Join(dir, "session.jsonl")}}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { s.RequestStop(); s.Wait() }()
	held, release := make(chan struct{}), make(chan struct{})
	testHookStreamLive = func() {
		select {
		case <-held:
		default:
			close(held)
		}
		<-release
	}
	t.Cleanup(func() { testHookStreamLive = nil })
	defer func() {
		select {
		case <-release:
		default:
			close(release) // a failed test must not leave the writer held
		}
	}()

	c, _, err := websocket.Dial(context.Background(), "ws://"+s.Addr()+"/v1/stream",
		&websocket.DialOptions{HTTPHeader: http.Header{"Authorization": {"Bearer " + string(s.token)}}})
	if err != nil {
		t.Fatal(err)
	}
	defer c.CloseNow()
	c.SetReadLimit(MaxStreamMessage)
	select {
	case <-held:
	case <-time.After(10 * time.Second):
		t.Fatal("the writer never reached the live loop")
	}

	// A delivery publishes its transitions with the writer held: the engine does not wait for it.
	inc, err := s.fakes[0].Incarnation()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		_, err := s.eng.Deliver(context.Background(), deliver.Mandate{MandateID: "m1", Target: "s", Incarnation: inc, Text: "hello"})
		done <- err
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(30 * time.Second): // a deadlock guard, not a timing bound
		t.Fatal("the delivery waited for the held slow client")
	}
	for range queue + 1 { // more than the queue holds, whatever the delivery published
		s.hub.publish("probe", nil)
	}
	close(release)

	var types []string
	var gap Gap
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, b, err := c.Read(ctx)
		cancel()
		if err != nil {
			t.Fatalf("after %v: %v before any gap", types, err)
		}
		var f frame
		json.Unmarshal(b, &f)
		types = append(types, f.Type)
		if f.Type == "gap" {
			json.Unmarshal(f.Data, &gap)
			break
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	_, _, err = c.Read(ctx)
	cancel()
	queued := len(types) - 2 // between hello and the gap
	if types[0] != "hello" || gap.Reason != "slow" || queued > queue || websocket.CloseStatus(err) != websocket.StatusPolicyViolation {
		t.Errorf("slow client: %v, gap %+v, then %v; want at most %d queued messages, a gap of reason slow, then close 1008",
			types, gap, err, queue)
	}
}
