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

type frame struct {
	Type string          `json:"type"`
	Seq  uint64          `json:"seq"`
	Data json.RawMessage `json:"data"`
}

type snapDelivery struct {
	MandateID string `json:"mandate_id"`
	State     string `json:"state"`
}

// snapshotOf reads a stale-epoch resume up to the end of its state, and returns the deliveries
// the state holds and the seq it covers. It reads the paged form (state-page … state-end) and the
// single state frame alike.
func snapshotOf(t *testing.T, c *websocket.Conn) ([]snapDelivery, uint64) {
	t.Helper()
	var out []snapDelivery
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		_, b, err := c.Read(ctx)
		cancel()
		if err != nil {
			t.Fatalf("stream read before the end of the state: %v", err)
		}
		var f frame
		json.Unmarshal(b, &f)
		switch f.Type {
		case "state", "state-page":
			var d struct {
				Deliveries []snapDelivery `json:"deliveries"`
			}
			json.Unmarshal(f.Data, &d)
			out = append(out, d.Deliveries...)
			if f.Type == "state" {
				return out, f.Seq
			}
		case "state-end":
			return out, f.Seq
		}
	}
}

// S3B-4 (Codex round 6, P2): the engine applies a transition, then the hub numbers it. A client
// resuming in between must not get it twice: in the snapshot and again as the next event.
func TestSnapshotAndStreamDoNotOverlap(t *testing.T) {
	dir := t.TempDir()
	s, err := Start(Config{Addr: "127.0.0.1:0", State: filepath.Join(dir, "state"), Targets: []Named{{Name: "s", Path: filepath.Join(dir, "session.jsonl")}}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { s.RequestStop(); s.Wait() }()
	var c *websocket.Conn
	var snap []snapDelivery
	var covered uint64
	testHookBeforePublish = func(tr deliver.Transition) {
		if tr.State != deliver.Delivered || c != nil {
			return
		}
		var err error
		c, _, err = websocket.Dial(context.Background(), "ws://"+s.Addr()+"/v1/stream?since=1&epoch=stale",
			&websocket.DialOptions{HTTPHeader: http.Header{"Authorization": {"Bearer " + string(s.token)}}})
		if err != nil {
			t.Errorf("dial: %v", err)
			return
		}
		c.SetReadLimit(MaxStreamMessage)
		snap, covered = snapshotOf(t, c)
	}
	t.Cleanup(func() { testHookBeforePublish = nil })
	inc, err := s.fakes[0].Incarnation()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.eng.Deliver(context.Background(), deliver.Mandate{MandateID: "m1", Target: "s", Incarnation: inc, Text: "hello"}); err != nil {
		t.Fatal(err)
	}
	if c == nil {
		t.Fatal("the hook never connected")
	}
	defer c.CloseNow()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, b, err := c.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var next frame
	json.Unmarshal(b, &next)
	var tr snapDelivery
	json.Unmarshal(next.Data, &tr)
	if next.Type != "mandate" || tr.MandateID != "m1" || tr.State != "delivered" || next.Seq != covered+1 {
		t.Errorf("after the state covering seq %d: %s seq %d %+v, want the m1 delivered event at seq %d", covered, next.Type, next.Seq, tr, covered+1)
	}
	for _, d := range snap {
		if d.MandateID == "m1" && d.State == "delivered" {
			t.Errorf("m1 delivered is both in the snapshot covering seq %d and in the event at seq %d", covered, next.Seq)
		}
	}
}
