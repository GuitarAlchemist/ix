package main

import (
	"context"
	"encoding/json"
	"sync/atomic"
	"testing"
	"time"
)

func nextFrame(t *testing.T, s *session) frameKind {
	t.Helper()
	select {
	case raw := <-s.Out():
		var f frameKind
		if err := json.Unmarshal(raw, &f); err != nil {
			t.Fatalf("frame %s: %v", raw, err)
		}
		return f
	case <-time.After(2 * time.Second):
		t.Fatal("no frame")
	}
	return frameKind{}
}

func wantReply(t *testing.T, s *session, id int64) {
	t.Helper()
	f := nextFrame(t, s)
	if f.Re == nil || *f.Re != id || !f.OK {
		t.Fatalf("want an ok reply to %d, got %+v", id, f)
	}
}

// Run markers are ordinary ring envelopes: a subscription skips them unless it asks,
// without losing the other replayed envelopes or changing their order.
func TestMemSessionRunMarkersAreOptInAndAcknowledged(t *testing.T) {
	h := NewHub("a")
	run := RunID{Origin: "a", Epoch: h.Epoch(), Seq: 1}
	begin := h.PublishRun("svc/x", event(runBeginKind, map[string]any{}), run)
	line1 := h.PublishRun("svc/x", event("log", map[string]any{"line": "one"}), run)
	end := h.PublishRun("svc/x", event(runEndKind, map[string]any{"outcome": "started"}), run)
	line2 := h.PublishRun("svc/x", event("log", map[string]any{"line": "two"}), run)

	plain := newMemSession(context.Background(), &hubOnly{h})
	defer plain.Close()
	plain.Handle([]byte(`{"id":1,"op":"subscribe","ch":"svc/x","since":0}`))
	wantReply(t, plain, 1)
	for _, want := range []*Envelope{line1, line2} {
		f := nextFrame(t, plain)
		if f.Env == nil || f.Env.Seq != want.Seq {
			t.Fatalf("plain replay got %+v, want seq %d", f, want.Seq)
		}
	}

	marked := newMemSession(context.Background(), &hubOnly{h})
	defer marked.Close()
	marked.Handle([]byte(`{"id":2,"op":"subscribe","ch":"svc/x","since":0,"markers":"run"}`))
	reply := nextFrame(t, marked)
	if reply.Re == nil || *reply.Re != 2 || !reply.OK {
		t.Fatalf("marker reply %+v", reply)
	}
	var ack struct {
		Markers string `json:"markers"`
	}
	if err := json.Unmarshal(reply.Raw, &ack); err != nil || ack.Markers != runMarkers {
		t.Fatalf("marker acknowledgement %s: %v", reply.Raw, err)
	}
	for _, want := range []*Envelope{begin, line1, end, line2} {
		f := nextFrame(t, marked)
		if f.Env == nil || f.Env.Seq != want.Seq {
			t.Fatalf("marked replay got %+v, want seq %d", f, want.Seq)
		}
	}

	bad := newMemSession(context.Background(), &hubOnly{h})
	defer bad.Close()
	bad.Handle([]byte(`{"id":3,"op":"subscribe","ch":"svc/x","markers":"other"}`))
	f := nextFrame(t, bad)
	if f.Re == nil || *f.Re != 3 || f.OK {
		t.Fatalf("unknown markers accepted: %+v", f)
	}
}

// Unsubscribe then subscribe again with the same id: the first pump, which
// leaves late, must not take the new subscription down with it. The first
// pump is held just before it removes its entry, so the order is fixed.
func TestMemSessionResubscribeSurvivesOldPump(t *testing.T) {
	h := NewHub("a")
	s := newMemSession(context.Background(), &hubOnly{h})
	defer s.Close()
	leaving, release, left := make(chan int64, 1), make(chan struct{}), make(chan int64, 4)
	var held atomic.Bool
	s.pumpLeaving = func(id int64) {
		if held.CompareAndSwap(false, true) {
			leaving <- id
			<-release
		}
	}
	s.pumpLeft = func(id int64) { left <- id }

	s.Handle([]byte(`{"id":7,"op":"subscribe","ch":"svc/"}`))
	wantReply(t, s, 7)
	s.Handle([]byte(`{"id":8,"op":"unsubscribe","sub":7}`))
	wantReply(t, s, 8)
	select {
	case <-leaving:
	case <-time.After(2 * time.Second):
		t.Fatal("the first pump did not leave after unsubscribe")
	}
	s.Handle([]byte(`{"id":7,"op":"subscribe","ch":"svc/"}`))
	wantReply(t, s, 7)
	close(release)
	select {
	case <-left:
	case <-time.After(2 * time.Second):
		t.Fatal("the first pump never finished leaving")
	}

	s.mu.Lock()
	_, ok := s.subs[7]
	s.mu.Unlock()
	if !ok {
		t.Fatal("the first pump removed the second subscription 7")
	}
	h.Publish("svc/x", event("tick", map[string]any{}))
	f := nextFrame(t, s)
	if f.Sub == nil || *f.Sub != 7 || f.Env == nil || f.Env.Ev.Kind != "tick" {
		t.Fatalf("the second subscription did not get the event: %+v", f)
	}
	select {
	case id := <-left:
		t.Fatalf("the second pump %d left too", id)
	default:
	}
}
