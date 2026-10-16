package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

func publishN(h interface {
	Publish(string, Event) *Envelope
}, ch string, n int) {
	for i := 0; i < n; i++ {
		h.Publish(ch, event("tick", map[string]any{"i": i}))
	}
}

func TestHubEnvelopeShape(t *testing.T) {
	h := NewHub("a")
	env := h.Publish("svc/x", event("stats", map[string]any{"nodes": []int{}}))
	var got map[string]any
	if err := json.Unmarshal(env.Raw(), &got); err != nil {
		t.Fatal(err)
	}
	if got["ch"] != "svc/x" || got["seq"] != 1.0 || got["origin"] != "a" || got["t"] == nil {
		t.Errorf("envelope %s", env.Raw())
	}
	ev := got["ev"].(map[string]any)
	if ev["kind"] != "stats" || ev["data"] == nil {
		t.Errorf("ev %v", ev)
	}
}

func TestHubCursorReplayThenLive(t *testing.T) {
	h := NewHub("a")
	publishN(h, "svc/x", 10)
	publishN(h, "ctl", 5)
	since := uint64(4)
	cur := h.Subscribe("svc/", &since)
	go func() {
		time.Sleep(20 * time.Millisecond)
		publishN(h, "svc/x", 10)
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var seqs []uint64
	for len(seqs) < 16 {
		envs, gap, err := cur.Next(ctx, 3)
		if err != nil || gap != nil {
			t.Fatalf("err %v gap %v", err, gap)
		}
		for _, e := range envs {
			seqs = append(seqs, e.Seq)
		}
	}
	want := []uint64{5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
	for i := range want {
		if seqs[i] != want[i] {
			t.Fatalf("got %v want %v", seqs, want)
		}
	}
}

func TestHubSinceTooOld(t *testing.T) {
	h := NewHub("a")
	publishN(h, "svc/x", ringSize+10)
	since := uint64(0)
	cur := h.Subscribe("", &since)
	_, gap, err := cur.Next(context.Background(), 10)
	if err != nil || gap == nil || gap.Reason != "since-too-old" || gap.From != 1 || gap.Oldest != 11 {
		t.Fatalf("gap %+v err %v", gap, err)
	}
	envs, _, _ := cur.Next(context.Background(), 1)
	if envs[0].Seq != 11 {
		t.Fatalf("resumes at the oldest: %d", envs[0].Seq)
	}
}

type frameKind struct {
	Re  *int64          `json:"re"`
	Sub *int64          `json:"sub"`
	Env *Envelope       `json:"env"`
	Gap *Gap            `json:"gap"`
	OK  bool            `json:"ok"`
	Raw json.RawMessage `json:"data"`
}

// The in-memory adapter: a live subscription whose queue fills up is
// dropped with an overflow gap that names the first seq it did not get.
func TestMemSessionOverflowGap(t *testing.T) {
	h := NewHub("a")
	b := &hubOnly{h}
	s := newMemSession(context.Background(), b)
	defer s.Close()
	s.Handle([]byte(`{"id":7,"op":"subscribe","ch":"svc/"}`))
	deadline := time.Now().Add(2 * time.Second)
	for len(s.out) == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	publishN(h, "svc/x", 300)
	for len(s.out) < sessionQueue && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	var frames []frameKind
	for len(frames) < sessionQueue+1 {
		select {
		case raw := <-s.Out():
			var f frameKind
			json.Unmarshal(raw, &f)
			frames = append(frames, f)
		case <-time.After(2 * time.Second):
			t.Fatalf("only %d frames", len(frames))
		}
	}
	if frames[0].Re == nil || !frames[0].OK {
		t.Fatalf("first frame is the subscribe reply: %+v", frames[0])
	}
	for i := 1; i < sessionQueue; i++ {
		if frames[i].Env == nil || frames[i].Env.Seq != uint64(i) {
			t.Fatalf("frame %d: %+v", i, frames[i])
		}
	}
	g := frames[sessionQueue]
	if g.Gap == nil || g.Gap.Reason != "overflow" || g.Gap.From != sessionQueue || *g.Sub != 7 {
		t.Fatalf("gap frame: %+v", g)
	}
	publishN(h, "svc/x", 5)
	select {
	case raw := <-s.Out():
		t.Fatalf("dropped subscription still delivers: %s", raw)
	case <-time.After(100 * time.Millisecond):
	}
}

// hubOnly is a Broker with no plane behind it.
type hubOnly struct{ *Hub }

func (hubOnly) Request(context.Context, Request) (any, error) { return nil, badRequest("no plane") }
