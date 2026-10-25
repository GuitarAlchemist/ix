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
	if got["epoch"] != h.Epoch() || len(h.Epoch()) != 12 || env.Epoch != h.Epoch() {
		t.Errorf("epoch %q in envelope %s", h.Epoch(), env.Raw())
	}
	if NewHub("a").Epoch() == h.Epoch() {
		t.Errorf("two boots share the epoch %q", h.Epoch())
	}
	ev := got["ev"].(map[string]any)
	if ev["kind"] != "stats" || ev["data"] == nil {
		t.Errorf("ev %v", ev)
	}
}

// A10: the adapter is the only test that fixes the fields of a run on the wire.
func TestRunIdentityWireForm(t *testing.T) {
	h := NewHub("a")
	env := h.PublishNewRun("ctl", event("service.starting", map[string]any{"service": "x"}))

	var wire struct {
		Run json.RawMessage `json:"run"`
	}
	if err := json.Unmarshal(env.Raw(), &wire); err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(wire.Run, &fields); err != nil {
		t.Fatal(err)
	}
	if len(fields) != 3 {
		t.Fatalf("run fields %v", fields)
	}
	for _, name := range []string{"origin", "epoch", "seq"} {
		if _, ok := fields[name]; !ok {
			t.Fatalf("run has no %q: %s", name, env.Raw())
		}
	}
	var got RunID
	if err := json.Unmarshal(wire.Run, &got); err != nil {
		t.Fatal(err)
	}
	want := RunID{Origin: "a", Epoch: h.Epoch(), Seq: env.Seq}
	if env.Run == nil || *env.Run != want || got != want {
		t.Fatalf("run %+v, decoded %+v, want %+v", env.Run, got, want)
	}
}

func TestHubCursorReplayThenLive(t *testing.T) {
	h := NewHub("a")
	publishN(h, "svc/x", 10)
	publishN(h, "ctl", 5)
	since := uint64(4)
	cur := h.Subscribe("svc/", &since, h.Epoch())
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
	cur := h.Subscribe("", &since, "")
	_, gap, err := cur.Next(context.Background(), 10)
	if err != nil || gap == nil || gap.Reason != "since-too-old" || gap.From != 1 || gap.Oldest != 11 || gap.Epoch != h.Epoch() {
		t.Fatalf("gap %+v err %v", gap, err)
	}
	envs, _, _ := cur.Next(context.Background(), 1)
	if envs[0].Seq != 11 {
		t.Fatalf("resumes at the oldest: %d", envs[0].Seq)
	}
}

// F1.1 of the one-way-doors doc: a since kept across the instance's restart.
// Named with its epoch, it gets an epoch-changed gap; a bare since past the
// new head gets since-ahead; both then replay everything this boot still
// holds. A bare since below the head is trusted, as before: nothing can tell
// it apart.
func TestHubSinceFromAnotherBoot(t *testing.T) {
	before := NewHub("a")
	publishN(before, "svc/x", 30)
	h := NewHub("a") // the restart: seq starts over, the epoch changes
	publishN(h, "svc/x", 5)

	for name, sub := range map[string]*Cursor{
		"with epoch": h.Subscribe("svc/", ptr(uint64(2)), before.Epoch()),
		"ahead":      h.Subscribe("svc/", ptr(uint64(30)), ""),
	} {
		_, gap, err := sub.Next(context.Background(), 10)
		if err != nil || gap == nil || gap.From != 1 || gap.Oldest != 1 || gap.Epoch != h.Epoch() {
			t.Fatalf("%s: gap %+v err %v", name, gap, err)
		}
		want := map[string]string{"with epoch": "epoch-changed", "ahead": "since-ahead"}[name]
		if gap.Reason != want {
			t.Fatalf("%s: reason %q", name, gap.Reason)
		}
		envs, gap, err := sub.Next(context.Background(), 10)
		if err != nil || gap != nil || len(envs) != 5 || envs[0].Seq != 1 || envs[0].Epoch != h.Epoch() {
			t.Fatalf("%s: replay %d envelopes, gap %+v, err %v", name, len(envs), gap, err)
		}
	}

	// The same epoch, or none with a since below the head: a plain replay.
	for name, sub := range map[string]*Cursor{
		"own epoch": h.Subscribe("svc/", ptr(uint64(2)), h.Epoch()),
		"bare":      h.Subscribe("svc/", ptr(uint64(2)), ""),
	} {
		envs, gap, err := sub.Next(context.Background(), 10)
		if err != nil || gap != nil || len(envs) != 3 || envs[0].Seq != 3 {
			t.Fatalf("%s: replay %d envelopes, gap %+v, err %v", name, len(envs), gap, err)
		}
	}

	// After an overflow the stale gap still comes first, pointing at the oldest kept.
	publishN(h, "svc/x", ringSize+10)
	sub := h.Subscribe("", ptr(uint64(1)), before.Epoch())
	_, gap, _ := sub.Next(context.Background(), 10)
	if gap == nil || gap.Reason != "epoch-changed" || gap.Oldest != 16 {
		t.Fatalf("gap after overflow %+v", gap)
	}
	envs, _, _ := sub.Next(context.Background(), 1)
	if envs[0].Seq != 16 {
		t.Fatalf("resumes at the oldest: %d", envs[0].Seq)
	}
}

// A since that names this boot's epoch is trusted even past the head: no since-ahead gap
// (that one is for a bare since, which may come from another boot), no replay, the stream
// goes live from the head. A #454 consumer expected the gap. Pins present behaviour.
func TestHubSinceAheadInItsOwnEpochGoesLive(t *testing.T) {
	h := NewHub("a")
	publishN(h, "svc/x", 5)
	sub := h.Subscribe("svc/", ptr(uint64(30)), h.Epoch())
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if envs, gap, err := sub.Next(ctx, 10); len(envs) != 0 || gap != nil || err == nil {
		t.Fatalf("before any live envelope: %d envelopes, gap %+v, err %v", len(envs), gap, err)
	}
	publishN(h, "svc/x", 1)
	envs, gap, err := sub.Next(context.Background(), 10)
	if err != nil || gap != nil || len(envs) != 1 || envs[0].Seq != 6 {
		t.Fatalf("live: %d envelopes, gap %+v, err %v", len(envs), gap, err)
	}
}

func ptr[T any](v T) *T { return &v }

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
	if g.Gap == nil || g.Gap.Reason != "overflow" || g.Gap.From != sessionQueue || g.Gap.Epoch != h.Epoch() || *g.Sub != 7 {
		t.Fatalf("gap frame: %+v", g)
	}
	publishN(h, "svc/x", 5)
	select {
	case raw := <-s.Out():
		t.Fatalf("dropped subscription still delivers: %s", raw)
	case <-time.After(100 * time.Millisecond):
	}
}

// A mirror copies the source run even when it is structurally malformed. The peer link is the
// validation boundary; the hub's mirror path must never reinterpret seq 0 as "make a local run".
func TestMirrorKeepsRunWithoutSynthesizingIt(t *testing.T) {
	h := NewHub("a")
	for _, source := range []RunID{
		{Origin: "b", Epoch: "source-boot", Seq: 17},
		{},
	} {
		env := h.Mirror(
			"peer/b/svc/x", "b", 123, Event{Kind: "log", Data: json.RawMessage(`{"line":"x"}`)},
			9, "source-boot", &source,
		)
		if env.Run == nil || *env.Run != source {
			t.Fatalf("mirrored run %+v, want source %+v", env.Run, source)
		}
		var wire Envelope
		if err := json.Unmarshal(env.Raw(), &wire); err != nil {
			t.Fatal(err)
		}
		if wire.Run == nil || *wire.Run != source {
			t.Fatalf("wire run %+v, want source %+v: %s", wire.Run, source, env.Raw())
		}
	}
}

// hubOnly is a Broker with no plane behind it.
type hubOnly struct{ *Hub }

func (hubOnly) Request(context.Context, Request) (any, error) { return nil, badRequest("no plane") }
