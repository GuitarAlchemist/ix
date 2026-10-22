package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// subscribe with since replays from the ring, then goes live, in order,
// with no hole and no repeat, while the publisher keeps going.
func TestWSSubscribeSinceReplaysThenLiveNoHole(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	base := in.plane.hub.Head()
	publishN(in.plane, "svc/x", 100)
	in.plane.Publish("ctl", event("noise", map[string]any{}))
	c := dial(t, in)

	const live = 1000
	go func() {
		for i := 0; i < live; i++ {
			in.plane.Publish("svc/x", event("tick", map[string]any{"i": i}))
			if i%10 == 9 {
				time.Sleep(time.Millisecond)
			}
		}
	}()
	since := base + 50
	w := subscribe(t, c, "svc/", &since)
	var seqs []uint64
	for len(seqs) < 50+live {
		f, err := w.next(10*time.Second, func(*Envelope) bool { return true })
		if err != nil {
			t.Fatalf("after %d events: %v", len(seqs), err)
		}
		seqs = append(seqs, f.Env.Seq)
	}
	for i := 1; i < len(seqs); i++ {
		want := seqs[i-1] + 1
		if seqs[i-1] == base+100 {
			want++ // the ctl event in between is not under svc/
		}
		if seqs[i] != want {
			t.Fatalf("hole or repeat at %d: %d then %d", i, seqs[i-1], seqs[i])
		}
	}
	if seqs[0] != base+51 {
		t.Fatalf("replay starts at %d, want %d", seqs[0], base+51)
	}
}

// A client that stops reading: its live subscription overflows the bounded
// queue and is dropped with a gap frame naming the first seq it missed.
func TestWSOverflowGivesGapFrame(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	c := dial(t, in)
	_, frames, err := c.Subscribe(context.Background(), "svc/", nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	pad := strings.Repeat("x", 1024)
	for i := 0; i < 20000; i++ {
		in.plane.Publish("svc/x", event("big", map[string]any{"pad": pad}))
	}
	var last uint64
	var gap *Gap
	for gap == nil {
		select {
		case f := <-frames:
			switch {
			case f.Gap != nil:
				gap = f.Gap
			case f.Env != nil:
				if last != 0 && f.Env.Seq != last+1 {
					t.Fatalf("hole before the gap: %d then %d", last, f.Env.Seq)
				}
				last = f.Env.Seq
			}
		case <-time.After(10 * time.Second):
			t.Fatalf("no gap frame; last seq %d", last)
		}
	}
	if gap.Reason != "overflow" || gap.From != last+1 || gap.Epoch != in.plane.Epoch() {
		t.Fatalf("gap %+v after seq %d", gap, last)
	}
	t.Logf("overflow after %d events", last)
	in.plane.Publish("svc/x", event("after", map[string]any{}))
	select {
	case f := <-frames:
		if f.Env != nil {
			t.Fatalf("dropped subscription still delivers seq %d", f.Env.Seq)
		}
	case <-time.After(200 * time.Millisecond):
	}
	// The connection itself is still fine.
	var p map[string]any
	json.Unmarshal(do(t, c, Request{Op: "ping"}), &p)
	if p["pong"] != true {
		t.Fatalf("ping %v", p)
	}
}

func TestWSSessionErrors(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	c := dial(t, in)
	ctx := context.Background()
	if _, _, err := c.Subscribe(ctx, "ctl", nil, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Do(ctx, Request{Op: "subscribe", Ch: "ctl", Peer: "b"}); err == nil {
		t.Error("subscribe with a peer accepted")
	}
	if _, err := c.Do(ctx, Request{Op: "unsubscribe", Sub: 999}); err == nil {
		t.Error("unsubscribe of nothing accepted")
	}
	if _, err := c.Do(ctx, Request{Op: "unsubscribe", Sub: 1}); err != nil {
		t.Errorf("unsubscribe: %v", err)
	}
	if _, err := c.Do(ctx, Request{Op: "apply", Peer: "zz", Desired: running("x")}); err == nil || !strings.Contains(err.Error(), "unknown peer") {
		t.Errorf("unknown peer: %v", err)
	}
}

// The same desired-state document with one unknown field, or another schema,
// gets the same refusal on POST /v1/apply and on an ixctl.v1 apply, and
// nothing is applied.
func TestUnknownFieldRefusedOnHTTPAndWS(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil, fakeSvc{name: "pod"}), "")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, "ws://"+in.addr+"/v1/ws", &websocket.DialOptions{
		HTTPClient: loopbackHTTP, HTTPHeader: http.Header{"Authorization": {"Bearer " + in.token}}, Subprotocols: []string{subprotocol}})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.CloseNow()
	for i, c := range []struct{ doc, refusal string }{
		{`{"services":{"pod":{"state":"running"}},"schema":"ixctl-desired/9"}`, `schema "ixctl-desired/9" is not "ixctl-desired/0"`},
		{`{"services":{"pod":{"state":"running","replicas":2}}}`, "unknown field"},
		{`{"services":{"pod":{"state":"running"}},"version":1}`, "unknown field"},
		// Routing is not state: a peer is named beside the document (F2.3).
		{`{"services":{"pod":{"state":"running"}},"peer":"b"}`, "unknown field"},
	} {
		doc := c.doc
		code, _, body := httpDo(t, in, "POST", "/v1/apply", json.RawMessage(doc), nil)
		var he struct {
			Error string `json:"error"`
		}
		json.Unmarshal(body, &he)
		if code != 400 || !strings.Contains(he.Error, c.refusal) {
			t.Fatalf("HTTP %s: %d %s, want %q", doc, code, body, c.refusal)
		}

		id := int64(i + 1)
		if err := conn.Write(ctx, websocket.MessageText, []byte(`{"id":`+strconv.FormatInt(id, 10)+`,"op":"apply","desired":`+doc+`}`)); err != nil {
			t.Fatal(err)
		}
		_, b, err := conn.Read(ctx)
		if err != nil {
			t.Fatal(err)
		}
		var f Frame
		json.Unmarshal(b, &f)
		if f.Re == nil || *f.Re != id || f.OK || f.Error != he.Error {
			t.Fatalf("WS %s: %s, want the refusal %q", doc, b, he.Error)
		}
	}
	// A peer that is named but empty is not "local": refused the same way on
	// every transport, before anything runs.
	for _, c := range []struct {
		method, path string
		body         any
	}{
		{"GET", "/v1/status?peer=", nil},
		{"POST", "/v1/plan?peer=", running("pod")},
		{"POST", "/v1/apply?peer=", running("pod")},
		{"POST", "/v1/apply?peer=b&peer=c", running("pod")},
		{"POST", "/v1/stop", json.RawMessage(`{"service":"pod","peer":""}`)},
		{"POST", "/v1/stop", json.RawMessage(`{"service":"pod","peer":null}`)},
		{"POST", "/v1/stop", json.RawMessage(`{"service":"pod","peer":7}`)},
	} {
		code, _, body := httpDo(t, in, c.method, c.path, c.body, nil)
		if code != 400 || !strings.Contains(string(body), "peer must not be empty") {
			t.Fatalf("HTTP %s %s with an empty peer: %d %s", c.method, c.path, code, body)
		}
	}
	// A malformed query is refused too: URL.Query() would drop the pair and
	// the op would read as local.
	if code, _, body := httpDo(t, in, "GET", "/v1/status?peer=b;peer=c", nil, nil); code != 400 || !strings.Contains(string(body), "bad query") {
		t.Fatalf("HTTP status with a malformed query: %d %s", code, body)
	}
	for _, frame := range []string{
		`{"id":90,"op":"status","peer":""}`,
		`{"id":90,"op":"apply","peer":"","desired":{"services":{"pod":{"state":"running"}}}}`,
		`{"id":90,"op":"stop","service":"pod","peer":""}`,
		`{"id":90,"op":"apply","peer":null,"desired":{"services":{"pod":{"state":"running"}}}}`,
	} {
		if err := conn.Write(ctx, websocket.MessageText, []byte(frame)); err != nil {
			t.Fatal(err)
		}
		_, b, err := conn.Read(ctx)
		if err != nil {
			t.Fatal(err)
		}
		var f Frame
		json.Unmarshal(b, &f)
		if f.Re == nil || *f.Re != 90 || f.OK || !strings.Contains(f.Error, "peer must not be empty") {
			t.Fatalf("WS %s: %s, want the empty-peer refusal", frame, b)
		}
	}
	var st StatusReply
	json.Unmarshal(do(t, dial(t, in), Request{Op: "status"}), &st)
	if st.Generation != 0 {
		t.Fatalf("a refused document was applied: generation %d", st.Generation)
	}

	// The document's own schema name is accepted on both transports, and so is
	// a document that names none.
	withSchema := json.RawMessage(`{"schema":"ixctl-desired/0","services":{"pod":{"state":"stopped"}}}`)
	code, _, body := httpDo(t, in, "POST", "/v1/apply", withSchema, nil)
	var ar ApplyReply
	json.Unmarshal(body, &ar)
	if code != 200 || ar.Generation != 1 {
		t.Fatalf("HTTP apply with schema: %d %s", code, body)
	}
	c := dial(t, in)
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: &Desired{Schema: desiredSchema, Services: map[string]DesiredService{"pod": {State: desiredStopped}}}}), &ar)
	if ar.Generation != 2 {
		t.Fatalf("WS apply with schema: generation %d, want 2", ar.Generation)
	}
	json.Unmarshal(do(t, c, Request{Op: "apply", Desired: running("pod")}), &ar)
	if ar.Generation != 3 {
		t.Fatalf("WS apply without schema: generation %d, want 3", ar.Generation)
	}
}
