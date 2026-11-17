package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

// sseItems reads /v1/events with the given query for d, or until n items, and returns them in
// order: "env:<ch>" for an event, "gap:<reason>" for a gap, "live" for the live mark when it is
// exactly `event: live` and `data: {}` with no `id:` (an id would move EventSource's
// Last-Event-ID off the last event), "live?" for any other shape of it. onLive, if given, runs
// once the live mark is read.
func sseItems(t *testing.T, in *instance, query string, n int, d time.Duration, onLive ...func()) []string {
	t.Helper()
	req, _ := http.NewRequest("GET", "http://"+in.addr+"/v1/events?"+query, nil)
	req.Header.Set("Authorization", "Bearer "+in.token)
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	resp, err := loopbackHTTP.Do(req.WithContext(ctx))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("GET /v1/events?%s: %d", query, resp.StatusCode)
	}
	var items []string
	sc := bufio.NewScanner(resp.Body)
	kind, id := "", false
	for len(items) < n && sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			kind, id = "", false
		case strings.HasPrefix(line, "id: "):
			id = true
		case strings.HasPrefix(line, "event: "):
			kind = line[7:]
		case strings.HasPrefix(line, "data: ") && kind == "gap":
			var g Gap
			json.Unmarshal([]byte(line[6:]), &g)
			items = append(items, "gap:"+g.Reason)
		case strings.HasPrefix(line, "data: ") && kind == "live":
			if line != "data: {}" || id {
				items = append(items, "live?")
				continue
			}
			items = append(items, "live")
			for _, f := range onLive {
				f()
			}
		case strings.HasPrefix(line, "data: "):
			var e Envelope
			if err := json.Unmarshal([]byte(line[6:]), &e); err != nil {
				t.Fatal(err)
			}
			items = append(items, "env:"+e.Ch)
		}
	}
	return items
}

// SSE takes exact as the WebSocket's subscribe does: the whole channel name, so svc/x does not
// cover svc/x-extra.
func TestSSEExactMatchesTheWholeChannel(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "a", nil), "")
	in.plane.Publish("svc/x-extra", event("stats", map[string]any{"n": 1}))
	in.plane.Publish("svc/x", event("stats", map[string]any{"n": 2}))
	in.plane.Publish("svc/x-extra", event("stats", map[string]any{"n": 3}))
	in.plane.Publish("svc/x", event("stats", map[string]any{"n": 4}))
	got := sseItems(t, in, "ch=svc/x&since=0&exact=1", 2, 2*time.Second)
	if strings.Join(got, " ") != "env:svc/x env:svc/x" {
		t.Fatalf("exact svc/x read %v, want svc/x's two events and nothing of svc/x-extra", got)
	}
}

// SSE takes live as the WebSocket's subscribe does: once the replay is over, one live mark,
// before the first live event, and at once on a channel whose replay holds nothing.
func TestSSELiveFollowsTheReplay(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "a", nil), "")
	for i := 1; i <= 3; i++ {
		in.plane.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	got := sseItems(t, in, "ch=svc/x&since=0&live=1", 5, 3*time.Second, func() {
		in.plane.Publish("svc/x", event("stats", map[string]any{"n": 4})) // live: the mark was read
	})
	if want := "env:svc/x env:svc/x env:svc/x live env:svc/x"; strings.Join(got, " ") != want {
		t.Fatalf("live after a replay read %v, want %s", got, want)
	}
	if got := sseItems(t, in, "ch=svc/quiet&since=0&live=true", 1, 2*time.Second); strings.Join(got, " ") != "live" {
		t.Fatalf("live on a quiet channel read %v, want the live mark at once", got)
	}
}

// A request that did not carry live or exact keeps its former meaning (ADR 0014, 1.2), and one
// whose value is neither 1 nor true is not refused (1.3: refusing a request that was valid takes
// a new name): it is read as before, a prefix with no live mark.
func TestSSEReadsOtherLiveOrExactValuesAsBefore(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "a", nil), "")
	in.plane.Publish("svc/x", event("stats", map[string]any{"n": 1}))
	in.plane.Publish("svc/x-extra", event("stats", map[string]any{"n": 2}))
	got := sseItems(t, in, "ch=svc/x&since=0&exact=yes&live=no", 3, time.Second)
	if strings.Join(got, " ") != "env:svc/x env:svc/x-extra" {
		t.Fatalf("exact=yes&live=no read %v, want the prefix's two events and no live mark", got)
	}
}

// The subscription ends after the overflow gap it returns: a reader that fell off the ring is
// told once, and nothing follows.
func TestASubscriptionEndsAfterAnOverflow(t *testing.T) {
	h := NewHub("a")
	sub := openSubscription(h, subRequest{prefix: "svc/x", live: true})
	for i := 0; i < ringSize+10; i++ {
		h.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	var kinds []string
	for {
		items, err := sub.Next(ctx, 256)
		if errors.Is(err, errSubscriptionEnded) {
			break
		}
		if err != nil {
			t.Fatalf("after %v: %v", kinds, err)
		}
		for _, it := range items {
			switch {
			case it.gap != nil:
				kinds = append(kinds, "gap:"+it.gap.Reason)
			case it.live:
				kinds = append(kinds, "live")
			}
		}
	}
	if strings.Join(kinds, " ") != "gap:overflow" {
		t.Fatalf("a subscription that fell off the ring said %v, want the overflow gap, then nothing", kinds)
	}
}

// itemsOf runs Next once and names its items: "r" a replayed event, "e" a live one, "live" the
// mark, "gap:<reason>" a gap.
func itemsOf(t *testing.T, sub *subscription, max int) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	items, err := sub.Next(ctx, max)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, it := range items {
		switch {
		case it.gap != nil:
			out = append(out, "gap:"+it.gap.Reason)
		case it.live:
			out = append(out, "live")
		case it.replay:
			out = append(out, "r")
		default:
			out = append(out, "e")
		}
	}
	return strings.Join(out, " ")
}

// A replay longer than one batch is not over at the end of the first batch: the mark comes
// after the last replayed event.
func TestTheLiveMarkWaitsForTheWholeReplay(t *testing.T) {
	h := NewHub("a")
	for i := 0; i < 5; i++ {
		h.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	zero := uint64(0)
	sub := openSubscription(h, subRequest{prefix: "svc/x", since: &zero, live: true})
	if got := itemsOf(t, sub, 3); got != "r r r" {
		t.Fatalf("first batch of a 5-event replay read %q, want three replayed events and no mark", got)
	}
	if got := itemsOf(t, sub, 3); got != "r r live" {
		t.Fatalf("second batch read %q, want the last two replayed events, then the mark", got)
	}
}

// A batch that runs past the head (events published between the subscribe and the first read)
// carries the mark between the last replayed event and the first live one.
func TestTheLiveMarkSitsBetweenReplayAndLiveInOneBatch(t *testing.T) {
	h := NewHub("a")
	for i := 0; i < 3; i++ {
		h.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	zero := uint64(0)
	sub := openSubscription(h, subRequest{prefix: "svc/x", since: &zero, live: true})
	for i := 3; i < 5; i++ {
		h.Publish("svc/x", event("stats", map[string]any{"n": i}))
	}
	if got := itemsOf(t, sub, 256); got != "r r r live e e" {
		t.Fatalf("a batch across the head read %q, want r r r live e e", got)
	}
}
