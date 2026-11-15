package main

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

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
