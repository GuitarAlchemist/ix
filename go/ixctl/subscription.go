package main

import (
	"context"
	"errors"
)

// A subscription is what a reader of the ring is told, in order: the events of its prefix
// (replay first, then live), the gaps, and, when it asked, one live mark once the replay is
// over. The WebSocket session and the SSE stream are two encodings of it: each frames the items
// and decides how to wait for its writer, neither decides what to say.
type subscription struct {
	cur     *Cursor
	head    uint64 // the hub's head when the subscription opened: up to it is replay
	told    bool   // the live mark was given, or was not asked for
	markers bool   // run markers pass (the reader asked for them)
	ended   bool   // an overflow gap was returned: nothing follows
}

// subRequest opens a subscription. since and epoch are resolved already (a bookmark is the
// caller's to resolve, with its own refusals).
type subRequest struct {
	prefix  string
	exact   bool // prefix is a whole channel name: svc/x does not cover svc/x-extra
	since   *uint64
	epoch   string
	live    bool // one live mark once the replay is over
	markers bool
}

// subItem is one thing to tell the reader: an event (replay says whether it is up to the head
// the subscription opened at), a gap, or the live mark.
type subItem struct {
	env    *Envelope
	replay bool
	gap    *Gap
	live   bool
}

// errSubscriptionEnded is returned by Next once the overflow gap was returned: the reader fell
// off the ring and was told; it resubscribes from the gap's oldest if it wants more.
var errSubscriptionEnded = errors.New("subscription ended after an overflow")

type subscriber interface {
	Subscribe(prefix string, since *uint64, epoch string) *Cursor
}

func openSubscription(b subscriber, r subRequest) *subscription {
	cur := b.Subscribe(r.prefix, r.since, r.epoch)
	cur.exact = r.exact // Broker.Subscribe takes no exact: the subscription is where it is set
	return &subscription{cur: cur, head: cur.Head(), told: !r.live, markers: r.markers}
}

// Head is the hub's head when the subscription opened: events up to it are replay.
func (s *subscription) Head() uint64 { return s.head }

// Epoch is the hub's boot id.
func (s *subscription) Epoch() string { return s.cur.Epoch() }

// liveDue gives the live mark once, when the cursor is past the head the subscription opened
// at: after the gap it may open with, after the last replayed event, and on a channel whose
// ring holds nothing up to the head (Catch; criterion 7 of ix#408, Codex round 43). The pump
// this replaces also waited while the cursor held a stale gap; the cursor clears it before
// returning anything, so it is always nil here.
func (s *subscription) liveDue() bool {
	if s.told || s.cur.next <= s.head {
		return false
	}
	s.told = true
	return true
}

// Next blocks until there is something to tell (at most max events, a gap, the live mark) and
// returns it in order. A gap comes alone, followed by the live mark if it is due; an overflow
// gap ends the subscription, and every later call returns errSubscriptionEnded. An error of ctx
// leaves the subscription where it was: the next call goes on from there.
func (s *subscription) Next(ctx context.Context, max int) ([]subItem, error) {
	for {
		if s.ended {
			return nil, errSubscriptionEnded
		}
		var envs []*Envelope
		var gap *Gap
		var err error
		if s.told {
			envs, gap, err = s.cur.Next(ctx, max)
		} else {
			envs, gap, err = s.cur.Catch(ctx, max) // empty once past the head: the replay is over
		}
		if err != nil {
			return nil, err
		}
		var items []subItem
		if gap != nil {
			items = append(items, subItem{gap: gap})
			if gap.Reason == "overflow" {
				s.ended = true
				return items, nil
			}
			if s.liveDue() {
				items = append(items, subItem{live: true})
			}
			return items, nil
		}
		for _, env := range envs {
			if !s.markers && isRunMarker(env) {
				continue
			}
			replay := env.Seq <= s.head
			if !replay && s.liveDue() {
				items = append(items, subItem{live: true})
			}
			items = append(items, subItem{env: env, replay: replay})
		}
		if s.liveDue() { // a batch that ended past the head, or an empty catch past it
			items = append(items, subItem{live: true})
		}
		if len(items) > 0 {
			return items, nil
		}
	}
}
