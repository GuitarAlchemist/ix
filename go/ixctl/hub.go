package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ringSize is how many envelopes an instance keeps for replay.
const ringSize = 4096

// Event is what a service (or the plane itself) says: {kind, t?, data}.
type Event struct {
	Kind string          `json:"kind"`
	T    int64           `json:"t,omitempty"`
	Data json.RawMessage `json:"data"`
}

// Envelope carries every event on every transport.
type Envelope struct {
	Ch     string `json:"ch"`
	Seq    uint64 `json:"seq"`
	Origin string `json:"origin"`
	T      int64  `json:"t"`
	Ev     Event  `json:"ev"`

	raw []byte // JSON encoding, built once at publish
}

// Raw returns the envelope's JSON encoding.
func (e *Envelope) Raw() []byte {
	if e.raw == nil {
		e.raw, _ = json.Marshal(e)
	}
	return e.raw
}

// Gap tells a subscriber that envelopes from From on are not coming.
type Gap struct {
	Reason string `json:"reason"` // "since-too-old" | "overflow"
	From   uint64 `json:"from"`   // first seq the subscriber did not get
	Oldest uint64 `json:"oldest,omitempty"`
}

// Request is one operation, whichever transport carried it. On WebSocket it
// is also the client frame, so it carries the subscription fields.
type Request struct {
	ID      int64    `json:"id,omitempty"`
	Op      string   `json:"op"`
	Ch      string   `json:"ch,omitempty"`
	Since   *uint64  `json:"since,omitempty"`
	Desired *Desired `json:"desired,omitempty"`
	Service string   `json:"service,omitempty"`
	Peer    string   `json:"peer,omitempty"`
	Sub     int64    `json:"sub,omitempty"` // unsubscribe: the subscription to drop
}

// Broker is all a transport sees of an instance: publish, subscribe with a
// prefix and since, request/reply. Nothing behind it knows which transport
// carried a request.
type Broker interface {
	Publish(ch string, ev Event) *Envelope
	Subscribe(prefix string, since *uint64) *Cursor
	Request(ctx context.Context, req Request) (any, error)
}

// opError is a refusal with the HTTP status that goes with it.
type opError struct {
	status int
	msg    string
}

func (e *opError) Error() string { return e.msg }

func badRequest(format string, a ...any) error {
	return &opError{status: 400, msg: fmt.Sprintf(format, a...)}
}

func statusOf(err error) int {
	var oe *opError
	if errors.As(err, &oe) {
		return oe.status
	}
	return 500
}

// Hub is the transport-agnostic core: a per-instance sequence, a replay ring
// and cursors that read it. Publishers never block on subscribers; a
// subscriber that falls more than the ring behind gets a gap.
type Hub struct {
	origin string

	mu   sync.Mutex
	ring [ringSize]*Envelope
	head uint64        // last seq assigned
	wake chan struct{} // closed and replaced on every publish
	tap  func(*Envelope)
}

func NewHub(origin string) *Hub {
	return &Hub{origin: origin, wake: make(chan struct{})}
}

// Publish stamps an event with this instance as origin.
func (h *Hub) Publish(ch string, ev Event) *Envelope {
	return h.publish(ch, h.origin, 0, ev)
}

// Mirror republishes an envelope received from a peer under a local channel,
// keeping the peer as origin and its time.
func (h *Hub) Mirror(ch, origin string, t int64, ev Event) *Envelope {
	return h.publish(ch, origin, t, ev)
}

func (h *Hub) publish(ch, origin string, t int64, ev Event) *Envelope {
	evRaw, err := json.Marshal(ev)
	if err != nil {
		ev.Data = nil
		evRaw, _ = json.Marshal(ev)
	}
	chRaw, _ := json.Marshal(ch)
	orRaw, _ := json.Marshal(origin)
	if t == 0 {
		t = time.Now().UnixMilli()
	}
	b := make([]byte, 0, len(evRaw)+len(chRaw)+len(orRaw)+64)

	h.mu.Lock()
	h.head++
	env := &Envelope{Ch: ch, Seq: h.head, Origin: origin, T: t, Ev: ev}
	b = append(b, `{"ch":`...)
	b = append(b, chRaw...)
	b = append(b, `,"seq":`...)
	b = strconv.AppendUint(b, env.Seq, 10)
	b = append(b, `,"origin":`...)
	b = append(b, orRaw...)
	b = append(b, `,"t":`...)
	b = strconv.AppendInt(b, t, 10)
	b = append(b, `,"ev":`...)
	b = append(b, evRaw...)
	env.raw = append(b, '}')
	h.ring[env.Seq%ringSize] = env
	if h.tap != nil {
		h.tap(env)
	}
	close(h.wake)
	h.wake = make(chan struct{})
	h.mu.Unlock()
	return env
}

// Head is the last seq assigned.
func (h *Hub) Head() uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.head
}

// Cursor reads the ring in seq order, replay and live alike, so a
// subscription can never see a hole without being told.
type Cursor struct {
	h      *Hub
	prefix string
	next   uint64 // next seq wanted
	start  uint64 // head when the cursor was made
	replay bool   // since was given: a first gap is since-too-old
}

// Subscribe returns a cursor over envelopes whose channel starts with prefix.
// With since nil it starts live; otherwise it replays every envelope with
// seq > *since still in the ring, then goes live.
func (h *Hub) Subscribe(prefix string, since *uint64) *Cursor {
	h.mu.Lock()
	defer h.mu.Unlock()
	c := &Cursor{h: h, prefix: prefix, next: h.head + 1, start: h.head}
	if since != nil && *since < h.head {
		c.next = *since + 1
		c.replay = true
	}
	return c
}

// Head is the hub's head when the cursor was made: envelopes up to it are
// replay, later ones are live.
func (c *Cursor) Head() uint64 { return c.start }

// Next blocks until matching envelopes (at most max) or a gap are available.
func (c *Cursor) Next(ctx context.Context, max int) ([]*Envelope, *Gap, error) {
	h := c.h
	for {
		h.mu.Lock()
		oldest := uint64(1)
		if h.head > ringSize {
			oldest = h.head - ringSize + 1
		}
		if c.next < oldest {
			g := &Gap{Reason: "overflow", From: c.next, Oldest: oldest}
			if c.replay {
				g.Reason = "since-too-old"
			}
			c.next, c.replay = oldest, false
			h.mu.Unlock()
			return nil, g, nil
		}
		c.replay = false
		var out []*Envelope
		for c.next <= h.head && len(out) < max {
			env := h.ring[c.next%ringSize]
			c.next++
			if strings.HasPrefix(env.Ch, c.prefix) {
				out = append(out, env)
			}
		}
		wake := h.wake
		h.mu.Unlock()
		if len(out) > 0 {
			return out, nil, nil
		}
		select {
		case <-wake:
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		}
	}
}
