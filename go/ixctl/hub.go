package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
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

const (
	runMarkers   = "run"
	runBeginKind = "ixctl.run.begin"
	runEndKind   = "ixctl.run.end"
)

// Event is what a service (or the plane itself) says: {kind, t?, data}. T is
// Unix milliseconds on the emitter's own clock, or absent when it gave none;
// the envelope's T is the plane's clock at publish.
type Event struct {
	Kind string          `json:"kind"`
	T    int64           `json:"t,omitempty"`
	Data json.RawMessage `json:"data"`
}

// RunID names one service launch attempt. Seq is the sequence of its service.starting
// envelope on the origin plane.
type RunID struct {
	Origin string `json:"origin"`
	Epoch  string `json:"epoch"`
	Seq    uint64 `json:"seq"`
}

func isRunMarker(env *Envelope) bool {
	return env.Ev.Kind == runBeginKind || env.Ev.Kind == runEndKind
}

func wantsRunMarkers(value *string) (bool, error) {
	if value == nil {
		return false, nil
	}
	if *value == runMarkers {
		return true, nil
	}
	return false, badRequest("markers must be %q", runMarkers)
}

// Envelope carries every event on every transport. Seq counts from 1 at each
// boot of the instance; Epoch names the boot, so a seq kept across a restart
// is never mistaken for one of the new boot.
type Envelope struct {
	Ch     string `json:"ch"`
	Seq    uint64 `json:"seq"`
	Epoch  string `json:"epoch"`
	Origin string `json:"origin"`
	Run    *RunID `json:"run,omitempty"`
	T      int64  `json:"t"`
	Ev     Event  `json:"ev"`
	// On a mirrored envelope (a peer/ channel): the source's own seq and epoch, so a consumer
	// here can key the series and dedup as one on the origin would (S2 slice 2, D4 ix#402).
	SrcSeq   uint64 `json:"src_seq,omitempty"`
	SrcEpoch string `json:"src_epoch,omitempty"`

	raw []byte // JSON encoding, built once at publish
}

// Raw returns the envelope's JSON encoding.
func (e *Envelope) Raw() []byte {
	if e.raw == nil {
		e.raw, _ = json.Marshal(e)
	}
	return e.raw
}

// Gap tells a subscriber that envelopes from From on are not coming. Epoch is
// the instance's current epoch, so a subscriber that resumed with a seq of an
// earlier boot ("epoch-changed", or "since-ahead" when it named no epoch)
// learns the new one even on SSE, where there is no subscribe reply.
type Gap struct {
	Reason string `json:"reason"` // "since-too-old" | "overflow" | "epoch-changed" | "since-ahead"
	From   uint64 `json:"from"`   // first seq the subscriber did not get
	Oldest uint64 `json:"oldest,omitempty"`
	Epoch  string `json:"epoch,omitempty"`
}

// Request is one operation, whichever transport carried it. On WebSocket it
// is also the client frame, so it carries the subscription fields. Epoch says
// which boot Since belongs to; without it a Since below the head is trusted.
type Request struct {
	ID      int64   `json:"id,omitempty"`
	Op      string  `json:"op"`
	Ch      string  `json:"ch,omitempty"`
	Exact   bool    `json:"exact,omitempty"`   // subscribe: Ch is one channel's exact name, not a prefix
	Live    bool    `json:"live,omitempty"`    // subscribe: send one {"sub","live":true} frame once the replay is past the head the subscription started from
	Markers *string `json:"markers,omitempty"` // subscribe: "run" includes ixctl.run.* markers
	// WithCursor: probe.list says the point of the plane's stream it stands at (`cursor`: the head
	// and the epoch), read under the tree lock with the nodes, so the list is exactly what ctl
	// says up to that seq (S2 slice 4).
	WithCursor bool     `json:"with_cursor,omitempty"`
	Since      *uint64  `json:"since,omitempty"`
	Epoch      string   `json:"epoch,omitempty"`
	Desired    *Desired `json:"desired,omitempty"`
	Service    string   `json:"service,omitempty"`
	Peer       string   `json:"peer,omitempty"`
	Wait       string   `json:"wait,omitempty"`
	Generation uint64   `json:"generation,omitempty"`
	Run        *RunID   `json:"run,omitempty"`
	TimeoutMS  int64    `json:"timeout_ms,omitempty"`
	Sub        int64    `json:"sub,omitempty"` // unsubscribe: the subscription to drop
	// Probe ops: the node to create, or the name of the node the op is about.
	Probe *NodeSpec `json:"probe,omitempty"`
	Node  string    `json:"node,omitempty"`
	// Bookmark ops: the bookmark's name; on subscribe, the point to replay from in place
	// of since/epoch. Note is bookmark.set's free text: absent keeps a moved bookmark's
	// note, present (even empty) replaces it.
	Bookmark string  `json:"bookmark,omitempty"`
	Note     *string `json:"note,omitempty"`
}

// Broker is all a transport sees of an instance: publish, subscribe with a
// prefix and since, request/reply. Nothing behind it knows which transport
// carried a request.
type Broker interface {
	Publish(ch string, ev Event) *Envelope
	Subscribe(prefix string, since *uint64, epoch string) *Cursor
	Epoch() string
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
	epoch  string // this boot's id, in every envelope and gap

	mu   sync.Mutex
	ring [ringSize]*Envelope
	head uint64        // last seq assigned
	wake chan struct{} // closed and replaced on every publish
	tap  func(*Envelope)
}

func NewHub(origin string) *Hub {
	return &Hub{origin: origin, epoch: newEpoch(), wake: make(chan struct{})}
}

// newEpoch draws a boot id: 12 hex characters, never repeated in practice.
func newEpoch() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		panic("epoch: " + err.Error())
	}
	return hex.EncodeToString(b)
}

// Epoch is this boot's id.
func (h *Hub) Epoch() string { return h.epoch }

// Publish stamps an event with this instance as origin.
func (h *Hub) Publish(ch string, ev Event) *Envelope {
	return h.publish(ch, h.origin, 0, ev, 0, "", nil)
}

// PublishRun stamps an event from an existing run.
func (h *Hub) PublishRun(ch string, ev Event, run RunID) *Envelope {
	return h.publish(ch, h.origin, 0, ev, 0, "", &run)
}

// PublishNewRun stamps an event as the start of a new run, whose seq is this envelope's.
func (h *Hub) PublishNewRun(ch string, ev Event) *Envelope {
	return h.publish(ch, h.origin, 0, ev, 0, "", &RunID{})
}

// Mirror republishes an envelope received from a peer under a local channel, keeping the
// peer as origin and its time, and saying which envelope of the source it was (srcSeq,
// srcEpoch: the source's own seq and epoch). Provenance is a pair: with either half missing
// (a seq of 0, an empty epoch) none is written, in the struct as in the bytes; the link
// refuses such a frame before it gets here.
func (h *Hub) Mirror(ch, origin string, t int64, ev Event, srcSeq uint64, srcEpoch string) *Envelope {
	return h.publish(ch, origin, t, ev, srcSeq, srcEpoch, nil)
}

func (h *Hub) publish(ch, origin string, t int64, ev Event, srcSeq uint64, srcEpoch string, run *RunID) *Envelope {
	evRaw, err := json.Marshal(ev)
	if err != nil {
		ev.Data = nil
		evRaw, _ = json.Marshal(ev)
	}
	chRaw, _ := json.Marshal(ch)
	orRaw, _ := json.Marshal(origin)
	var srcRaw []byte
	if srcSeq == 0 || srcEpoch == "" {
		srcSeq, srcEpoch = 0, ""
	} else {
		epRaw, _ := json.Marshal(srcEpoch) // a peer's epoch came over the wire: escaped like any string
		srcRaw = append(srcRaw, `,"src_seq":`...)
		srcRaw = strconv.AppendUint(srcRaw, srcSeq, 10)
		srcRaw = append(srcRaw, `,"src_epoch":`...)
		srcRaw = append(srcRaw, epRaw...)
	}
	if t == 0 {
		t = time.Now().UnixMilli()
	}

	h.mu.Lock()
	h.head++
	var runRaw []byte
	if run != nil {
		r := *run
		if r.Seq == 0 {
			r = RunID{Origin: origin, Epoch: h.epoch, Seq: h.head}
		}
		run = &r
		encoded, _ := json.Marshal(r)
		runRaw = append(runRaw, `,"run":`...)
		runRaw = append(runRaw, encoded...)
	}
	b := make([]byte, 0, len(evRaw)+len(chRaw)+len(orRaw)+len(srcRaw)+len(runRaw)+64)
	env := &Envelope{Ch: ch, Seq: h.head, Epoch: h.epoch, Origin: origin, Run: run, T: t, Ev: ev, SrcSeq: srcSeq, SrcEpoch: srcEpoch}
	b = append(b, `{"ch":`...)
	b = append(b, chRaw...)
	b = append(b, `,"seq":`...)
	b = strconv.AppendUint(b, env.Seq, 10)
	b = append(b, `,"epoch":"`...)
	b = append(b, h.epoch...) // hex, needs no escaping
	b = append(b, `","origin":`...)
	b = append(b, orRaw...)
	b = append(b, runRaw...)
	b = append(b, srcRaw...)
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

// Span is the head and the oldest seq the ring still holds, read together: a replay
// from since gives every seq in (since, head] only when since + 1 >= oldest; otherwise
// it opens with a since-too-old gap. Cursor.Next computes the same oldest.
func (h *Hub) Span() (head, oldest uint64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	oldest = 1
	if h.head > ringSize {
		oldest = h.head - ringSize + 1
	}
	return h.head, oldest
}

// Cursor reads the ring in seq order, replay and live alike, so a
// subscription can never see a hole without being told.
type Cursor struct {
	h      *Hub
	prefix string
	exact  bool   // prefix is a whole channel name: probe/cpu does not cover probe/cpu-extra
	next   uint64 // next seq wanted
	start  uint64 // head when the cursor was made
	replay bool   // since was given: a first gap is since-too-old
	stale  *Gap   // since came from another boot: told first, then replay from the oldest
}

// Subscribe returns a cursor over envelopes whose channel starts with prefix.
// With since nil it starts live; otherwise it replays every envelope with
// seq > *since still in the ring, then goes live. A since of another boot
// (epoch names a different one, or no epoch and since is past the head) is
// answered with a gap naming this epoch, then everything the ring still holds.
func (h *Hub) Subscribe(prefix string, since *uint64, epoch string) *Cursor {
	h.mu.Lock()
	defer h.mu.Unlock()
	c := &Cursor{h: h, prefix: prefix, next: h.head + 1, start: h.head}
	switch {
	case since == nil:
	case epoch != "" && epoch != h.epoch:
		c.stale = &Gap{Reason: "epoch-changed", From: 1, Epoch: h.epoch}
	case epoch == "" && *since > h.head:
		c.stale = &Gap{Reason: "since-ahead", From: 1, Epoch: h.epoch}
	case *since < h.head:
		c.next = *since + 1
		c.replay = true
	}
	if c.stale != nil {
		c.next = 1
	}
	return c
}

func (c *Cursor) matches(ch string) bool {
	if c.exact {
		return ch == c.prefix
	}
	return strings.HasPrefix(ch, c.prefix)
}

// Head is the hub's head when the cursor was made: envelopes up to it are
// replay, later ones are live.
func (c *Cursor) Head() uint64 { return c.start }

// Epoch is the hub's boot id.
func (c *Cursor) Epoch() string { return c.h.epoch }

// Next blocks until matching envelopes (at most max) or a gap are available.
func (c *Cursor) Next(ctx context.Context, max int) ([]*Envelope, *Gap, error) {
	return c.advance(ctx, max, false)
}

// Catch is Next for a replay: once the cursor is past the head it started from with nothing
// more to return (the ring held no matching envelope up to it, or the last batch ended there),
// it returns empty rather than waiting for a live envelope, so a caller knows the replay is over
// on a quiet channel too (Codex round 43). A pending gap is returned first, as by Next.
func (c *Cursor) Catch(ctx context.Context, max int) ([]*Envelope, *Gap, error) {
	return c.advance(ctx, max, true)
}

func (c *Cursor) advance(ctx context.Context, max int, catch bool) ([]*Envelope, *Gap, error) {
	h := c.h
	for {
		h.mu.Lock()
		oldest := uint64(1)
		if h.head > ringSize {
			oldest = h.head - ringSize + 1
		}
		if g := c.stale; g != nil {
			g.Oldest = oldest
			c.stale, c.next = nil, oldest
			h.mu.Unlock()
			return nil, g, nil
		}
		if c.next < oldest {
			g := &Gap{Reason: "overflow", From: c.next, Oldest: oldest, Epoch: h.epoch}
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
			if c.matches(env.Ch) {
				out = append(out, env)
			}
		}
		wake := h.wake
		h.mu.Unlock()
		if len(out) > 0 || (catch && c.next > c.start) {
			return out, nil, nil
		}
		select {
		case <-wake:
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		}
	}
}
