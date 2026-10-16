package main

import (
	"context"
	"encoding/json"
	"strconv"
	"sync"
)

// sessionQueue bounds what one connection may have waiting to be written.
const sessionQueue = 256

// session is the ixctl.v1 protocol over any duplex frame transport: the
// transport feeds it client frames with Handle and writes what comes out of
// Out. WebSocket is one such transport; the in-memory adapter is another.
type session struct {
	b      Broker
	out    chan []byte
	ctx    context.Context
	cancel context.CancelFunc

	mu   sync.Mutex
	subs map[int64]*subEntry

	// Test seams, nil in production: pumpLeaving runs as a pump starts to
	// leave (before it removes its entry), pumpLeft once it has gone.
	pumpLeaving, pumpLeft func(id int64)
}

type subEntry struct {
	prefix string
	cancel context.CancelFunc
}

func newSession(parent context.Context, b Broker, queue int) *session {
	ctx, cancel := context.WithCancel(parent)
	return &session{b: b, out: make(chan []byte, queue), ctx: ctx, cancel: cancel, subs: map[int64]*subEntry{}}
}

// newMemSession is the in-memory adapter: the caller is the transport.
func newMemSession(parent context.Context, b Broker) *session {
	return newSession(parent, b, sessionQueue)
}

// Out carries every frame for the client, in the order it must be written.
func (s *session) Out() <-chan []byte { return s.out }

func (s *session) Close() { s.cancel() }

// Handle takes one client frame. Subscriptions are set up in line, so their
// reply is queued before their first event; other ops run on their own so a
// slow stop never holds the connection.
func (s *session) Handle(frame []byte) {
	var req Request
	if err := json.Unmarshal(frame, &req); err != nil {
		s.reply(0, nil, badRequest("bad frame: %v", err))
		return
	}
	switch req.Op {
	case "subscribe":
		s.subscribe(req)
	case "unsubscribe":
		s.unsubscribe(req)
	default:
		go func() {
			data, err := s.b.Request(s.ctx, req)
			s.reply(req.ID, data, err)
		}()
	}
}

type replyFrame struct {
	Re    int64  `json:"re"`
	OK    bool   `json:"ok"`
	Data  any    `json:"data,omitempty"`
	Error string `json:"error,omitempty"`
}

func (s *session) reply(id int64, data any, err error) {
	f := replyFrame{Re: id, OK: err == nil, Data: data}
	if err != nil {
		f.Error = err.Error()
	}
	b, merr := json.Marshal(f)
	if merr != nil {
		b, _ = json.Marshal(replyFrame{Re: id, Error: merr.Error()})
	}
	s.send(s.ctx, b)
}

// send queues a frame, waiting for room.
func (s *session) send(ctx context.Context, b []byte) bool {
	select {
	case s.out <- b:
		return true
	case <-ctx.Done():
		return false
	}
}

func (s *session) subscribe(req Request) {
	if req.Peer != "" {
		s.reply(req.ID, nil, badRequest("subscribe is local: a peer's channels are mirrored under peer/%s/", req.Peer))
		return
	}
	s.mu.Lock()
	if _, dup := s.subs[req.ID]; dup {
		s.mu.Unlock()
		s.reply(req.ID, nil, badRequest("subscription %d already exists", req.ID))
		return
	}
	cur := s.b.Subscribe(req.Ch, req.Since)
	ctx, cancel := context.WithCancel(s.ctx)
	e := &subEntry{prefix: req.Ch, cancel: cancel}
	s.subs[req.ID] = e
	s.mu.Unlock()
	s.reply(req.ID, map[string]any{"sub": req.ID, "head": cur.Head()}, nil)
	go s.pump(ctx, e, req.ID, cur)
}

func (s *session) unsubscribe(req Request) {
	s.mu.Lock()
	n := 0
	for id, e := range s.subs {
		if (req.Sub != 0 && id == req.Sub) || (req.Sub == 0 && req.Ch != "" && e.prefix == req.Ch) {
			e.cancel()
			delete(s.subs, id)
			n++
		}
	}
	s.mu.Unlock()
	if n == 0 {
		s.reply(req.ID, nil, badRequest("no such subscription"))
		return
	}
	s.reply(req.ID, map[string]any{"dropped": n}, nil)
}

// dropIf ends a pump's own subscription, and removes its entry only while
// the id still names it: after an unsubscribe the same id may already name
// a newer subscription.
func (s *session) dropIf(id int64, mine *subEntry) {
	mine.cancel()
	s.mu.Lock()
	if s.subs[id] == mine {
		delete(s.subs, id)
	}
	s.mu.Unlock()
}

// pump moves one subscription from the ring to the connection queue. Replay
// waits for room (the client asked for it, and it is bounded by the ring);
// live events never wait: a full queue drops the subscription with a gap.
func (s *session) pump(ctx context.Context, e *subEntry, id int64, cur *Cursor) {
	defer func() {
		if s.pumpLeaving != nil {
			s.pumpLeaving(id)
		}
		s.dropIf(id, e)
		if s.pumpLeft != nil {
			s.pumpLeft(id)
		}
	}()
	replayUntil := cur.Head()
	for {
		envs, gap, err := cur.Next(ctx, 256)
		if err != nil {
			return
		}
		if gap != nil {
			s.send(ctx, gapFrame(id, gap))
			if gap.Reason == "overflow" {
				return
			}
			continue
		}
		for _, env := range envs {
			f := eventFrame(id, env)
			if env.Seq <= replayUntil {
				if !s.send(ctx, f) {
					return
				}
				continue
			}
			select {
			case s.out <- f:
			case <-ctx.Done():
				return
			default:
				s.dropIf(id, e)
				s.send(s.ctx, gapFrame(id, &Gap{Reason: "overflow", From: env.Seq}))
				return
			}
		}
	}
}

func eventFrame(id int64, env *Envelope) []byte {
	raw := env.Raw()
	b := make([]byte, 0, len(raw)+32)
	b = append(b, `{"sub":`...)
	b = strconv.AppendInt(b, id, 10)
	b = append(b, `,"env":`...)
	b = append(b, raw...)
	return append(b, '}')
}

func gapFrame(id int64, g *Gap) []byte {
	b, _ := json.Marshal(struct {
		Sub int64 `json:"sub"`
		Gap *Gap  `json:"gap"`
	}{id, g})
	return b
}
