package main

import (
	"context"
	"errors"
	"fmt"

	"github.com/coder/websocket"
)

// frameConn is the transport under Client: a duplex of ixctl.v1 frames, one JSON value each.
// Client keeps everything above it (request ids, replies, subscriptions, live frames). Two
// adapters:
//   - wsFrames, a WebSocket: what every process uses.
//   - memFrames, a plane's session served in the same process: what tests use to federate
//     planes with no socket, no state file and no HTTP server.
type frameConn interface {
	// Read blocks for the next frame; an error ends the connection.
	Read(ctx context.Context) ([]byte, error)
	Write(ctx context.Context, frame []byte) error
	Close() error
}

type wsFrames struct{ c *websocket.Conn }

func (w wsFrames) Read(ctx context.Context) ([]byte, error) {
	_, b, err := w.c.Read(ctx)
	return b, err
}

func (w wsFrames) Write(ctx context.Context, frame []byte) error {
	return w.c.Write(ctx, websocket.MessageText, frame)
}

func (w wsFrames) Close() error { return w.c.Close(websocket.StatusNormalClosure, "") }

// errMemClosed is what a memFrames connection reports once its session has ended.
var errMemClosed = errors.New("in-memory session closed")

// clientReadLimit is the largest frame a Client reads from its server.
const clientReadLimit = 16 << 20

// memInQueue bounds the client frames a memFrames connection holds before its session takes them,
// as a socket's buffer does: past it, a write waits.
const memInQueue = 64

// memFrames serves a plane's ixctl.v1 session in memory, as ws.go serves it over a WebSocket.
// Write queues the client's frame and returns while the queue has room (memInQueue frames), as a
// socket write returns once the frame is in its buffer; past it, the write waits. One loop hands
// the queued frames to the session one at a time, in order, as ws.go's read loop does, so a
// session waiting for room in its output holds that loop, and the writers only once the queue is
// full. Read takes what the session queues, in order. The connection ends when the session's
// context does: Close, or the plane the session was opened on shutting down; what is still
// queued either way is dropped, as a socket's buffers are. A frame over wsReadLimit is refused
// and ends the connection, as the WebSocket refuses it; one over clientReadLimit is not read and
// ends it too, as the client's WebSocket does.
//
// @ai:invariant nothing reaches the session or the client once the session has ended, though a select may take a frame queued before [T:test conf:0.9 src:peers_mem_test.go:TestMemFramesHandsNothingAfterTheSessionEnds+TestMemFramesReadsNothingAfterTheSessionEnds]
// @ai:invariant the session handles the frames one at a time, in the order written [T:test conf:0.85 src:peers_mem_test.go:TestMemFramesHandsFramesInOrder]
type memFrames struct {
	s  *session
	in chan []byte
}

func newMemFrames(s *session) *memFrames {
	m := &memFrames{s: s, in: make(chan []byte, memInQueue)}
	go m.serve()
	return m
}

func (m *memFrames) serve() {
	for {
		select {
		case f := <-m.in:
			if m.s.ctx.Err() != nil {
				return // the select may take a queued frame once the session has ended
			}
			m.s.Handle(f)
		case <-m.s.ctx.Done():
			return
		}
	}
}

func (m *memFrames) Read(ctx context.Context) ([]byte, error) {
	select {
	case f := <-m.s.Out():
		if m.s.ctx.Err() != nil {
			return nil, errMemClosed // the select may take a queued frame once the session has ended
		}
		if len(f) > clientReadLimit {
			m.s.Close()
			return nil, fmt.Errorf("frame of %d bytes is over the client's %d-byte limit", len(f), clientReadLimit)
		}
		return f, nil
	case <-m.s.ctx.Done():
		return nil, errMemClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (m *memFrames) Write(ctx context.Context, frame []byte) error {
	switch {
	case m.s.ctx.Err() != nil:
		return errMemClosed
	case len(frame) > wsReadLimit:
		m.s.Close()
		return fmt.Errorf("frame of %d bytes is over the %d-byte limit", len(frame), wsReadLimit)
	}
	f := append([]byte(nil), frame...) // the session may hold it after Write returns
	select {
	case m.in <- f:
		return nil
	case <-m.s.ctx.Done():
		return errMemClosed
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *memFrames) Close() error {
	m.s.Close()
	return nil
}

// dialMem opens a Client on b's ixctl.v1 session in memory. The session lives until the Client
// is closed or parent ends: a plane passes its own lifetime, so its shutdown ends the session.
func dialMem(parent context.Context, b Broker) *Client {
	return newClient(newMemFrames(newMemSession(parent, b)))
}
