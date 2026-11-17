package main

import (
	"context"
	"errors"
	"fmt"
)

// The in-memory adapter of frameConn and of the session: tests federate planes through it with
// no socket, no state file and no HTTP server. No production code serves a session in memory.

// newMemSession is the in-memory adapter: the caller is the transport.
func newMemSession(parent context.Context, b Broker) *session {
	return newSession(parent, b, sessionQueue)
}

// errMemClosed is what a memFrames connection reports once its session has ended.
var errMemClosed = errors.New("in-memory session closed")

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
// ends it too, as the client's WebSocket does. A write returns nil only for a frame queued
// before the end, as a WebSocket write that commits after its connection closed fails.
//
// Pinned by TestMemFramesHandsNothingAfterTheSessionEnds, TestMemFramesReadsNothingAfterTheSessionEnds,
// TestMemFramesWriteFailsOnceClosed and TestMemFramesHandsFramesInOrder:
// @ai:invariant a frame serve or Read takes once the session has ended goes no further; one taken just before may still be handled or read, as ws.go may handle a frame it read just before [P:assumed conf:0.8]
// @ai:invariant Write returns nil only if its frame was queued while the session ran [P:assumed conf:0.8]
// @ai:invariant serve invokes Handle for one frame at a time, in the order the frames were queued (the order of sequential writes) [P:assumed conf:0.8]
type memFrames struct {
	s      *session
	in     chan []byte
	served chan struct{} // closed when serve has returned: nothing more reaches the session

	// beforeSend, nil unless a test sets it: Write calls it once its frame passed the
	// checks, before it is queued.
	beforeSend func()
}

func newMemFrames(s *session) *memFrames {
	m := &memFrames{s: s, in: make(chan []byte, memInQueue), served: make(chan struct{})}
	go m.serve()
	return m
}

func (m *memFrames) serve() {
	defer close(m.served)
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
	if m.beforeSend != nil {
		m.beforeSend()
	}
	select {
	case m.in <- f:
		if m.s.ctx.Err() != nil {
			return errMemClosed // queued once the session ended (serve freed a slot as it left): never handled
		}
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
