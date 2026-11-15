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

// memFrames serves a plane's ixctl.v1 session in memory, as ws.go serves it over a WebSocket:
// Write hands the client's frame to the session, Read takes what the session queues, in order.
// The connection ends when the session's context does: Close, or the plane the session was
// opened on shutting down. A frame over wsReadLimit is refused, as the WebSocket refuses it.
type memFrames struct{ s *session }

func (m memFrames) Read(ctx context.Context) ([]byte, error) {
	select {
	case f := <-m.s.Out():
		return f, nil
	case <-m.s.ctx.Done():
		return nil, errMemClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (m memFrames) Write(ctx context.Context, frame []byte) error {
	switch {
	case m.s.ctx.Err() != nil:
		return errMemClosed
	case ctx.Err() != nil:
		return ctx.Err()
	case len(frame) > wsReadLimit:
		m.s.Close()
		return fmt.Errorf("frame of %d bytes is over the %d-byte limit", len(frame), wsReadLimit)
	}
	m.s.Handle(frame)
	return nil
}

func (m memFrames) Close() error {
	m.s.Close()
	return nil
}

// dialMem opens a Client on b's ixctl.v1 session in memory. The session lives until the Client
// is closed or parent ends: a plane passes its own lifetime, so its shutdown ends the session.
func dialMem(parent context.Context, b Broker) *Client {
	return newClient(memFrames{newMemSession(parent, b)})
}
