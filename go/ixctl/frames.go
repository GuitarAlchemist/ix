package main

import (
	"context"

	"github.com/coder/websocket"
)

// frameConn is the transport under Client: a duplex of ixctl.v1 frames, one JSON value each.
// Client keeps everything above it (request ids, replies, subscriptions, live frames). Two
// adapters:
//   - wsFrames, a WebSocket: what every process uses.
//   - memFrames (frames_mem_test.go), a plane's session served in the same process: what tests
//     use to federate planes with no socket, no state file and no HTTP server.
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

// clientReadLimit is the largest frame a Client reads from its server.
const clientReadLimit = 16 << 20
