package main

import (
	"context"
	"net/http"
	"time"

	"github.com/coder/websocket"
)

// wsWriteLimit closes a connection whose client stops reading for this long.
const wsWriteLimit = 10 * time.Second

// ws upgrades to ixctl.v1. Host, Origin and token were checked by ServeHTTP,
// so the library's own origin check is off; a client that does not ask for
// ixctl.v1 is refused.
func (s *server) ws(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		Subprotocols:       []string{subprotocol},
		InsecureSkipVerify: true,
	})
	if err != nil {
		return
	}
	if c.Subprotocol() != subprotocol {
		c.Close(websocket.StatusPolicyViolation, "subprotocol "+subprotocol+" required")
		return
	}
	c.SetReadLimit(1 << 20)
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	stop := context.AfterFunc(s.ctx, cancel)
	defer stop()

	sess := newSession(ctx, s.b, sessionQueue)
	sess.runMarkersUnsupported = &s.runMarkersUnsupported
	defer sess.Close()
	go func() {
		defer cancel()
		for {
			select {
			case f := <-sess.Out():
				if d := s.writeDelay.Load(); d > 0 {
					time.Sleep(time.Duration(d)) // the test knob: a slow socket
				}
				wctx, wcancel := context.WithTimeout(ctx, wsWriteLimit)
				err := c.Write(wctx, websocket.MessageText, f)
				wcancel()
				if err != nil {
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	for {
		_, msg, err := c.Read(ctx)
		if err != nil {
			break
		}
		sess.Handle(msg)
	}
	c.CloseNow()
}
