package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/coder/websocket"
)

const subprotocol = "ixctl.v1"

// loopbackHTTP never goes through a proxy (every address ixctl talks to is
// loopback) and never follows a redirect: Go keeps Authorization on a
// same-host redirect, even to another port.
var loopbackHTTP = &http.Client{
	Transport:     &http.Transport{Proxy: nil, MaxIdleConnsPerHost: 4},
	CheckRedirect: noRedirect,
}

func noRedirect(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }

// Frame is anything the plane sends on ixctl.v1: a reply, an event or a gap.
type Frame struct {
	Re    *int64          `json:"re,omitempty"`
	OK    bool            `json:"ok"`
	Data  json.RawMessage `json:"data,omitempty"`
	Error string          `json:"error,omitempty"`
	Sub   *int64          `json:"sub,omitempty"`
	Env   *Envelope       `json:"env,omitempty"`
	Gap   *Gap            `json:"gap,omitempty"`

	At int64 `json:"-"` // hrnow when the frame was read
}

// remoteError is a refusal sent back by the far side.
type remoteError struct{ msg string }

func (e *remoteError) Error() string { return e.msg }

// Client speaks ixctl.v1 over one WebSocket: requests and subscriptions
// multiplexed on it.
type Client struct {
	c      *websocket.Conn
	ctx    context.Context
	cancel context.CancelFunc

	mu      sync.Mutex
	nextID  int64
	pending map[int64]chan Frame
	subs    map[int64]chan Frame
	err     error
}

// DialWS sends the token only to a loopback IP literal and port.
func DialWS(ctx context.Context, addr, token string) (*Client, error) {
	if err := checkLoopback(addr); err != nil {
		return nil, err
	}
	c, _, err := websocket.Dial(ctx, "ws://"+addr+"/v1/ws", &websocket.DialOptions{
		HTTPClient:   loopbackHTTP,
		HTTPHeader:   http.Header{"Authorization": {"Bearer " + token}},
		Subprotocols: []string{subprotocol},
	})
	if err != nil {
		return nil, err
	}
	if c.Subprotocol() != subprotocol {
		c.CloseNow()
		return nil, fmt.Errorf("server did not accept %s", subprotocol)
	}
	c.SetReadLimit(16 << 20)
	cctx, cancel := context.WithCancel(context.Background())
	cl := &Client{c: c, ctx: cctx, cancel: cancel, pending: map[int64]chan Frame{}, subs: map[int64]chan Frame{}}
	go cl.readLoop()
	return cl, nil
}

func (cl *Client) readLoop() {
	var err error
	for {
		var b []byte
		_, b, err = cl.c.Read(cl.ctx)
		if err != nil {
			break
		}
		at := hrnow()
		var f Frame
		if json.Unmarshal(b, &f) != nil {
			continue
		}
		f.At = at
		cl.mu.Lock()
		var ch chan Frame
		switch {
		case f.Re != nil:
			ch = cl.pending[*f.Re]
			delete(cl.pending, *f.Re)
		case f.Sub != nil:
			ch = cl.subs[*f.Sub]
		}
		cl.mu.Unlock()
		if ch == nil {
			continue
		}
		select {
		case ch <- f:
		case <-cl.ctx.Done():
		}
	}
	cl.mu.Lock()
	cl.err = err
	for id, ch := range cl.subs {
		close(ch)
		delete(cl.subs, id)
	}
	cl.mu.Unlock()
	cl.cancel()
}

// Done is closed when the connection is gone.
func (cl *Client) Done() <-chan struct{} { return cl.ctx.Done() }

func (cl *Client) Err() error {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	return cl.err
}

func (cl *Client) Close() error {
	cl.cancel()
	return cl.c.Close(websocket.StatusNormalClosure, "")
}

func (cl *Client) send(ctx context.Context, req Request) error {
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	return cl.c.Write(ctx, websocket.MessageText, b)
}

func (cl *Client) register(sub chan Frame) (int64, chan Frame) {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	cl.nextID++
	id := cl.nextID
	reply := make(chan Frame, 1)
	cl.pending[id] = reply
	if sub != nil {
		cl.subs[id] = sub
	}
	return id, reply
}

// forget drops a request; only readLoop ever closes a subscription channel.
func (cl *Client) forget(id int64) {
	cl.mu.Lock()
	delete(cl.pending, id)
	delete(cl.subs, id)
	cl.mu.Unlock()
}

func (cl *Client) wait(ctx context.Context, id int64, reply chan Frame) (json.RawMessage, error) {
	select {
	case f := <-reply:
		if !f.OK {
			return nil, &remoteError{msg: f.Error}
		}
		return f.Data, nil
	case <-ctx.Done():
		cl.forget(id)
		return nil, ctx.Err()
	case <-cl.ctx.Done():
		cl.forget(id)
		return nil, errors.New("connection closed")
	}
}

// Do sends one request and waits for its reply.
func (cl *Client) Do(ctx context.Context, req Request) (json.RawMessage, error) {
	id, reply := cl.register(nil)
	req.ID = id
	if err := cl.send(ctx, req); err != nil {
		cl.forget(id)
		return nil, err
	}
	return cl.wait(ctx, id, reply)
}

// Subscribe opens a subscription; its events and gaps arrive on the channel,
// which is closed with the connection. A full channel holds the reader back.
func (cl *Client) Subscribe(ctx context.Context, prefix string, since *uint64, buf int) (int64, <-chan Frame, error) {
	return cl.SubscribeFrom(ctx, prefix, since, "", buf)
}

// SubscribeFrom is Subscribe with the epoch since belongs to: a plane that
// has rebooted since then answers with an epoch-changed gap, then replays
// what it still holds, instead of reading since as one of its own seqs.
func (cl *Client) SubscribeFrom(ctx context.Context, prefix string, since *uint64, epoch string, buf int) (int64, <-chan Frame, error) {
	frames := make(chan Frame, buf)
	id, reply := cl.register(frames)
	if err := cl.send(ctx, Request{ID: id, Op: "subscribe", Ch: prefix, Since: since, Epoch: epoch}); err != nil {
		cl.forget(id)
		return 0, nil, err
	}
	if _, err := cl.wait(ctx, id, reply); err != nil {
		cl.forget(id)
		return 0, nil, err
	}
	return id, frames, nil
}

// readStateFile reads an instance's state file: where it listens and its token.
func readStateFile(path string) (stateFile, error) {
	var st stateFile
	b, err := os.ReadFile(path)
	if err != nil {
		return st, err
	}
	if err := json.Unmarshal(b, &st); err != nil {
		return st, fmt.Errorf("%s: %w", path, err)
	}
	if st.Addr == "" || st.Token == "" {
		return st, fmt.Errorf("%s: no addr or token", path)
	}
	return st, nil
}

type stateFile struct {
	Name  string `json:"name"`
	Addr  string `json:"addr"`
	Pid   int    `json:"pid"`
	Token string `json:"token"`
}

func waitFor(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}
