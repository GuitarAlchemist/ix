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

	At    int64 `json:"-"` // hrnow when the frame was read
	Bytes int   `json:"-"` // the frame's size on the wire, for a reader that bounds what it holds
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
	subs    map[int64]*sub
	err     error
}

// sub is one subscription's delivery: its frames, and done, closed once, by forget (an
// Unsubscribe, a failed open) or by readLoop when the connection ends. Only readLoop closes
// frames, since it may be sending on it; a reader that may unsubscribe ends on done.
type sub struct {
	frames chan Frame
	done   chan struct{}
}

// Sub is an open subscription: its frames, and Done, closed when the subscription is dropped
// (Unsubscribe) or the connection ends. Frames closes with the connection, and not at all once
// the subscription was dropped first (the reader no longer knows it): a reader ends on Done.
type Sub struct {
	ID     int64
	Frames <-chan Frame
	Done   <-chan struct{}
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
	cl := &Client{c: c, ctx: cctx, cancel: cancel, pending: map[int64]chan Frame{}, subs: map[int64]*sub{}}
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
		f.At, f.Bytes = at, len(b)
		cl.mu.Lock()
		var reply chan Frame
		var s *sub
		switch {
		case f.Re != nil:
			reply = cl.pending[*f.Re]
			delete(cl.pending, *f.Re)
		case f.Sub != nil:
			s = cl.subs[*f.Sub]
		}
		cl.mu.Unlock()
		switch {
		case reply != nil:
			select {
			case reply <- f:
			case <-cl.ctx.Done():
			}
		case s != nil:
			select {
			case s.frames <- f:
			case <-s.done: // dropped while this frame waited for room: nobody reads it any more
			case <-cl.ctx.Done():
			}
		}
	}
	cl.mu.Lock()
	cl.err = err
	for id, s := range cl.subs {
		close(s.frames)
		close(s.done)
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

func (cl *Client) register(frames chan Frame) (int64, chan Frame, *sub) {
	cl.mu.Lock()
	defer cl.mu.Unlock()
	cl.nextID++
	id := cl.nextID
	reply := make(chan Frame, 1)
	cl.pending[id] = reply
	var s *sub
	if frames != nil {
		s = &sub{frames: frames, done: make(chan struct{})}
		cl.subs[id] = s
	}
	return id, reply, s
}

// forget drops a request and, for a subscription, closes its done (once: the entry leaves the
// map in the same critical section). Only readLoop ever closes a subscription's frames.
func (cl *Client) forget(id int64) {
	cl.mu.Lock()
	delete(cl.pending, id)
	if s := cl.subs[id]; s != nil {
		delete(cl.subs, id)
		close(s.done)
	}
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
	id, reply, _ := cl.register(nil)
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
	return cl.subscribe(ctx, Request{Op: "subscribe", Ch: prefix, Since: since, Epoch: epoch}, buf)
}

// SubscribeExact subscribes to one channel by its exact name, not as a prefix: a peer link
// mirroring probe/cpu must not receive probe/cpu-extra. It returns the Sub, Done included,
// because such a subscription is the one a caller drops while the connection goes on.
func (cl *Client) SubscribeExact(ctx context.Context, ch string, since *uint64, epoch string, buf int) (*Sub, error) {
	return cl.open(ctx, Request{Op: "subscribe", Ch: ch, Since: since, Epoch: epoch, Exact: true}, buf)
}

// Unsubscribe drops a subscription on both sides and closes its Done. Frames already queued on
// its channel stay readable; the channel itself is never closed after this (the reader has
// forgotten it), so a reader that unsubscribes ends on Done, not on the channel's close.
func (cl *Client) Unsubscribe(ctx context.Context, id int64) error {
	_, err := cl.Do(ctx, Request{Op: "unsubscribe", Sub: id})
	cl.forget(id)
	return err
}

// SubscribeBookmark replays from a bookmark the plane holds: everything on prefix after
// the point the bookmark names, then live. A bookmark of an earlier boot gives an
// epoch-changed gap first; one older than the ring a since-too-old gap.
func (cl *Client) SubscribeBookmark(ctx context.Context, prefix, bookmark string, buf int) (int64, <-chan Frame, error) {
	return cl.subscribe(ctx, Request{Op: "subscribe", Ch: prefix, Bookmark: bookmark}, buf)
}

func (cl *Client) subscribe(ctx context.Context, req Request, buf int) (int64, <-chan Frame, error) {
	s, err := cl.open(ctx, req, buf)
	if err != nil {
		return 0, nil, err
	}
	return s.ID, s.Frames, nil
}

func (cl *Client) open(ctx context.Context, req Request, buf int) (*Sub, error) {
	frames := make(chan Frame, buf)
	id, reply, s := cl.register(frames)
	req.ID = id
	if err := cl.send(ctx, req); err != nil {
		cl.forget(id)
		return nil, err
	}
	if _, err := cl.wait(ctx, id, reply); err != nil {
		cl.forget(id)
		return nil, err
	}
	return &Sub{ID: id, Frames: frames, Done: s.done}, nil
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
