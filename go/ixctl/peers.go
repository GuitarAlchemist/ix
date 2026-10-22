package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"time"
)

const (
	peerBackoffMin = 200 * time.Millisecond
	peerBackoffMax = 5 * time.Second
	peerRefresh    = 100 * time.Millisecond // at most one status refresh per interval
)

// peerLink federates with one peer: it mirrors the peer's ctl and svc/
// channels as peer/<name>/<ch>, and forwards ops to it. The peer decides.
type peerLink struct {
	spec PeerSpec
	hub  *Hub

	mu       sync.Mutex
	client   *Client
	up       bool
	services json.RawMessage
	token    string            // the peer instance we last saw
	since    map[string]uint64 // last mirrored seq per subscribed prefix
	epoch    string            // the peer boot those seqs belong to
}

func newPeerLink(spec PeerSpec, hub *Hub) *peerLink {
	return &peerLink{spec: spec, hub: hub, since: map[string]uint64{}}
}

func (l *peerLink) status() PeerStatus {
	l.mu.Lock()
	defer l.mu.Unlock()
	st := PeerStatus{Name: l.spec.Name, State: "down", Services: l.services}
	if l.up {
		st.State = "up"
	}
	if st.Services == nil {
		st.Services = json.RawMessage("[]")
	}
	return st
}

func (l *peerLink) forward(ctx context.Context, req Request) (any, error) {
	l.mu.Lock()
	c := l.client
	l.mu.Unlock()
	if c == nil {
		return nil, &opError{status: 502, msg: "peer " + l.spec.Name + " is down"}
	}
	data, err := c.Do(ctx, req)
	if err != nil {
		var re *remoteError
		if errors.As(err, &re) {
			return nil, &opError{status: 400, msg: "peer " + l.spec.Name + ": " + re.msg}
		}
		return nil, &opError{status: 502, msg: "peer " + l.spec.Name + ": " + err.Error()}
	}
	return data, nil
}

// run keeps the link up, reconnecting with backoff.
func (l *peerLink) run(ctx context.Context) {
	backoff := peerBackoffMin
	for ctx.Err() == nil {
		t0 := time.Now()
		wasUp, err := l.session(ctx)
		if wasUp {
			reason := "closed"
			if err != nil {
				reason = err.Error()
			}
			l.hub.Publish("ctl", event("peer.down", map[string]any{"peer": l.spec.Name, "reason": reason}))
			if time.Since(t0) > peerBackoffMax {
				backoff = peerBackoffMin
			}
		}
		if !waitFor(ctx, backoff) {
			return
		}
		if backoff *= 2; backoff > peerBackoffMax {
			backoff = peerBackoffMax
		}
	}
}

// session is one connection to the peer, from dial to loss.
func (l *peerLink) session(ctx context.Context) (bool, error) {
	st, err := readStateFile(l.spec.StateFile)
	if err != nil {
		return false, err
	}
	if err := checkLoopback(st.Addr); err != nil {
		return false, err
	}
	dctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	c, err := DialWS(dctx, st.Addr, st.Token)
	cancel()
	if err != nil {
		return false, err
	}
	defer c.Close()
	go func() {
		select {
		case <-ctx.Done():
			c.Close()
		case <-c.Done():
		}
	}()

	l.mu.Lock()
	if st.Token != l.token { // a new peer instance: its seqs start over
		l.token, l.since = st.Token, map[string]uint64{}
	}
	since := map[string]*uint64{}
	for _, p := range []string{"ctl", "svc/"} {
		if s, ok := l.since[p]; ok {
			since[p] = &s
		}
	}
	epoch := l.epoch
	l.mu.Unlock()

	merged := make(chan peerFrame, 1024)
	for _, prefix := range []string{"ctl", "svc/"} {
		_, frames, err := c.SubscribeFrom(ctx, prefix, since[prefix], epoch, 1024)
		if err != nil {
			return false, err
		}
		go func() {
			for f := range frames {
				select {
				case merged <- peerFrame{prefix, f}:
				case <-c.Done():
					return
				}
			}
		}()
	}

	if data, err := c.Do(ctx, Request{Op: "status"}); err == nil {
		l.setServices(data)
	}
	refresh := make(chan struct{}, 1)
	go l.refresher(c, refresh)

	l.mu.Lock()
	l.client, l.up = c, true
	l.mu.Unlock()
	l.hub.Publish("ctl", event("peer.up", map[string]any{"peer": l.spec.Name, "addr": st.Addr, "origin": st.Name}))
	defer func() {
		l.mu.Lock()
		l.client, l.up = nil, false
		l.mu.Unlock()
	}()

	for {
		select {
		case pf := <-merged:
			if pf.f.Gap != nil && pf.f.Gap.Reason == "overflow" {
				// The peer dropped this subscription: reconnect, and resume
				// from the last mirrored seq.
				return true, errors.New("mirror of " + pf.prefix + " overflowed")
			}
			if pf.f.Env != nil {
				l.mirror(st.Name, pf.prefix, pf.f.Env)
				select {
				case refresh <- struct{}{}:
				default:
				}
			}
		case <-c.Done():
			return true, c.Err()
		}
	}
}

type peerFrame struct {
	prefix string
	f      Frame
}

// mirror republishes one of the peer's own envelopes. Envelopes the peer
// itself mirrored (another origin, or a peer/ channel) are never forwarded.
func (l *peerLink) mirror(origin, prefix string, env *Envelope) {
	l.mu.Lock()
	l.since[prefix], l.epoch = env.Seq, env.Epoch
	l.mu.Unlock()
	if env.Origin != origin || strings.HasPrefix(env.Ch, "peer/") {
		return
	}
	l.hub.Mirror("peer/"+l.spec.Name+"/"+env.Ch, env.Origin, env.T, env.Ev)
}

// refresher keeps the peer's service list for status, at most once per
// peerRefresh, on demand.
func (l *peerLink) refresher(c *Client, kick <-chan struct{}) {
	for {
		select {
		case <-kick:
		case <-c.Done():
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		data, err := c.Do(ctx, Request{Op: "status"})
		cancel()
		if err == nil {
			l.setServices(data)
		}
		if !waitFor(c.ctx, peerRefresh) {
			return
		}
	}
}

func (l *peerLink) setServices(status json.RawMessage) {
	var st struct {
		Services json.RawMessage `json:"services"`
	}
	if json.Unmarshal(status, &st) == nil {
		l.mu.Lock()
		l.services = st.Services
		l.mu.Unlock()
	}
}
