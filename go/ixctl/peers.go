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
// channels as peer/<name>/<ch>, plus the probe pipes the spec asks for while the peer
// exports them (D4, ix#402), and forwards ops to it. The peer decides.
type peerLink struct {
	spec PeerSpec
	hub  *Hub

	mu       sync.Mutex
	client   *Client
	up       bool
	services json.RawMessage
	token    string              // the peer instance we last saw
	since    map[string]uint64   // last mirrored seq per subscribed prefix or exact channel
	epoch    string              // the peer boot those seqs belong to
	pipes    map[string]*pipeSub // per wanted probe node, what it is now
}

// pipeSub is one wanted probe node's standing on the link.
type pipeSub struct {
	state string // pipeAbsent | pipeNotExportable | pipeMirrored
	sub   int64  // the subscription while mirrored
}

const (
	pipeAbsent        = "absent"
	pipeNotExportable = "not-exportable"
	pipeMirrored      = "mirrored"
)

func newPeerLink(spec PeerSpec, hub *Hub) *peerLink {
	l := &peerLink{spec: spec, hub: hub, since: map[string]uint64{}, pipes: map[string]*pipeSub{}}
	for _, n := range spec.Probes {
		l.pipes[n] = &pipeSub{state: pipeAbsent}
	}
	return l
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
	for _, n := range l.spec.Probes {
		st.Probes = append(st.Probes, PeerProbeStatus{Name: n, State: l.pipes[n].state})
	}
	return st
}

// wanted says whether the spec asks for this probe node's pipe.
func (l *peerLink) wanted(node string) bool {
	_, ok := l.pipes[node]
	return ok
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
		for _, p := range l.pipes {
			*p = pipeSub{state: pipeAbsent}
		}
		l.mu.Unlock()
	}()

	// The wanted probe pipes: the peer's list says which exist and are exportable now; the
	// ctl mirror, subscribed above and live before the list is read, says every change after.
	// An event between the two is applied after the list, and both steps are idempotent, so
	// the order needs no snapshot here (slice 1 of S2; the cursor-bound snapshot is slice 4).
	if err := l.reconcilePipes(ctx, c, merged); err != nil {
		return true, err
	}

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
				if pf.prefix == "ctl" && pf.f.Env.Origin == st.Name {
					if err := l.followPipe(ctx, c, merged, pf.f.Env); err != nil {
						return true, err
					}
				}
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

// reconcilePipes reads the peer's probe list once and mirrors every wanted node it exports.
func (l *peerLink) reconcilePipes(ctx context.Context, c *Client, merged chan<- peerFrame) error {
	if len(l.spec.Probes) == 0 {
		return nil
	}
	data, err := c.Do(ctx, Request{Op: "probe.list"})
	if err != nil {
		return err
	}
	var list struct {
		Nodes []struct {
			Name   string `json:"name"`
			Export bool   `json:"export"`
		} `json:"nodes"`
	}
	if err := json.Unmarshal(data, &list); err != nil {
		return err
	}
	for _, n := range list.Nodes {
		if err := l.setPipe(ctx, c, merged, n.Name, n.Export); err != nil {
			return err
		}
	}
	return nil
}

// followPipe applies one of the peer's own ctl events to the wanted pipes: a probe.created
// opens or declines the pipe by its export flag, a probe.removed closes it.
func (l *peerLink) followPipe(ctx context.Context, c *Client, merged chan<- peerFrame, env *Envelope) error {
	if env.Ev.Kind != "probe.created" && env.Ev.Kind != "probe.removed" {
		return nil
	}
	var d struct {
		Node   string `json:"node"`
		Export bool   `json:"export"`
	}
	if json.Unmarshal(env.Ev.Data, &d) != nil || !l.wanted(d.Node) {
		return nil
	}
	if env.Ev.Kind == "probe.removed" {
		return l.dropPipe(ctx, c, d.Node)
	}
	return l.setPipe(ctx, c, merged, d.Node, d.Export)
}

// setPipe makes a wanted node mirrored (subscribing to its exact channel, resuming from the
// last mirrored seq when there is one) or not-exportable. Idempotent.
func (l *peerLink) setPipe(ctx context.Context, c *Client, merged chan<- peerFrame, node string, export bool) error {
	if !l.wanted(node) {
		return nil
	}
	if !export {
		if err := l.dropPipe(ctx, c, node); err != nil {
			return err
		}
		l.mu.Lock()
		l.pipes[node].state = pipeNotExportable
		l.mu.Unlock()
		return nil
	}
	ch := "probe/" + node
	l.mu.Lock()
	if l.pipes[node].state == pipeMirrored {
		l.mu.Unlock()
		return nil
	}
	var since *uint64
	if s, ok := l.since[ch]; ok {
		since = &s
	}
	epoch := l.epoch
	l.mu.Unlock()
	sctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	id, frames, err := c.SubscribeExact(sctx, ch, since, epoch, 1024)
	cancel()
	if err != nil {
		return err
	}
	go func() {
		for f := range frames {
			select {
			case merged <- peerFrame{ch, f}:
			case <-c.Done():
				return
			}
		}
	}()
	l.mu.Lock()
	*l.pipes[node] = pipeSub{state: pipeMirrored, sub: id}
	l.mu.Unlock()
	return nil
}

// dropPipe ends a wanted node's mirror, if any, and marks it absent.
func (l *peerLink) dropPipe(ctx context.Context, c *Client, node string) error {
	l.mu.Lock()
	p := l.pipes[node]
	was := *p
	*p = pipeSub{state: pipeAbsent}
	l.mu.Unlock()
	if was.state != pipeMirrored {
		return nil
	}
	uctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	return c.Unsubscribe(uctx, was.sub)
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
