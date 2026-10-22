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
	since    map[string]uint64   // last mirrored seq per prefix (ctl, svc/: resumed on reconnect) or pipe (recorded, never resumed)
	epoch    string              // the peer boot those seqs belong to
	pipes    map[string]*pipeSub // per wanted probe node, what it is now
}

// pipeSub is one wanted probe node's standing on the link. gen counts its changes: a subscribe
// or unsubscribe that completes after the pipe moved on finds another gen and stands down, and
// a frame from a subscription that is no longer the pipe's is dropped.
type pipeSub struct {
	state string // pipeAbsent | pipeNotExportable | pipeOpening | pipeMirrored
	sub   int64  // the subscription while mirrored
	gen   uint64
}

const (
	pipeAbsent        = "absent"
	pipeNotExportable = "not-exportable"
	pipeOpening       = "opening" // subscribing to the peer's pipe
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
				case merged <- peerFrame{prefix, f, 0}:
				case <-c.Done():
					return
				}
			}
		}()
	}

	// The peer's status and, when pipes are wanted, its probe list, before any mirrored frame is
	// applied. Both replies come through the client's reader, which a resumed subscription's
	// replay could hold back if nothing drained merged and the subscription channels filled
	// (a reconnect after a long outage replays up to the ring); so merged is drained into a
	// backlog meanwhile, bounded by that replay plus what arrives during the bound, and applied
	// in order once the list is. A status that cannot be read is skipped as before; a list that
	// cannot be read ends the session.
	type reply struct {
		op   string
		data json.RawMessage
		err  error
	}
	ops := []string{"status"}
	if len(l.spec.Probes) > 0 {
		ops = append(ops, "probe.list")
	}
	got := make(chan reply, len(ops))
	for _, op := range ops {
		go func(op string) {
			rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			data, err := c.Do(rctx, Request{Op: op})
			got <- reply{op, data, err}
		}(op)
	}
	var backlog []peerFrame
	var list json.RawMessage
	for n := 0; n < len(ops); {
		select {
		case pf := <-merged:
			backlog = append(backlog, pf)
		case r := <-got:
			n++
			switch {
			case r.op == "status" && r.err == nil:
				l.setServices(r.data)
			case r.op == "probe.list" && r.err != nil:
				return true, r.err
			case r.op == "probe.list":
				list = r.data
			}
		case <-c.Done():
			return true, c.Err()
		}
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
			p.gen++
			p.state, p.sub = pipeAbsent, 0
		}
		l.mu.Unlock()
	}()

	// The wanted probe pipes: the list says which exist and are exportable now; the ctl mirror,
	// subscribed above and so live before the list, says every change after, applied in arrival
	// order. A lifecycle event older than the list is applied after it (it may reopen a pipe
	// the list no longer has, until its removal follows a moment later), and one lost to a gap
	// is not seen: that window is what the cursor-bound snapshot closes (slice 4 of S2); the
	// list is read again on every reconnect. Subscribing and unsubscribing run on their own
	// goroutines: this loop is the one consumer of merged, and a reply the peer sends can only
	// reach us while the client's reader is not held back by a full subscription channel, so
	// the loop never waits on one.
	fail := make(chan error, 1)
	if err := l.reconcilePipes(ctx, c, merged, fail, st.Name, list, backlog); err != nil {
		return true, err
	}
	for _, pf := range backlog {
		if err := l.apply(ctx, c, merged, fail, refresh, st.Name, pf); err != nil {
			return true, err
		}
	}

	for {
		select {
		case pf := <-merged:
			if err := l.apply(ctx, c, merged, fail, refresh, st.Name, pf); err != nil {
				return true, err
			}
		case err := <-fail:
			return true, err
		case <-c.Done():
			return true, c.Err()
		}
	}
}

// apply mirrors one frame the link received, follows the peer's own lifecycle events on the
// wanted pipes, and kicks the status refresher. An overflow ends the session: the peer dropped
// that subscription, and a reconnect resumes it from the last mirrored seq.
func (l *peerLink) apply(ctx context.Context, c *Client, merged chan<- peerFrame, fail chan<- error, refresh chan<- struct{}, origin string, pf peerFrame) error {
	if pf.f.Gap != nil && pf.f.Gap.Reason == "overflow" {
		return errors.New("mirror of " + pf.prefix + " overflowed")
	}
	if pf.f.Env == nil {
		return nil
	}
	if pf.sub != 0 && !l.current(pf.prefix, pf.sub) {
		return nil // a frame of a subscription that is no longer the pipe's
	}
	l.mirror(origin, pf.prefix, pf.f.Env)
	if pf.prefix == "ctl" && pf.f.Env.Origin == origin {
		l.followPipe(ctx, c, merged, fail, pf.f.Env)
	}
	select {
	case refresh <- struct{}{}:
	default:
	}
	return nil
}

// current says whether sub is still the subscription of the pipe channel ch.
func (l *peerLink) current(ch string, sub int64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	p := l.pipes[strings.TrimPrefix(ch, "probe/")]
	return p != nil && p.state == pipeMirrored && p.sub == sub
}

// reconcilePipes applies the peer's probe list, read before any frame of the backlog is
// applied (so no lifecycle event already queued is undone by a list arriving later, Codex
// round 36), and mirrors every wanted node it exports. A pipe resumes from its last mirrored
// seq only when the ctl replay in the backlog proves it is still the incarnation the link last
// mirrored: no gap on ctl, and no probe.created / probe.removed of that name since. Otherwise
// it starts live: the same channel name may carry another incarnation's records, and the link
// cannot tell them apart yet (slice 2 brings the created seq, slice 3 the bounded replay and
// the continuity marker for what a live start skips).
func (l *peerLink) reconcilePipes(ctx context.Context, c *Client, merged chan<- peerFrame, fail chan<- error, origin string, data json.RawMessage, backlog []peerFrame) error {
	if len(l.spec.Probes) == 0 {
		return nil
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
	ctlGap, moved := false, map[string]bool{}
	for _, pf := range backlog {
		if pf.prefix != "ctl" {
			continue
		}
		if pf.f.Gap != nil {
			ctlGap = true
		}
		if pf.f.Env == nil || pf.f.Env.Origin != origin {
			continue
		}
		if k := pf.f.Env.Ev.Kind; k == "probe.created" || k == "probe.removed" {
			var d struct {
				Node string `json:"node"`
			}
			if json.Unmarshal(pf.f.Env.Ev.Data, &d) == nil {
				moved[d.Node] = true
			}
		}
	}
	for _, n := range list.Nodes {
		l.setPipe(ctx, c, merged, fail, n.Name, n.Export, !ctlGap && !moved[n.Name])
	}
	return nil
}

func (l *peerLink) failWith(fail chan<- error, err error) {
	select {
	case fail <- err:
	default:
	}
}

// followPipe applies one of the peer's own ctl events to the wanted pipes: a probe.created
// opens or declines the pipe by its export flag, a probe.removed closes it.
func (l *peerLink) followPipe(ctx context.Context, c *Client, merged chan<- peerFrame, fail chan<- error, env *Envelope) {
	if env.Ev.Kind != "probe.created" && env.Ev.Kind != "probe.removed" {
		return
	}
	var d struct {
		Node   string `json:"node"`
		Export bool   `json:"export"`
	}
	if json.Unmarshal(env.Ev.Data, &d) != nil || !l.wanted(d.Node) {
		return
	}
	if env.Ev.Kind == "probe.removed" {
		l.dropPipe(ctx, c, d.Node, pipeAbsent)
		return
	}
	l.setPipe(ctx, c, merged, fail, d.Node, d.Export, false)
}

// setPipe makes a wanted node mirrored (opening: a subscription to its exact channel on its
// own goroutine) or not-exportable. Idempotent. The pipe resumes from the last seq the link
// mirrored of it only when the caller proved that seq belongs to the node's current
// incarnation (reconcilePipes, from the ctl replay); a node just created starts live, and a
// live start forgets the seq, so a later resume can never reach back past it.
func (l *peerLink) setPipe(ctx context.Context, c *Client, merged chan<- peerFrame, fail chan<- error, node string, export, resume bool) {
	if !l.wanted(node) {
		return
	}
	if !export {
		l.dropPipe(ctx, c, node, pipeNotExportable)
		return
	}
	ch := "probe/" + node
	l.mu.Lock()
	p := l.pipes[node]
	if p.state == pipeMirrored || p.state == pipeOpening {
		l.mu.Unlock()
		return
	}
	p.gen++
	p.state, p.sub = pipeOpening, 0
	gen := p.gen
	var since *uint64
	epoch := ""
	if s, ok := l.since[ch]; ok && resume {
		since, epoch = &s, l.epoch
	} else {
		delete(l.since, ch)
	}
	l.mu.Unlock()
	go func() {
		sctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		s, err := c.SubscribeExact(sctx, ch, since, epoch, 1024)
		cancel()
		l.mu.Lock()
		stale := p.gen != gen
		if err == nil && !stale {
			p.state, p.sub = pipeMirrored, s.ID
		} else if !stale {
			p.state = pipeAbsent
		}
		l.mu.Unlock()
		if err != nil {
			l.failWith(fail, err)
			return
		}
		if stale { // the pipe moved on while we subscribed: not ours any more
			uctx, cancel := context.WithTimeout(ctx, 5*time.Second)
			c.Unsubscribe(uctx, s.ID)
			cancel()
			return
		}
		for {
			select {
			case f, ok := <-s.Frames:
				if !ok {
					return
				}
				select {
				case merged <- peerFrame{ch, f, s.ID}:
				case <-s.Done:
					return
				case <-c.Done():
					return
				}
			case <-s.Done: // dropped: Unsubscribe closed it, and the connection goes on
				return
			}
		}
	}()
}

// dropPipe ends a wanted node's mirror, if any, forgets where it was (the seq belongs to the
// incarnation that just ended), and marks it `to`.
func (l *peerLink) dropPipe(ctx context.Context, c *Client, node, to string) {
	l.mu.Lock()
	p := l.pipes[node]
	was := *p
	p.gen++
	p.state, p.sub = to, 0
	delete(l.since, "probe/"+node)
	l.mu.Unlock()
	if was.state != pipeMirrored {
		return
	}
	go func() {
		uctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		c.Unsubscribe(uctx, was.sub) // best effort: stale frames are dropped by gen anyway; its Done ends the forwarder
	}()
}

type peerFrame struct {
	prefix string
	f      Frame
	sub    int64 // the subscription, for a probe pipe (0 for ctl and svc/)
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
