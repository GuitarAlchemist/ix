package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	peerBackoffMin = 200 * time.Millisecond
	peerBackoffMax = 5 * time.Second
	peerRefresh    = 100 * time.Millisecond // at most one status refresh per interval
	// maxLinkBacklog and maxLinkBacklogBytes bound what a session holds of ctl and svc/ while
	// its first replies are awaited, in frames and in bytes on the wire (frames vary in size,
	// so a count alone does not bound memory): a resumed replay is at most the ring (4096),
	// the rest is five seconds of lifecycle and service events; past either bound the session
	// gives up and reconnects.
	maxLinkBacklog      = 16384
	maxLinkBacklogBytes = 16 << 20
	// maxPipeQueue and maxPipeQueueBytes bound what a session holds of the wanted probe pipes,
	// in records and in bytes on the wire (criterion 6 of ix#408): past either, a pipe frame is
	// dropped rather than queued, so a parent slower than the offered load loses pipe records,
	// visibly, and never ctl or svc/, which have their own queue and go first.
	maxPipeQueue      = 1024
	maxPipeQueueBytes = 4 << 20
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
	boot     string              // the peer's epoch as its status last said it (the boot a live start reads)
	last     map[string]srcPoint // per pipe channel, the last frame mirrored, across sessions and peer boots: named by the next live start's marker
	pipes    map[string]*pipeSub // per wanted probe node, what it is now

	// pipeDelay is a test knob: nanoseconds the mirroring loop spends on each pipe frame, a
	// durably slow parent (zero in production).
	pipeDelay atomic.Int64
}

// srcPoint is one envelope of the peer, by its own seq and boot.
type srcPoint struct {
	seq   uint64
	epoch string
}

// linkGapKind is the continuity marker the link publishes in a mirrored channel, as its own
// envelope (the parent's origin, no provenance), when the mirror skipped something it cannot
// replay (S2 slice 3, D4 ix#402 Q6; contract G8). On a pipe, every live start: `live-start`
// with the boot now read (`src_epoch`) and the last frame mirrored before (`last_src_seq`,
// `last_src_epoch`, absent when none). On the peer's mirrored ctl, a gap the peer answered on
// the link's resumed ctl or svc/ subscription (`prefix`): its reason (`since-too-old`: history
// expired; `epoch-changed`: the peer restarted), `src_epoch`, `from`, `oldest` as the peer
// said; or, without `prefix`, `epoch-changed` the link saw itself when the peer came back as a
// new instance (`src_epoch` the new boot, `last_src_epoch` the one mirrored before). On a pipe
// again (criterion 6): `dropped`, before the record that followed a run of pipe frames the link
// itself discarded for want of room in its bounded pipe queue, with their number (`dropped`: the
// link's own exact count, never an inference from the source's seqs); and `overflow` when the
// peer dropped the pipe's subscription (`from`, `src_epoch` as it said), the pipe then reopened
// live and marked as at any live start. A boundary, not a count, except for what the link itself
// discarded: nothing says how many records a skip on the source covered.
const linkGapKind = "link.gap"

// pipeSub is one wanted probe node's standing on the link. gen counts its changes: a subscribe
// or unsubscribe that completes after the pipe moved on finds another gen and stands down, and
// a frame from a subscription that is no longer the pipe's is dropped.
type pipeSub struct {
	state   string // pipeAbsent | pipeNotExportable | pipeOpening | pipeMirrored
	sub     int64  // the subscription while mirrored
	gen     uint64
	dropped uint64 // frames the link discarded for want of room, since the link was made
}

const (
	pipeAbsent        = "absent"
	pipeNotExportable = "not-exportable"
	pipeOpening       = "opening" // subscribing to the peer's pipe
	pipeMirrored      = "mirrored"
)

func newPeerLink(spec PeerSpec, hub *Hub) *peerLink {
	l := &peerLink{spec: spec, hub: hub, since: map[string]uint64{}, last: map[string]srcPoint{}, pipes: map[string]*pipeSub{}}
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
		st.Probes = append(st.Probes, PeerProbeStatus{Name: n, State: l.pipes[n].state, Dropped: l.pipes[n].dropped})
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

	// Two queues by class (criterion 6 of ix#408). merged holds ctl and svc/: lossless, their
	// forwarders wait for room (the peer's own overflow of such a subscription then ends the
	// session, which resumes them). The pipe queue, made below once the first replies are in,
	// holds the wanted probe pipes, bounded in records and bytes: a pipe forwarder never waits,
	// so a load of pipes can neither hold the reader back nor grow memory; what does not fit is
	// dropped, counted and marked. The loop reads merged first.
	merged := make(chan peerFrame, 1024)
	for _, prefix := range []string{"ctl", "svc/"} {
		_, frames, err := c.SubscribeFrom(ctx, prefix, since[prefix], epoch, 1024)
		if err != nil {
			return false, err
		}
		go func() {
			for f := range frames {
				select {
				case merged <- peerFrame{prefix: prefix, f: f}:
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
	var held int
	var list json.RawMessage
	for n := 0; n < len(ops); {
		select {
		case pf := <-merged:
			if held += pf.f.Bytes; len(backlog) == maxLinkBacklog || held > maxLinkBacklogBytes {
				return true, fmt.Errorf("link backlog over %d frames or %d bytes while the peer's first replies were awaited", maxLinkBacklog, maxLinkBacklogBytes)
			}
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
	// The peer's boot, when the status could not say it: the first queued frame's (and failing
	// both, the first frame mirrored says it, in mirror). A new boot is marked by noteBoot before
	// anything of it is mirrored.
	if len(backlog) > 0 && backlog[0].f.Env != nil {
		l.noteBoot(backlog[0].f.Env.Epoch)
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
	q := newPipeQueue()
	if err := l.reconcilePipes(ctx, c, q, fail, list); err != nil {
		return true, err
	}
	for _, pf := range backlog {
		if err := l.apply(ctx, c, q, fail, refresh, st.Name, pf); err != nil {
			return true, err
		}
	}

	for {
		// Control first: a ctl or svc/ frame is applied while there is one, a pipe frame only
		// when there is none, so a load of pipes never holds the lifecycle back (and a
		// probe.removed reaches the pipe it closes before that pipe's frames pile up further).
		select {
		case pf := <-merged:
			if err := l.apply(ctx, c, q, fail, refresh, st.Name, pf); err != nil {
				return true, err
			}
			continue
		case err := <-fail:
			return true, err
		case <-c.Done():
			return true, c.Err()
		default:
		}
		select {
		case pf := <-merged:
			if err := l.apply(ctx, c, q, fail, refresh, st.Name, pf); err != nil {
				return true, err
			}
		case pf := <-q.ch:
			q.took(pf)
			if err := l.apply(ctx, c, q, fail, refresh, st.Name, pf); err != nil {
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
// wanted pipes, and kicks the status refresher. An overflow of ctl or svc/ ends the session:
// the peer dropped that subscription, and a reconnect resumes both from the last mirrored seq;
// an overflow of a pipe reopens that pipe alone, live and marked (reopenPipe). The other gap
// reasons answer the since the link gave for ctl / svc/ at subscribe time (the peer's history
// expired, or it restarted): what they skip cannot be replayed, so they are relayed on the
// peer's mirrored ctl as a continuity marker. A pipe frame that follows a run of frames its
// forwarder dropped is preceded by the `dropped` marker, on the pipe, in order.
func (l *peerLink) apply(ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, refresh chan<- struct{}, origin string, pf peerFrame) error {
	if pf.f.Gap != nil && pf.f.Gap.Reason == "overflow" {
		if pf.sub == 0 {
			return errors.New("mirror of " + pf.prefix + " overflowed")
		}
		l.reopenPipe(ctx, c, q, fail, pf)
		return nil
	}
	if pf.f.Gap != nil {
		l.markGap(pf.prefix, pf.f.Gap)
		return nil
	}
	if pf.f.Env == nil {
		return nil
	}
	if pf.sub != 0 && !l.current(pf.prefix, pf.sub) {
		return nil // a frame of a subscription that is no longer the pipe's
	}
	if pf.dropped > 0 {
		l.markDropped(pf.prefix, pf.dropped)
	}
	if d := l.pipeDelay.Load(); d > 0 && pf.sub != 0 {
		time.Sleep(time.Duration(d)) // the test knob: a durably slow parent
	}
	if err := l.mirror(origin, pf.prefix, pf.f.Env); err != nil {
		return err
	}
	if pf.prefix == "ctl" && pf.f.Env.Origin == origin {
		l.followPipe(ctx, c, q, fail, pf.f.Env)
	}
	select {
	case refresh <- struct{}{}:
	default:
	}
	return nil
}

// markGap relays a gap the peer answered on prefix as a continuity marker on the peer's
// mirrored ctl (a pipe is never resumed, so only ctl and svc/ can be answered one).
func (l *peerLink) markGap(prefix string, g *Gap) {
	data := map[string]any{"peer": l.spec.Name, "prefix": prefix, "reason": g.Reason, "src_epoch": g.Epoch}
	if g.From != 0 {
		data["from"] = g.From
	}
	if g.Oldest != 0 {
		data["oldest"] = g.Oldest
	}
	l.hub.Publish("peer/"+l.spec.Name+"/ctl", event(linkGapKind, data))
}

// markDropped publishes, on a mirrored pipe, the marker for a run of n frames the link itself
// discarded for want of room, right before the record that followed them: the count is the
// link's own, exact.
func (l *peerLink) markDropped(ch string, n uint64) {
	l.mu.Lock()
	data := map[string]any{"peer": l.spec.Name, "reason": "dropped", "dropped": n}
	if l.boot != "" {
		data["src_epoch"] = l.boot
	}
	l.mu.Unlock()
	l.hub.Publish("peer/"+l.spec.Name+"/"+ch, event(linkGapKind, data))
}

// countDrop counts one pipe frame discarded for want of room, on the node's status.
func (l *peerLink) countDrop(node string) {
	l.mu.Lock()
	if p := l.pipes[node]; p != nil {
		p.dropped++
	}
	l.mu.Unlock()
}

// reopenPipe answers the peer dropping a pipe's subscription (its queue toward this link
// overflowed, the gap its last frame): if the subscription is still the pipe's, the pipe is
// marked `overflow` on its mirrored channel with what the peer said, the client forgets the
// subscription (the peer already did), and the pipe is opened live again, marked at that live
// start as any. The session goes on: ctl and svc/ were not dropped.
func (l *peerLink) reopenPipe(ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, pf peerFrame) {
	node := strings.TrimPrefix(pf.prefix, "probe/")
	l.mu.Lock()
	p := l.pipes[node]
	if p == nil || p.state != pipeMirrored || p.sub != pf.sub {
		l.mu.Unlock()
		return // not the pipe's subscription any more: nothing to reopen
	}
	p.gen++
	p.state, p.sub = pipeAbsent, 0
	data := map[string]any{"peer": l.spec.Name, "reason": "overflow", "src_epoch": pf.f.Gap.Epoch}
	if pf.f.Gap.From != 0 {
		data["from"] = pf.f.Gap.From
	}
	l.hub.Publish("peer/"+l.spec.Name+"/"+pf.prefix, event(linkGapKind, data)) // under l.mu, one step with the state
	l.mu.Unlock()
	c.forget(pf.sub)
	l.setPipe(ctx, c, q, fail, node, true)
}

// noteBoot records the peer's boot as last seen and, when it is not the boot the link knew,
// says so on the peer's mirrored ctl: the link resumes nothing from a new instance, so the peer
// answers no gap for a restart. Whichever names the boot first in a session marks the boundary
// once: the status, the first queued frame, or the first frame mirrored (Codex round 40: the
// status alone left a restart behind a failed status unmarked).
func (l *peerLink) noteBoot(epoch string) {
	if epoch == "" {
		return
	}
	l.mu.Lock()
	prev := l.boot
	l.boot = epoch
	l.mu.Unlock()
	if prev != "" && prev != epoch {
		l.hub.Publish("peer/"+l.spec.Name+"/ctl", event(linkGapKind, map[string]any{
			"peer": l.spec.Name, "reason": "epoch-changed", "src_epoch": epoch, "last_src_epoch": prev,
		}))
	}
}

// markLiveStartLocked publishes the marker that opens a pipe's live start, before its first
// mirrored frame: the boot it reads (absent while the link has not seen it), and the last frame
// the link mirrored on it before, if any. Called with l.mu held, in the same critical section
// that makes the pipe mirrored, so a subscription the pipe moved past never marks (Codex round
// 40: marking outside the lock let a superseded subscription publish a late marker after the
// current one's samples).
func (l *peerLink) markLiveStartLocked(ch string) {
	data := map[string]any{"peer": l.spec.Name, "reason": "live-start"}
	if l.boot != "" {
		data["src_epoch"] = l.boot
	}
	if last, had := l.last[ch]; had {
		data["last_src_seq"], data["last_src_epoch"] = last.seq, last.epoch
	}
	l.hub.Publish("peer/"+l.spec.Name+"/"+ch, event(linkGapKind, data))
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
// round 36), and mirrors every wanted node it exports. Every pipe starts live: a resume from
// the last mirrored seq would need proof that the name is still the incarnation the link
// mirrored, and the ctl replay cannot give it (the list's reply may overtake the replay, so
// an absent lifecycle event proves nothing, Codex round 38); the cursor-bound list of slice 4
// is that proof, slice 2 the created seq, slice 3 the bounded replay and the continuity
// marker for what a live start skips (contract G8).
func (l *peerLink) reconcilePipes(ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, data json.RawMessage) error {
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
	for _, n := range list.Nodes {
		l.setPipe(ctx, c, q, fail, n.Name, n.Export)
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
func (l *peerLink) followPipe(ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, env *Envelope) {
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
	l.setPipe(ctx, c, q, fail, d.Node, d.Export)
}

// setPipe makes a wanted node mirrored (opening: a subscription to its exact channel on its
// own goroutine, live, never from a seq: see reconcilePipes) or not-exportable. Idempotent.
// A live start is marked on the mirrored channel in the step that makes the pipe `mirrored`,
// before its first frame crosses, so a reader that waited for the state sees no marker, one
// that was on the channel sees the marker first, and a superseded subscription marks nothing.
func (l *peerLink) setPipe(ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, node string, export bool) {
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
	l.mu.Unlock()
	go func() {
		sctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		s, err := c.SubscribeExact(sctx, ch, nil, "", 1024)
		cancel()
		l.mu.Lock()
		stale := p.gen != gen
		switch {
		case stale:
		case err != nil:
			p.state = pipeAbsent
		default:
			l.markLiveStartLocked(ch) // under l.mu: the marker and the state change are one step
			p.state, p.sub = pipeMirrored, s.ID
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
		// The forwarder never waits for room in the pipe queue (the client's reader would wait
		// behind it, ctl and the replies with it): a frame that does not fit is dropped and
		// counted, and the next that fits carries the count, for the loop to mark before it.
		var dropped uint64
		for {
			select {
			case f, ok := <-s.Frames:
				if !ok {
					return
				}
				if f.Gap != nil && f.Gap.Reason == "overflow" {
					// The peer dropped this subscription: this is its last frame, queued whatever
					// the room (nothing follows it here) for the loop to reopen the pipe from.
					pf := peerFrame{prefix: ch, f: f, sub: s.ID}
					q.bytes.Add(int64(f.Bytes))
					select {
					case q.ch <- pf:
					case <-s.Done:
						q.bytes.Add(-int64(f.Bytes))
					case <-c.Done():
						q.bytes.Add(-int64(f.Bytes))
					}
					return
				}
				if q.offer(peerFrame{prefix: ch, f: f, sub: s.ID, dropped: dropped}) {
					dropped = 0
				} else {
					dropped++
					l.countDrop(node)
				}
			case <-s.Done: // dropped: Unsubscribe closed it, and the connection goes on
				return
			}
		}
	}()
}

// dropPipe ends a wanted node's mirror, if any, and marks it `to`.
func (l *peerLink) dropPipe(ctx context.Context, c *Client, node, to string) {
	l.mu.Lock()
	p := l.pipes[node]
	was := *p
	p.gen++
	p.state, p.sub = to, 0
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
	prefix  string
	f       Frame
	sub     int64  // the subscription, for a probe pipe (0 for ctl and svc/)
	dropped uint64 // pipe frames of this subscription the link discarded right before this one
}

// pipeQueue is what a session holds of the wanted probe pipes between the link's reader and
// the loop that mirrors, bounded in records (the channel) and in bytes on the wire (criterion
// 6 of ix#408). A pipe forwarder never waits for room: what does not fit is dropped, counted
// on the pipe and marked before the next frame that fits.
type pipeQueue struct {
	ch    chan peerFrame
	bytes atomic.Int64
}

func newPipeQueue() *pipeQueue { return &pipeQueue{ch: make(chan peerFrame, maxPipeQueue)} }

// offer queues pf if it fits both bounds, at once or not at all. The bytes are taken before the
// record is queued and given back when it is not, so the count never understates what the queue
// holds.
func (q *pipeQueue) offer(pf peerFrame) bool {
	n := int64(pf.f.Bytes)
	if q.bytes.Add(n) > maxPipeQueueBytes {
		q.bytes.Add(-n)
		return false
	}
	select {
	case q.ch <- pf:
		return true
	default:
		q.bytes.Add(-n)
		return false
	}
}

// took gives back the bytes of a frame read from the queue.
func (q *pipeQueue) took(pf peerFrame) { q.bytes.Add(-int64(pf.f.Bytes)) }

// mirror republishes one of the peer's own envelopes. Envelopes the peer
// itself mirrored (another origin, or a peer/ channel) are never forwarded.
// A frame without a seq or an epoch is malformed (the hub assigns seqs from 1,
// every boot has an epoch): it has no provenance to mirror and ends the session.
func (l *peerLink) mirror(origin, prefix string, env *Envelope) error {
	if env.Seq == 0 || env.Epoch == "" {
		return fmt.Errorf("frame on %s without provenance (seq %d, epoch %q)", prefix, env.Seq, env.Epoch)
	}
	l.noteBoot(env.Epoch) // a boot the status and the backlog did not name is marked here, first
	l.mu.Lock()
	l.since[prefix], l.epoch = env.Seq, env.Epoch
	if strings.HasPrefix(prefix, "probe/") {
		l.last[prefix] = srcPoint{env.Seq, env.Epoch}
	}
	l.mu.Unlock()
	if env.Origin != origin || strings.HasPrefix(env.Ch, "peer/") {
		return nil
	}
	l.hub.Mirror("peer/"+l.spec.Name+"/"+env.Ch, env.Origin, env.T, env.Ev, env.Seq, env.Epoch)
	return nil
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
		Epoch    string          `json:"epoch"`
	}
	if json.Unmarshal(status, &st) == nil {
		l.mu.Lock()
		l.services = st.Services
		l.mu.Unlock()
		l.noteBoot(st.Epoch)
	}
}
