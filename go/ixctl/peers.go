package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
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
	// replayAllowFrames and replayAllowBytes bound what the link replays of the peer's ctl and svc/
	// per link (criterion 7 of ix#408): one allowance across resubscribes and reconnects, full when
	// the link is made, drained by every frame of a catch-up, refilled to full over replayRefill,
	// never reset by a reconnect; replayFor bounds one catch-up in time. Past any of them the link
	// stops catching up and subscribes live, marked `jumped`: one bounded replay, then the jump,
	// never a replay without end at a parent that cannot keep up.
	replayAllowFrames = 2 * ringSize
	replayAllowBytes  = 32 << 20
	replayRefill      = time.Minute
	replayFor         = 30 * time.Second
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
	boot     string              // the peer's boot as read in this session ("" until something named it: a marker then omits src_epoch)
	lastBoot string              // the last boot ever named, across sessions and instances: the one an epoch-changed marker leaves
	sess     uint64              // the session now running; an observation made by an older one is ignored
	last     map[string]srcPoint // per pipe channel, the last frame mirrored, across sessions and peer boots: named by the next live start's marker
	pipes    map[string]*pipeSub // per wanted probe node, what it is now
	ctl      map[string]*ctlSub  // per control prefix (ctl, svc/): its subscription now, and whether it is live
	listed   bool                // the peer's probe list was applied this session (false while down)
	unlisted uint64              // times a jump on ctl cleared listed: a list asked for before the last one cannot say reconciled (snapshot)
	cursor   srcPoint            // the point of the peer's stream the last snapshot stood at: a lifecycle event at or before it is already reflected
	jumps    uint64              // catch-ups the link gave up on since it was made, each marked `jumped`

	// replay is what is left of the allowance (frames and bytes) and when it was last refilled;
	// allowFrames and allowBytes its capacity: replayAllowFrames / replayAllowBytes, set lower by
	// tests.
	replay struct {
		frames, bytes float64
		at            time.Time
	}
	allowFrames, allowBytes atomic.Int64

	// pipeDelay and ctlDelay are test knobs: nanoseconds the mirroring loop spends on each pipe
	// frame, or on each control frame, a durably slow parent (zero in production).
	pipeDelay, ctlDelay atomic.Int64
	// held counts the control frames resnapshot held behind a re-list since the link was made,
	// those taken before its request was sent and those its select took while the reply was
	// awaited: a test observation. A frame queued in merged when the reply wins the select is
	// applied by the loop after the list, in order, without passing through the count.
	held atomic.Int64
}

// unlistedGen is the generation a list is asked for at: snapshot says reconciled only if no jump on
// ctl cleared listed since (Codex round 45: an older list's reply restored it after a newer jump).
func (l *peerLink) unlistedGen() uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.unlisted
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
// live and marked as at any live start. On the peer's mirrored ctl again, with `prefix`
// (criterion 7): `jumped`, when the link gave a catch-up of ctl or svc/ up (its replay allowance
// spent, or the catch-up too long) and subscribed live instead: `from` the last seq it mirrored
// on that prefix (absent when none), `to` the peer's head the live subscription starts from,
// `src_epoch` the boot read; the envelope's `t` dates it. A boundary, not a count, except for
// what the link itself discarded: nothing says how many records a skip on the source covered.
const linkGapKind = "link.gap"

// pipeSub is one wanted probe node's standing on the link. gen counts its changes: a subscribe
// or unsubscribe that completes after the pipe moved on finds another gen and stands down, and
// a frame from a subscription that is no longer the pipe's is dropped.
type pipeSub struct {
	state   string // pipeAbsent | pipeNotExportable | pipeOpening | pipeMirrored
	sub     int64  // the subscription while mirrored
	gen     uint64
	dropped uint64   // frames the link discarded for want of room, since the link was made
	created srcPoint // the incarnation under the name: the seq and boot of its probe.created (zero while absent)
}

// ctlSub is one control prefix's standing on the link (the peer's ctl or svc/): the subscription
// now, the head it started from, whether it is past its replay (live: the peer said so, or it
// started live), and when its catch-up began. gen counts its changes as pipeSub's does: a
// subscription that completes after the prefix moved on stands down, and a frame of one that is
// no longer the prefix's is dropped.
type ctlSub struct {
	sub   int64
	gen   uint64
	live  bool
	since time.Time
}

// isControl says whether prefix is one of the peer's control prefixes, mirrored losslessly by
// the link (as opposed to a probe pipe).
func isControl(prefix string) bool { return prefix == "ctl" || prefix == "svc/" }

const (
	pipeAbsent        = "absent"
	pipeNotExportable = "not-exportable"
	pipeOpening       = "opening" // subscribing to the peer's pipe
	pipeMirrored      = "mirrored"
)

func newPeerLink(spec PeerSpec, hub *Hub) *peerLink {
	l := &peerLink{spec: spec, hub: hub, since: map[string]uint64{}, last: map[string]srcPoint{}, pipes: map[string]*pipeSub{},
		ctl: map[string]*ctlSub{"ctl": {}, "svc/": {}}}
	for _, n := range spec.Probes {
		l.pipes[n] = &pipeSub{state: pipeAbsent}
	}
	l.allowFrames.Store(replayAllowFrames)
	l.allowBytes.Store(replayAllowBytes)
	l.replay.frames, l.replay.bytes, l.replay.at = replayAllowFrames, replayAllowBytes, time.Now()
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
	st.Live = l.up && l.ctl["ctl"].live && l.ctl["svc/"].live
	st.Reconciled = l.up && l.listed
	for _, n := range l.spec.Probes {
		if l.pipes[n].state == pipeOpening {
			st.Reconciled = false
		}
		p := l.pipes[n]
		st.Probes = append(st.Probes, PeerProbeStatus{Name: n, State: p.state, Dropped: p.dropped, CreatedSeq: p.created.seq, SrcEpoch: p.created.epoch})
	}
	st.Degraded, st.Jumps = !(st.Live && st.Reconciled), l.jumps
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

	since, epoch, sess := l.beginSession(st.Token)

	// Two queues by class (criterion 6 of ix#408). merged holds ctl and svc/: lossless, their
	// forwarders wait for room (the peer's own overflow of such a subscription reopens it in this
	// session, resumed within the replay allowance or live and marked: openControl). The pipe
	// queue, made below once the first replies are in, holds the wanted probe pipes, bounded in
	// records and bytes: a pipe forwarder never waits, so a load of pipes can neither hold the
	// reader back nor grow memory; what does not fit is dropped, counted and marked. The loop
	// reads merged first.
	merged := make(chan peerFrame, 1024)
	for _, prefix := range []string{"ctl", "svc/"} {
		if err := l.openControl(ctx, c, nil, merged, nil, prefix, since[prefix], epoch, l.nextControlGen(prefix), false); err != nil {
			return false, err
		}
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
	asked := l.unlistedGen() // the control subscriptions stand (a jump at start cleared listed already): the list is asked for now
	for _, op := range ops {
		go func(op string) {
			rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			data, err := c.Do(rctx, Request{Op: op, WithCursor: op == "probe.list"})
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
				l.setServices(sess, r.data)
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
		l.noteBoot(sess, backlog[0].f.Env.Epoch)
	}
	refresh := make(chan struct{}, 1)
	go l.refresher(c, sess, refresh)

	l.mu.Lock()
	l.client, l.up = c, true
	l.mu.Unlock()
	l.hub.Publish("ctl", event("peer.up", map[string]any{"peer": l.spec.Name, "addr": st.Addr, "origin": st.Name}))
	defer func() {
		l.mu.Lock()
		l.client, l.up, l.listed, l.cursor = nil, false, false, srcPoint{}
		for _, p := range l.pipes {
			p.gen++
			p.state, p.sub, p.created = pipeAbsent, 0, srcPoint{}
		}
		for _, cs := range l.ctl {
			cs.gen++
			cs.sub, cs.live = 0, false
		}
		l.mu.Unlock()
	}()

	// The wanted probe pipes: the list, bound to a cursor of the peer's stream, says which exist
	// and are exportable at that point and which incarnation each is (its created_seq); the ctl
	// mirror, subscribed above and so live before the list, says every change after it, applied
	// in arrival order, while a lifecycle event at or before the cursor is already reflected and
	// not followed (slice 4 of S2). One lost to a jump on ctl is not seen: this loop reads the list
	// again after such a jump (resnapshot); the list is read again on every reconnect. Subscribing and
	// unsubscribing run on their own goroutines: this loop is the one consumer of merged, and a
	// reply the peer sends can only reach us while the client's reader is not held back by a
	// full subscription channel, so the loop never waits on one. Once the list is applied the
	// state is reconciled, but for the pipes it set opening (status).
	fail := make(chan error, 1)
	q := newPipeQueue()
	relist := make(chan struct{}, 1) // a jump on ctl asks this loop to read the list again (resnapshot)
	if err := l.snapshot(ctx, c, q, fail, asked, list); err != nil {
		return true, err
	}
	for _, pf := range backlog {
		if err := l.apply(ctx, c, q, merged, fail, refresh, relist, sess, st.Name, pf); err != nil {
			return true, err
		}
	}

	for {
		// Control first: a ctl or svc/ frame is applied while there is one, a pipe frame only
		// when there is none, so a load of pipes never holds the lifecycle
		// back (and a probe.removed reaches the pipe it closes before that pipe's frames pile up
		// further). Preferential, not strict: a control frame that arrives while the second
		// select is pending may wait one pipe frame's apply, never a queue of them; and a
		// sustained control stream is applied before any pipe frame, by design.
		select {
		case pf := <-merged:
			if err := l.apply(ctx, c, q, merged, fail, refresh, relist, sess, st.Name, pf); err != nil {
				return true, err
			}
			continue
		case <-relist:
			if err := l.resnapshot(ctx, c, q, merged, fail, refresh, relist, sess, st.Name); err != nil {
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
			if err := l.apply(ctx, c, q, merged, fail, refresh, relist, sess, st.Name, pf); err != nil {
				return true, err
			}
		case <-relist:
			if err := l.resnapshot(ctx, c, q, merged, fail, refresh, relist, sess, st.Name); err != nil {
				return true, err
			}
		case pf := <-q.ch:
			q.took(pf)
			if err := l.apply(ctx, c, q, merged, fail, refresh, relist, sess, st.Name, pf); err != nil {
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
// wanted pipes, and kicks the status refresher. An overflow (the peer dropped the subscription
// toward this link) reopens that subscription alone, in the session: a pipe live and marked
// (reopenPipe), ctl or svc/ resumed within the replay allowance or live and marked `jumped`
// (reopenControl); one that names no subscription is not ours and ends the session. The other
// gap reasons answer the since the link gave for ctl / svc/ at subscribe time (the peer's
// history expired, or it restarted): what they skip cannot be replayed, so they are relayed on
// the peer's mirrored ctl as a continuity marker. The live frame says a control subscription is
// past its replay. A control frame of a catch-up is charged to the allowance, and the catch-up
// is given up (the subscription replaced by a live one, marked) when it runs out or lasts past
// replayFor. A pipe frame that follows a run of frames its forwarder dropped is preceded by the
// `dropped` marker, on the pipe, in order.
func (l *peerLink) apply(ctx context.Context, c *Client, q *pipeQueue, merged chan<- peerFrame, fail chan<- error, refresh chan<- struct{}, relist chan<- struct{}, sess uint64, origin string, pf peerFrame) error {
	if pf.f.Gap != nil && pf.f.Gap.Reason == "overflow" {
		switch {
		case pf.sub == 0:
			return errors.New("mirror of " + pf.prefix + " overflowed")
		case isControl(pf.prefix):
			l.reopenControl(ctx, c, relist, merged, fail, pf)
		default:
			l.reopenPipe(ctx, c, q, fail, pf)
		}
		return nil
	}
	if pf.f.Gap != nil {
		l.markGap(pf.prefix, pf.f.Gap)
		return nil
	}
	if pf.f.Live {
		l.setLive(pf.prefix, pf.sub)
		return nil
	}
	if pf.f.Env == nil {
		return nil
	}
	if pf.sub != 0 && !l.current(pf.prefix, pf.sub) {
		return nil // a frame of a subscription that is no longer the prefix's or the pipe's
	}
	if pf.dropped > 0 {
		l.markDropped(pf.prefix, pf.dropped)
	}
	if isControl(pf.prefix) {
		if d := l.ctlDelay.Load(); d > 0 {
			time.Sleep(time.Duration(d)) // the test knob: a durably slow parent
		}
		if !l.grantReplay(pf.prefix, pf.sub, pf.f.Bytes) {
			l.jumpControl(ctx, c, relist, merged, fail, pf.prefix, pf.sub)
			return nil // not mirrored: the jump is marked from the last frame that was
		}
	} else if d := l.pipeDelay.Load(); d > 0 && pf.sub != 0 {
		time.Sleep(time.Duration(d)) // the test knob: a durably slow parent
	}
	if err := l.mirror(sess, origin, pf.prefix, pf.f.Env); err != nil {
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
	created := p.created // the same incarnation, reopened
	data := map[string]any{"peer": l.spec.Name, "reason": "overflow", "src_epoch": pf.f.Gap.Epoch}
	if pf.f.Gap.From != 0 {
		data["from"] = pf.f.Gap.From
	}
	l.hub.Publish("peer/"+l.spec.Name+"/"+pf.prefix, event(linkGapKind, data)) // under l.mu, one step with the state
	l.mu.Unlock()
	c.forget(pf.sub)
	l.setPipe(ctx, c, q, fail, node, true, created)
}

// noteBoot records the peer's boot as now read and, when it is not the last boot ever named,
// says so on the peer's mirrored ctl: the link resumes nothing from a new instance, so the peer
// answers no gap for a restart. Whichever names the boot first in a session marks the boundary
// once: the status, the first queued frame, or the first frame mirrored (Codex round 40: the
// status alone left a restart behind a failed status unmarked). An observation of an older
// session is ignored (Codex round 41).
func (l *peerLink) noteBoot(sess uint64, epoch string) {
	if epoch == "" {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if sess < l.sess {
		return // an older session's observation (its refresher answering late): not the boot now read
	}
	prev := l.lastBoot
	l.boot, l.lastBoot = epoch, epoch
	if prev != "" && prev != epoch {
		// Under l.mu: two boundaries cannot cross, and no marker of this boot goes out before it.
		l.hub.Publish("peer/"+l.spec.Name+"/ctl", event(linkGapKind, map[string]any{
			"peer": l.spec.Name, "reason": "epoch-changed", "src_epoch": epoch, "last_src_epoch": prev,
		}))
	}
}

// beginSession notes the instance a session connects to and says what its ctl and svc/
// subscriptions resume from. A new instance (another token) is a boot the link has not read
// yet: a marker published before anything of it was seen omits src_epoch rather than name the
// boot before (Codex round 41); the boundary is marked by whatever names the new boot first,
// from the last boot ever named.
func (l *peerLink) beginSession(token string) (since map[string]*uint64, epoch string, sess uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.sess++
	if token != l.token { // a new peer instance: its seqs start over, its boot is not read yet
		l.token, l.since, l.boot = token, map[string]uint64{}, ""
	}
	since = map[string]*uint64{}
	for _, p := range []string{"ctl", "svc/"} {
		if s, ok := l.since[p]; ok {
			since[p] = &s
		}
	}
	return since, l.epoch, l.sess
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

// current says whether sub is still the subscription of the control prefix or pipe channel ch.
func (l *peerLink) current(ch string, sub int64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if cs := l.ctl[ch]; cs != nil {
		return cs.sub == sub
	}
	p := l.pipes[strings.TrimPrefix(ch, "probe/")]
	return p != nil && p.state == pipeMirrored && p.sub == sub
}

// nextControlGen moves a control prefix on: whatever subscription was its stands down, and the
// gen returned is the one an openControl in flight must still find to become the prefix's.
func (l *peerLink) nextControlGen(prefix string) uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	cs := l.ctl[prefix]
	cs.gen++
	cs.sub, cs.live = 0, false
	return cs.gen
}

// openControl subscribes to one of the peer's control prefixes (ctl or svc/) and forwards its
// frames into merged, in order, on a goroutine of its own, until the peer drops the subscription
// (overflow: its gap is the last frame forwarded) or the link does. A resume (since given) is a
// catch-up, granted only while the allowance has a frame and a byte left; otherwise, as when
// jump asks for it, the subscription starts live and the jump is marked on the peer's mirrored
// ctl (`jumped`, `prefix`, `from` the last seq mirrored on the prefix, `to` the head the live
// subscription starts from, `src_epoch` the boot read), under l.mu, in the step that makes the
// subscription the prefix's, unless nothing lies between the two (the head is the last seq
// mirrored, in the same boot): a live start that misses nothing is no jump (Codex round 43). A
// jump of ctl loses the lifecycle events of its span, so the state is no longer reconciled and the
// session loop is asked to read the list again (relist, signalled once the live subscription
// stands, so the list's cursor is at or past the head it starts from and nothing falls between
// the two: resnapshot; nil at session start, whose own list follows). A subscription that
// completes after the prefix moved on (gen) stands down.
func (l *peerLink) openControl(ctx context.Context, c *Client, relist chan<- struct{}, merged chan<- peerFrame, fail chan<- error, prefix string, since *uint64, epoch string, gen uint64, jump bool) error {
	if since != nil && !l.canReplay() {
		since, jump = nil, true
	}
	sctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	s, err := c.SubscribeFromLive(sctx, prefix, since, epoch, 1024)
	cancel()
	if err != nil {
		return err
	}
	l.mu.Lock()
	cs := l.ctl[prefix]
	if cs.gen != gen {
		l.mu.Unlock()
		uctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		c.Unsubscribe(uctx, s.ID)
		cancel()
		return nil
	}
	cs.sub, cs.live, cs.since = s.ID, since == nil, time.Now()
	if from, had := l.since[prefix]; jump && had && from == s.Head && s.Epoch == l.epoch {
		jump = false // the live subscription starts where the mirror stopped: nothing skipped
	}
	resnap := false
	if jump {
		l.jumps++
		if prefix == "ctl" {
			// A lifecycle event in the jumped span is lost, so the pipes may no longer stand for
			// the peer's tree: not reconciled until a new snapshot is applied (jumpControl said so
			// already when the jump is its; a reopen past the allowance says it here).
			l.listed, resnap = false, relist != nil
			l.unlisted++
		}
		data := map[string]any{"peer": l.spec.Name, "prefix": prefix, "reason": "jumped", "to": s.Head}
		if from, ok := l.since[prefix]; ok {
			data["from"] = from
		}
		if l.boot != "" {
			data["src_epoch"] = l.boot
		}
		l.hub.Publish("peer/"+l.spec.Name+"/ctl", event(linkGapKind, data)) // under l.mu, one step with the subscription
	}
	l.mu.Unlock()
	if resnap {
		select { // one pending signal covers every jump before the list it brings is read
		case relist <- struct{}{}:
		default:
		}
	}
	go func() {
		for {
			select {
			case f, ok := <-s.Frames:
				if !ok {
					return
				}
				select {
				case merged <- peerFrame{prefix: prefix, f: f, sub: s.ID}:
				case <-s.Done:
					return
				case <-c.Done():
					return
				}
				if f.Gap != nil && f.Gap.Reason == "overflow" {
					return // the peer dropped the subscription: nothing follows; the loop reopens it
				}
			case <-s.Done: // the link dropped it (a jump): the connection goes on
				return
			}
		}
	}()
	return nil
}

// reopenControl answers the peer dropping a control subscription (its queue toward this link
// overflowed, the gap its last frame): if the subscription is still the prefix's, the client
// forgets it (the peer already did) and the prefix is opened again from the last seq mirrored,
// a catch-up within the allowance, or live and marked when none is left. The session goes on:
// the pipes and the other prefix were not dropped, and nothing reconnects.
func (l *peerLink) reopenControl(ctx context.Context, c *Client, relist chan<- struct{}, merged chan<- peerFrame, fail chan<- error, pf peerFrame) {
	l.mu.Lock()
	cs := l.ctl[pf.prefix]
	if cs.sub != pf.sub {
		l.mu.Unlock()
		return // not the prefix's subscription any more
	}
	cs.gen++
	cs.sub, cs.live = 0, false
	gen, epoch := cs.gen, l.epoch
	var since *uint64
	if s, ok := l.since[pf.prefix]; ok {
		since = &s
	}
	l.mu.Unlock()
	c.forget(pf.sub)
	go func() {
		if err := l.openControl(ctx, c, relist, merged, fail, pf.prefix, since, epoch, gen, false); err != nil {
			l.failWith(fail, err)
		}
	}()
}

// jumpControl gives a catch-up up: the subscription, if still the prefix's, stands down (its
// remaining replay is dropped as stale, and unsubscribed on a goroutine), and the prefix is
// opened live, marked `jumped` from the last seq mirrored. On ctl the frame given up is a
// lifecycle event the link may need, so the state is no longer reconciled from this step, not
// from the one that stands the live subscription up (Codex round 44: the status said reconciled
// for the seconds between the two).
func (l *peerLink) jumpControl(ctx context.Context, c *Client, relist chan<- struct{}, merged chan<- peerFrame, fail chan<- error, prefix string, sub int64) {
	l.mu.Lock()
	cs := l.ctl[prefix]
	if cs.sub != sub {
		l.mu.Unlock()
		return
	}
	cs.gen++
	cs.sub, cs.live = 0, false
	if prefix == "ctl" {
		l.listed = false
		l.unlisted++
	}
	gen := cs.gen
	l.mu.Unlock()
	go func() {
		uctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		c.Unsubscribe(uctx, sub) // best effort: its Done ends the forwarder, gen drops what it queued
		cancel()
		if err := l.openControl(ctx, c, relist, merged, fail, prefix, nil, "", gen, true); err != nil {
			l.failWith(fail, err)
		}
	}()
}

// resnapshot reads the peer's list again, bound to a cursor, and reconciles the pipes to it:
// after a jump on ctl, whose span may hold lifecycle events the link did not see. On the session
// loop, which asked for it (relist) once the live subscription stood, so the list's cursor is at
// or past the head that subscription starts from: nothing falls between the two. While the reply
// is awaited every control frame is held (bounded as at session start; past the bound the session
// ends), a pipe frame applied as usual; then the list is applied as a snapshot and the held frames
// in order (one at or before the cursor is skipped by followPipe). The loop owns it, so no frame is
// applied while the list is (Codex round 44: a probe.removed applied before an older list was
// undone by it), two jumps in a row read the list twice in order, and nothing of it outlives the
// session: a reply that comes once the loop is gone is dropped with the goroutine awaiting it. A
// list that cannot be read ends the session, as at start. A jump on ctl after the list was asked
// for (another reopen past the allowance while it was in flight) leaves the state unreconciled:
// the list cannot hold what that span lost, and the re-list it asked for follows (Codex round 45).
func (l *peerLink) resnapshot(ctx context.Context, c *Client, q *pipeQueue, merged chan peerFrame, fail chan error, refresh chan<- struct{}, relist chan struct{}, sess uint64, origin string) error {
	asked := l.unlistedGen()
	if len(l.spec.Probes) == 0 {
		return l.snapshot(ctx, c, q, fail, asked, nil) // nothing to reconcile but the state
	}
	var held []peerFrame
	var bytes int
	hold := func(pf peerFrame) error {
		if bytes += pf.f.Bytes; len(held) == maxLinkBacklog || bytes > maxLinkBacklogBytes {
			return fmt.Errorf("link backlog over %d frames or %d bytes while the peer's list was read again", maxLinkBacklog, maxLinkBacklogBytes)
		}
		held = append(held, pf)
		l.held.Add(1)
		return nil
	}
	// What merged holds now arrived before the list is asked for: taken first, so the reply never
	// overtakes it whatever the select below picks (Codex round 46).
	for taken := true; taken; {
		select {
		case pf := <-merged:
			if err := hold(pf); err != nil {
				return err
			}
		default:
			taken = false
		}
	}
	type reply struct {
		data json.RawMessage
		err  error
	}
	got := make(chan reply, 1)
	go func() {
		rctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		data, err := c.Do(rctx, Request{Op: "probe.list", WithCursor: true})
		got <- reply{data, err}
	}()
	for {
		select {
		case pf := <-merged:
			if err := hold(pf); err != nil {
				return err
			}
		case pf := <-q.ch:
			q.took(pf)
			if err := l.apply(ctx, c, q, merged, fail, refresh, relist, sess, origin, pf); err != nil {
				return err
			}
		case r := <-got:
			if r.err != nil {
				return r.err
			}
			if err := l.snapshot(ctx, c, q, fail, asked, r.data); err != nil {
				return err
			}
			for _, pf := range held {
				if err := l.apply(ctx, c, q, merged, fail, refresh, relist, sess, origin, pf); err != nil {
					return err
				}
			}
			return nil
		case err := <-fail:
			return err
		case <-c.Done():
			return c.Err()
		}
	}
}

// setLive notes the peer saying a control subscription is past its replay.
func (l *peerLink) setLive(prefix string, sub int64) {
	l.mu.Lock()
	if cs := l.ctl[prefix]; cs != nil && cs.sub == sub {
		cs.live = true
	}
	l.mu.Unlock()
}

// refillLocked tops the allowance up with the share of its capacity the time since the last
// refill earns (a full capacity per replayRefill), to the capacity at most. Called with l.mu held.
func (l *peerLink) refillLocked() {
	now := time.Now()
	capF, capB := float64(l.allowFrames.Load()), float64(l.allowBytes.Load())
	share := now.Sub(l.replay.at).Seconds() / replayRefill.Seconds()
	l.replay.frames = math.Min(capF, l.replay.frames+capF*share)
	l.replay.bytes = math.Min(capB, l.replay.bytes+capB*share)
	l.replay.at = now
}

// canReplay says whether a catch-up may begin: the allowance has a frame and a byte left (it ends
// at the first frame the allowance cannot cover, grantReplay).
func (l *peerLink) canReplay() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.refillLocked()
	return l.replay.frames >= 1 && l.replay.bytes >= 1
}

// grantReplay charges one frame of a control subscription's catch-up to the allowance and says
// whether the catch-up goes on: not when the allowance is out of frames or bytes, nor past
// replayFor since the catch-up began. A frame of a live subscription (or of none the link
// tracks) is not a catch-up and always goes on.
func (l *peerLink) grantReplay(prefix string, sub int64, bytes int) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	cs := l.ctl[prefix]
	if sub == 0 || cs == nil || cs.sub != sub || cs.live {
		return true
	}
	if time.Since(cs.since) > replayFor {
		return false
	}
	l.refillLocked()
	if l.replay.frames < 1 || l.replay.bytes < float64(bytes) {
		return false
	}
	l.replay.frames--
	l.replay.bytes -= float64(bytes)
	return true
}

// snapshot applies the peer's probe list as the snapshot it is: bound to a cursor of the peer's
// stream (every lifecycle event at or before it is reflected, none after: followPipe skips the
// former), each node with the identity of its incarnation (created_seq, the boot). Every wanted
// node it holds is set from it (mirrored while exported, declined otherwise; a name whose
// identity changed is reopened as the new incarnation), every wanted node it does not hold is
// dropped, and the state is reconciled (listed). A list that is no snapshot (no cursor, a wanted
// node without its created_seq) is refused before anything moves, and ends the session (Codex
// round 44: a zero identity stood for any). At session start it is read before any frame of the
// backlog is applied (so no lifecycle event already queued is undone by a list arriving later,
// Codex round 36); after a jump on ctl it is read again on the loop, the frames that arrive
// meanwhile held behind it (resnapshot). The state is reconciled only if no jump on ctl cleared it
// since the list was asked for (asked, from unlistedGen): such a jump lost lifecycle events past the
// list's cursor, and asked for a list of its own. Every pipe still
// starts live: with the identity, a resume from the last mirrored seq would be sound, and is
// left for later (what a live start skips is marked, slice 3).
func (l *peerLink) snapshot(ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, asked uint64, data json.RawMessage) error {
	var list struct {
		Cursor struct {
			Seq   uint64 `json:"seq"`
			Epoch string `json:"epoch"`
		} `json:"cursor"`
		Nodes []struct {
			Name       string `json:"name"`
			Export     bool   `json:"export"`
			CreatedSeq uint64 `json:"created_seq"`
		} `json:"nodes"`
	}
	if len(l.spec.Probes) > 0 {
		if err := json.Unmarshal(data, &list); err != nil {
			return err
		}
		if list.Cursor.Epoch == "" {
			return errors.New("the peer's probe list names no cursor: not a snapshot")
		}
		for _, n := range list.Nodes {
			if l.wanted(n.Name) && n.CreatedSeq == 0 {
				return fmt.Errorf("the peer's probe list names no created_seq for %s: not a snapshot", n.Name)
			}
		}
	}
	l.mu.Lock()
	l.cursor = srcPoint{list.Cursor.Seq, list.Cursor.Epoch}
	l.mu.Unlock()
	seen := map[string]bool{}
	for _, n := range list.Nodes {
		if !l.wanted(n.Name) {
			continue
		}
		seen[n.Name] = true
		l.setPipe(ctx, c, q, fail, n.Name, n.Export, srcPoint{n.CreatedSeq, list.Cursor.Epoch})
	}
	for _, n := range l.spec.Probes {
		if !seen[n] {
			l.dropPipe(ctx, c, n, pipeAbsent, srcPoint{})
		}
	}
	l.mu.Lock()
	l.listed = l.unlisted == asked // a jump on ctl since the list was asked for: not reconciled until its own list
	l.mu.Unlock()
	return nil
}

func (l *peerLink) failWith(fail chan<- error, err error) {
	select {
	case fail <- err:
	default:
	}
}

// followPipe applies one of the peer's own ctl events to the wanted pipes: a probe.created
// opens or declines the pipe by its export flag, with the event as the incarnation's identity,
// a probe.removed closes it. An event at or before the last snapshot's cursor is already
// reflected by it and not followed (it would reopen a pipe the list no longer has, or close one
// it has).
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
	l.mu.Lock()
	held := env.Epoch == l.cursor.epoch && env.Seq <= l.cursor.seq
	l.mu.Unlock()
	if held {
		return
	}
	if env.Ev.Kind == "probe.removed" {
		l.dropPipe(ctx, c, d.Node, pipeAbsent, srcPoint{})
		return
	}
	l.setPipe(ctx, c, q, fail, d.Node, d.Export, srcPoint{env.Seq, env.Epoch})
}

// setPipe makes a wanted node mirrored (opening: a subscription to its exact channel on its
// own goroutine, live, never from a seq: see snapshot) or not-exportable, as the incarnation
// `created` names (never zero: the list and the event both name one). Idempotent for the same
// incarnation; another incarnation under a name already
// mirrored or opening (a same-name recreation a new snapshot found, its removal lost to a jump)
// stands the pipe down first and opens it again as the new one, its live start marked as any
// (naming the last frame mirrored, the previous incarnation's). A live start is marked on the
// mirrored channel in the step that makes the pipe `mirrored`, before its first frame crosses,
// so a reader that waited for the state sees no marker, one that was on the channel sees the
// marker first, and a superseded subscription marks nothing.
func (l *peerLink) setPipe(ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, node string, export bool, created srcPoint) {
	if !l.wanted(node) {
		return
	}
	if !export {
		l.dropPipe(ctx, c, node, pipeNotExportable, created)
		return
	}
	ch := "probe/" + node
	l.mu.Lock()
	p := l.pipes[node]
	if p.state == pipeMirrored || p.state == pipeOpening {
		if p.created == created {
			l.mu.Unlock()
			return
		}
		if was := *p; was.state == pipeMirrored { // another incarnation: the old subscription stands down
			go func() {
				uctx, cancel := context.WithTimeout(ctx, 5*time.Second)
				defer cancel()
				c.Unsubscribe(uctx, was.sub)
			}()
		}
	}
	p.gen++
	p.state, p.sub, p.created = pipeOpening, 0, created
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
		// The forwarder never waits for room for a record (the client's reader would wait
		// behind it, ctl and the replies with it): a frame that does not fit is dropped and
		// counted, and the next that fits carries the count, for the loop to mark before it. The
		// one frame it waits to hand over is the subscription's overflow gap, its last.
		var dropped uint64
		for {
			select {
			case f, ok := <-s.Frames:
				if !ok {
					return
				}
				if f.Gap != nil && f.Gap.Reason == "overflow" {
					// The peer dropped this subscription: this is its last frame (nothing follows
					// it here), queued behind its records for the loop to reopen the pipe from.
					q.putLast(peerFrame{prefix: ch, f: f, sub: s.ID}, s.Done, c.Done())
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

// dropPipe ends a wanted node's mirror, if any, and marks it `to`, with the identity `created`
// (zero when the name is absent).
func (l *peerLink) dropPipe(ctx context.Context, c *Client, node, to string, created srcPoint) {
	l.mu.Lock()
	p := l.pipes[node]
	was := *p
	p.gen++
	p.state, p.sub, p.created = to, 0, created
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
// 6 of ix#408). A pipe forwarder never waits for room for a record: what does not fit is
// dropped, counted on the pipe and marked before the next frame that fits; a subscription's
// last frame, the peer's overflow gap, is the one it waits to queue (putLast).
type pipeQueue struct {
	ch    chan peerFrame
	bytes atomic.Int64
}

func newPipeQueue() *pipeQueue { return &pipeQueue{ch: make(chan peerFrame, maxPipeQueue)} }

// putLast queues a subscription's last frame, the peer's overflow gap, behind every record of
// it already queued, waiting for a slot (the record bound holds; the gap carries no record, so
// no bytes are counted) until the subscription or the connection is gone. The forwarder that
// waits here has nothing left to forward: the peer dropped the subscription, so the client's
// reader is never held behind it. Codex round 42: a slot of its own per pipe could be taken by
// a stale subscription's gap after a remove and recreate, and ended the session.
func (q *pipeQueue) putLast(pf peerFrame, done, conn <-chan struct{}) bool {
	select {
	case q.ch <- pf:
		return true
	case <-done:
		return false
	case <-conn:
		return false
	}
}

// weight is what a queued frame counts against the byte bound: its wire bytes for a record,
// nothing for a gap (putLast counts none, so took gives none back; Codex round 43: a gap taken
// back drove the count below zero).
func (pf peerFrame) weight() int64 {
	if pf.f.Gap != nil {
		return 0
	}
	return int64(pf.f.Bytes)
}

// offer queues pf if it fits both bounds, at once or not at all. The bytes are taken before the
// record is queued and given back when it is not, so the count never understates what the queue
// holds.
func (q *pipeQueue) offer(pf peerFrame) bool {
	n := pf.weight()
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
func (q *pipeQueue) took(pf peerFrame) { q.bytes.Add(-pf.weight()) }

// mirror republishes one of the peer's own envelopes. Envelopes the peer
// itself mirrored (another origin, or a peer/ channel) are never forwarded.
// A frame without a seq or an epoch is malformed (the hub assigns seqs from 1,
// every boot has an epoch): it has no provenance to mirror and ends the session.
func (l *peerLink) mirror(sess uint64, origin, prefix string, env *Envelope) error {
	if env.Seq == 0 || env.Epoch == "" {
		return fmt.Errorf("frame on %s without provenance (seq %d, epoch %q)", prefix, env.Seq, env.Epoch)
	}
	l.noteBoot(sess, env.Epoch) // a boot the status and the backlog did not name is marked here, first
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
// peerRefresh, on demand. Its answers are the session's: one that lands after the next session
// began is ignored (Codex round 41: a late status of the old boot regressed the boot and marked
// false boundaries).
func (l *peerLink) refresher(c *Client, sess uint64, kick <-chan struct{}) {
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
			l.setServices(sess, data)
		}
		if !waitFor(c.ctx, peerRefresh) {
			return
		}
	}
}

func (l *peerLink) setServices(sess uint64, status json.RawMessage) {
	var st struct {
		Services json.RawMessage `json:"services"`
		Epoch    string          `json:"epoch"`
	}
	if json.Unmarshal(status, &st) == nil {
		l.mu.Lock()
		stale := sess < l.sess
		if !stale {
			l.services = st.Services
		}
		l.mu.Unlock()
		if !stale {
			l.noteBoot(sess, st.Epoch)
		}
	}
}
