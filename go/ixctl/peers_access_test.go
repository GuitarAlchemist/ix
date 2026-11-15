package main

import (
	"context"
	"encoding/json"
	"sync"
)

// Test access to the peer link's per-connection state and steps. Every white-box peer test
// reaches them through this file, so moving that state behind a session object changed this
// file alone and not the assertions of the tests that use it. Slice 1c of the ixctl deepening
// drives those tests through the link's frames instead and removes the file.

// sessOf is the state of the session the link runs now, or ran last (client, ctl, pipes, listed,
// cursor). The caller holds l.mu when it reads or writes through it while the link runs.
func sessOf(l *peerLink) *linkState {
	if l.latest == nil { // a link a test made by hand, with no session yet
		l.latest = &linkState{}
	}
	return l.latest
}

// The tests name sessions by number, as the link did before it named them by identity: a link's
// sessions are numbered in the order this file first sees them, the session a link starts with
// is 0 when beginSessionT makes the next, and a number below the current one is an older session.
var sessNumbers = struct {
	sync.Mutex
	of map[*peerLink]*sessIndex
}{of: map[*peerLink]*sessIndex{}}

type sessIndex struct {
	no   map[*linkState]uint64
	at   map[uint64]*linkState
	next uint64
}

// sessNo is st's number among l's sessions, given on first sight.
func sessNo(l *peerLink, st *linkState) uint64 {
	sessNumbers.Lock()
	defer sessNumbers.Unlock()
	ix := sessNumbers.of[l]
	if ix == nil {
		ix = &sessIndex{no: map[*linkState]uint64{}, at: map[uint64]*linkState{}}
		sessNumbers.of[l] = ix
	}
	n, ok := ix.no[st]
	if !ok {
		n = ix.next
		ix.next++
		ix.no[st], ix.at[n] = n, st
	}
	return n
}

// sessID names the session the link runs now, for a test that checks it did not reconnect. The
// caller holds l.mu.
func sessID(l *peerLink) uint64 { return sessNo(l, sessOf(l)) }

// stateAt is the state of l's session number n: the current one for n at or past its number (an
// observation of the session now running), an older one's otherwise.
func stateAt(l *peerLink, n uint64) *linkState {
	l.mu.Lock()
	cur := sessOf(l)
	l.mu.Unlock()
	if n >= sessNo(l, cur) {
		return cur
	}
	sessNumbers.Lock()
	defer sessNumbers.Unlock()
	if st := sessNumbers.of[l].at[n]; st != nil {
		return st
	}
	return &linkState{} // an older session this file never saw
}

// wired is a session of l on state st whose steps reach the connection through what the test
// gives: several wired sessions share st, as the steps of one session share its state.
func wired(l *peerLink, st *linkState, ctx context.Context, c *Client, q *pipeQueue, merged chan peerFrame, fail chan error, refresh chan struct{}, relist chan struct{}, origin string) *linkSession {
	return &linkSession{linkState: st, l: l, ctx: ctx, c: c, origin: origin, q: q, merged: merged, fail: fail, refresh: refresh, relist: relist}
}

func latestOf(l *peerLink) *linkState {
	l.mu.Lock()
	defer l.mu.Unlock()
	return sessOf(l)
}

func applyT(l *peerLink, ctx context.Context, c *Client, q *pipeQueue, merged chan peerFrame, fail chan error, refresh chan struct{}, relist chan struct{}, sess uint64, origin string, pf peerFrame) error {
	return wired(l, stateAt(l, sess), ctx, c, q, merged, fail, refresh, relist, origin).apply(pf)
}

func snapshotT(l *peerLink, ctx context.Context, c *Client, q *pipeQueue, fail chan error, asked uint64, data json.RawMessage) error {
	return wired(l, latestOf(l), ctx, c, q, nil, fail, nil, nil, "").snapshot(asked, data)
}

func resnapshotT(l *peerLink, ctx context.Context, c *Client, q *pipeQueue, merged chan peerFrame, fail chan error, refresh chan struct{}, relist chan struct{}, sess uint64, origin string) error {
	return wired(l, stateAt(l, sess), ctx, c, q, merged, fail, refresh, relist, origin).resnapshot()
}

func jumpControlT(l *peerLink, ctx context.Context, c *Client, relist chan struct{}, merged chan peerFrame, fail chan error, prefix string, sub int64) {
	wired(l, latestOf(l), ctx, c, nil, merged, fail, nil, relist, "").jumpControl(prefix, sub)
}

func reopenControlT(l *peerLink, ctx context.Context, c *Client, relist chan struct{}, merged chan peerFrame, fail chan error, pf peerFrame) {
	wired(l, latestOf(l), ctx, c, nil, merged, fail, nil, relist, "").reopenControl(pf)
}

func mirrorT(l *peerLink, sess uint64, origin, prefix string, env *Envelope) error {
	return wired(l, stateAt(l, sess), nil, nil, nil, nil, nil, nil, nil, origin).mirror(prefix, env)
}

func noteBootT(l *peerLink, sess uint64, epoch string) {
	wired(l, stateAt(l, sess), nil, nil, nil, nil, nil, nil, nil, "").noteBoot(epoch)
}

func setServicesT(l *peerLink, sess uint64, status json.RawMessage) {
	wired(l, stateAt(l, sess), nil, nil, nil, nil, nil, nil, nil, "").setServices(status)
}

// beginSessionT begins a session on l (no connection: the steps the test runs on it give their
// own) and says what it resumes from and its number.
func beginSessionT(l *peerLink, token string) (map[string]*uint64, string, uint64) {
	sessNo(l, latestOf(l)) // the session the link had numbered before the new one
	s, since, epoch := l.beginSession(context.Background(), nil, stateFile{Token: token})
	return since, epoch, sessNo(l, s.linkState)
}

// markLiveStartLockedT is called with l.mu held, as the step it stands for.
func markLiveStartLockedT(l *peerLink, ch string) { l.markLiveStartLocked(ch) }
