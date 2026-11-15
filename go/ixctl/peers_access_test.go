package main

import (
	"context"
	"encoding/json"
)

// Test access to the peer link's per-connection state and steps. Every white-box peer test
// reaches them through this file, so moving that state behind a session object changes this
// file alone and not the assertions of the tests that use it. Slice 1c of the ixctl deepening
// drives those tests through the link's frames instead and removes the file.

// sessOf is what holds the link's per-connection state (client, ctl, pipes, listed, cursor). The
// caller holds l.mu when it reads or writes through it while the link runs.
func sessOf(l *peerLink) *peerLink { return l }

// sessID names the session the link runs now, for a test that checks it did not reconnect. The
// caller holds l.mu.
func sessID(l *peerLink) uint64 { return l.sess }

func applyT(l *peerLink, ctx context.Context, c *Client, q *pipeQueue, merged chan<- peerFrame, fail chan<- error, refresh chan<- struct{}, relist chan<- struct{}, sess uint64, origin string, pf peerFrame) error {
	return l.apply(ctx, c, q, merged, fail, refresh, relist, sess, origin, pf)
}

func snapshotT(l *peerLink, ctx context.Context, c *Client, q *pipeQueue, fail chan<- error, asked uint64, data json.RawMessage) error {
	return l.snapshot(ctx, c, q, fail, asked, data)
}

func resnapshotT(l *peerLink, ctx context.Context, c *Client, q *pipeQueue, merged chan peerFrame, fail chan error, refresh chan<- struct{}, relist chan struct{}, sess uint64, origin string) error {
	return l.resnapshot(ctx, c, q, merged, fail, refresh, relist, sess, origin)
}

func jumpControlT(l *peerLink, ctx context.Context, c *Client, relist chan<- struct{}, merged chan<- peerFrame, fail chan<- error, prefix string, sub int64) {
	l.jumpControl(ctx, c, relist, merged, fail, prefix, sub)
}

func reopenControlT(l *peerLink, ctx context.Context, c *Client, relist chan<- struct{}, merged chan<- peerFrame, fail chan<- error, pf peerFrame) {
	l.reopenControl(ctx, c, relist, merged, fail, pf)
}

func mirrorT(l *peerLink, sess uint64, origin, prefix string, env *Envelope) error {
	return l.mirror(sess, origin, prefix, env)
}

func noteBootT(l *peerLink, sess uint64, epoch string) { l.noteBoot(sess, epoch) }

func setServicesT(l *peerLink, sess uint64, status json.RawMessage) { l.setServices(sess, status) }

func beginSessionT(l *peerLink, token string) (map[string]*uint64, string, uint64) {
	return l.beginSession(token)
}

// markLiveStartLockedT is called with l.mu held, as the step it stands for.
func markLiveStartLockedT(l *peerLink, ch string) { l.markLiveStartLocked(ch) }
