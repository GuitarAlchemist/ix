package main

import "context"

// planeOp is one op of the plane, declared once: Plane.Request runs it, forward relays it to a
// peer when relay is set, and the tests walk the table. The transports decode their own
// encoding of an op (an HTTP route's body, a WebSocket frame) into a Request and hand it to
// Request; subscribe and unsubscribe are the WebSocket session's own and are not here.
// @ai:invariant the ops not relayed are exactly shutdown and need.*; every other op in the table reaches a named peer [P:assumed conf:0.8] (pinned by TestEveryOpOfTheTableIsRelayedOrRefusedByName)
type planeOp struct {
	relay bool // a request naming a peer is handed on to it (forward); otherwise refused by name
	run   func(p *Plane, ctx context.Context, req Request) (any, error)
}

// planeOps is filled by init: its handlers call into the plane, which reads the table.
var planeOps map[string]planeOp

func init() {
	needOp := planeOp{run: func(p *Plane, ctx context.Context, req Request) (any, error) { return p.needRequest(ctx, req) }}
	planeOps = map[string]planeOp{
		"status": {relay: true, run: func(p *Plane, _ context.Context, _ Request) (any, error) { return p.Status(), nil }},
		"ping": {relay: true, run: func(p *Plane, _ context.Context, _ Request) (any, error) {
			return map[string]any{"pong": true, "name": p.name}, nil
		}},
		"plan": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) {
			if req.Desired == nil {
				return nil, badRequest("plan needs a desired-state document")
			}
			return p.Plan(req.Desired)
		}},
		"apply": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) {
			if req.Desired == nil {
				return nil, badRequest("apply needs a desired-state document")
			}
			r, _, err := p.Apply(req.Desired)
			return r, err
		}},
		"wait": {relay: true, run: func(p *Plane, ctx context.Context, req Request) (any, error) { return p.Wait(ctx, req) }},
		"stop": {relay: true, run: func(p *Plane, ctx context.Context, req Request) (any, error) { return p.Stop(ctx, req.Service) }},
		"shutdown": {run: func(p *Plane, _ context.Context, _ Request) (any, error) {
			p.RequestShutdown()
			return map[string]any{"shutdown": true}, nil
		}},
		"probe.kinds": {relay: true, run: func(p *Plane, _ context.Context, _ Request) (any, error) {
			return kindsReply(p.probes.budgetNow()), nil
		}},
		"probe.create": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) {
			if req.Probe == nil {
				return nil, badRequest("probe.create needs a node under `probe`")
			}
			return p.probes.create(req.Probe)
		}},
		"probe.list":  {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) { return p.probes.list(req.WithCursor), nil }},
		"probe.start": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) { return p.probes.start(req.Node) }},
		"probe.stop": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) {
			return p.probes.stop(req.Node, "requested")
		}},
		"probe.touch": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) { return p.probes.touch(req.Node) }},
		"probe.remove": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) {
			return p.probes.remove(req.Node, "removed")
		}},
		"bookmark.set":    {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) { return p.bookmarks.set(req) }},
		"bookmark.get":    {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) { return p.bookmarks.get(req.Bookmark) }},
		"bookmark.list":   {relay: true, run: func(p *Plane, _ context.Context, _ Request) (any, error) { return p.bookmarks.list(), nil }},
		"bookmark.remove": {relay: true, run: func(p *Plane, _ context.Context, req Request) (any, error) { return p.bookmarks.remove(req.Bookmark) }},
	}
	for _, op := range needOps {
		planeOps["need."+op] = needOp
	}
}
