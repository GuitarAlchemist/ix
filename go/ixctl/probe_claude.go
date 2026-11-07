package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"time"
)

// The claude-sessions@1 kind reads `claude agents --json`, Claude Code's own list of the live
// sessions of the user running the plane (interactive and background), and publishes one
// snapshot per tick. The plane must run in that user's session (a desk peer): a service in
// session 0 sees no session. Reading costs no model call; it takes seconds, hence the kind's
// minimum interval.
//
// Per session, the record keeps apart what Claude Code declares and what the plane verified:
// `session_id` is the stable identity; `declared` is the status (or state) as listed;
// `incarnation` is `pid:creation_ms`, set only when the process at that pid was created at most
// incarnationSlack before the session's `startedAt` (a process created after it holds a reused
// pid; a correlation, not an authentication: it does not prove the process is Claude Code, and
// it trusts the list as written). `availability` is `idle` or `busy` only for a verified
// incarnation, `blocked` for a declared wait, and `unknown` with a `reason` otherwise. A session missing from a later list
// is said once under `gone`, `unknown`, reason `absent`. A failed or slow read is an error
// record: it carries no availability, and no earlier one is repeated. Availability grants no
// authority to dispatch work: it is an observation, read_at_ms says how old.

var claudeSessionsFields = []string{"read_at_ms", "read_ms", "count", "sessions", "gone"}

const (
	// claudeSessionsMinEveryMs bounds the kind's interval: one read starts a process that takes
	// seconds (3 to 8 s measured on 2026-10-31), and a faster tick would only queue reads.
	claudeSessionsMinEveryMs = 5000
	// claudeReadTimeout bounds one read; past it the CLI's whole tree is killed and the tick is
	// an error. A probe stop or the plane's shutdown cancels the read sooner.
	claudeReadTimeout = 20 * time.Second
	// claudeWaitDelay bounds how long the CLI's output may stay open once the CLI has ended (a
	// descendant inheriting it): past it the pipe is closed and the read refused.
	claudeWaitDelay = time.Second
	// claudeMaxOutput bounds what one read may return.
	claudeMaxOutput = 4 << 20
	// incarnationSlack is how long before its session's startedAt a process may have been
	// created and still be that session's (0.7 to 7 s measured).
	incarnationSlack = 60 * time.Second
)

// claudeAgent is one entry of `claude agents --json`, the fields the kind reads.
type claudeAgent struct {
	SessionID string `json:"sessionId"`
	Name      string `json:"name"`
	Kind      string `json:"kind"`
	Status    string `json:"status"`
	State     string `json:"state"`
	Pid       *int   `json:"pid"`
	StartedAt *int64 `json:"startedAt"`
}

// claudeSessions is the claude-sessions@1 sampler. read and procStart are its ports to the
// CLI and the OS; tests replace them.
type claudeSessions struct {
	read      func(ctx context.Context) ([]byte, error)
	procStart func(pid int) (int64, error)
	timeout   time.Duration
	seen      map[string]claudeSeen // the sessions of the last good read, by session id
}

type claudeSeen struct {
	name        string
	incarnation string // empty when it was not verified
}

func newClaudeSessions(bin string) *claudeSessions {
	return &claudeSessions{read: func(ctx context.Context) ([]byte, error) { return readAgents(ctx, bin) },
		procStart: procStartMs, timeout: claudeReadTimeout}
}

// readAgents runs `bin agents --json` as a process tree of its own (a job object on Windows, a
// process group on Linux, as a service runs), so that nothing it starts outlives the read: the
// tree is killed when ctx ends and once the CLI has ended. Its output must close within
// claudeWaitDelay of the CLI's end; an output a descendant kept open is refused, not trusted.
func readAgents(ctx context.Context, bin string) ([]byte, error) {
	cmd := exec.Command(bin, "agents", "--json")
	var out bytes.Buffer
	cmd.Stdout = &limitedBuffer{buf: &out, max: claudeMaxOutput}
	cmd.WaitDelay = claudeWaitDelay
	tree, err := startTree(cmd, nil)
	if err != nil {
		return nil, err
	}
	defer func() { tree.kill(); tree.close() }()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err = <-done:
	case <-ctx.Done():
		tree.kill()
		<-done // the tree is dead: its pipe closes, and WaitDelay bounds the rest
		return nil, ctx.Err()
	}
	if errors.Is(err, exec.ErrWaitDelay) {
		return nil, fmt.Errorf("its output stayed open %s after it ended (a descendant held it)", claudeWaitDelay)
	}
	return out.Bytes(), err
}

func (c *claudeSessions) sample() (map[string]any, error) {
	return c.sampleCtx(context.Background())
}

// sampleCtx is a sample tied to ctx: the probe tree passes the node's context, so a probe stop
// or the plane's shutdown cancels a read in flight instead of waiting for its bound.
func (c *claudeSessions) sampleCtx(parent context.Context) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(parent, c.timeout)
	defer cancel()
	t0 := time.Now()
	raw, err := c.read(ctx)
	if err != nil {
		return nil, fmt.Errorf("claude agents --json: %w", err)
	}
	var agents []claudeAgent
	if err := json.Unmarshal(raw, &agents); err != nil {
		return nil, fmt.Errorf("claude agents --json: not a list of sessions: %v", err)
	}
	if agents == nil { // `null`: no list at all, which must not read as every session gone
		return nil, errors.New("claude agents --json: null, not a list of sessions")
	}
	ids := make(map[string]bool, len(agents))
	for _, a := range agents {
		if a.SessionID != "" && ids[a.SessionID] {
			return nil, fmt.Errorf("claude agents --json: session %s listed twice", a.SessionID)
		}
		ids[a.SessionID] = true
	}
	readAt := time.Now()

	sessions := make([]map[string]any, 0, len(agents))
	now := make(map[string]claudeSeen, len(agents))
	for _, a := range agents {
		s := c.observe(a)
		if prev, ok := c.seen[a.SessionID]; ok && prev.incarnation != "" && s["incarnation"] != nil &&
			prev.incarnation != s["incarnation"] {
			s["incarnation_changed"] = true
		}
		if a.SessionID != "" { // an entry without an id is listed, unknown, and never tracked
			inc, _ := s["incarnation"].(string)
			now[a.SessionID] = claudeSeen{name: a.Name, incarnation: inc}
		}
		sessions = append(sessions, s)
	}
	gone := []map[string]any{}
	for id, prev := range c.seen {
		if _, ok := now[id]; !ok {
			g := map[string]any{"session_id": id, "name": prev.name, "availability": "unknown", "reason": "absent"}
			if prev.incarnation != "" {
				g["last_incarnation"] = prev.incarnation
			}
			gone = append(gone, g)
		}
	}
	c.seen = now
	return map[string]any{
		"read_at_ms": readAt.UnixMilli(),
		"read_ms":    readAt.Sub(t0).Milliseconds(),
		"count":      len(sessions),
		"sessions":   sessions,
		"gone":       gone,
	}, nil
}

// observe is one session's record: what was declared, what could be verified, and the
// availability that follows.
func (c *claudeSessions) observe(a claudeAgent) map[string]any {
	declared := a.Status
	if declared == "" {
		declared = a.State
	}
	s := map[string]any{"session_id": a.SessionID, "name": a.Name, "kind": a.Kind, "declared": declared,
		"identity": "unknown"}
	if a.Pid != nil {
		s["pid"] = *a.Pid
	}
	if a.StartedAt != nil {
		s["started_at_ms"] = *a.StartedAt
	}
	reason := ""
	switch {
	case a.SessionID == "":
		reason = "no-session-id"
	case a.Pid == nil:
		reason = "no-pid"
	case a.StartedAt == nil:
		reason = "no-started-at"
	default:
		created, err := c.procStart(*a.Pid)
		switch {
		case err != nil:
			reason = "process-unreadable"
		case created > *a.StartedAt:
			reason = "pid-reused"
		case *a.StartedAt-created > incarnationSlack.Milliseconds():
			reason = "start-mismatch"
		default:
			s["identity"] = "verified"
			s["proc_start_ms"] = created
			s["incarnation"] = strconv.Itoa(*a.Pid) + ":" + strconv.FormatInt(created, 10)
		}
	}
	switch {
	case reason != "":
		s["availability"], s["reason"] = "unknown", reason
	case declared == "idle":
		s["availability"] = "idle"
	case declared == "busy":
		s["availability"] = "busy"
	case declared == "waiting" || declared == "blocked":
		s["availability"] = "blocked"
	default:
		s["availability"], s["reason"] = "unknown", "declared "+strconv.Quote(declared)
	}
	return s
}

// limitedBuffer keeps at most max bytes and fails the write past them, which ends the read.
type limitedBuffer struct {
	buf *bytes.Buffer
	max int
}

func (l *limitedBuffer) Write(p []byte) (int, error) {
	if l.buf.Len()+len(p) > l.max {
		return 0, fmt.Errorf("output past %d bytes", l.max)
	}
	return l.buf.Write(p)
}
