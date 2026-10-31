package main

import (
	"bytes"
	"context"
	"encoding/json"
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
// pid). `availability` is `idle` or `busy` only for a verified incarnation, `blocked` for a
// declared wait, and `unknown` with a `reason` otherwise. A session missing from a later list
// is said once under `gone`, `unknown`, reason `absent`. A failed or slow read is an error
// record: it carries no availability, and no earlier one is repeated. Availability grants no
// authority to dispatch work: it is an observation, read_at_ms says how old.

var claudeSessionsFields = []string{"read_at_ms", "read_ms", "count", "sessions", "gone"}

const (
	// claudeSessionsMinEveryMs bounds the kind's interval: one read starts a process that takes
	// seconds (3 to 8 s measured on 2026-10-31), and a faster tick would only queue reads.
	claudeSessionsMinEveryMs = 5000
	// claudeReadTimeout bounds one read; past it the process is killed and the tick is an error.
	claudeReadTimeout = 20 * time.Second
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
	return &claudeSessions{
		read: func(ctx context.Context) ([]byte, error) {
			cmd := exec.CommandContext(ctx, bin, "agents", "--json")
			var out bytes.Buffer
			cmd.Stdout = &limitedBuffer{buf: &out, max: claudeMaxOutput}
			err := cmd.Run()
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return out.Bytes(), err
		},
		procStart: procStartMs,
		timeout:   claudeReadTimeout,
	}
}

func (c *claudeSessions) sample() (map[string]any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
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
	readAt := time.Now()

	sessions := make([]map[string]any, 0, len(agents))
	now := make(map[string]claudeSeen, len(agents))
	for _, a := range agents {
		s := c.observe(a)
		if prev, ok := c.seen[a.SessionID]; ok && prev.incarnation != "" && s["incarnation"] != nil &&
			prev.incarnation != s["incarnation"] {
			s["incarnation_changed"] = true
		}
		inc, _ := s["incarnation"].(string)
		now[a.SessionID] = claudeSeen{name: a.Name, incarnation: inc}
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
