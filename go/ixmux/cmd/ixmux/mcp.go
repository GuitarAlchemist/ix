package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// ixmux mcp is a local MCP server on stdio (JSON-RPC 2.0, one message per line) over the HTTP API
// of the ixmux serve whose state directory it is given. It reads the address and the token there
// at each tool call, so it follows an ixmux restarted since; it never takes the token as an
// argument or a variable, and never writes it. It has no tool that stops ixmux.

// Named bounds of the adapter.
const (
	MCPMaxLine       = 1 << 20                 // bytes of one JSON-RPC line read on stdin
	MCPReceiptWait   = 1500 * time.Millisecond // ixmux_submit answers within this, plus one HTTP read
	MCPMaxResultText = 4 << 10                 // bytes of a result's text (claudestream keeps as many)
	MCPCallTimeout   = 5 * time.Second         // one GET of ixmux_status or ixmux_result
	MCPMaxIDBytes    = 128                     // bytes of a mandate id
	MCPMaxPosts      = 4                       // POSTs of ixmux_submit still running at once
)

// mcpPostExtra is what a POST is given beyond ixmux's delivery deadline (its bounds): the time
// ixmux itself gives the answer (server.WriteTimeout), plus a margin. A test lowers it.
var mcpPostExtra = server.WriteTimeout + 5*time.Second

// The protocol versions the adapter speaks, the latest first. initialize answers the version asked
// if it is one of them, else the latest.
var mcpProtocols = []string{"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}

// A mandate id is a path segment of GET /v1/mandates/{id}: letters, digits, '.', '_' and '-', not
// starting with '.' (no "." or ".." segment).
var mandateIDRE = regexp.MustCompile(`^[A-Za-z0-9_-][A-Za-z0-9._-]*$`)

// Typed reasons the adapter gives itself; ixmux's refusal codes are passed on as they are.
const (
	reasonUnavailable = "ixmux-unavailable"
	reasonBadArgs     = "bad-arguments"
	reasonBadID       = "bad-mandate-id"
	reasonLineTooLong = "line-too-long"
	reasonUnknownTool = "unknown-tool"
	reasonIxmuxError  = "ixmux-error"
	// reasonPostsInFlight: MCPMaxPosts submissions still wait for ixmux's answer.
	reasonPostsInFlight = "submits-in-flight"
)

type rpcError struct {
	Code    int            `json:"code"`
	Message string         `json:"message"`
	Data    map[string]any `json:"data,omitempty"`
}

type mcp struct {
	state  string
	out    *json.Encoder
	client *http.Client
	posts  chan struct{} // one per POST of ixmux_submit still running
}

func newMCP(state string, stdout io.Writer) *mcp {
	return &mcp{state: state, out: json.NewEncoder(stdout), posts: make(chan struct{}, MCPMaxPosts),
		// Loopback only, no proxy, no connection kept: an ixmux restarted since is a new address.
		client: &http.Client{Transport: &http.Transport{Proxy: nil, DisableKeepAlives: true}}}
}

func runMCP(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("mcp", flag.ContinueOnError)
	fs.SetOutput(stderr)
	state := fs.String("state", "", "the state directory of the ixmux serve to talk to")
	if err := fs.Parse(args); err != nil {
		return server.ExitUsage
	}
	if *state == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: ixmux mcp --state <dir>")
		return server.ExitUsage
	}
	m := newMCP(*state, stdout)
	br := bufio.NewReaderSize(stdin, 64<<10)
	for {
		line, tooLong, err := readRPCLine(br, MCPMaxLine)
		if tooLong {
			m.reply(nil, nil, &rpcError{-32600, "line too long", map[string]any{"reason": reasonLineTooLong, "max_bytes": MCPMaxLine}})
		} else if len(bytes.TrimSpace(line)) > 0 {
			m.handle(line)
		}
		if err != nil { // stdin closed: a clean end; ixmux goes on
			return server.ExitOK
		}
	}
}

// readRPCLine reads one line without its end of line. A line over max bytes is read to its end
// and dropped (tooLong).
func readRPCLine(br *bufio.Reader, max int) (line []byte, tooLong bool, err error) {
	for {
		chunk, err := br.ReadSlice('\n')
		if !tooLong {
			if len(line)+len(chunk) > max+2 { // the end of line is not counted
				tooLong, line = true, nil
			} else {
				line = append(line, chunk...)
			}
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		return bytes.TrimRight(line, "\r\n"), tooLong, err
	}
}

func (m *mcp) reply(id json.RawMessage, result any, rerr *rpcError) {
	msg := map[string]any{"jsonrpc": "2.0", "id": id}
	if id == nil {
		msg["id"] = nil
	}
	if rerr != nil {
		msg["error"] = rerr
	} else {
		msg["result"] = result
	}
	m.out.Encode(msg) // one line: Encode ends it with '\n'
}

func (m *mcp) handle(line []byte) {
	if line[0] == '[' {
		m.reply(nil, nil, &rpcError{-32600, "batches are not supported", nil})
		return
	}
	var req struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Method  string          `json:"method"`
		Params  json.RawMessage `json:"params"`
	}
	if err := json.Unmarshal(line, &req); err != nil {
		m.reply(nil, nil, &rpcError{-32700, "parse error", nil})
		return
	}
	notification := req.ID == nil || string(req.ID) == "null"
	if req.JSONRPC != "2.0" || req.Method == "" {
		if !notification {
			m.reply(req.ID, nil, &rpcError{-32600, "invalid request", nil})
		}
		return
	}
	if notification { // notifications/initialized and the others: nothing to answer
		return
	}
	switch req.Method {
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		json.Unmarshal(req.Params, &p)
		v := mcpProtocols[0]
		for _, s := range mcpProtocols {
			if s == p.ProtocolVersion {
				v = s
			}
		}
		m.reply(req.ID, map[string]any{
			"protocolVersion": v,
			"capabilities":    map[string]any{"tools": map[string]any{"listChanged": false}},
			"serverInfo":      map[string]any{"name": "ixmux", "version": version},
			"instructions":    "ixmux_submit hands a mandate to a session ixmux manages and answers a receipt; ixmux_result reads its outcome by mandate_id.",
		}, nil)
	case "ping":
		m.reply(req.ID, map[string]any{}, nil)
	case "tools/list":
		m.reply(req.ID, map[string]any{"tools": mcpTools}, nil)
	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			m.reply(req.ID, nil, &rpcError{-32602, "invalid params", map[string]any{"reason": reasonBadArgs}})
			return
		}
		var v any
		var terr *toolErr
		switch p.Name {
		case "ixmux_status":
			v, terr = m.status(p.Arguments)
		case "ixmux_submit":
			v, terr = m.submit(p.Arguments)
		case "ixmux_result":
			v, terr = m.result(p.Arguments)
		default:
			m.reply(req.ID, nil, &rpcError{-32602, "unknown tool " + p.Name, map[string]any{"reason": reasonUnknownTool}})
			return
		}
		if terr != nil {
			v = terr
		}
		text, _ := json.Marshal(v)
		m.reply(req.ID, map[string]any{"content": []any{map[string]any{"type": "text", "text": string(text)}},
			"structuredContent": v, "isError": terr != nil}, nil)
	default:
		m.reply(req.ID, nil, &rpcError{-32601, "method not found: " + req.Method, nil})
	}
}

// toolErr is the result of a tool that failed: a typed reason, the adapter staying up.
type toolErr struct {
	Error  string `json:"error"`
	Detail string `json:"detail,omitempty"`
}

func fail(code, detail string) *toolErr { return &toolErr{code, detail} }

// args decodes a tool's arguments; an unknown field or a wrong type is bad-arguments.
func args(raw json.RawMessage, v any) *toolErr {
	if len(bytes.TrimSpace(raw)) == 0 || string(raw) == "null" {
		raw = []byte("{}")
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fail(reasonBadArgs, err.Error())
	}
	return nil
}

func checkID(id string) *toolErr {
	if len(id) == 0 || len(id) > MCPMaxIDBytes || !mandateIDRE.MatchString(id) {
		return fail(reasonBadID, fmt.Sprintf("a mandate id is 1 to %d bytes of letters, digits, '.', '_' and '-', not starting with '.'", MCPMaxIDBytes))
	}
	return nil
}

// --- talking to ixmux ------------------------------------------------------------------------

type conn struct{ addr, token string }

// connect reads the address and the token ixmux serve wrote in the state directory.
func (m *mcp) connect() (conn, *toolErr) {
	b, err := os.ReadFile(filepath.Join(m.state, server.InfoFile))
	if err != nil {
		return conn{}, fail(reasonUnavailable, "no "+server.InfoFile+" in the state directory")
	}
	var info server.Info
	if json.Unmarshal(b, &info) != nil {
		return conn{}, fail(reasonUnavailable, server.InfoFile+" is unreadable")
	}
	if info.State != "running" {
		return conn{}, fail(reasonUnavailable, "ixmux is "+info.State)
	}
	host, _, err := net.SplitHostPort(info.Addr)
	if ip := net.ParseIP(host); err != nil || ip == nil || !ip.IsLoopback() {
		return conn{}, fail(reasonUnavailable, "the address in "+server.InfoFile+" is not loopback")
	}
	tok, err := os.ReadFile(filepath.Join(m.state, server.TokenFile))
	if err != nil || len(tok) == 0 {
		return conn{}, fail(reasonUnavailable, "no token in the state directory")
	}
	return conn{info.Addr, string(tok)}, nil
}

// do sends one request to ixmux and decodes a 200 answer into out. A refusal of ixmux comes back
// with its own code; no answer at all is ixmux-unavailable. No error names the token.
func (m *mcp) do(ctx context.Context, c conn, method, path string, body, out any) *toolErr {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, "http://"+c.addr+path, rd)
	if err != nil {
		return fail(reasonIxmuxError, err.Error())
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := m.client.Do(req)
	if err != nil { // err.Error() would name the URL, never the header; the address is enough
		if ctx.Err() != nil {
			return fail(reasonUnavailable, "no answer from ixmux in time")
		}
		return fail(reasonUnavailable, "ixmux does not answer at "+c.addr)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return fail(reasonUnavailable, "the answer of ixmux was cut")
	}
	if resp.StatusCode == http.StatusUnauthorized {
		return fail(reasonUnavailable, "ixmux refused the token of the state directory")
	}
	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error  string `json:"error"`
			Detail string `json:"detail"`
		}
		if json.Unmarshal(b, &e) == nil && e.Error != "" {
			return fail(e.Error, e.Detail)
		}
		return fail(reasonIxmuxError, fmt.Sprintf("HTTP %d", resp.StatusCode))
	}
	if out != nil && json.Unmarshal(b, out) != nil {
		return fail(reasonIxmuxError, "an answer of ixmux is unreadable")
	}
	return nil
}

type ixStatus struct {
	Version string `json:"version"`
	Epoch   string `json:"epoch"`
	Targets []struct {
		Name        string `json:"name"`
		Kind        string `json:"kind"`
		Incarnation string `json:"incarnation"`
		Alive       bool   `json:"alive"`
		Error       string `json:"error"`
	} `json:"targets"`
	Deliveries []deliver.Entry `json:"deliveries"`
	Bounds     struct {
		DeliveryTimeout string `json:"delivery_timeout"`
	} `json:"bounds"`
}

// --- tools -----------------------------------------------------------------------------------

func (m *mcp) status(raw json.RawMessage) (any, *toolErr) {
	var a struct{}
	if e := args(raw, &a); e != nil {
		return nil, e
	}
	c, e := m.connect()
	if e != nil {
		return nil, e
	}
	ctx, cancel := context.WithTimeout(context.Background(), MCPCallTimeout)
	defer cancel()
	var st ixStatus
	if e := m.do(ctx, c, "GET", "/v1/status", nil, &st); e != nil {
		return nil, e
	}
	type inFlight struct {
		MandateID    string        `json:"mandate_id"`
		State        deliver.State `json:"state"`
		StillRunning bool          `json:"still_running"`
	}
	type target struct {
		Name        string     `json:"name"`
		Kind        string     `json:"kind"`
		Incarnation string     `json:"incarnation"`
		Alive       bool       `json:"alive,omitempty"`
		Error       string     `json:"error,omitempty"`
		InFlight    []inFlight `json:"in_flight"`
	}
	out := []target{}
	for _, t := range st.Targets {
		tg := target{Name: t.Name, Kind: t.Kind, Incarnation: t.Incarnation, Alive: t.Alive, Error: t.Error, InFlight: []inFlight{}}
		for _, d := range st.Deliveries {
			if d.Target == t.Name && (d.State == deliver.Accepted || d.State == deliver.Delivering || d.StillRunning) {
				tg.InFlight = append(tg.InFlight, inFlight{d.MandateID, d.State, d.StillRunning})
			}
		}
		out = append(out, tg)
	}
	return map[string]any{"version": st.Version, "epoch": st.Epoch, "targets": out}, nil
}

// submit hands a mandate to ixmux and answers a receipt within MCPReceiptWait (plus one bounded
// read). POST /v1/mandates waits for the whole turn, so it runs on; the receipt says accepted only
// once ixmux answered, or GET /v1/mandates/{id} shows the mandate in a durable state: the engine
// shows a transition only after its line is written and synced in the command log. A POST cut at
// the transport proves nothing either way: the entry is read on until the bound.
func (m *mcp) submit(raw json.RawMessage) (any, *toolErr) {
	began := time.Now()
	var a struct {
		Target      string `json:"target"`
		MandateID   string `json:"mandate_id"`
		Text        string `json:"text"`
		Incarnation string `json:"incarnation"`
	}
	if e := args(raw, &a); e != nil {
		return nil, e
	}
	if e := checkID(a.MandateID); e != nil {
		return nil, e
	}
	if a.Target == "" {
		return nil, fail(reasonBadArgs, "target is required")
	}
	if len(a.Text) > deliver.MaxText {
		return nil, fail(deliver.TextTooLong, fmt.Sprintf("text of %d bytes, bound %d", len(a.Text), deliver.MaxText))
	}
	c, e := m.connect()
	if e != nil {
		return nil, e
	}
	budget, cancel := context.WithDeadline(context.Background(), began.Add(MCPReceiptWait))
	defer cancel()
	// The status gives ixmux's delivery deadline, and the target's current incarnation when none
	// is given.
	var st ixStatus
	if e := m.do(budget, c, "GET", "/v1/status", nil, &st); e != nil {
		return nil, e
	}
	if a.Incarnation == "" {
		found := false
		for _, t := range st.Targets {
			if t.Name == a.Target {
				a.Incarnation, found = t.Incarnation, true
			}
		}
		if !found {
			return nil, fail(deliver.UnknownTarget, "no target "+a.Target)
		}
	}
	select {
	case m.posts <- struct{}{}:
	default:
		return nil, fail(reasonPostsInFlight, fmt.Sprintf("%d submissions still wait for ixmux's answer; try again once one ends", MCPMaxPosts))
	}
	deliveryTimeout, err := time.ParseDuration(st.Bounds.DeliveryTimeout)
	if err != nil || deliveryTimeout <= 0 {
		deliveryTimeout = deliver.DefaultTimeout
	}
	type posted struct {
		res deliver.Result
		err *toolErr
	}
	done := make(chan posted, 1)
	// @ai:invariant at most MCPMaxPosts POSTs run at once, each within ixmux's delivery deadline plus mcpPostExtra [T:test conf:0.85 src:go/ixmux/cmd/ixmux/s6b_mcp_test.go:TestDetachedPOSTsAreBounded]
	go func() { // not bound to the receipt: ixmux delivers whether or not anyone still waits
		defer func() { <-m.posts }()
		ctx, cancel := context.WithTimeout(context.Background(), deliveryTimeout+mcpPostExtra)
		defer cancel()
		var res deliver.Result
		e := m.do(ctx, c, "POST", "/v1/mandates",
			map[string]string{"mandate_id": a.MandateID, "target": a.Target, "incarnation": a.Incarnation, "text": a.Text}, &res)
		done <- posted{res, e}
	}()
	receipt := map[string]any{"mandate_id": a.MandateID, "target": a.Target, "incarnation": a.Incarnation}
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case p := <-done:
			if p.err != nil && p.err.Error == reasonUnavailable {
				// Cut at the transport: ixmux may have taken it all the same. Its entry tells,
				// until the receipt's bound.
				done = nil
				continue
			}
			if p.err != nil {
				return nil, p.err
			}
			receipt["accepted"], receipt["state"], receipt["replayed"] = true, p.res.State, p.res.Replayed
			if p.res.Ack != nil && p.res.Ack.Receipt != "" {
				receipt["receipt"] = p.res.Ack.Receipt
			}
			return receipt, nil
		case <-tick.C:
			// Any durable state proves the mandate is in the command log: the engine shows an
			// entry only once its accepted line is synced.
			// @ai:invariant the receipt says accepted only on ixmux's answer or a durable state of the entry [T:test conf:0.85 src:go/ixmux/cmd/ixmux/s6b_mcp_test.go:TestReceiptAfterACutPOST]
			var en deliver.Entry
			if m.do(budget, c, "GET", "/v1/mandates/"+a.MandateID, nil, &en) == nil && durable(en.State) {
				receipt["accepted"], receipt["state"], receipt["replayed"] = true, en.State, false
				return receipt, nil
			}
		case <-budget.Done():
			receipt["accepted"], receipt["state"] = false, "submitted"
			receipt["detail"] = "ixmux did not confirm it within the receipt wait; ixmux_result tells, and the same submission again is safe"
			return receipt, nil
		}
	}
}

// durable: a state the engine writes to the command log.
func durable(s deliver.State) bool {
	switch s {
	case deliver.Accepted, deliver.Delivering, deliver.Delivered, deliver.Refused, deliver.Uncertain:
		return true
	}
	return false
}

func (m *mcp) result(raw json.RawMessage) (any, *toolErr) {
	var a struct {
		MandateID string `json:"mandate_id"`
	}
	if e := args(raw, &a); e != nil {
		return nil, e
	}
	if e := checkID(a.MandateID); e != nil {
		return nil, e
	}
	c, e := m.connect()
	if e != nil {
		return nil, e
	}
	ctx, cancel := context.WithTimeout(context.Background(), MCPCallTimeout)
	defer cancel()
	var v struct {
		deliver.Entry
		Ack *deliver.Ack `json:"ack"`
	}
	if e := m.do(ctx, c, "GET", "/v1/mandates/"+a.MandateID, nil, &v); e != nil {
		return nil, e
	}
	out := map[string]any{"mandate_id": v.MandateID, "target": v.Target, "state": v.State, "last": v.Last,
		"still_running": v.StillRunning}
	if v.Reason != "" {
		out["reason"] = v.Reason
	}
	if v.Ack != nil {
		text, cut := v.Ack.Text, v.Ack.Truncated
		if len(text) > MCPMaxResultText {
			text, cut = strings.ToValidUTF8(text[:MCPMaxResultText], ""), true
		}
		out["receipt"], out["text"], out["ack_seen"], out["truncated"] = v.Ack.Receipt, text, v.Ack.AckSeen, cut
	}
	return out, nil
}

// --- tools/list ------------------------------------------------------------------------------

func obj(props map[string]any, required ...string) map[string]any {
	s := map[string]any{"type": "object", "properties": props, "additionalProperties": false}
	if len(required) > 0 {
		s["required"] = required
	}
	return s
}

var idSchema = map[string]any{"type": "string", "minLength": 1, "maxLength": MCPMaxIDBytes, "pattern": mandateIDRE.String(),
	"description": "unique in the whole command log of ixmux"}

var mcpTools = []map[string]any{
	{"name": "ixmux_status", "title": "ixmux status",
		"description": "The ixmux server and each session it manages: name, kind, incarnation, the deliveries in flight. Metadata only.",
		"inputSchema": obj(map[string]any{})},
	{"name": "ixmux_submit", "title": "Submit a mandate",
		"description": "Hand a mandate (a text) to a session ixmux manages, at most once per mandate_id. Answers a receipt within 2 s, even when the turn lasts longer; accepted is true only once the mandate is in ixmux's command log. Read the outcome with ixmux_result.",
		"inputSchema": obj(map[string]any{
			"target":      map[string]any{"type": "string", "description": "the session's name in ixmux_status"},
			"mandate_id":  idSchema,
			"text":        map[string]any{"type": "string", "maxLength": deliver.MaxText, "description": "the mandate, at most 64 KiB"},
			"incarnation": map[string]any{"type": "string", "description": "the session incarnation it is meant for; default the current one. Give the original one when submitting again after a restart."},
		}, "target", "mandate_id", "text")},
	{"name": "ixmux_result", "title": "Result of a mandate",
		"description": "The state of a mandate by its mandate_id and, once delivered, its result: receipt, the answer's text (at most 4 KiB, truncated says it was cut), ack_seen. Read from ixmux's command log: it survives a restart of the adapter and of ixmux.",
		"inputSchema": obj(map[string]any{"mandate_id": idSchema}, "mandate_id")},
}
