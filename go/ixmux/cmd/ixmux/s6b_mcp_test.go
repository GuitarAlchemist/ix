package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// The S6B oracles of the Codex round 8 drive the adapter in-process (m.handle) against a fake
// ixmux: an HTTP server on loopback whose ixmux.json and token are written in a state directory.

const fakeToken = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

// fakeIxmux serves h on loopback and writes the state directory the adapter reads.
func fakeIxmux(t *testing.T, h http.HandlerFunc) string {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	state := t.TempDir()
	info, _ := json.Marshal(server.Info{Addr: strings.TrimPrefix(srv.URL, "http://"), State: "running"})
	os.WriteFile(filepath.Join(state, server.InfoFile), info, 0o600)
	os.WriteFile(filepath.Join(state, server.TokenFile), []byte(fakeToken), 0o600)
	return state
}

// release returns a channel closed when the test ends. Call it after fakeIxmux: cleanups run last
// first, so the channel is closed before the fake server's Close waits for its held handlers.
func release(t *testing.T) chan struct{} {
	c := make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(c) }) })
	return c
}

type inproc struct {
	t   *testing.T
	m   *mcp
	out *bytes.Buffer
	id  int
}

func newInproc(t *testing.T, state string) *inproc {
	out := &bytes.Buffer{}
	return &inproc{t: t, m: newMCP(state, out), out: out}
}

// raw sends one line and returns the answers it produced (none for a notification).
func (p *inproc) raw(line string) []rpcResponse {
	p.t.Helper()
	p.out.Reset()
	p.m.handle([]byte(line))
	var rs []rpcResponse
	for _, l := range bytes.Split(bytes.TrimSpace(p.out.Bytes()), []byte("\n")) {
		if len(l) == 0 {
			continue
		}
		var r rpcResponse
		if err := json.Unmarshal(l, &r); err != nil {
			p.t.Fatalf("not JSON-RPC: %q", l)
		}
		rs = append(rs, r)
	}
	return rs
}

func (p *inproc) tool(name string, args map[string]any) toolResult {
	p.t.Helper()
	p.id++
	b, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": p.id, "method": "tools/call", "params": map[string]any{"name": name, "arguments": args}})
	rs := p.raw(string(b))
	if len(rs) != 1 || rs[0].Error != nil {
		p.t.Fatalf("%s: %+v", name, rs)
	}
	var tr toolResult
	json.Unmarshal(rs[0].Result, &tr)
	return tr
}

// statusJSON is a /v1/status of one claude target c.
const statusJSON = `{"version":"fake","epoch":"e","targets":[{"name":"c","kind":"claude","incarnation":"inc-1"}],"deliveries":[],"bounds":{"delivery_timeout":"30s"}}`

// S6B Codex 6: only the versions whose behaviour is held are negotiated, from 2025-06-18: an
// older one asked gets the latest (batches, refused here, belong to 2025-03-26 and before).
func TestOnlyHeldVersionsAreNegotiated(t *testing.T) {
	p := newInproc(t, t.TempDir())
	for asked, want := range map[string]string{"2025-11-25": "2025-11-25", "2025-06-18": "2025-06-18",
		"2025-03-26": "2025-11-25", "2024-11-05": "2025-11-25", "1999-01-01": "2025-11-25"} {
		rs := p.raw(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"` + asked +
			`","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}`)
		var v struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		if len(rs) == 1 {
			json.Unmarshal(rs[0].Result, &v)
		}
		if v.ProtocolVersion != want {
			t.Errorf("asked %s: answered %q, want %s", asked, v.ProtocolVersion, want)
		}
	}
}

// S6B Codex 5: JSON-RPC 2.0 as written: an id is a string or a number; id null is no
// notification; an invalid request is answered -32600 even without an id; params are structured;
// initialize has its required fields (-32602).
func TestJSONRPCValidation(t *testing.T) {
	p := newInproc(t, t.TempDir())
	init := `"params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}`
	for _, c := range []struct {
		line   string
		code   int    // 0: a result
		id     string // the id of the answer, "" for none at all
		answer bool
	}{
		{`{"jsonrpc":"2.0","id":"a","method":"ping"}`, 0, `"a"`, true},
		{`{"jsonrpc":"2.0","id":7,"method":"ping"}`, 0, `7`, true},
		{`{"jsonrpc":"2.0","id":true,"method":"ping"}`, -32600, `null`, true},
		{`{"jsonrpc":"2.0","id":{},"method":"ping"}`, -32600, `null`, true},
		{`{"jsonrpc":"2.0","id":[1],"method":"ping"}`, -32600, `null`, true},
		{`{"jsonrpc":"2.0","id":null,"method":"ping"}`, -32600, `null`, true},
		{`{"jsonrpc":"2.0","method":5}`, -32600, `null`, true},
		{`{"id":1,"method":"ping"}`, -32600, `1`, true},
		{`"a string"`, -32600, `null`, true},
		{`{"jsonrpc":"2.0","id":1,"method":"ping","params":null}`, -32600, `1`, true},
		{`{"jsonrpc":"2.0","id":1,"method":"ping","params":3}`, -32600, `1`, true},
		{`{"jsonrpc":"2.0","method":"notifications/initialized"}`, 0, "", false},
		{`{"jsonrpc":"2.0","id":1,"method":"initialize","params":[]}`, -32602, `1`, true},
		{`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`, -32602, `1`, true},
		{`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":5,"capabilities":{},"clientInfo":{"name":"t","version":"0"}}}`, -32602, `1`, true},
		{`{"jsonrpc":"2.0","id":1,"method":"initialize",` + init + `}`, 0, `1`, true},
		{`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ixmux_status","arguments":[]}}`, -32602, `1`, true},
		{`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"arguments":{}}}`, -32602, `1`, true},
	} {
		rs := p.raw(c.line)
		switch {
		case !c.answer:
			if len(rs) != 0 {
				t.Errorf("%s: %d answers, want none", c.line, len(rs))
			}
			continue
		case len(rs) != 1:
			t.Errorf("%s: %d answers, want one", c.line, len(rs))
			continue
		}
		r := rs[0]
		got := 0
		if r.Error != nil {
			got = r.Error.Code
		}
		if got != c.code || string(r.ID) != c.id {
			t.Errorf("%s: code %d id %s, want code %d id %s", c.line, got, r.ID, c.code, c.id)
		}
	}
}

// S6B Codex 3: a local service that took the port and echoes the Authorization header: no text
// from the network reaches stdout with the token's bytes in it.
func TestEchoedTokenIsNeverShown(t *testing.T) {
	state := fakeIxmux(t, func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if r.URL.Path == "/v1/status" {
			b, _ := json.Marshal(map[string]any{"version": auth, "targets": []any{map[string]any{"name": "c", "kind": auth, "incarnation": auth}}})
			w.Write(b)
			return
		}
		w.WriteHeader(http.StatusConflict)
		b, _ := json.Marshal(map[string]string{"error": "echo " + auth, "detail": "you sent " + auth})
		w.Write(b)
	})
	p := newInproc(t, state)
	var all bytes.Buffer
	for _, c := range []struct {
		name string
		args map[string]any
	}{{"ixmux_status", map[string]any{}}, {"ixmux_result", map[string]any{"mandate_id": "m1"}},
		{"ixmux_submit", map[string]any{"target": "c", "mandate_id": "m1", "text": "x", "incarnation": "i"}}} {
		p.tool(c.name, c.args)
		all.Write(p.out.Bytes())
	}
	if bytes.Contains(all.Bytes(), []byte(fakeToken)) {
		t.Errorf("the token echoed by the network is in the adapter's output:\n%s", all.String())
	}
}

// S6B Codex 2: POSTs left running are bounded: at most MCPMaxPosts at once (beyond, a typed
// refusal), and each within ixmux's delivery deadline plus a margin, after which its slot is free.
func TestDetachedPOSTsAreBounded(t *testing.T) {
	var hold chan struct{}
	state := fakeIxmux(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/status":
			w.Write([]byte(strings.Replace(statusJSON, `"30s"`, `"1ms"`, 1)))
		case r.Method == "POST":
			select { // accepted, never answered
			case <-hold:
			case <-r.Context().Done():
			}
		default:
			w.Write([]byte(`{"target":"c","mandate_id":"x","state":"accepted"}`))
		}
	})
	hold = release(t)
	old := mcpPostExtra
	mcpPostExtra = 300 * time.Millisecond // the POST's deadline: 1ms of ixmux, plus this
	t.Cleanup(func() { mcpPostExtra = old })
	p := newInproc(t, state)
	for i := range MCPMaxPosts {
		if rc := p.tool("ixmux_submit", map[string]any{"target": "c", "mandate_id": fmt.Sprintf("m%d", i), "text": "x"}); rc.IsError {
			t.Fatalf("submit %d: %v", i, rc.Structured)
		}
	}
	if rc := p.tool("ixmux_submit", map[string]any{"target": "c", "mandate_id": "over", "text": "x"}); !rc.IsError || rc.Structured["error"] != "submits-in-flight" {
		t.Errorf("one POST over MCPMaxPosts: %v, want isError submits-in-flight", rc.Structured)
	}
	deadline := time.Now().Add(10 * time.Second) // a guard: the POSTs end at their 300 ms deadline
	for {
		rc := p.tool("ixmux_submit", map[string]any{"target": "c", "mandate_id": "after", "text": "x"})
		if !rc.IsError {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("no slot freed after the POSTs' deadline: %v", rc.Structured)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// S6B Codex 1: a POST cut at the transport is no proof of anything: the adapter goes on reading
// the entry until the receipt's bound, and any durable state proves the mandate was accepted.
func TestReceiptAfterACutPOST(t *testing.T) {
	for _, tc := range []struct {
		name, entry string
		cut         bool // the POST's connection is closed without an answer; else it is held
	}{
		{"cut, then delivered", `{"target":"c","mandate_id":"m1","state":"delivered"}`, true},
		{"held, uncertain", `{"target":"c","mandate_id":"m1","state":"uncertain"}`, false},
		{"held, refused", `{"target":"c","mandate_id":"m1","state":"refused"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var hold chan struct{}
			state := fakeIxmux(t, func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.URL.Path == "/v1/status":
					w.Write([]byte(statusJSON))
				case r.Method == "POST" && tc.cut:
					c, _, _ := http.NewResponseController(w).Hijack()
					c.Close()
				case r.Method == "POST":
					<-hold
				case r.URL.Path == "/v1/mandates/m1":
					w.Write([]byte(tc.entry))
				}
			})
			hold = release(t)
			p := newInproc(t, state)
			rc := p.tool("ixmux_submit", map[string]any{"target": "c", "mandate_id": "m1", "text": "x"})
			var want map[string]any
			json.Unmarshal([]byte(tc.entry), &want)
			if rc.IsError || rc.Structured["accepted"] != true || rc.Structured["state"] != want["state"] {
				t.Errorf("receipt: %v, want accepted with state %v", rc.Structured, want["state"])
			}
		})
	}
}
