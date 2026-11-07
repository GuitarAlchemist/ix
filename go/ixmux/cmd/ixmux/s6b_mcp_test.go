package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

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
