package main

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
	"testing"
)

// S6E oracle 1 (Codex 11): a hostile local service holds ixmux's port. It fails the hello and
// echoes the Authorization header in version. Every tool answers ixmux-unverified, and the
// service never receives an Authorization header.
func TestHostileServerNeverGetsTheToken(t *testing.T) {
	var sawAuth atomic.Int32
	state := rawIxmux(t, func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "" {
			sawAuth.Add(1)
		}
		if r.URL.Path == "/v1/hello" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		b, _ := json.Marshal(map[string]any{"version": auth, "epoch": "e", "bounds": map[string]any{"delivery_timeout": "30s"},
			"targets": []any{map[string]any{"name": "c", "kind": "claude", "incarnation": "i"}}})
		w.Write(b)
	})
	p := newInproc(t, state)
	for _, c := range []struct {
		name string
		args map[string]any
	}{{"ixmux_status", map[string]any{}}, {"ixmux_result", map[string]any{"mandate_id": "m1"}},
		{"ixmux_submit", map[string]any{"target": "c", "mandate_id": "m1", "text": "x", "incarnation": "i"}}} {
		rc := p.tool(c.name, c.args)
		if !rc.IsError || rc.Structured["error"] != "ixmux-unverified" {
			t.Errorf("%s against a server that fails the hello: %v, want isError ixmux-unverified", c.name, rc.Structured)
		}
	}
	if n := sawAuth.Load(); n != 0 {
		t.Errorf("the hostile server received the Authorization header %d times", n)
	}
}
