package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// S6D P1-1 (Codex 10): no free text from the network reaches a tool answer. A fragment of the
// token encoded alone (base64 of 8 of its bytes) in a refusal's detail, a mandate's reason or a
// target's error: none of it comes out. A refusal answers a known code with the adapter's own
// message.
func TestNoFreeTextFromTheNetwork(t *testing.T) {
	// A token with no period (fakeToken repeats every 16 bytes, so any of its fragments falls on a
	// base64 group of the whole), and a fragment not on a group boundary.
	const tok = "9f3c2a71e0b84d56c1a7f02e8b6d4935a0e7c3f19b2d58e64c0a7f3e1d9b2c86"
	frag := base64.StdEncoding.EncodeToString([]byte(tok[1:9]))
	var state string
	state = fakeIxmux(t, func(w http.ResponseWriter, r *http.Request) {
		var v any
		switch {
		case r.URL.Path == "/v1/status":
			v = map[string]any{"version": "fake", "epoch": "e", "bounds": map[string]any{"delivery_timeout": "30s"},
				"targets": []any{map[string]any{"name": "c", "kind": "claude", "incarnation": "inc-1", "error": "lost: " + frag}}}
		case r.URL.Path == "/v1/mandates/m1":
			v = map[string]any{"target": "c", "mandate_id": "m1", "state": "uncertain", "reason": "adapter error: " + frag, "cause": "adapter-error " + frag}
		default:
			w.WriteHeader(http.StatusConflict)
			v = map[string]any{"error": "target-busy", "detail": frag}
		}
		b, _ := json.Marshal(v)
		w.Write(b)
	})
	os.WriteFile(filepath.Join(state, server.TokenFile), []byte(tok), 0o600)
	p := newInproc(t, state)
	var all bytes.Buffer
	for _, c := range []struct {
		name string
		args map[string]any
	}{{"ixmux_status", map[string]any{}}, {"ixmux_result", map[string]any{"mandate_id": "m1"}},
		{"ixmux_result", map[string]any{"mandate_id": "m2"}}} {
		p.tool(c.name, c.args)
		all.Write(p.out.Bytes())
	}
	if bytes.Contains(all.Bytes(), []byte(frag)) {
		t.Errorf("free text from the network (%q) is in the adapter's output:\n%s", frag, all.String())
	}
	rc := p.tool("ixmux_result", map[string]any{"mandate_id": "m2"})
	if d, _ := rc.Structured["detail"].(string); rc.Structured["error"] != "target-busy" || d == "" {
		t.Errorf("a refusal: %v, want target-busy with the adapter's own message", rc.Structured)
	}
}
