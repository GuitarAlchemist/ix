package main

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"testing"
)

// leak names the first 8-byte window of the token, raw, in base64 or in hex, found in out.
func leak(out []byte) string {
	for _, form := range []string{fakeToken, base64.StdEncoding.EncodeToString([]byte(fakeToken)), hex.EncodeToString([]byte(fakeToken))} {
		for i := 0; i+8 <= len(form); i++ {
			if bytes.Contains(out, []byte(form[i:i+8])) {
				return form[i : i+8]
			}
		}
	}
	return ""
}

// S6C P1-3 (Codex 9): text from the network that holds the token in pieces or encoded, in a
// refusal's code or detail, or in a field of a 200 answer, never reaches stdout.
func TestTokenPiecesAreNeverShown(t *testing.T) {
	b64 := base64.StdEncoding.EncodeToString([]byte(fakeToken))
	for name, pc := range map[string][2]string{
		"prefix": {fakeToken[:12], ""},
		"halves": {fakeToken[:32], fakeToken[32:]},
		"base64": {b64, ""},
	} {
		t.Run(name, func(t *testing.T) {
			state := fakeIxmux(t, func(w http.ResponseWriter, r *http.Request) {
				var v any
				switch {
				case r.URL.Path == "/v1/status":
					v = map[string]any{"version": pc[0], "epoch": pc[1], "bounds": map[string]any{"delivery_timeout": "30s"},
						"targets": []any{map[string]any{"name": "c", "kind": pc[1], "incarnation": "inc-1"}}}
				case r.Method == "GET":
					v = map[string]any{"target": "c", "mandate_id": "m1", "state": "uncertain", "reason": pc[0] + " " + pc[1]}
				default:
					w.WriteHeader(http.StatusConflict)
					v = map[string]any{"error": "x-" + pc[0], "detail": pc[1] + " " + pc[0]}
				}
				b, _ := json.Marshal(v)
				w.Write(b)
			})
			p := newInproc(t, state)
			var all bytes.Buffer
			for _, c := range []struct {
				name string
				args map[string]any
			}{{"ixmux_status", map[string]any{}}, {"ixmux_result", map[string]any{"mandate_id": "m1"}},
				{"ixmux_submit", map[string]any{"target": "c", "mandate_id": "m1", "text": "x"}}} {
				p.tool(c.name, c.args)
				all.Write(p.out.Bytes())
			}
			if w := leak(all.Bytes()); w != "" {
				t.Errorf("a piece of the token (%q) is in the adapter's output:\n%s", w, all.String())
			}
		})
	}
}

// S6C P1-4 (Codex 9): arguments absent is {}; arguments null, or any type but an object, is
// -32602, as MCP.md says.
func TestNullArgumentsAreRefused(t *testing.T) {
	p := newInproc(t, t.TempDir())
	for args, want := range map[string]int{`,"arguments":null`: -32602, `,"arguments":[]`: -32602, `,"arguments":"x"`: -32602, ``: 0} {
		rs := p.raw(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ixmux_status"` + args + `}}`)
		got := 0
		if len(rs) == 1 && rs[0].Error != nil {
			got = rs[0].Error.Code
		}
		if len(rs) != 1 || got != want {
			t.Errorf("tools/call with %q: %+v, want code %d", args, rs, want)
		}
	}
}

// S6C P1-3: only ixmux's known refusal codes pass, without their detail; an unknown code is
// ixmux-error.
func TestOnlyKnownCodesPass(t *testing.T) {
	for code, want := range map[string]string{"target-busy": "target-busy", "made-up-code": reasonIxmuxError} {
		t.Run(code, func(t *testing.T) {
			state := fakeIxmux(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusConflict)
				b, _ := json.Marshal(map[string]string{"error": code, "detail": "a detail from the network"})
				w.Write(b)
			})
			rc := newInproc(t, state).tool("ixmux_result", map[string]any{"mandate_id": "m1"})
			if !rc.IsError || rc.Structured["error"] != want {
				t.Errorf("a refusal %q: %v, want isError %s", code, rc.Structured, want)
			}
			if d, _ := rc.Structured["detail"].(string); bytes.Contains([]byte(d), []byte("from the network")) || bytes.Contains([]byte(d), []byte(code)) && want != code {
				t.Errorf("the network's text passes in detail: %q", d)
			}
		})
	}
}
