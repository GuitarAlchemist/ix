package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"
)

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
