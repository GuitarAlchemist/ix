package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// S6E oracle 2: a mac that is malformed, empty, missing, or the mac of another nonce (a replay):
// refused ixmux-unverified, and the token is never sent.
func TestBadOrReplayedMACIsRefused(t *testing.T) {
	other := make([]byte, server.HelloNonceBytes)
	rand.Read(other)
	for name, answer := range map[string]string{
		"malformed": `{"mac":"zz"}`,
		"empty":     `{"mac":""}`,
		"missing":   `{}`,
		"replayed":  `{"mac":"` + hex.EncodeToString(server.HelloMAC([]byte(fakeToken), other)) + `"}`,
		"not json":  `mac`,
	} {
		t.Run(name, func(t *testing.T) {
			var sawAuth atomic.Int32
			state := rawIxmux(t, func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Authorization") != "" {
					sawAuth.Add(1)
				}
				if r.URL.Path == "/v1/hello" {
					w.Write([]byte(answer))
					return
				}
				w.Write([]byte(statusJSON))
			})
			rc := newInproc(t, state).tool("ixmux_status", map[string]any{})
			if !rc.IsError || rc.Structured["error"] != reasonUnverified {
				t.Errorf("hello answered %s: %v, want isError %s", answer, rc.Structured, reasonUnverified)
			}
			if sawAuth.Load() != 0 {
				t.Error("the token was sent after a bad hello")
			}
		})
	}
}

// S6E oracle 4: one TCP connection per request, the hello first on it, then the authenticated
// request; ixmux_status and ixmux_result are one request each, so one connection per tool call.
// ixmux_submit makes several requests (its POST holds a connection for the whole turn): each on its
// own connection, each after its own hello.
func TestOneConnectionHelloFirst(t *testing.T) {
	var mu sync.Mutex
	conns := map[string][]string{} // remote address -> the requests on it, in order ("hello", "auth ...", "bare ...")
	state := rawIxmux(t, func(w http.ResponseWriter, r *http.Request) {
		kind := "bare " + r.URL.Path
		if r.Header.Get("Authorization") != "" {
			kind = "auth " + r.URL.Path
		}
		if r.URL.Path == "/v1/hello" {
			kind = "hello"
		}
		mu.Lock()
		conns[r.RemoteAddr] = append(conns[r.RemoteAddr], kind)
		mu.Unlock()
		switch {
		case r.URL.Path == "/v1/hello":
			answerHello(w, r, []byte(fakeToken))
		case r.URL.Path == "/v1/status":
			w.Write([]byte(statusJSON))
		default:
			w.Write([]byte(`{"target":"c","mandate_id":"m1","state":"delivered"}`))
		}
	})
	p := newInproc(t, state)
	for _, c := range []struct {
		name  string
		args  map[string]any
		conns int // -1: at least one
	}{{"ixmux_status", map[string]any{}, 1}, {"ixmux_result", map[string]any{"mandate_id": "m1"}, 1},
		{"ixmux_submit", map[string]any{"target": "c", "mandate_id": "m1", "text": "x"}, -1}} {
		mu.Lock()
		clear(conns)
		mu.Unlock()
		if rc := p.tool(c.name, c.args); rc.IsError {
			t.Fatalf("%s: %v", c.name, rc.Structured)
		}
		mu.Lock()
		if c.conns > 0 && len(conns) != c.conns {
			t.Errorf("%s: %d connections %v, want %d", c.name, len(conns), conns, c.conns)
		}
		for addr, reqs := range conns {
			if len(reqs) != 2 || reqs[0] != "hello" || !strings.HasPrefix(reqs[1], "auth ") {
				t.Errorf("%s: connection %s carried %v, want the hello, then one authenticated request", c.name, addr, reqs)
			}
		}
		mu.Unlock()
	}
}

// S6E oracle 5, against the real ixmux: /v1/hello answers the mac of the derived key, refuses a
// malformed nonce typed and an Origin; its mac is no signature of a cursor (the cursors are signed
// by another key), even for a nonce chosen to be a cursor's payload.
func TestHelloEndpoint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	os.WriteFile(path, []byte(`{"n":1}`+"\n"+`{"n":2}`+"\n"), 0o600)
	p := start(t, filepath.Join(dir, "state"), nil, "--journal", "j="+path)
	hello := func(body string, origin bool) (int, map[string]string) {
		req, _ := http.NewRequest("POST", "http://"+p.addr+"/v1/hello", strings.NewReader(body))
		if origin {
			req.Header.Set("Origin", "http://127.0.0.1")
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]string
		json.NewDecoder(resp.Body).Decode(&out)
		return resp.StatusCode, out
	}
	good := strings.Repeat("ab", server.HelloNonceBytes)
	for _, bad := range []string{`{"nonce":"xyz"}`, `{"nonce":"` + strings.ToUpper(good) + `"}`, `{"nonce":"` + good[2:] + `"}`,
		`{"nonce":"` + good + `","extra":1}`, `{}`, `nonce`} {
		if code, out := hello(bad, false); code != http.StatusBadRequest || out["error"] != "bad-nonce" {
			t.Errorf("hello %s: %d %v, want 400 bad-nonce", bad, code, out)
		}
	}
	if code, _ := hello(`{"nonce":"`+good+`"}`, true); code != http.StatusForbidden {
		t.Errorf("hello with an Origin: %d, want 403", code)
	}
	// The mac, computed here from the definition: HMAC-SHA-256(HMAC-SHA-256(token, domain), nonce).
	macOf := func(nonce []byte) []byte {
		k := hmac.New(sha256.New, []byte(p.token))
		k.Write([]byte("ixmux/hello/v1"))
		m := hmac.New(sha256.New, k.Sum(nil))
		m.Write(nonce)
		return m.Sum(nil)
	}
	nonce, _ := hex.DecodeString(good)
	if code, out := hello(`{"nonce":"`+good+`"}`, false); code != 200 || out["mac"] != hex.EncodeToString(macOf(nonce)) {
		t.Errorf("hello: %d %v, want the mac of the derived key", code, out)
	}

	// A nonce that is a cursor's payload: 24 bytes of JSON, 32 of base64url. Its mac is offered as
	// the cursor's signature: ignored.
	js := []byte(`{"j":"j","c":{}}`)
	js = append(js, bytes.Repeat([]byte(" "), 24-len(js))...)
	payload := base64.RawURLEncoding.EncodeToString(js)
	code, out := hello(`{"nonce":"`+hex.EncodeToString([]byte(payload))+`"}`, false)
	mac, _ := hex.DecodeString(out["mac"])
	if code != 200 || len(mac) == 0 {
		t.Fatalf("hello for a cursor payload: %d %v", code, out)
	}
	var page struct {
		CursorIgnored bool `json:"cursor_ignored"`
	}
	cursor := payload + "." + base64.RawURLEncoding.EncodeToString(mac)
	p.do("GET", "/v1/journal/j?from=0&cursor="+url.QueryEscape(cursor), nil, &page)
	if !page.CursorIgnored {
		t.Error("a hello mac was taken as the signature of a cursor")
	}
}
