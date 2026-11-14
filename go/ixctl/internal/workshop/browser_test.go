package workshop

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func TestBrowserIntegration(t *testing.T) {
	if os.Getenv("WORKSHOP_BROWSER") != "1" {
		t.Skip("set WORKSHOP_BROWSER=1 for isolated fresh-profile Chrome integration")
	}
	path := filepath.Join(t.TempDir(), "fixture-workshop.sqlite")
	store, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	defer func() { store.Close() }()
	var origin string
	var handler http.Handler
	version := 1
	proxyDown := false
	key, err := randomID()
	if err != nil {
		t.Fatal(err)
	}
	fixture := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if r.URL.Path != "/fixture" {
			if proxyDown {
				w.WriteHeader(502)
				return
			}
			if version == 2 && (r.URL.Path == "/workshop/sw.js" || r.URL.Path == "/workshop/branding.json") {
				w.Header().Set("Cache-Control", "no-store")
				if r.URL.Path == "/workshop/sw.js" {
					body, _ := assets.ReadFile("web/sw.js")
					w.Header().Set("Content-Type", "text/javascript")
					w.Header().Set("Service-Worker-Allowed", "/workshop/")
					brand, _ := assets.ReadFile("web/branding.json")
					brand = []byte(strings.ReplaceAll(string(brand), "0.1.1", "0.2.0"))
					cache := fixtureCacheName(map[string][]byte{"branding.json": brand})
					w.Write([]byte(regexp.MustCompile(`const CACHE = '[^']+';`).ReplaceAllString(string(body), "const CACHE = '"+cache+"';")))
				} else {
					body, _ := assets.ReadFile("web/branding.json")
					w.Header().Set("Content-Type", "application/json")
					w.Write([]byte(strings.ReplaceAll(string(body), "0.1.1", "0.2.0")))
				}
				return
			}
			handler.ServeHTTP(w, r)
			return
		}
		if r.Method != "POST" || r.Header.Get("X-Workshop-Fixture") != key {
			w.WriteHeader(403)
			return
		}
		var body struct {
			Action string `json:"action"`
			ID     string `json:"id"`
		}
		if json.NewDecoder(io.LimitReader(r.Body, 1024)).Decode(&body) != nil {
			w.WriteHeader(400)
			return
		}
		var result any
		var err error
		switch body.Action {
		case "identity":
			result = map[string]string{"fixture": key}
		case "upgrade":
			version = 2
			result = map[string]bool{"ok": true}
		case "proxy-down":
			proxyDown = true
			result = map[string]bool{"ok": true}
		case "proxy-up":
			proxyDown = false
			result = map[string]bool{"ok": true}
		case "invite":
			var invite Invite
			invite, err = store.NewInvite(origin)
			result = map[string]any{"code": invite.Code, "url": invite.URL, "svg": invite.QRSVG}
		case "devices":
			result, err = store.Devices()
		case "approved", "refused", "revoked":
			err = store.Decide(body.ID, body.Action)
			result = map[string]bool{"ok": err == nil}
		case "restart":
			err = store.Close()
			if err == nil {
				store, err = Open(path)
			}
			if err == nil {
				handler, err = Handler(store, origin)
			}
			result = map[string]bool{"ok": err == nil}
		default:
			w.WriteHeader(400)
			return
		}
		if err != nil {
			t.Log(err)
			w.WriteHeader(500)
			return
		}
		respond(w, 200, result)
	})
	server := httptest.NewTLSServer(fixture)
	defer server.Close()
	u, _ := url.Parse(server.URL)
	port, _ := strconv.Atoi(u.Port())
	if port >= 8737 && port <= 8740 {
		t.Fatal("production port refused")
	}
	origin = "https://workshop.invalid:" + u.Port()
	handler, err = Handler(store, origin)
	if err != nil {
		t.Fatal(err)
	}
	pin := sha256.Sum256(server.Certificate().RawSubjectPublicKeyInfo)
	command := exec.Command("node", "e2e/browser.mjs")
	command.Env = append(os.Environ(), "WORKSHOP_FIXTURE_URL="+server.URL, "WORKSHOP_FIXTURE_ORIGIN="+origin, "WORKSHOP_FIXTURE_PIN="+base64.StdEncoding.EncodeToString(pin[:]), "WORKSHOP_FIXTURE_CERT="+base64.StdEncoding.EncodeToString(server.Certificate().Raw), "WORKSHOP_FIXTURE_KEY="+key, "WORKSHOP_ARTIFACT_DIR="+os.Getenv("WORKSHOP_ARTIFACT_DIR"))
	output, err := command.CombinedOutput()
	t.Log(string(output))
	if err != nil {
		t.Fatal(err)
	}
}
