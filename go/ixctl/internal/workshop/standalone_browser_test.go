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
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// siteFixture serves the standalone app as GitHub Pages would (spareilleux.github.io/apps/ixforge/: a folder of a
// shared origin, no custom headers), and a fake Claude API on the same origin, so no real request, key or cost is
// involved. Every request is counted.
type siteFixture struct {
	origin, chrome, pin string
	app                 string       // the folder served at /apps/ixforge/
	apiCalls            atomic.Int64 // requests to the Desktop API, which a standalone app must never make
	unreachable         atomic.Bool
	claude              atomic.Value // the fake Claude's next answer: "ok", "unauthorized" or "refusal"
	mu                  sync.Mutex
	asked               []claudeCall
}

type claudeCall struct {
	Header http.Header
	Body   map[string]any
}

func newSiteFixture(t *testing.T) *siteFixture {
	t.Helper()
	if os.Getenv("WORKSHOP_CDP") != "1" {
		t.Skip("set WORKSHOP_CDP=1 for the fresh-profile headless Chrome checks of the standalone app")
	}
	f := &siteFixture{chrome: os.Getenv("CHROME")}
	if f.chrome == "" {
		f.chrome = `C:\Program Files\Google\Chrome\Application\chrome.exe`
	}
	f.claude.Store("ok")
	dir := t.TempDir()
	f.app = filepath.Join(dir, "apps", "ixforge")
	if err := WriteApp(f.app); err != nil {
		t.Fatal(err)
	}
	files := http.FileServer(http.Dir(dir))
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.unreachable.Load() {
			if conn, _, err := http.NewResponseController(w).Hijack(); err == nil {
				conn.Close()
			}
			return
		}
		if strings.Contains(r.URL.Path, "/api/") {
			f.apiCalls.Add(1)
			http.NotFound(w, r)
			return
		}
		if r.URL.Path == "/fake-anthropic/v1/messages" {
			f.answer(w, r)
			return
		}
		w.Header().Set("Cache-Control", "max-age=600") // what GitHub Pages sends for every file
		files.ServeHTTP(w, r)
	}))
	t.Cleanup(server.Close)
	u, _ := url.Parse(server.URL)
	if port, _ := strconv.Atoi(u.Port()); port >= 8737 && port <= 8740 {
		t.Fatal("production port refused")
	}
	f.origin = "https://workshop.invalid:" + u.Port()
	sum := sha256.Sum256(server.Certificate().RawSubjectPublicKeyInfo)
	f.pin = base64.StdEncoding.EncodeToString(sum[:])
	return f
}

func (f *siteFixture) answer(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var request map[string]any
	_ = json.Unmarshal(body, &request)
	f.mu.Lock()
	f.asked = append(f.asked, claudeCall{r.Header.Clone(), request})
	f.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	switch f.claude.Load() {
	case "unauthorized":
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`))
	case "refusal":
		w.Write([]byte(`{"type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":"refusal","stop_details":{"type":"refusal","category":null}}`))
	default:
		tool, _ := json.Marshal(map[string]any{"kind": "checklist", "title": "Camping", "value": 0,
			"items": []map[string]any{{"text": "Tente", "done": false}, {"text": "Lampe frontale", "done": false}}})
		reply, _ := json.Marshal(map[string]any{"type": "message", "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "end_turn",
			"content": []map[string]any{{"type": "text", "text": string(tool)}}})
		w.Write(reply)
	}
}

func (f *siteFixture) open(t *testing.T) (*cdpPage, string, func()) {
	t.Helper()
	profile := t.TempDir()
	page, quit := startChromeIn(t, f.chrome, f.pin, profile)
	page.navigate(f.origin + "/apps/ixforge/")
	settle(t, page)
	return page, profile, quit
}

// TestTheStandaloneAppIsInstallableAndWorksAlone: a friend opens the site and installs it. Without any Desktop,
// IxForge runs on that device only and says so, offers no pairing, keeps working with the site unreachable, and
// never calls a Desktop API.
func TestTheStandaloneAppIsInstallableAndWorksAlone(t *testing.T) {
	f := newSiteFixture(t)
	page, profile, quit := f.open(t)
	page.until("the app runs from its folder", `location.pathname==='/apps/ixforge/' && document.querySelector('meta[http-equiv="Content-Security-Policy"]')!==null`)

	var install struct {
		InstallabilityErrors []struct{ ErrorID string }
	}
	raw := page.call("Page.getInstallabilityErrors", map[string]any{})
	if err := json.Unmarshal(raw, &install); err != nil || len(install.InstallabilityErrors) != 0 {
		t.Fatalf("Chrome must find the app installable: %s", raw)
	}

	page.until("on this device only", `document.getElementById('connection').textContent==='Sur cet appareil seulement'`)
	page.until("no AI without a key", `document.getElementById('ai-create').hidden===true`)
	page.click("devices-open")
	page.until("settings, with no pairing", `document.getElementById('devices').hidden===false && document.getElementById('devices-open').getAttribute('aria-label')==='Réglages' && document.getElementById('pair-panel').hidden===true && document.getElementById('admin').hidden===true && document.getElementById('ai-panel').hidden===false`)
	page.click("devices-close")
	newTool(t, page, "Épicerie", "new-list")
	addItem(t, page, "Lait")
	page.screenshot("autonome.png")

	// Reopen with the site unreachable: the installed app comes from its cache, the list from the browser.
	quit()
	f.unreachable.Store(true)
	page, _ = startChromeIn(t, f.chrome, f.pin, profile)
	page.navigate(f.origin + "/apps/ixforge/")
	page.until("boot with the site unreachable", clientBooted)
	openTool(t, page, "Épicerie")
	page.until("works with the site unreachable", shownList+`==='[ ] Lait'`)
	if n := f.apiCalls.Load(); n != 0 {
		t.Errorf("a standalone app must never call a Desktop API: %d requests", n)
	}
}

// TestANewVersionOfTheSiteReplacesTheOldOne: GitHub Pages lets browsers keep every file for ten minutes. When a new
// version is published within those ten minutes, the installed app must still end up with the new files, in its
// offline copy and on screen, not the old files under the new version's name.
func TestANewVersionOfTheSiteReplacesTheOldOne(t *testing.T) {
	f := newSiteFixture(t)
	page, profile, quit := f.open(t)

	appJS := filepath.Join(f.app, "app.js")
	old, err := os.ReadFile(appJS)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(appJS, append(old, []byte("\n// ixforge-new-version\n")...), 0o644); err != nil {
		t.Fatal(err)
	}
	sw := filepath.Join(f.app, "sw.js")
	worker, _ := os.ReadFile(sw)
	newWorker := cacheLine.ReplaceAll(worker, []byte("const CACHE = 'ix-workshop-shell-new-version';"))
	if string(newWorker) == string(worker) {
		t.Fatal("the worker's cache name was not found")
	}
	if err := os.WriteFile(sw, newWorker, 0o644); err != nil {
		t.Fatal(err)
	}

	// The new cache exists from the start of the install, before its files are in it: wait for app.js itself.
	const offlineCopy = `(async()=>{for(let i=0;i<150;i++){if((await caches.keys()).includes('ix-workshop-shell-new-version')){const kept=await (await caches.open('ix-workshop-shell-new-version')).match('./app.js');if(kept)return kept.text()}await new Promise(d=>setTimeout(d,100))}return 'no new version installed'})()`
	page.navigate(f.origin + "/apps/ixforge/")
	var copied string
	page.evalInto(offlineCopy, &copied)
	if !strings.Contains(copied, "ixforge-new-version") {
		t.Fatalf("the new version's offline copy holds the old app.js (%d bytes, %q…)", len(copied), copied[:min(len(copied), 40)])
	}

	// Reopened, the new version runs and serves the new file, and keeps it offline.
	quit()
	page, _ = startChromeIn(t, f.chrome, f.pin, profile)
	page.navigate(f.origin + "/apps/ixforge/")
	page.until("boot on the new version", clientBooted)
	var served, kept string
	page.evalInto(`fetch('app.js').then(r=>r.text())`, &served)
	page.evalInto(offlineCopy, &kept)
	if !strings.Contains(served, "ixforge-new-version") || !strings.Contains(kept, "ixforge-new-version") {
		t.Errorf("the new version must serve and keep the new app.js: served new=%v, kept new=%v", strings.Contains(served, "ixforge-new-version"), strings.Contains(kept, "ixforge-new-version"))
	}
}

// TestTheOptionalAIUsesTheUsersOwnKey: with no key, IxForge has no AI and lacks nothing else. With the person's own
// Claude key, kept on the device, a sentence becomes a tool. The request goes to Claude straight from the browser
// with that key, asks for the tool as JSON matching a schema, and what comes back is validated before anything is
// created. A refused key or a declined request says so and creates nothing. Forgetting the key removes it.
func TestTheOptionalAIUsesTheUsersOwnKey(t *testing.T) {
	f := newSiteFixture(t)
	page, _, _ := f.open(t)
	const key = "sk-ant-test-0123456789abcdef"
	page.click("devices-open")
	page.typeInto("ai-key", key)
	page.click("ai-save")
	page.until("key kept, not shown", `document.getElementById('ai-key').value==='' && document.getElementById('ai-state').textContent.includes('cdef') && !document.getElementById('ai-state').textContent.includes('0123456789')`)
	page.click("devices-close")
	page.until("AI offered", `document.getElementById('ai-create').hidden===false`)
	// Tests talk to the fake Claude on the site's own origin; people never see or set this.
	page.evalInto(`import('./sync.js').then(m=>m.Replica.open()).then(r=>r.change(s=>{s.aiEndpoint='/fake-anthropic/v1/messages'})).then(()=>true)`, nil)

	page.typeInto("new-name", "Ce qu'il faut emporter pour le camping")
	page.press("#ai-create")
	page.until("a tool made by Claude", `document.getElementById('tool-title').value==='Camping' && `+shownList+`==='[ ] Tente, [ ] Lampe frontale' && document.querySelector('#widget-host [data-xlet-local]')!==null && document.getElementById('new-name').value===''`)
	page.screenshot("ia-camping.png")

	f.mu.Lock()
	call := f.asked[0]
	f.mu.Unlock()
	if call.Header.Get("x-api-key") != key || call.Header.Get("anthropic-version") == "" || call.Header.Get("anthropic-dangerous-direct-browser-access") != "true" {
		t.Errorf("the request must carry the person's key and Claude's headers: %v", call.Header)
	}
	format, _ := call.Body["output_config"].(map[string]any)["format"].(map[string]any)
	messages, _ := json.Marshal(call.Body["messages"])
	if call.Body["model"] != "claude-opus-5-5" || format["type"] != "json_schema" || !strings.Contains(string(messages), "camping") {
		t.Errorf("the request must ask Claude for the tool as JSON: %v", call.Body)
	}

	f.claude.Store("unauthorized")
	page.typeInto("new-name", "Compter les verres d'eau")
	page.press("#ai-create")
	page.until("a refused key says so", `document.getElementById('notice').textContent.startsWith('Claude refuse cette clé') && document.querySelectorAll('#tools [data-tool]').length===1`)
	f.claude.Store("refusal")
	page.press("#ai-create")
	page.until("a declined request says so", `document.getElementById('notice').textContent.startsWith('Claude n’a pas fabriqué cet outil') && document.querySelectorAll('#tools [data-tool]').length===1`)

	page.click("devices-open")
	page.click("ai-forget")
	page.until("key forgotten", `document.getElementById('ai-create').hidden===true`)
	var kept string
	page.evalInto(`import('./sync.js').then(m=>m.Replica.open()).then(r=>r.read()).then(s=>(s.ai&&s.ai.key)||'')`, &kept)
	if kept != "" {
		t.Errorf("forgetting the key must remove it from the device")
	}
}
