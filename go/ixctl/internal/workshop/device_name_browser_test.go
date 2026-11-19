package workshop

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// TestDeviceNameIsHonestAboutWhenItIsSent drives the real tablet client in a fresh-profile headless Chrome over the
// DevTools protocol, from Go only (no Node). The device name is sent to Desktop once, with the pairing request; there is
// no rename API. The field must therefore say so, keep the typed name across a reload until pairing, then show the name
// Desktop registered, read-only, instead of an editable field that looks saved but is not.
func TestDeviceNameIsHonestAboutWhenItIsSent(t *testing.T) {
	if os.Getenv("WORKSHOP_CDP") != "1" {
		t.Skip("set WORKSHOP_CDP=1 for the fresh-profile headless Chrome check of the device name field")
	}
	chrome := os.Getenv("CHROME")
	if chrome == "" {
		chrome = `C:\Program Files\Google\Chrome\Application\chrome.exe`
	}
	if _, err := os.Stat(chrome); err != nil {
		t.Fatalf("Chrome not found at %s (set CHROME): %v", chrome, err)
	}
	store, err := Open(filepath.Join(t.TempDir(), "fixture-workshop.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var handler http.Handler
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { handler.ServeHTTP(w, r) }))
	defer server.Close()
	u, _ := url.Parse(server.URL)
	if port, _ := strconv.Atoi(u.Port()); port >= 8737 && port <= 8740 {
		t.Fatal("production port refused")
	}
	origin := "https://workshop.invalid:" + u.Port()
	if handler, err = Handler(store, origin); err != nil {
		t.Fatal(err)
	}
	pin := sha256.Sum256(server.Certificate().RawSubjectPublicKeyInfo)
	page := startChrome(t, chrome, base64.StdEncoding.EncodeToString(pin[:]))

	boot := func(step string) {
		// A fresh profile installs the service worker first and briefly locks the page; type only once it says current.
		page.until(step, clientReady)
		// The device's name and pairing live in the Devices drawer.
		page.click("devices-open")
		page.until(step+": devices drawer", `document.getElementById('devices').hidden===false`)
	}
	field := func() (value string, disabled bool, help string) {
		var got struct {
			Value    string
			Disabled bool
			Help     string
		}
		page.evalInto(`(()=>{const f=document.getElementById('device-name');const h=document.getElementById('device-name-help');return {Value:f.value,Disabled:f.disabled,Help:h?h.textContent:''}})()`, &got)
		return got.Value, got.Disabled, got.Help
	}

	page.navigate(origin + "/workshop/")
	boot("first boot")
	if _, _, help := field(); !strings.Contains(help, "pairage") {
		t.Errorf("before pairing, the field must say the name is sent only with the pairing request; help=%q", help)
	}
	page.typeInto("device-name", "PC")
	page.reload()
	boot("reload before pairing")
	if value, disabled, _ := field(); value != "PC" || disabled {
		t.Errorf("before pairing, a reload must keep the typed name editable: value=%q disabled=%v, want \"PC\" editable", value, disabled)
	}

	invite, err := store.NewInvite(origin)
	if err != nil {
		t.Fatal(err)
	}
	page.typeInto("pair-code", invite.Code)
	page.click("pair")
	page.until("pairing requested", `document.getElementById('pair').disabled===true`)
	devices, err := store.Devices()
	if err != nil || len(devices) != 1 || devices[0].Status != "pending" {
		t.Fatalf("Desktop must have registered exactly one pending device: %+v %v", devices, err)
	}
	if devices[0].Name != "PC" {
		t.Errorf("Desktop registered %q, not the name typed before the reload (\"PC\")", devices[0].Name)
	}
	if value, disabled, help := field(); value != "PC" || !disabled || !strings.Contains(help, "PC") {
		t.Errorf("after pairing, the field must show the registered name read-only: value=%q disabled=%v help=%q", value, disabled, help)
	}
	page.reload()
	boot("reload after pairing")
	if value, disabled, help := field(); value != "PC" || !disabled || !strings.Contains(help, "PC") {
		t.Errorf("after pairing and a reload, the field must still show the registered name read-only: value=%q disabled=%v help=%q", value, disabled, help)
	}

	if err := store.Decide(devices[0].ID, "refused"); err != nil {
		t.Fatal(err)
	}
	page.click("sync")
	page.until("refusal shown", `document.getElementById('replace-access').hidden===false`)
	page.click("replace-access")
	if value, disabled, _ := field(); value != "PC" || disabled {
		t.Errorf("re-pairing must make the name editable again, starting from the registered one: value=%q disabled=%v", value, disabled)
	}
}

type cdpPage struct {
	t    *testing.T
	ctx  context.Context
	conn *websocket.Conn
	next int
	port string // the browser's own debugging port, to reach the other windows it opens
}

// window waits for another page of the same browser whose URL contains part, and connects to it.
func (p *cdpPage) window(part string) *cdpPage {
	p.t.Helper()
	for deadline := time.Now().Add(15 * time.Second); time.Now().Before(deadline); time.Sleep(200 * time.Millisecond) {
		resp, err := http.Get("http://127.0.0.1:" + p.port + "/json/list")
		if err != nil {
			continue
		}
		var targets []struct{ Type, URL, WebSocketDebuggerURL string }
		err = json.NewDecoder(resp.Body).Decode(&targets)
		resp.Body.Close()
		for _, target := range targets {
			if err == nil && target.Type == "page" && strings.Contains(target.URL, part) {
				conn, _, err := websocket.Dial(p.ctx, target.WebSocketDebuggerURL, nil)
				if err != nil {
					p.t.Fatal(err)
				}
				conn.SetReadLimit(16 << 20)
				p.t.Cleanup(func() { conn.CloseNow() })
				return &cdpPage{t: p.t, ctx: p.ctx, conn: conn, port: p.port}
			}
		}
	}
	p.t.Fatalf("no window opened at %q", part)
	return nil
}

// windows counts the browser's open pages whose URL contains part.
func (p *cdpPage) windows(part string) int {
	p.t.Helper()
	resp, err := http.Get("http://127.0.0.1:" + p.port + "/json/list")
	if err != nil {
		p.t.Fatal(err)
	}
	defer resp.Body.Close()
	var targets []struct{ Type, URL string }
	_ = json.NewDecoder(resp.Body).Decode(&targets)
	n := 0
	for _, target := range targets {
		if target.Type == "page" && strings.Contains(target.URL, part) {
			n++
		}
	}
	return n
}

// press clicks the first element matching selector as a person does (a user gesture), so the page may open a window.
func (p *cdpPage) press(selector string) {
	p.t.Helper()
	raw := p.call("Runtime.evaluate", map[string]any{"expression": `(()=>{const e=document.querySelector(` + strconv.Quote(selector) + `);if(!e)return false;e.click();return true})()`, "userGesture": true, "returnByValue": true})
	var result struct{ Result struct{ Value bool } }
	if json.Unmarshal(raw, &result) != nil || !result.Result.Value {
		p.t.Fatalf("nothing to press at %s", selector)
	}
}

func startChrome(t *testing.T, chrome, pin string) *cdpPage {
	t.Helper()
	page, _ := startChromeIn(t, chrome, pin, t.TempDir())
	return page
}

// startChromeIn starts Chrome on the given profile folder; quit stops it, so that the same profile can be reopened the
// way a person closes and reopens the app.
// rules are extra host-resolver rules ("MAP host ip", or "MAP host ~NOTFOUND" for an unreachable host).
func startChromeIn(t *testing.T, chrome, pin, profile string, rules ...string) (*cdpPage, func()) {
	t.Helper()
	cmd := exec.Command(chrome, "--headless=new", "--remote-debugging-port=0", "--user-data-dir="+profile, "--no-first-run",
		"--no-default-browser-check", "--disable-background-networking", "--no-proxy-server",
		"--ignore-certificate-errors-spki-list="+pin,
		"--host-resolver-rules="+strings.Join(append(rules, "MAP workshop.invalid 127.0.0.1, MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"), ", "), "about:blank")
	_ = os.Remove(filepath.Join(profile, "DevToolsActivePort")) // a reopened profile keeps the previous run's endpoint
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	var once sync.Once
	quit := func() {
		once.Do(func() {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
			t.Logf("stopped owned Chrome PID %d", cmd.Process.Pid)
		})
	}
	t.Cleanup(quit)
	var port string
	for deadline := time.Now().Add(20 * time.Second); port == "" && time.Now().Before(deadline); time.Sleep(100 * time.Millisecond) {
		if f, err := os.Open(filepath.Join(profile, "DevToolsActivePort")); err == nil {
			s := bufio.NewScanner(f)
			if s.Scan() {
				port = strings.TrimSpace(s.Text())
			}
			f.Close()
		}
	}
	if port == "" {
		t.Fatal("Chrome did not publish its own debugging endpoint")
	}
	var targets []struct{ Type, WebSocketDebuggerURL string }
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(100 * time.Millisecond) {
		resp, err := http.Get("http://127.0.0.1:" + port + "/json/list")
		if err != nil {
			continue
		}
		targets = nil
		err = json.NewDecoder(resp.Body).Decode(&targets)
		resp.Body.Close()
		if err == nil && len(targets) > 0 {
			break
		}
	}
	for _, target := range targets {
		if target.Type == "page" {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			t.Cleanup(cancel)
			conn, _, err := websocket.Dial(ctx, target.WebSocketDebuggerURL, nil)
			if err != nil {
				t.Fatal(err)
			}
			conn.SetReadLimit(16 << 20)
			t.Cleanup(func() { conn.CloseNow() })
			return &cdpPage{t: t, ctx: ctx, conn: conn, port: port}, quit
		}
	}
	t.Fatal("Chrome exposed no page target")
	return nil, nil
}

func (p *cdpPage) call(method string, params any) json.RawMessage {
	p.t.Helper()
	p.next++
	id := p.next
	raw, _ := json.Marshal(map[string]any{"id": id, "method": method, "params": params})
	if err := p.conn.Write(p.ctx, websocket.MessageText, raw); err != nil {
		p.t.Fatalf("%s: %v", method, err)
	}
	for {
		_, data, err := p.conn.Read(p.ctx)
		if err != nil {
			p.t.Fatalf("%s: %v", method, err)
		}
		var msg struct {
			ID     int
			Result json.RawMessage
			Error  *struct{ Message string }
		}
		if json.Unmarshal(data, &msg) != nil || msg.ID != id {
			continue // events and other replies
		}
		if msg.Error != nil {
			p.t.Fatalf("%s: %s", method, msg.Error.Message)
		}
		return msg.Result
	}
}

func (p *cdpPage) evalInto(expression string, out any) {
	p.t.Helper()
	var result struct {
		Result           struct{ Value json.RawMessage }
		ExceptionDetails *struct {
			Text      string
			Exception struct{ Description string }
		}
	}
	_ = json.Unmarshal(p.call("Runtime.evaluate", map[string]any{"expression": expression, "awaitPromise": true, "returnByValue": true}), &result)
	if result.ExceptionDetails != nil {
		p.t.Fatalf("page error: %s %s", result.ExceptionDetails.Text, result.ExceptionDetails.Exception.Description)
	}
	if out != nil {
		if err := json.Unmarshal(result.Result.Value, out); err != nil {
			p.t.Fatalf("page value %s: %v", result.Result.Value, err)
		}
	}
}

func (p *cdpPage) until(step, condition string) {
	p.t.Helper()
	for deadline := time.Now().Add(15 * time.Second); time.Now().Before(deadline); time.Sleep(100 * time.Millisecond) {
		var ok bool
		p.evalInto(`(()=>{try{return Boolean(`+condition+`)}catch{return false}})()`, &ok)
		if ok {
			return
		}
	}
	var notice string
	p.evalInto(`document.getElementById('notice')?.textContent||''`, &notice)
	p.t.Fatalf("%s: condition never held: %s (notice %q)", step, condition, notice)
}

func (p *cdpPage) navigate(target string) {
	p.t.Helper()
	p.call("Page.navigate", map[string]any{"url": target})
}

func (p *cdpPage) reload() {
	p.t.Helper()
	p.evalInto(`(()=>{const f=document.getElementById('device-name');if(f)f.removeAttribute('id');location.reload();return true})()`, nil)
}

// typeInto replaces the field's text the way a person does: focus, select all, type, leave the field.
func (p *cdpPage) typeInto(id, text string) {
	p.t.Helper()
	p.evalInto(fmt.Sprintf(`(()=>{const f=document.getElementById(%q);f.focus();f.select();return true})()`, id), nil)
	p.call("Input.insertText", map[string]any{"text": text})
	p.evalInto(fmt.Sprintf(`(()=>{document.getElementById(%q).blur();return true})()`, id), nil)
	time.Sleep(300 * time.Millisecond) // let the client's own input handler finish its local write
}

// screenshot saves the page as PNG when WORKSHOP_ARTIFACT_DIR names an evidence folder; otherwise it does nothing.
func (p *cdpPage) screenshot(name string) {
	p.t.Helper()
	dir := os.Getenv("WORKSHOP_ARTIFACT_DIR")
	if dir == "" {
		return
	}
	var shot struct{ Data string }
	_ = json.Unmarshal(p.call("Page.captureScreenshot", map[string]any{"format": "png", "captureBeyondViewport": true}), &shot)
	png, err := base64.StdEncoding.DecodeString(shot.Data)
	if err != nil {
		p.t.Fatalf("screenshot: %v", err)
	}
	if err = os.WriteFile(filepath.Join(dir, name), png, 0o644); err != nil {
		p.t.Fatalf("screenshot: %v", err)
	}
	p.t.Logf("screenshot %s", filepath.Join(dir, name))
}

func (p *cdpPage) click(id string) {
	p.t.Helper()
	p.evalInto(fmt.Sprintf(`(()=>{document.getElementById(%q).click();return true})()`, id), nil)
}
