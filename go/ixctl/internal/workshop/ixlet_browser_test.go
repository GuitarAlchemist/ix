package workshop

import (
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"
)

// browserFixture is a real Workshop handler behind an ephemeral loopback TLS listener, for fresh-profile headless
// Chromes driven from Go (no Node). Action requests are counted; loseReplies makes the Desktop run an action while its
// reply is lost on the way back.
type browserFixture struct {
	t           *testing.T
	store       *Store
	origin      string
	chrome, pin string
	reader      string
	actions     atomic.Int64
	loseReplies atomic.Bool
	unreachable atomic.Bool // every request's connection is dropped, as when the Desktop cannot be reached
}

func newBrowserFixture(t *testing.T) *browserFixture {
	t.Helper()
	if os.Getenv("WORKSHOP_CDP") != "1" {
		t.Skip("set WORKSHOP_CDP=1 for the fresh-profile headless Chrome checks of the IXLET runtime")
	}
	f := &browserFixture{t: t, chrome: os.Getenv("CHROME")}
	if f.chrome == "" {
		f.chrome = `C:\Program Files\Google\Chrome\Application\chrome.exe`
	}
	if _, err := os.Stat(f.chrome); err != nil {
		t.Fatalf("Chrome not found at %s (set CHROME): %v", f.chrome, err)
	}
	var err error
	if f.store, err = Open(filepath.Join(t.TempDir(), "fixture-workshop.sqlite")); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.store.Close() })
	var handler http.Handler
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if f.unreachable.Load() {
			if conn, _, err := http.NewResponseController(w).Hijack(); err == nil {
				conn.Close()
			}
			return
		}
		if r.Method == http.MethodPost && r.URL.Path == "/workshop/api/xlets/actions" {
			f.actions.Add(1)
			if f.loseReplies.Load() {
				handler.ServeHTTP(httptest.NewRecorder(), r)
				w.WriteHeader(http.StatusBadGateway)
				return
			}
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(server.Close)
	u, _ := url.Parse(server.URL)
	if port, _ := strconv.Atoi(u.Port()); port >= 8737 && port <= 8740 {
		t.Fatal("production port refused")
	}
	f.origin = "https://workshop.invalid:" + u.Port()
	if handler, err = Handler(f.store, f.origin); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(server.Certificate().RawSubjectPublicKeyInfo)
	f.pin = base64.StdEncoding.EncodeToString(sum[:])
	return f
}

// clientReady: the client has booted, its service worker is settled, and the page is not locked for an update (typing
// during that lock is lost).
const clientReady = `document.getElementById('new-list')?.textContent==='Liste' && document.getElementById('update-status').textContent.startsWith('Client à jour') && !document.querySelector('main').inert`

// join opens a new device, pairs it under name from the Devices drawer and approves it on the Desktop, then waits
// until it is connected.
func (f *browserFixture) join(name string) *cdpPage {
	t := f.t
	t.Helper()
	page := startChrome(t, f.chrome, f.pin)
	page.navigate(f.origin + "/workshop/")
	settle(t, page) // typing during the first load's takeover lock is lost
	invite, err := f.store.NewInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	page.click("devices-open")
	page.until("devices drawer", `document.getElementById('devices').hidden===false`)
	page.typeInto("device-name", name)
	page.typeInto("pair-code", invite.Code)
	// "Code expiré" came back now and then (R21, R24, R25) for a code just made: say what the fields held if they
	// do not hold what was typed, instead of failing on the Desktop's answer.
	var fields struct {
		Name, Code string
		Inert      bool
	}
	page.evalInto(`({name:document.getElementById('device-name').value,code:document.getElementById('pair-code').value,inert:document.querySelector('main').inert})`, &fields)
	if fields.Name != name || fields.Code != invite.Code {
		t.Fatalf("%s: the pairing fields lost what was typed: name %q, code %q (typed %q), page inert %v", name, fields.Name, fields.Code, invite.Code, fields.Inert)
	}
	page.click("pair")
	page.until(name+" pairing requested", `document.getElementById('pair').disabled===true`)
	devices, err := f.store.Devices()
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devices {
		if d.Name == name && d.Status == "pending" {
			if err = f.store.Decide(d.ID, "approved"); err != nil {
				t.Fatal(err)
			}
		}
	}
	page.click("sync")
	page.until(name+" connected", `document.getElementById('connection').textContent==='Desktop connecté'`)
	page.click("devices-close")
	return page
}

// create types a command (or a name) in the new-tool field and presses Enter. There is nothing else to press: the
// sync saves the project on the Desktop and activates its IXLET there, which then serves the widget.
func (f *browserFixture) create(page *cdpPage, command string) {
	f.t.Helper()
	page.typeInto("new-name", command)
	page.press("#new-list")
	page.until("shared by the Desktop", `document.querySelector('#widget-host [data-xlet-live]')!==null && document.getElementById('tool-status').textContent.startsWith('Partagé par le Desktop')`)
}

// TestTabletCreatesAndRunsACounterIXLET is the counter's useful path end to end: dictate-or-type a command, create the
// workspace with its counter widget and IXLET definition, save it to the Desktop, activate the IXLET, drive it from the
// widget, see the same value on a second device, and get an explicit "uncertain" (never a silent resend) when an
// action's reply is lost.
func TestTabletCreatesAndRunsACounterIXLET(t *testing.T) {
	f := newBrowserFixture(t)
	value := `document.querySelector('[data-xlet-value]')?.textContent`
	desktopValue := func() (v, version int64) {
		if err := f.store.db.QueryRow(`SELECT value,version FROM workshop_xlets`).Scan(&v, &version); err != nil {
			t.Fatalf("Desktop instance: %v", err)
		}
		return
	}

	tablet := f.join("Tablette")
	f.create(tablet, `Créer un workspace "Maison" avec un compteur "Lectures"`)
	tablet.until("widget live", value+`==='0'`)
	press := func(page *cdpPage, action, want string) {
		t.Helper()
		page.until("buttons enabled", `document.querySelector('[data-xlet-action="`+action+`"]')?.disabled===false`)
		page.evalInto(`(()=>{document.querySelector('[data-xlet-action="`+action+`"]').click();return true})()`, nil)
		page.until(action+" → "+want, value+`==='`+want+`'`)
	}
	press(tablet, "increment", "1")
	press(tablet, "increment", "2")
	press(tablet, "increment", "3")
	press(tablet, "decrement", "2")
	if v, version := desktopValue(); v != 2 || version != 5 {
		t.Fatalf("the Desktop instance must hold the value: value=%d version=%d, want 2 at version 5", v, version)
	}

	pc := f.join("PC")
	pc.until("second device sees the shared value", value+`==='2'`)
	press(pc, "increment", "3")
	tablet.click("sync")
	tablet.until("first device sees the second device's action", value+`==='3'`)
	tablet.screenshot("tablette-ixlet-actif.png")

	tablet.reload()
	tablet.until("value survives a reload", value+`==='3' && document.getElementById('connection').textContent==='Desktop connecté'`)

	before := f.actions.Load()
	f.loseReplies.Store(true)
	tablet.until("buttons enabled", `document.querySelector('[data-xlet-action="increment"]')?.disabled===false`)
	tablet.evalInto(`(()=>{document.querySelector('[data-xlet-action="increment"]').click();return true})()`, nil)
	tablet.until("lost reply reported as uncertain", `document.getElementById('notice').textContent.startsWith('Résultat incertain')`)
	f.loseReplies.Store(false)
	tablet.until("the Desktop's real value is reread", value+`==='4'`)
	time.Sleep(6 * time.Second) // one more automatic sync period: nothing may be resent behind the user's back
	if sent := f.actions.Load() - before; sent != 1 {
		t.Fatalf("a lost reply must not be resent automatically: %d action requests", sent)
	}
	if v, _ := desktopValue(); v != 4 {
		t.Fatalf("the Desktop ran the action exactly once: value=%d, want 4", v)
	}
	press(tablet, "reset", "0")
}
