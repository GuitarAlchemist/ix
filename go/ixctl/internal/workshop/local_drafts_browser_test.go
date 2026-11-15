package workshop

import (
	"net"
	"net/url"
	"os"
	"testing"
)

// TestWorkspaceDraftsAreUsableWithoutPairing: on a PC that is not paired (no tablet, no approval), a person must still
// be able to create a workspace from a command, edit its draft, close the app and find the draft again, then reopen it
// while the Desktop cannot be reached. Drafts live in the browser's IndexedDB (ix-workshop-replica); nothing reaches
// the Desktop before pairing.
func TestWorkspaceDraftsAreUsableWithoutPairing(t *testing.T) {
	f := newBrowserFixture(t)
	checkLocalDrafts(t, f.origin, func(profile string, unreachable bool) (*cdpPage, func()) {
		f.unreachable.Store(unreachable)
		return startChromeIn(t, f.chrome, f.pin, profile)
	})
	if devices, err := f.store.Devices(); err != nil || len(devices) != 0 {
		t.Errorf("nothing may reach the Desktop before pairing: devices=%v err=%v", devices, err)
	}
}

// TestLiveWorkspaceDraftsAreUsableWithoutPairing runs the same path against the deployed service, in a throwaway
// profile that never pairs: the client holds no token, so it writes nothing on the Desktop. Opt-in only:
// WORKSHOP_LIVE_ORIGIN=https://<host>. The Desktop is made unreachable by resolving its host to nothing.
func TestLiveWorkspaceDraftsAreUsableWithoutPairing(t *testing.T) {
	origin := os.Getenv("WORKSHOP_LIVE_ORIGIN")
	if origin == "" {
		t.Skip("set WORKSHOP_LIVE_ORIGIN to check the deployed service with a throwaway, never-paired profile")
	}
	u, err := url.Parse(origin)
	if err != nil || u.Scheme != "https" {
		t.Fatalf("WORKSHOP_LIVE_ORIGIN must be an https origin: %q", origin)
	}
	addrs, err := net.LookupHost(u.Hostname())
	if err != nil || len(addrs) == 0 {
		t.Fatalf("resolve %s: %v", u.Hostname(), err)
	}
	chrome := os.Getenv("CHROME")
	if chrome == "" {
		chrome = `C:\Program Files\Google\Chrome\Application\chrome.exe`
	}
	checkLocalDrafts(t, origin, func(profile string, unreachable bool) (*cdpPage, func()) {
		rule := "MAP " + u.Hostname() + " " + addrs[0]
		if unreachable {
			rule = "MAP " + u.Hostname() + " ~NOTFOUND"
		}
		return startChromeIn(t, chrome, "", profile, rule)
	})
}

func checkLocalDrafts(t *testing.T, origin string, start func(profile string, unreachable bool) (*cdpPage, func())) {
	t.Helper()
	profile := t.TempDir()
	boot := clientReady
	page, quit := start(profile, false)
	page.navigate(origin + "/workshop/")
	page.until("boot", boot)
	var served struct{ Origin, Version string }
	page.evalInto(`fetch('/workshop/branding.json',{cache:'no-store'}).then(r=>r.json()).then(b=>({origin:location.origin,version:b.version}))`, &served)
	if served.Origin != origin {
		t.Fatalf("the client was served by %q, want %q", served.Origin, origin)
	}
	t.Logf("client served by %s (client %s)", served.Origin, served.Version)
	page.until("on this device only", `document.getElementById('connection').textContent==='Sur cet appareil seulement'`)
	settle(t, page) // typing during the first load's takeover lock is lost

	names :=`[...document.querySelectorAll('#tools [data-tool-title]')].map(e=>e.textContent).join(', ')`
	page.typeInto("new-name", "Créer un workspace « Courses » avec une liste « Épicerie »")
	page.press("#new-list")
	page.until("tool kept on this device", names+`==='Épicerie' && document.getElementById('tool-status').textContent==='Sur cet appareil seulement'`)
	page.typeInto("tool-title", "Courses de la semaine")
	page.until("tool renamed", names+`==='Courses de la semaine'`)
	page.typeInto("new-name", "Lectures")
	page.press("#new-counter")
	page.until("second tool", names+`==='Courses de la semaine, Lectures'`)
	page.screenshot("local-drafts-before-close.png")

	// Close the app, reopen it on the same profile.
	quit()
	page, quit = start(profile, false)
	page.navigate(origin + "/workshop/")
	page.until("boot after reopening", boot)
	page.until("tools after reopening", names+`==='Courses de la semaine, Lectures'`)

	// Reopen once more with the Desktop unreachable: the shell comes from the client cache, the tools from IndexedDB.
	quit()
	page, _ = start(profile, true)
	page.navigate(origin + "/workshop/")
	page.until("boot with the Desktop unreachable", clientBooted)
	page.evalInto(`(()=>{[...document.querySelectorAll('#tools [data-tool]')].find(b=>b.textContent.includes('Courses de la semaine')).click();return true})()`, nil)
	page.until("tool editable with the Desktop unreachable", `document.getElementById('tool-title').value==='Courses de la semaine' && document.querySelector('#widget-host ix-checklist')!==null`)
	page.typeInto("tool-title", "Courses hors ligne")
	page.until("edit kept with the Desktop unreachable", `document.querySelector('#tools [data-tool][aria-current="true"] [data-tool-title]')?.textContent==='Courses hors ligne'`)
	var outbox int
	page.evalInto(`import('/workshop/sync.js').then(m=>m.Replica.open()).then(r=>r.read()).then(s=>s.outbox.length)`, &outbox)
	if outbox < 2 {
		t.Errorf("the drafts must wait in the local outbox for a future pairing: outbox=%d", outbox)
	}
	page.screenshot("local-drafts-desktop-unreachable.png")
}
