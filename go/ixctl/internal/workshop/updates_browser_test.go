package workshop

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestTheOwnerUpdatesTheDesktopFromTheApp: on the linked PC (an admin device), the Devices drawer says which version
// the Desktop runs and offers the newer release built on this PC, with its notes. While the updater is not installed,
// it says so and installs nothing. Once the updater runs, one press asks for the release; the drawer says the request
// waits for the updater, then shows the updater's result.
func TestTheOwnerUpdatesTheDesktopFromTheApp(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the browser fixture runs on Windows")
	}
	u := &Updates{Installed: "0.1.0-ixforge.16", Releases: t.TempDir(), Requests: t.TempDir(), Now: time.Now}
	f := newBrowserFixture(t, WithUpdates(u))
	addRelease(t, u.Releases, "0.1.0-ixforge.17", "0.1.0-ixforge.17", []byte("seventeen"), "")

	link, err := f.store.NewLinkInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	pc, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	pc.navigate(link.URL)
	pc.until("linked", `document.getElementById('connection').textContent==='Desktop connecté'`)
	settle(t, pc)
	pc.click("devices-open")
	const block = `document.getElementById('desktop-update')`
	pc.until("the installed version and the newer release", block+`.hidden===false && document.getElementById('desktop-installed').textContent==='Version installée : 0.1.0-ixforge.16' && document.querySelector('[data-release="0.1.0-ixforge.17"] [data-release-notes]')?.textContent==='Notes of 0.1.0-ixforge.17'`)
	pc.until("no updater: said, and nothing to press", `document.getElementById('updater-state').textContent.startsWith('Assistant de mise à jour non installé') && document.querySelector('[data-release="0.1.0-ixforge.17"] [data-install]').disabled===true`)
	shot := func(name string) {
		pc.evalInto(`(()=>{document.getElementById('desktop-update').scrollIntoView({block:'end'});return true})()`, nil)
		pc.screenshot(name)
	}
	shot("mise-a-jour-sans-assistant.png")

	seenUpdater(t, u, time.Now())
	pc.click("sync")
	pc.until("the updater runs: the release can be installed", `document.getElementById('updater-state').textContent==='Assistant de mise à jour actif' && document.querySelector('[data-release="0.1.0-ixforge.17"] [data-install]').disabled===false && document.querySelector('[data-release="0.1.0-ixforge.17"] [data-install]').textContent==='Installer'`)
	pc.evalInto(`(()=>{document.querySelector('[data-release="0.1.0-ixforge.17"] [data-install]').click();return true})()`, nil)
	pc.until("the request waits for the updater", `document.getElementById('update-request')?.textContent.startsWith('Installation de 0.1.0-ixforge.17 demandée') && document.querySelector('[data-release="0.1.0-ixforge.17"] [data-install]').disabled===true`)
	shot("mise-a-jour-demandee.png")
	var asked struct {
		Version    string `json:"version"`
		DeviceName string `json:"device_name"`
	}
	raw, err := os.ReadFile(filepath.Join(u.Requests, "update-request.json"))
	if err != nil || json.Unmarshal(raw, &asked) != nil || asked.Version != "0.1.0-ixforge.17" || asked.DeviceName != "PC" {
		t.Fatalf("one press writes the request for the updater: %s %v", raw, err)
	}

	// The updater takes the request and writes its result.
	if err := os.Remove(filepath.Join(u.Requests, "update-request.json")); err != nil {
		t.Fatal(err)
	}
	result, _ := json.Marshal(map[string]string{"version": "0.1.0-ixforge.17", "state": "completed", "receipt": "update-0123456789abcdef0123456789abcdef", "finished_at": time.Now().UTC().Format(time.RFC3339)})
	if err := os.WriteFile(filepath.Join(u.Requests, "update-result.json"), result, 0o644); err != nil {
		t.Fatal(err)
	}
	pc.click("sync")
	pc.until("the updater's result", `document.getElementById('update-result').textContent.startsWith('Dernière mise à jour : 0.1.0-ixforge.17 installée') && document.getElementById('update-request')===null`)
	shot("mise-a-jour-installee.png")
}
