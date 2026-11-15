package workshop

import (
	"encoding/json"
	"net/url"
	"strings"
	"testing"
)

// localPage opens a fresh, unpaired IxForge and waits until it is settled.
func (f *browserFixture) localPage() *cdpPage {
	f.t.Helper()
	page, _ := startChromeIn(f.t, f.chrome, f.pin, f.t.TempDir())
	page.navigate(f.origin + "/workshop/")
	settle(f.t, page)
	return page
}

// TestAWidgetDetachesAndComesBack: a tool opens alone in its own window (to put on another screen) and the main
// window says where it went. Both windows show the same state. It comes back from either side: "Ramener ici" in the
// main window closes the other one, and closing the detached window gives the tool back.
func TestAWidgetDetachesAndComesBack(t *testing.T) {
	f := newBrowserFixture(t)
	page := f.localPage()
	newTool(t, page, "Tasses", "new-counter")
	pressCounter(page, "increment")
	page.until("counter", shownCounter+`==='1'`)

	page.press("#detach")
	away := page.window("#widget=")
	away.until("the detached window shows the tool alone", `document.body.dataset.mode==='widget' && document.getElementById('tool-title').value==='Tasses' && `+shownCounter+`==='1' && getComputedStyle(document.getElementById('rail')).display==='none' && getComputedStyle(document.getElementById('reattach')).display!=='none'`)
	page.until("the main window says where the tool is", `document.getElementById('detached-note').hidden===false && document.querySelector('#widget-host ix-widget')===null`)
	pressCounter(away, "increment")
	away.until("acted in the detached window", shownCounter+`==='2'`)
	away.screenshot("widget-detache.png")
	page.screenshot("fenetre-principale-widget-detache.png")

	page.click("reattach-main")
	page.until("back in the main window, with the detached window's change", `document.getElementById('detached-note').hidden===true && `+shownCounter+`==='2'`)
	if n := waitWindows(page, "#widget=", 0); n != 0 {
		t.Fatalf("bringing the tool back must close its detached window: %d left", n)
	}

	page.press("#detach")
	away = page.window("#widget=")
	away.until("detached again", shownCounter+`==='2'`)
	page.until("away again", `document.getElementById('detached-note').hidden===false`)
	away.press("#reattach")
	page.until("closing the detached window gives the tool back", `document.getElementById('detached-note').hidden===true && `+shownCounter+`==='2'`)
}

// waitWindows waits up to a few seconds for the number of windows whose URL contains part to reach want.
func waitWindows(page *cdpPage, part string, want int) int {
	n := -1
	for i := 0; i < 50 && n != want; i++ {
		n = page.windows(part)
		if n != want {
			page.evalInto(`new Promise(r=>setTimeout(()=>r(true),100))`, nil)
		}
	}
	return n
}

// TestPinnedToolsShowOnTheSurface: the surface is IxForge reduced to widgets, side by side, in a window of its own.
// Pinned tools appear there and stay usable; a change made on the surface shows in IxForge; unpinning removes it.
func TestPinnedToolsShowOnTheSurface(t *testing.T) {
	f := newBrowserFixture(t)
	page := f.localPage()
	newTool(t, page, "Valise", "new-list")
	addItem(t, page, "Passeport")
	page.click("pin")
	page.until("pinned", `document.getElementById('pin').getAttribute('aria-pressed')==='true'`)
	newTool(t, page, "Tasses", "new-counter")
	page.click("pin")
	newTool(t, page, "Pas épinglé", "new-counter")

	page.press("#surface-open")
	surface := page.window("#surface")
	tiles := `[...document.querySelectorAll('#surface [data-surface-tool] h2')].map(h=>h.textContent).join(', ')`
	surface.until("the pinned tools, side by side, and nothing else", `document.body.dataset.mode==='surface' && `+tiles+`==='Valise, Tasses' && getComputedStyle(document.getElementById('rail')).display==='none' && document.getElementById('stage').offsetParent===null`)
	surface.until("usable on the surface", `[...document.querySelectorAll('#surface [data-item] .item-text')].map(e=>e.textContent).join()==='Passeport'`)
	surface.evalInto(`(()=>{document.querySelector('#surface [data-xlet-action="increment"]').click();return true})()`, nil)
	surface.screenshot("surface.png")
	openTool(t, page, "Tasses")
	page.until("the surface's change shows in IxForge", shownCounter+`==='1'`)

	surface.evalInto(`(()=>{[...document.querySelectorAll('#surface [data-surface-tool]')].find(a=>a.querySelector('h2').textContent==='Valise').querySelector('[data-unpin]').click();return true})()`, nil)
	surface.until("unpinned", tiles+`==='Tasses'`)
}

// TestAWidgetGoesToAClaudeArtifactAndBack: a tool exports as a standalone page (what a Claude artifact is) carrying
// its state as an ixforge-widget/1 block. The page works by itself, offline. What it holds comes back into IxForge,
// as a new tool, by pasting its JSON or the whole page in the new-tool field.
func TestAWidgetGoesToAClaudeArtifactAndBack(t *testing.T) {
	f := newBrowserFixture(t)
	page := f.localPage()
	newTool(t, page, "Valise", "new-list")
	addItem(t, page, "Passeport")
	addItem(t, page, "Chargeur")
	onItem(page, "Passeport", "[data-item-check]")
	page.until("list", shownList+`==='[x] Passeport, [ ] Chargeur'`)

	// Catch the download: the page exported, as a file would hold it.
	page.evalInto(`(()=>{window.__exported=null;URL.createObjectURL=b=>{window.__exported=b;return 'about:blank'};HTMLAnchorElement.prototype.click=function(){window.__file=this.download};return true})()`, nil)
	page.click("export")
	var exported string
	page.evalInto(`(async()=>{for(let i=0;i<50&&!window.__exported;i++)await new Promise(r=>setTimeout(r,100));return window.__exported?await window.__exported.text():''})()`, &exported)
	if !strings.Contains(exported, `<script type="application/json" id="ixforge-widget">`) || !strings.Contains(exported, `"format":"ixforge-widget/1"`) || !strings.Contains(exported, `"text":"Passeport","done":true`) {
		t.Fatalf("the export must carry the widget's state as an ixforge-widget/1 block:\n%s", exported)
	}
	var file string
	page.evalInto(`window.__file`, &file)
	if file != "Valise.ixforge.html" {
		t.Errorf("the export is saved as %q, want Valise.ixforge.html", file)
	}
	if strings.Contains(exported, "/workshop/") || strings.Contains(exported, "src=") {
		t.Fatalf("the exported page must work by itself, without IxForge or the network:\n%s", exported)
	}

	// The page on its own, in another browser with no IxForge and no network: check the second item there, as someone
	// would in the artifact.
	alone, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	source, _ := json.Marshal(exported)
	alone.evalInto(`(()=>{document.open();document.write(`+string(source)+`);document.close();return true})()`, nil)
	alone.until("the exported page works by itself", `document.querySelectorAll('li').length===2 && document.getElementById('title').textContent==='Valise'`)
	alone.evalInto(`(()=>{[...document.querySelectorAll('li')].find(li=>li.textContent.includes('Chargeur')).querySelector('input').click();return true})()`, nil)
	alone.screenshot("widget-exporte.png")
	var block string
	alone.evalInto(`document.getElementById('ixforge-widget').textContent`, &block)
	var data struct {
		Format, Kind, Title string
		Items               []struct {
			Text string
			Done bool
		}
	}
	if err := json.Unmarshal([]byte(block), &data); err != nil || data.Format != "ixforge-widget/1" || len(data.Items) != 2 || !data.Items[1].Done {
		t.Fatalf("the exported page must keep its state in its block: %s %v", block, err)
	}

	// Back into IxForge: the JSON, pasted in the new-tool field.
	page.navigate(f.origin + "/workshop/")
	page.until("boot", clientBooted)
	page.typeInto("new-name", block)
	page.press("#new-list")
	page.until("imported from the JSON", `document.querySelectorAll('#tools [data-tool]').length===2 && document.getElementById('tool-title').value==='Valise' && `+shownList+`==='[x] Passeport, [x] Chargeur' && document.querySelector('#widget-host [data-xlet-local]')!==null`)
	// And the whole page, pasted as it is.
	page.typeInto("new-name", strings.Replace(exported, `"title":"Valise"`, `"title":"Valise 2"`, 1))
	page.press("#new-list")
	page.until("imported from the page", `document.querySelectorAll('#tools [data-tool]').length===3 && document.getElementById('tool-title').value==='Valise 2' && `+shownList+`==='[x] Passeport, [ ] Chargeur'`)
	// Something else is refused, and says why.
	page.typeInto("new-name", `{"format":"autre","title":"x"}`)
	page.press("#new-list")
	page.until("refused", `document.getElementById('notice').textContent.startsWith('Ce n’est pas un widget IxForge') && document.querySelectorAll('#tools [data-tool]').length===3`)
}

// TestTheLinkedPCAdministersDevicesFromTheApp: on the PC the owner linked, the Devices drawer lists the devices,
// invites one with a QR code and approves it, without going to the owner page. An ordinary device does not see this.
func TestTheLinkedPCAdministersDevicesInTheApp(t *testing.T) {
	f := newBrowserFixture(t)
	link, err := f.store.NewLinkInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	pc, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	pc.navigate(link.URL)
	pc.until("linked", `document.getElementById('connection').textContent==='Desktop connecté'`)
	pc.click("devices-open")
	devices := `[...document.querySelectorAll('#device-list [data-device]')].map(li=>li.querySelector('.name').textContent+' '+li.querySelector('.state').textContent).join(', ')`
	pc.until("the PC administers", `document.getElementById('admin').hidden===false && `+devices+`==='PC cet appareil'`)
	pc.click("invite")
	pc.until("an invite with its QR code", `document.getElementById('invite-view').hidden===false && document.querySelector('#invite-qr svg')!==null && document.getElementById('invite-link').href.includes('#pair=')`)
	var href string
	pc.evalInto(`document.getElementById('invite-link').href`, &href)
	u, err := url.Parse(href)
	if err != nil {
		t.Fatal(err)
	}
	fragment, _ := url.ParseQuery(u.Fragment)
	phone, _, err := f.store.Claim(fragment.Get("pair"), "Téléphone")
	if err != nil || phone.Status != "pending" {
		t.Fatalf("the invite must be an ordinary one: %+v %v", phone, err)
	}
	pc.until("the new device waits for approval in the drawer", devices+`==='PC cet appareil, Téléphone en attente'`)
	pc.screenshot("appareils-administration.png")
	pc.evalInto(`(()=>{document.querySelector('#device-list [data-device="`+phone.ID+`"] [data-decide="approved"]').click();return true})()`, nil)
	pc.until("approved from the app", devices+`==='PC cet appareil, Téléphone approuvé'`)
	list, err := f.store.Devices()
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range list {
		if d.ID == phone.ID && d.Status != "approved" {
			t.Fatalf("the Desktop must hold the approval: %+v", d)
		}
	}

	tablet := f.join("Tablette")
	tablet.click("devices-open")
	tablet.until("an ordinary device does not administer", `document.getElementById('devices').hidden===false && document.getElementById('admin').hidden===true`)
}
