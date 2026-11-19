package workshop

import (
	"fmt"
	"testing"
	"time"
)

// clientBooted: the client is up and unlocked, whether or not its update check could reach the Desktop.
const clientBooted = `document.getElementById('new-list')?.textContent==='Liste' && !document.querySelector('main').inert`

// settle waits, on a first load, until one service worker is active and controlling, none is installing or waiting,
// and the page is unlocked: typing during the takeover lock is lost.
func settle(t *testing.T, page *cdpPage) {
	t.Helper()
	var settled bool
	page.evalInto(`(async()=>{for(let i=0;i<150;i++){const r=await navigator.serviceWorker.getRegistration();if(r?.active&&!r.waiting&&!r.installing&&navigator.serviceWorker.controller&&!document.querySelector('main').inert&&document.getElementById('update-status').textContent.startsWith('Client à jour'))return true;await new Promise(d=>setTimeout(d,100))}return false})()`, &settled)
	if !settled {
		t.Fatal("the client never settled after its first load")
	}
}

// Page helpers shared by the widget tests.
const shownList = `[...document.querySelectorAll('#widget-host [data-item]')].map(li=>(li.querySelector('[data-item-check]').checked?'[x] ':'[ ] ')+li.querySelector('.item-text').textContent).join(', ')`
const shownCounter = `document.querySelector('#widget-host [data-xlet-value]')?.textContent`

// newTool types a name or a sentence in the new-tool field and presses the List or Counter button.
func newTool(t *testing.T, page *cdpPage, text, button string) {
	t.Helper()
	var before int
	page.evalInto(`document.querySelectorAll('#tools [data-tool]').length`, &before)
	page.typeInto("new-name", text)
	page.press("#" + button)
	page.until("tool created", fmt.Sprintf(`document.querySelectorAll('#tools [data-tool]').length===%d && document.getElementById('new-name').value===''`, before+1))
}

// openTool selects a tool in the rail by its name.
func openTool(t *testing.T, page *cdpPage, title string) {
	t.Helper()
	q := fmt.Sprintf("%q", title)
	page.until("tool "+title+" listed", `[...document.querySelectorAll('#tools [data-tool]')].some(b=>b.querySelector('[data-tool-title]').textContent===`+q+`)`)
	page.evalInto(`(()=>{[...document.querySelectorAll('#tools [data-tool]')].find(b=>b.querySelector('[data-tool-title]').textContent===`+q+`).click();return true})()`, nil)
	page.until("tool "+title+" open", `document.getElementById('tool-title').value===`+q)
}

func addItem(t *testing.T, page *cdpPage, text string) {
	t.Helper()
	page.until("add enabled", `document.querySelector('#widget-host [data-item-add]')?.disabled===false`)
	page.evalInto(`(()=>{const f=document.querySelector('#widget-host [data-item-text]');f.focus();f.select();return true})()`, nil)
	page.call("Input.insertText", map[string]any{"text": text})
	page.evalInto(`(()=>{document.querySelector('#widget-host [data-item-add]').click();return true})()`, nil)
	page.until("added "+text, `[...document.querySelectorAll('#widget-host [data-item] .item-text')].some(e=>e.textContent===`+fmt.Sprintf("%q", text)+`) && document.querySelector('#widget-host [data-item-text]').value===''`)
}

func onItem(page *cdpPage, text, selector string) {
	page.evalInto(`(()=>{const li=[...document.querySelectorAll('#widget-host [data-item]')].find(li=>li.querySelector('.item-text').textContent===`+fmt.Sprintf("%q", text)+`);li.querySelector('`+selector+`').click();return true})()`, nil)
}

func pressCounter(page *cdpPage, action string) {
	page.evalInto(`(()=>{document.querySelector('#widget-host [data-xlet-action="`+action+`"]').click();return true})()`, nil)
}

// TestUnpairedDeviceRunsItsIXLETsLocally: on a device that is not paired, a checklist and a counter are usable at
// once, on this device only, and say so. Their state lives in the browser's IndexedDB: it is found again after the app
// is closed and reopened, and stays usable while the Desktop cannot be reached. Nothing is sent to the Desktop.
func TestUnpairedDeviceRunsItsIXLETsLocally(t *testing.T) {
	f := newBrowserFixture(t)
	profile := t.TempDir()
	open := func(unreachable bool) (*cdpPage, func()) {
		f.unreachable.Store(unreachable)
		page, quit := startChromeIn(t, f.chrome, f.pin, profile)
		page.navigate(f.origin + "/workshop/")
		page.until("boot", clientBooted)
		return page, quit
	}
	shows := func(page *cdpPage, step, wantList, wantCounter string) {
		t.Helper()
		openTool(t, page, "Épicerie")
		page.until(step+" (list)", shownList+`===`+fmt.Sprintf("%q", wantList))
		openTool(t, page, "Lectures")
		page.until(step+" (counter)", shownCounter+`===`+fmt.Sprintf("%q", wantCounter))
	}
	// The open widget says it is local, and nothing claims the Desktop.
	honest := `document.querySelector('#widget-host [data-xlet-local]')!==null && document.querySelectorAll('[data-xlet-live]').length===0 && document.getElementById('tool-status').textContent==='Sur cet appareil seulement' && document.getElementById('connection').textContent==='Sur cet appareil seulement'`

	page, quit := open(false)
	settle(t, page)
	newTool(t, page, "Créer un workspace « Courses » avec une liste « Épicerie »", "new-list")
	newTool(t, page, "Lectures", "new-counter")
	shows(page, "both widgets start empty", "", "0")
	openTool(t, page, "Épicerie")
	addItem(t, page, "Lait")
	addItem(t, page, "Pain")
	addItem(t, page, "Œufs")
	onItem(page, "Pain", "[data-item-check]")
	onItem(page, "Œufs", "[data-item-remove]")
	page.until("progress", `document.querySelector('[data-xlet-progress]').textContent.startsWith('1/2')`)
	page.until("the list says it is local", honest)
	openTool(t, page, "Lectures")
	for _, action := range []string{"increment", "increment", "increment", "decrement"} {
		pressCounter(page, action)
	}
	shows(page, "local list and counter", "[ ] Lait, [x] Pain", "2")
	page.until("the counter says it is local", honest)
	page.until("the rail shows each tool's state", `[...document.querySelectorAll('#tools .badge')].map(b=>b.textContent).join(' ')==='1/2 2'`)
	page.screenshot("local-ixlets-before-close.png")
	page.call("Emulation.setDeviceMetricsOverride", map[string]any{"width": 1280, "height": 800, "deviceScaleFactor": 1, "mobile": false})
	openTool(t, page, "Épicerie")
	page.screenshot("local-ixlets-grand-ecran.png")
	page.call("Emulation.clearDeviceMetricsOverride", map[string]any{})

	// Close the app and reopen it on the same profile: the state is found again.
	quit()
	page, quit = open(false)
	shows(page, "state after reopening", "[ ] Lait, [x] Pain", "2")
	page.until("still local after reopening", honest)

	// Reopen with the Desktop unreachable: still usable.
	quit()
	page, _ = open(true)
	shows(page, "state with the Desktop unreachable", "[ ] Lait, [x] Pain", "2")
	pressCounter(page, "reset")
	openTool(t, page, "Épicerie")
	onItem(page, "Lait", "[data-item-check]")
	shows(page, "usable with the Desktop unreachable", "[x] Lait, [x] Pain", "0")
	page.screenshot("local-ixlets-desktop-unreachable.png")

	if sent := f.actions.Load(); sent != 0 {
		t.Errorf("a local IXLET must not send actions to the Desktop: %d requests", sent)
	}
	if devices, err := f.store.Devices(); err != nil || len(devices) != 0 {
		t.Errorf("nothing may reach the Desktop before pairing: devices=%v err=%v", devices, err)
	}
}

// TestLocalWorkGoesToTheDesktopWhenThePCIsLinked: what was done on the PC before it was linked is not left behind.
// Once linked, with nothing to press, the list is saved on the Desktop, activated there with its local items, shared
// with the other devices, and the local copy is dropped. No save or activate button remains.
func TestLocalWorkGoesToTheDesktopWhenThePCIsLinked(t *testing.T) {
	f := newBrowserFixture(t)
	page, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	page.navigate(f.origin + "/workshop/")
	settle(t, page)
	newTool(t, page, "Épicerie", "new-list")
	addItem(t, page, "Lait")
	addItem(t, page, "Pain")
	onItem(page, "Lait", "[data-item-check]")
	newTool(t, page, "Tasses", "new-counter")
	for range 3 {
		pressCounter(page, "increment")
	}
	page.until("local counter", shownCounter+`==='3'`)
	if sent := f.actions.Load(); sent != 0 {
		t.Fatalf("nothing may reach the Desktop before the link: %d requests", sent)
	}

	link, err := f.store.NewLinkInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	page.navigate(link.URL)
	page.until("linked", `document.getElementById('connection').textContent==='Desktop connecté'`)
	page.until("the counter is shared, from its local value", `document.querySelector('#widget-host [data-xlet-live]')!==null && `+shownCounter+`==='3' && document.getElementById('tool-status').textContent.startsWith('Partagé par le Desktop') && document.getElementById('pending').textContent===''`)
	openTool(t, page, "Épicerie")
	page.until("the list is shared, from its local items", `document.querySelector('#widget-host [data-xlet-live]')!==null && `+shownList+`==='[x] Lait, [ ] Pain'`)
	page.until("no button to save or activate", `document.querySelector('#save,[data-activate]')===null`)
	var local int
	page.evalInto(`import('/workshop/sync.js').then(m=>m.Replica.open()).then(r=>r.read()).then(s=>Object.keys(s.localXlets||{}).length)`, &local)
	if local != 0 {
		t.Errorf("the local copies must be dropped once the Desktop holds them: %d left", local)
	}
	page.screenshot("pc-relie-partage.png")

	states, err := f.store.Xlets(f.anyToken(t))
	if err != nil || len(states) != 2 {
		t.Fatalf("Desktop instances: %+v %v", states, err)
	}
	for _, x := range states {
		if (x.Kind == "checklist" && items(x) != "[x] Lait, [ ] Pain") || (x.Kind == "counter" && x.Value != 3) {
			t.Fatalf("the Desktop must start from the local state: %+v", x)
		}
	}
	tablet := f.join("Tablette")
	openTool(t, tablet, "Épicerie")
	tablet.until("another device sees the list", shownList+`==='[x] Lait, [ ] Pain'`)
}

// TestALocalEditDuringActivationIsKept (Codex P1 on #544): a counter pressed on the PC while its activation is on its
// way to the Desktop is not lost when the activation's reply arrives. The Desktop starts from the value that was sent;
// the newer local value stays on this device, and the app says so.
func TestALocalEditDuringActivationIsKept(t *testing.T) {
	f := newBrowserFixture(t)
	page, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	page.navigate(f.origin + "/workshop/")
	settle(t, page)
	newTool(t, page, "Tasses", "new-counter")
	pressCounter(page, "increment")
	page.until("local counter", shownCounter+`==='1'`)

	hold := make(chan struct{})
	f.holdActions.Store(&hold)
	link, err := f.store.NewLinkInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	page.navigate(link.URL)
	deadline := time.Now().Add(20 * time.Second)
	for f.actions.Load() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("the activation never left")
		}
		time.Sleep(50 * time.Millisecond)
	}
	pressCounter(page, "increment")
	page.until("the local edit while the activation is in flight", shownCounter+`==='2'`)
	f.holdActions.Store(nil)
	close(hold)

	page.until("the Desktop holds the sent value and the local edit is kept", `document.querySelector('#widget-host [data-xlet-live]')!==null && `+shownCounter+`==='1' && !document.getElementById('local-kept').hidden`)
	var local int
	page.evalInto(`import('/workshop/sync.js').then(m=>m.Replica.open()).then(r=>r.read()).then(s=>(Object.values(s.localXlets||{})[0]||{}).value||0)`, &local)
	if local != 2 {
		t.Fatalf("the local value pressed during the activation must be kept: %d", local)
	}
}
