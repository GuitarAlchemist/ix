package workshop

import (
	"fmt"
	"testing"
)

// phone and tablet widths, emulated in a real headless Chrome: this is NOT a real device.
func emulate(page *cdpPage, width, height int) {
	page.call("Emulation.setDeviceMetricsOverride", map[string]any{"width": width, "height": height, "deviceScaleFactor": 2, "mobile": true})
	page.call("Emulation.setTouchEmulationEnabled", map[string]any{"enabled": true})
}

// openFromCatalog presses the Widgets or IXlets navigation (when it is not already the open one) and opens a tool by its title.
func openFromCatalog(t *testing.T, page *cdpPage, nav, title string) {
	t.Helper()
	q := fmt.Sprintf("%q", title)
	page.evalInto(`(()=>{const b=document.querySelector("[data-nav='`+nav+`']");if(b.getAttribute('aria-pressed')!=='true')b.click();return true})()`, nil)
	page.until("catalog lists "+title, `[...document.querySelectorAll('#catalog [data-inst] .inst-title')].some(e=>e.textContent===`+q+`)`)
	page.evalInto(`(()=>{[...document.querySelectorAll('#catalog [data-inst]')].find(b=>b.querySelector('.inst-title').textContent===`+q+`).click();return true})()`, nil)
	page.until(title+" open", `document.getElementById('tool-title').value===`+q+` && document.getElementById('catalog').hidden===true`)
}

func addListItem(t *testing.T, page *cdpPage, text string) {
	t.Helper()
	page.until("add enabled", `document.querySelector('[data-item-add]')?.disabled===false`)
	page.evalInto(`(()=>{const f=document.querySelector('[data-item-text]');f.focus();f.select();return true})()`, nil)
	page.call("Input.insertText", map[string]any{"text": text})
	page.evalInto(`(()=>{document.querySelector('[data-item-add]').click();return true})()`, nil)
	page.until("added "+text, `[...document.querySelectorAll('[data-item] .item-text')].some(e=>e.textContent===`+fmt.Sprintf("%q", text)+`) && document.querySelector('[data-item-text]').value===''`)
}

const shownItems = `[...document.querySelectorAll('[data-item]')].map(li=>(li.querySelector('[data-item-check]').checked?'[x] ':'[ ] ')+li.querySelector('.item-text').textContent).join(', ')`

// TestPhoneNavigationNamesTypesAndStates: on a phone width, an unpaired device shows its version and display mode apart
// from the connection, keeps a Widgets / IXlets navigation in view, lists the widget types that exist (List, Counter,
// Queries) with their names visible, tells drafts from anything else, and keeps what was typed when going back and forth.
func TestPhoneNavigationNamesTypesAndStates(t *testing.T) {
	f := newBrowserFixture(t)
	page := f.localPage()
	emulate(page, 390, 844)

	page.until("version and display mode are shown, apart from the connection", `(()=>{const a=document.getElementById('app-info').textContent;return /\d/.test(a)&&a.includes('Affichage : onglet du navigateur')&&!document.getElementById('sync').contains(document.getElementById('app-info'))&&document.getElementById('connection').textContent==='Sur cet appareil seulement'})()`)
	page.until("installation is not claimed", `!/install/i.test(document.getElementById('app-info').textContent)`)
	page.until("the navigation stays in view with both names", `(()=>{const n=document.getElementById('mobile-nav'),r=n.getBoundingClientRect();return r.height>=44&&r.bottom<=innerHeight+1&&r.top>0&&[...n.querySelectorAll('button')].map(b=>b.textContent.trim()).join('|')==='Widgets|IXlets'})()`)
	page.until("List and Counter labels stay visible on a phone", `['new-list','new-counter'].every(id=>{const s=document.querySelector('#'+id+' span[data-i18n]');return s&&s.getBoundingClientRect().width>0&&s.textContent.length>0})`)

	page.press("[data-nav='widgets']")
	page.until("three widget types are named", `[...document.querySelectorAll('#catalog [data-type] .kind-chip span')].map(e=>e.textContent).join('|')==='Liste|Compteur|Requêtes'`)
	page.until("only List and Counter can be made by hand", `[...document.querySelectorAll('#catalog [data-create]')].map(b=>b.dataset.create).join('|')==='checklist|counter'`)
	page.until("every type has a preview", `document.querySelectorAll('#catalog [data-type] .pv').length===3`)
	page.until("nothing is made yet", `document.querySelectorAll('#catalog [data-inst]').length===0`)

	// The name is optional: an empty field makes "Liste", then "Liste 2".
	page.press("[data-create='checklist']")
	page.until("an unnamed list is called Liste", `[...document.querySelectorAll('#catalog [data-inst] .inst-title')].some(e=>e.textContent==='Liste')`)
	page.press("[data-create='checklist']")
	page.until("the second unnamed list is Liste 2", `[...document.querySelectorAll('#catalog [data-inst] .inst-title')].some(e=>e.textContent==='Liste 2')`)
	page.typeInto("new-name", "Épicerie")
	page.press("[data-create='checklist']")
	page.until("the list is a draft, with its step and its blocker", `(()=>{const e=[...document.querySelectorAll("#catalog [data-group='groupDraft'] [data-inst]")].find(b=>b.querySelector('.inst-title').textContent==='Épicerie');return e&&e.dataset.state==='draft'&&e.textContent.includes('Brouillon sur cet appareil')&&e.textContent.includes('Étape 1 sur 3')&&e.textContent.includes('Bloqué : cet appareil n’est pas appairé')&&!/%/.test(e.textContent)})()`)
	page.typeInto("new-name", "Tasses")
	page.press("[data-create='counter']")
	page.until("four drafts", `document.querySelectorAll("#catalog [data-group='groupDraft'] [data-inst]").length===4`)
	page.until("no tool is shared yet", `document.querySelectorAll("#catalog [data-group='groupShared'] [data-inst]").length===0`)
	page.screenshot("telephone-widgets.png")

	page.press("[data-nav='ixlets']")
	page.until("IXlets: both run only on this device", `document.querySelectorAll("#catalog [data-group='ixletLocal'] [data-inst]").length===4 && document.querySelectorAll("#catalog [data-group='ixletActive'] [data-inst]").length===0`)
	page.screenshot("telephone-ixlets.png")

	openFromCatalog(t, page, "widgets", "Épicerie")
	page.until("the widget has a visible boundary and a named kind and state", `(()=>{const h=getComputedStyle(document.getElementById('widget-host'));return h.borderTopWidth==='1px'&&document.getElementById('tool-kind').textContent==='Liste'&&document.getElementById('tool-step').textContent.startsWith('Brouillon sur cet appareil')})()`)
	page.until("its actions have visible names", `[...document.querySelectorAll('.tool-actions .lbl')].filter(e=>e.getBoundingClientRect().width>0).map(e=>e.textContent).join('|').includes('Épingler')`)
	addListItem(t, page, "Lait")
	addListItem(t, page, "Pain")
	page.press("#pin")
	page.until("pinning is visible: the label and a notice change", `document.querySelector('#pin .lbl').textContent==='Épinglé' && document.getElementById('pin').getAttribute('aria-pressed')==='true' && document.getElementById('notice').textContent.includes('Épinglé')`)
	page.press("#surface-open")
	page.until("on a phone the surface opens in this window, with the pinned tool and a way back", `location.hash==='#surface' && [...document.querySelectorAll('#surface [data-surface-tool] h2')].map(h=>h.textContent).join()==='Épicerie' && document.querySelector('#surface .surface-back')!==null`)
	page.press("#surface .surface-back")
	page.until("back in IxForge", `document.body.dataset.mode==='app'`)
	page.until("nothing is pending at rest", `document.getElementById('action-busy').textContent===''`)
	page.screenshot("telephone-liste.png")

	openFromCatalog(t, page, "widgets", "Tasses")
	page.until("counter shows its kind", `document.getElementById('tool-kind').textContent==='Compteur'`)
	page.evalInto(`(()=>{document.querySelector('[data-xlet-action="increment"]').click();return true})()`, nil)
	page.until("counter at 1", `document.querySelector('[data-xlet-value]')?.textContent==='1'`)

	openFromCatalog(t, page, "widgets", "Épicerie")
	page.until("the list kept its contents", shownItems+`==='[ ] Lait, [ ] Pain'`)
	page.until("the counter kept its value", `(()=>{[...document.querySelectorAll('#tools [data-tool]')].find(b=>b.querySelector('[data-tool-title]').textContent==='Tasses').click();return true})()`)
	page.until("counter still 1", `document.querySelector('[data-xlet-value]')?.textContent==='1'`)
}

// TestTabletDiscoversCreatesACounterAndKeepsTheSharedList: a paired tablet finds the Counter type, makes one, comes back
// to the shared grocery list with its contents intact, and a second device sees both after a sync.
func TestTabletDiscoversCreatesACounterAndKeepsTheSharedList(t *testing.T) {
	f := newBrowserFixture(t)
	tablet := f.join("Tablette")
	f.create(tablet, `Créer un workspace "Courses" avec une liste "Épicerie"`)
	emulate(tablet, 820, 1180)
	addListItem(t, tablet, "Lait")
	addListItem(t, tablet, "Pain")

	tablet.until("version and mode apart from 'Desktop connecté'", `(()=>{const a=document.getElementById('app-info').textContent;return /\d/.test(a)&&document.getElementById('connection').textContent==='Desktop connecté'&&!a.includes('Desktop')})()`)
	tablet.press("[data-nav='widgets']")
	tablet.until("the shared list is held by the Desktop", `(()=>{const e=document.querySelector("#catalog [data-group='groupShared'] [data-inst]");return e&&e.dataset.state==='shared'&&e.textContent.includes('Partagé par le Desktop')&&e.textContent.includes('Étape 3 sur 3')})()`)
	tablet.screenshot("tablette-widgets.png")
	tablet.typeInto("new-name", "Tasses")
	tablet.press("[data-create='counter']")
	tablet.until("the counter is made and becomes shared", `(()=>{const e=[...document.querySelectorAll("#catalog [data-group='groupShared'] [data-inst]")].find(b=>b.querySelector('.inst-title').textContent==='Tasses');return Boolean(e)&&document.querySelectorAll("#catalog [data-group='groupShared'] [data-inst]").length===2})()`)

	openFromCatalog(t, tablet, "widgets", "Tasses")
	tablet.until("counter live", `document.querySelector('[data-xlet-value]')?.textContent==='0'`)
	tablet.evalInto(`(()=>{document.querySelector('[data-xlet-action="increment"]').click();return true})()`, nil)
	tablet.until("counter at 1", `document.querySelector('[data-xlet-value]')?.textContent==='1'`)

	openFromCatalog(t, tablet, "ixlets", "Épicerie")
	tablet.until("the shared list kept its contents", shownItems+`==='[ ] Lait, [ ] Pain'`)
	tablet.screenshot("tablette-liste-retrouvee.png")

	states, err := f.store.Xlets(f.anyToken(t))
	if err != nil || len(states) != 2 {
		t.Fatalf("the Desktop must hold both instances: %d %v", len(states), err)
	}

	pc := f.join("PC")
	pc.until("second device sees the list", shownItems+`==='[ ] Lait, [ ] Pain'`)
	openTool(t, pc, "Tasses")
	pc.until("second device sees the counter at 1", `document.querySelector('[data-xlet-value]')?.textContent==='1'`)
}
