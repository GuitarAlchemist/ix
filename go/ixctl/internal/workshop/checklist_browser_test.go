package workshop

import (
	"fmt"
	"testing"
	"time"
)

// TestTabletCreatesAndRunsAChecklistIXLET: a sentence creates a workspace with a checklist widget and IXLET; once
// activated, the Desktop holds the items; two devices add, check, uncheck and remove them and see each other's changes;
// text being typed survives a sync; a lost reply is uncertain and never resent.
func TestTabletCreatesAndRunsAChecklistIXLET(t *testing.T) {
	f := newBrowserFixture(t)
	list := `[...document.querySelectorAll('[data-item]')].map(li=>(li.querySelector('[data-item-check]').checked?'[x] ':'[ ] ')+li.querySelector('.item-text').textContent).join(', ')`
	shows := func(page *cdpPage, step, want string) {
		t.Helper()
		page.until(step, list+`===`+fmt.Sprintf("%q", want))
	}
	typeItem := func(page *cdpPage, text string) {
		t.Helper()
		page.until("add enabled", `document.querySelector('[data-item-add]')?.disabled===false`)
		page.evalInto(`(()=>{const f=document.querySelector('[data-item-text]');f.focus();f.select();return true})()`, nil)
		page.call("Input.insertText", map[string]any{"text": text})
	}
	add := func(page *cdpPage, text string) {
		t.Helper()
		page.evalInto(`(()=>{document.querySelector('[data-item-add]').click();return true})()`, nil)
		page.until("added "+text, `[...document.querySelectorAll('[data-item] .item-text')].some(e=>e.textContent===`+fmt.Sprintf("%q", text)+`) && document.querySelector('[data-item-text]').value===''`)
	}
	onItem := func(page *cdpPage, text, selector string) {
		t.Helper()
		page.evalInto(`(()=>{const li=[...document.querySelectorAll('[data-item]')].find(li=>li.querySelector('.item-text').textContent===`+fmt.Sprintf("%q", text)+`);li.querySelector('`+selector+`').click();return true})()`, nil)
	}
	desktop := func() string {
		states, err := f.store.Xlets(f.anyToken(t))
		if err != nil || len(states) != 1 {
			t.Fatalf("Desktop instances: %+v %v", states, err)
		}
		return items(states[0])
	}

	tablet := f.join("Tablette")
	f.create(tablet, `Créer un workspace "Courses" avec une liste "Épicerie"`)
	tablet.until("a checklist named after the sentence", `document.querySelector('#widget-host ix-checklist')!==null && document.getElementById('tool-title').value==='Épicerie'`)
	shows(tablet, "empty list", "")

	typeItem(tablet, "Lait")
	add(tablet, "Lait")
	typeItem(tablet, "Pain")
	// Syncs re-render the page while the item is still being written: the text must survive them.
	tablet.click("sync")
	time.Sleep(6 * time.Second)
	tablet.until("text kept across syncs", `document.querySelector('[data-item-text]').value==='Pain'`)
	add(tablet, "Pain")
	typeItem(tablet, "Œufs")
	add(tablet, "Œufs")
	shows(tablet, "three items", "[ ] Lait, [ ] Pain, [ ] Œufs")
	onItem(tablet, "Pain", "[data-item-check]")
	shows(tablet, "Pain checked", "[ ] Lait, [x] Pain, [ ] Œufs")
	onItem(tablet, "Œufs", "[data-item-remove]")
	shows(tablet, "Œufs removed", "[ ] Lait, [x] Pain")
	tablet.until("progress", `document.querySelector('[data-xlet-progress]').textContent.startsWith('1/2')`)
	if got := desktop(); got != "[ ] Lait, [x] Pain" {
		t.Fatalf("the Desktop must hold the list: %q", got)
	}

	pc := f.join("PC")
	shows(pc, "second device sees the list", "[ ] Lait, [x] Pain")
	onItem(pc, "Lait", "[data-item-check]")
	shows(pc, "PC checks Lait", "[x] Lait, [x] Pain")
	onItem(pc, "Pain", "[data-item-check]")
	shows(pc, "PC unchecks Pain", "[x] Lait, [ ] Pain")
	tablet.click("sync")
	shows(tablet, "tablet sees the PC's changes", "[x] Lait, [ ] Pain")
	tablet.screenshot("tablette-liste-active.png")

	before := f.actions.Load()
	f.loseReplies.Store(true)
	typeItem(tablet, "Thé")
	tablet.evalInto(`(()=>{document.querySelector('[data-item-add]').click();return true})()`, nil)
	tablet.until("lost reply reported as uncertain", `document.getElementById('notice').textContent.startsWith('Résultat incertain')`)
	f.loseReplies.Store(false)
	shows(tablet, "the Desktop's real list is reread", "[x] Lait, [ ] Pain, [ ] Thé")
	time.Sleep(6 * time.Second) // one more automatic sync period: nothing may be resent behind the user's back
	if sent := f.actions.Load() - before; sent != 1 {
		t.Fatalf("a lost reply must not be resent automatically: %d action requests", sent)
	}
	if got := desktop(); got != "[x] Lait, [ ] Pain, [ ] Thé" {
		t.Fatalf("the Desktop ran the add exactly once: %q", got)
	}
}

// anyToken pairs a hidden Desktop-side reader so the test can read instances through the same access checks.
func (f *browserFixture) anyToken(t *testing.T) string {
	t.Helper()
	if f.reader == "" {
		f.reader, _ = paired(t, f.store)
	}
	return f.reader
}
