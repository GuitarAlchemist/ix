package workshop

import (
	"runtime"
	"testing"
)

// TestTheLinkedPCRunsCommandsInItsTerminal: on the PC the owner linked, IxForge offers a terminal. A command typed there
// runs on the Desktop's PC and its output and exit code appear under it; nothing of it is kept in the browser. Another
// device is not offered the terminal.
func TestTheLinkedPCRunsCommandsInItsTerminal(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the terminal runs Windows PowerShell")
	}
	f := newBrowserFixture(t)
	link, err := f.store.NewLinkInvite(f.origin)
	if err != nil {
		t.Fatal(err)
	}
	pc, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	pc.navigate(link.URL)
	pc.until("linked", `document.getElementById('connection').textContent==='Desktop connecté'`)
	settle(t, pc) // typing during the first load's takeover lock is lost
	pc.until("terminal offered", `document.getElementById('terminal-open').hidden===false`)
	pc.click("terminal-open")
	pc.until("terminal shown", `document.getElementById('terminal').hidden===false && document.getElementById('stage').hidden===true`)
	pc.typeInto("terminal-command", "Write-Output ('bon'+'jour'); exit 2")
	pc.press("#terminal-run")
	last := `[...document.querySelectorAll('#terminal-log [data-run]')].at(-1)`
	pc.until("output and exit code shown", last+`?.querySelector('[data-run-output]').textContent.trim()==='bonjour' && `+last+`.querySelector('[data-run-exit]').textContent.includes('2') && document.getElementById('terminal-command').value===''`)
	pc.screenshot("terminal-pc.png")
	var kept bool
	pc.evalInto(`import('./sync.js').then(m=>m.Replica.open()).then(r=>r.read()).then(s=>JSON.stringify(s).includes('bonjour'))`, &kept)
	if kept {
		t.Error("the terminal's output must not be kept in the browser")
	}

	tablet := f.join("Tablette")
	tablet.until("another device has no terminal", `document.getElementById('connection').textContent==='Desktop connecté' && document.getElementById('terminal-open').hidden===true`)
}
