package workshop

import (
	"runtime"
	"testing"
)

// TestTheAppSaysWhichDesktopItTalksTo: a paired tablet (not an admin) shows the Desktop's version beside the connection
// pill, and "À propos" tells the Desktop's version from the interface's own number.
func TestTheAppSaysWhichDesktopItTalksTo(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the browser fixture runs on Windows")
	}
	f := newBrowserFixture(t, WithVersion("0.1.0-ixforge.18"))
	tablet := f.join("Tablette")
	tablet.until("the Desktop's version beside the pill", `document.getElementById('desktop-version').textContent==='Desktop 0.1.0-ixforge.18' && document.getElementById('desktop-version').hidden===false`)
	tablet.until("both numbers in À propos", `/^IxForge · Desktop 0\.1\.0-ixforge\.18 · interface \d+\.\d+\.\d+$/.test(document.getElementById('client-version').textContent)`)
	tablet.screenshot("version-desktop.png")
}
