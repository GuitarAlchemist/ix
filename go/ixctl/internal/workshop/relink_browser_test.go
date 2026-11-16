package workshop

import "testing"

// TestARevokedPCIsLinkedAgainByTheOwnersLink: the owner revoked the PC's IxForge, then pressed "Relier ce PC" again
// (2026-11-15, live: "Un accès existe déjà sur cet appareil."). The owner's link is the owner's own approval: the PC
// opens it in a new window, takes the new access in place of the revoked one, administers again, and keeps its tools.
func TestARevokedPCIsLinkedAgainByTheOwnersLink(t *testing.T) {
	f := newBrowserFixture(t)
	owner := newOwnerFace(t, f.store, f.origin)
	profile := t.TempDir()
	page, quit := startChromeIn(t, f.chrome, f.pin, profile)
	page.navigate(f.origin + "/workshop/")
	settle(t, page)
	page.navigate(owner.link())
	page.until("linked", `document.getElementById('connection').textContent==='Desktop connecté'`)
	f.create(page, `Créer un workspace "Courses" avec une liste "Épicerie"`)
	devices, err := f.store.Devices()
	if err != nil || len(devices) != 1 {
		t.Fatalf("one linked PC expected: %+v %v", devices, err)
	}
	if err := f.store.Decide(devices[0].ID, "revoked"); err != nil {
		t.Fatal(err)
	}
	page.click("sync")
	page.until("revocation shown", `document.getElementById('replace-access').hidden===false`)
	quit()

	// "Ouvrir IxForge relié" opens a new window of the same profile.
	page, _ = startChromeIn(t, f.chrome, f.pin, profile)
	page.navigate(owner.link())
	page.until("linked again", `document.getElementById('connection').textContent==='Desktop connecté' || document.getElementById('notice').textContent!==''`)
	var notice string
	page.evalInto(`document.getElementById('notice').textContent`, &notice)
	page.until("linked again", `document.getElementById('connection').textContent==='Desktop connecté'`)
	if notice != "" {
		t.Errorf("the owner's link must not be refused: %q", notice)
	}
	page.until("the tool is still there", `document.getElementById('tools').textContent.includes('Épicerie')`)
	devices, err = f.store.Devices()
	if err != nil || len(devices) != 2 || devices[1].Status != "approved" || !devices[1].Admin {
		t.Fatalf("the Desktop must hold the revoked PC and a new approved, administering PC: %+v %v", devices, err)
	}
	page.screenshot("pc-relie-apres-revocation.png")
}
