package workshop

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

// saved commits a project holding one checklist and one counter, and returns their definition ids.
func saved(t *testing.T, s *Store, token string) (checklist, counter string) {
	t.Helper()
	p := Project{ID: "project-seed", Title: "Maison", Workspaces: []Workspace{{ID: "ws-seed", Title: "Maison",
		Widgets:          []Widget{{ID: "w-list", Title: "Épicerie", Kind: "checklist"}, {ID: "w-count", Title: "Lectures", Kind: "counter", InitialValue: 0}},
		IXLETDefinitions: []IXLETDefinition{{ID: "d-list", Title: "Épicerie", Kind: "checklist", Status: "draft"}, {ID: "d-count", Title: "Lectures", Kind: "counter", Status: "draft"}}}}}
	if _, err := s.Save(token, Operation{OpID: "op-seed", BaseRevision: 0, Project: p}); err != nil {
		t.Fatal(err)
	}
	return "d-list", "d-count"
}

// TestActivationCarriesTheLocalState: what a device did locally before the Desktop ran the IXLET goes with the
// activation, atomically, so nothing typed offline is lost. A running instance is never overwritten by a seed.
func TestActivationCarriesTheLocalState(t *testing.T) {
	s, token, _ := fixture(t)
	list, count := saved(t, s, token)
	seeded, err := s.Act(token, XletAction{OpID: "a1", ProjectID: "project-seed", DefinitionID: list, Action: "activate",
		Seed: &XletSeed{Items: []XletSeedItem{{Text: "Lait", Done: true}, {Text: "Pain"}}}})
	if err != nil {
		t.Fatal(err)
	}
	if !seeded.Seeded || items(seeded) != "[x] Lait, [ ] Pain" || seeded.Value != 2 {
		t.Fatalf("the checklist must start from the local items: %+v", seeded)
	}
	c, err := s.Act(token, XletAction{OpID: "a2", ProjectID: "project-seed", DefinitionID: count, Action: "activate", Seed: &XletSeed{Value: 7}})
	if err != nil || !c.Seeded || c.Value != 7 {
		t.Fatalf("the counter must start from the local value: %+v %v", c, err)
	}
	again, err := s.Act(token, XletAction{OpID: "a3", ProjectID: "project-seed", DefinitionID: list, Action: "activate",
		Seed: &XletSeed{Items: []XletSeedItem{{Text: "Autre"}}}})
	if err != nil || again.Seeded || items(again) != "[x] Lait, [ ] Pain" {
		t.Fatalf("a running instance must not be overwritten by a seed: %+v %v", again, err)
	}
	replay, err := s.Act(token, XletAction{OpID: "a1", ProjectID: "project-seed", DefinitionID: list, Action: "activate",
		Seed: &XletSeed{Items: []XletSeedItem{{Text: "Lait", Done: true}, {Text: "Pain"}}}})
	if err != nil || !replay.Seeded || items(replay) != "[x] Lait, [ ] Pain" {
		t.Fatalf("the same op_id must return the first result: %+v %v", replay, err)
	}
	for name, bad := range map[string]XletAction{
		"seed on another action": {OpID: "b1", ProjectID: "project-seed", DefinitionID: list, Action: "add", Text: "x", Seed: &XletSeed{Value: 1}},
		"items on a counter":     {OpID: "b2", ProjectID: "project-seed", DefinitionID: count, Action: "activate", Seed: &XletSeed{Items: []XletSeedItem{{Text: "x"}}}},
		"empty item text":        {OpID: "b3", ProjectID: "project-seed", DefinitionID: list, Action: "activate", Seed: &XletSeed{Items: []XletSeedItem{{Text: " "}}}},
		"value out of range":     {OpID: "b4", ProjectID: "project-seed", DefinitionID: count, Action: "activate", Seed: &XletSeed{Value: maxSafe + 1}},
	} {
		if _, err := s.Act(token, bad); !errors.Is(err, ErrInvalid) {
			t.Errorf("%s must be refused: %v", name, err)
		}
	}
	many := make([]XletSeedItem, maxChecklistItems+1)
	for i := range many {
		many[i] = XletSeedItem{Text: "x"}
	}
	s2, token2, _ := fixture(t)
	list2, _ := saved(t, s2, token2)
	if _, err := s2.Act(token2, XletAction{OpID: "c1", ProjectID: "project-seed", DefinitionID: list2, Action: "activate", Seed: &XletSeed{Items: many}}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("more than %d seeded items must be refused: %v", maxChecklistItems, err)
	}
}

// TestTheLinkedPCManagesDevicesFromTheApp: the PC the owner linked from the owner page administers the other devices
// from IxForge itself: list, invite, approve, refuse, revoke. Other devices cannot, and the PC cannot lock itself out.
// The owner page can also grant this to a device already approved.
func TestTheLinkedPCManagesDevicesFromTheApp(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "admin.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	const app = "https://workshop.invalid"
	h, err := Handler(s, app)
	if err != nil {
		t.Fatal(err)
	}
	link, err := s.NewLinkInvite(app)
	if err != nil {
		t.Fatal(err)
	}
	pc, pcToken, err := s.Claim(link.Code, "PC")
	if err != nil {
		t.Fatal(err)
	}
	tabletToken, tabletID := paired(t, s)

	var me struct{ Admin bool }
	if w := request(t, h, "GET", "/workshop/api/me", pcToken, ""); w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &me) != nil || !me.Admin {
		t.Fatalf("the linked PC must know it administers: %d %s", w.Code, w.Body)
	}
	if w := request(t, h, "GET", "/workshop/api/me", tabletToken, ""); w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &me) != nil || me.Admin {
		t.Fatalf("an ordinary device does not administer: %d %s", w.Code, w.Body)
	}
	for _, call := range [][3]string{{"GET", "/workshop/api/devices", ""}, {"POST", "/workshop/api/devices/invite", "{}"}, {"POST", "/workshop/api/devices/" + pc.ID + "/revoked", "{}"}} {
		if w := request(t, h, call[0], call[1], tabletToken, call[2]); w.Code != 403 {
			t.Fatalf("an ordinary device must not %s %s: %d", call[0], call[1], w.Code)
		}
		if w := request(t, h, call[0], call[1], "", call[2]); w.Code != 401 {
			t.Fatalf("an anonymous request must not %s %s: %d", call[0], call[1], w.Code)
		}
	}
	var list struct {
		Devices []struct {
			ID, Name, Status string
			Admin, Self      bool
		}
	}
	if w := request(t, h, "GET", "/workshop/api/devices", pcToken, ""); w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &list) != nil || len(list.Devices) != 2 {
		t.Fatalf("device list: %d %s", w.Code, w.Body)
	}
	for _, d := range list.Devices {
		if (d.ID == pc.ID) != (d.Admin && d.Self) {
			t.Fatalf("the list must mark the PC as the administering device, and as this device: %+v", d)
		}
	}
	var invite struct {
		URL   string `json:"url"`
		QRSVG string `json:"qr_svg"`
	}
	w := request(t, h, "POST", "/workshop/api/devices/invite", pcToken, "{}")
	if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &invite) != nil || !strings.HasPrefix(invite.URL, app+"/workshop/#pair=") || strings.Contains(invite.URL, "link=1") || !strings.HasPrefix(invite.QRSVG, "<svg") {
		t.Fatalf("the PC must create an ordinary invite: %d %s", w.Code, w.Body)
	}
	code := strings.TrimPrefix(invite.URL, app+"/workshop/#pair=")
	phone, phoneToken, err := s.Claim(code, "Téléphone")
	if err != nil || phone.Status != "pending" {
		t.Fatalf("a device invited from the app waits for approval: %+v %v", phone, err)
	}
	if w := request(t, h, "POST", "/workshop/api/devices/"+phone.ID+"/approved", pcToken, "{}"); w.Code != 200 {
		t.Fatalf("the PC must approve from the app: %d %s", w.Code, w.Body)
	}
	if _, err := s.Authenticate(phoneToken); err != nil {
		t.Fatalf("approved from the app: %v", err)
	}
	if w := request(t, h, "POST", "/workshop/api/devices/"+tabletID+"/revoked", pcToken, "{}"); w.Code != 200 {
		t.Fatalf("the PC must revoke from the app: %d", w.Code)
	}
	if _, err := s.Authenticate(tabletToken); !errors.Is(err, ErrRevoked) {
		t.Fatalf("revoked from the app: %v", err)
	}
	if w := request(t, h, "POST", "/workshop/api/devices/"+pc.ID+"/revoked", pcToken, "{}"); w.Code != 400 {
		t.Fatalf("the PC must not revoke itself from the app: %d", w.Code)
	}

	// The owner page grants administration to a device approved before this existed.
	owner := newOwnerFace(t, s, app)
	session := owner.session()
	if code := owner.do("POST", "/workshop-owner/api/devices/"+phone.ID+"/admin", "{}", session, nil); code != 200 {
		t.Fatalf("the owner must grant administration: %d", code)
	}
	if w := request(t, h, "GET", "/workshop/api/me", phoneToken, ""); json.Unmarshal(w.Body.Bytes(), &me) != nil || !me.Admin {
		t.Fatalf("granted from the owner page: %s", w.Body)
	}
	if code := owner.do("POST", "/workshop-owner/api/devices/"+tabletID+"/admin", "{}", session, nil); code == 200 {
		t.Fatal("a revoked device must not be granted administration")
	}
	var state struct {
		Devices []struct {
			ID    string
			Admin bool
		}
	}
	if code := owner.do("GET", "/workshop-owner/api/state", "", session, &state); code != 200 {
		t.Fatalf("owner state: %d", code)
	}
	admins := 0
	for _, d := range state.Devices {
		if d.Admin {
			admins++
		}
	}
	if admins != 2 {
		t.Fatalf("the owner page must show who administers: %+v", state.Devices)
	}
}

// TestLitWidgetsAreServed: the widget module and the vendored Lit are part of the client the Desktop serves.
func TestLitWidgetsAreServed(t *testing.T) {
	s, _, _ := fixture(t)
	h, err := Handler(s, "https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"/workshop/widgets.js", "/workshop/vendor/lit-all.min.js"} {
		if w := request(t, h, "GET", p, "", ""); w.Code != 200 || !strings.Contains(w.Header().Get("Content-Type"), "javascript") {
			t.Fatalf("%s: %d %q", p, w.Code, w.Header().Get("Content-Type"))
		}
	}
}
