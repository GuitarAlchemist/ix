package workshop

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// ownerFace is the loopback owner face over a store, with a way to open an owner session the way the Desktop shortcut
// does (HMAC proof with the persistent key, which never travels).
type ownerFace struct {
	t        *testing.T
	handler  http.Handler
	origin   string
	key      []byte
	sessions int
}

func newOwnerFace(t *testing.T, s *Store, appOrigin string) *ownerFace {
	return newOwnerFaceAt(t, s, appOrigin, "http://127.0.0.1:18999")
}

func newOwnerFaceAt(t *testing.T, s *Store, appOrigin, origin string) *ownerFace {
	t.Helper()
	o := &ownerFace{t: t, origin: origin, key: []byte(strings.Repeat("k", 32))}
	var err error
	if o.handler, err = OwnerHandler(s, hex.EncodeToString(o.key), o.origin, appOrigin); err != nil {
		t.Fatal(err)
	}
	return o
}

func (o *ownerFace) do(method, path, body, bearer string, out any) int {
	o.t.Helper()
	req := httptest.NewRequest(method, o.origin+path, strings.NewReader(body))
	if method != http.MethodGet {
		req.Header.Set("Origin", o.origin)
		req.Header.Set("Content-Type", "application/json")
	}
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	rec := httptest.NewRecorder()
	o.handler.ServeHTTP(rec, req)
	if out != nil && rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), out); err != nil {
			o.t.Fatalf("%s %s: %v in %s", method, path, err, rec.Body.String())
		}
	}
	return rec.Code
}

func (o *ownerFace) session() string {
	o.t.Helper()
	o.sessions++ // a nonce opens one session only
	nonce := strings.Repeat("c", 60) + strconv.FormatInt(int64(1000+o.sessions), 10)
	var challenge struct {
		Epoch   string
		Expires int64
	}
	if code := o.do("GET", "/workshop-owner/challenge?nonce="+nonce, "", "", &challenge); code != 200 {
		o.t.Fatalf("challenge: %d", code)
	}
	mac := hmac.New(sha256.New, o.key)
	mac.Write([]byte(strings.Join([]string{"ixforge-owner/1", "session", o.origin, nonce, challenge.Epoch, strconv.FormatInt(challenge.Expires, 10)}, "\n")))
	body, _ := json.Marshal(map[string]any{"nonce": nonce, "epoch": challenge.Epoch, "expires": challenge.Expires, "proof": hex.EncodeToString(mac.Sum(nil))})
	var session struct{ Session string }
	if code := o.do("POST", "/workshop-owner/session", string(body), "", &session); code != 200 || session.Session == "" {
		o.t.Fatalf("session: %d", code)
	}
	return session.Session
}

// link presses "Relier ce PC" on the owner face and returns the link it hands to the PC's own IxForge.
func (o *ownerFace) link() string {
	o.t.Helper()
	var result struct{ URL string }
	if code := o.do("POST", "/workshop-owner/api/link", "{}", o.session(), &result); code != 200 {
		o.t.Fatalf("the owner face must link this PC in one request: status %d", code)
	}
	return result.URL
}

// TestOwnerLinksThisPCWithoutACode: the owner, on the PC that hosts the Desktop, links that PC's IxForge with one
// action. The owner's request is the approval, so the device is approved at once; the link works once, and it does not
// change ordinary pairing, which stays pending until approved.
func TestOwnerLinksThisPCWithoutACode(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "link-pc.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	const app = "https://workshop.invalid"
	owner := newOwnerFace(t, s, app)
	if code := owner.do("POST", "/workshop-owner/api/link", "{}", "", nil); code == 200 {
		t.Fatal("linking must require an owner session")
	}
	if code := owner.do("POST", "/workshop-owner/api/link", "{}", "not-a-session", nil); code == 200 {
		t.Fatal("linking must refuse an unknown session")
	}
	link := owner.link()
	u, err := url.Parse(link)
	if err != nil || u.Scheme+"://"+u.Host != app || u.Path != "/workshop/" {
		t.Fatalf("link must open this Desktop's own app: %q %v", link, err)
	}
	fragment, _ := url.ParseQuery(u.Fragment)
	code := fragment.Get("pair")
	if code == "" || fragment.Get("link") != "1" {
		t.Fatalf("link must carry a single-use secret in its fragment and say it links this PC: %q", u.Fragment)
	}
	d, token, err := s.Claim(code, "PC")
	if err != nil || d.Status != "approved" {
		t.Fatalf("a linked PC is approved by the owner's own action: %+v %v", d, err)
	}
	if _, err = s.Authenticate(token); err != nil {
		t.Fatalf("the linked PC must be usable at once: %v", err)
	}
	if _, _, err = s.Claim(code, "PC again"); !errors.Is(err, ErrInvite) {
		t.Fatalf("the link must work once: %v", err)
	}
	invite, err := s.NewInvite(app)
	if err != nil {
		t.Fatal(err)
	}
	if d, _, err = s.Claim(invite.Code, "Tablet"); err != nil || d.Status != "pending" {
		t.Fatalf("ordinary pairing must stay pending until the owner approves: %+v %v", d, err)
	}
	// A link that was not used must not make the next ordinary invite approved.
	owner.link()
	if invite, err = s.NewInvite(app); err != nil {
		t.Fatal(err)
	}
	if d, _, err = s.Claim(invite.Code, "Phone"); err != nil || d.Status != "pending" {
		t.Fatalf("an unused link must not leak approval into the next invite: %+v %v", d, err)
	}
}

// TestThisPCIsLinkedInOneStepAndSendsItsDrafts: an unpaired IxForge on the PC holds a draft waiting to be sent and a
// local checklist. Opening the owner's link connects it without typing a code or a second approval, under the name
// "PC"; the waiting draft is saved on the Desktop, and the local checklist is still there.
func TestThisPCIsLinkedInOneStepAndSendsItsDrafts(t *testing.T) {
	f := newBrowserFixture(t)
	owner := newOwnerFace(t, f.store, f.origin)
	profile := t.TempDir()
	page, quit := startChromeIn(t, f.chrome, f.pin, profile)
	page.navigate(f.origin + "/workshop/")
	var settled bool
	page.evalInto(`(async()=>{for(let i=0;i<150;i++){const r=await navigator.serviceWorker.getRegistration('/workshop/');if(r?.active&&!r.waiting&&!r.installing&&navigator.serviceWorker.controller&&!document.querySelector('main').inert&&document.getElementById('update-status').textContent.startsWith('Client à jour'))return true;await new Promise(d=>setTimeout(d,100))}return false})()`, &settled)
	if !settled {
		t.Fatal("the client never settled after its first load")
	}
	page.typeInto("new-name", `Créer un workspace "Courses" avec une liste "Épicerie"`)
	page.press("#new-list")
	page.until("tool on this device", `document.getElementById('connection').textContent==='Sur cet appareil seulement' && document.querySelector('[data-item-add]')?.disabled===false`)
	page.evalInto(`(()=>{const f=document.querySelector('[data-item-text]');f.focus();f.select();return true})()`, nil)
	page.call("Input.insertText", map[string]any{"text": "Lait"})
	page.evalInto(`(()=>{document.querySelector('[data-item-add]').click();return true})()`, nil)
	page.until("local item", `document.querySelector('[data-xlet-local] .item-text')?.textContent==='Lait'`)
	quit()

	// The owner presses "Relier ce PC"; the link opens the PC's own IxForge.
	page, _ = startChromeIn(t, f.chrome, f.pin, profile)
	page.navigate(owner.link())
	page.until("connected without a code", `document.getElementById('connection').textContent==='Desktop connecté'`)
	page.until("the waiting draft is saved on the Desktop", `document.getElementById('pending').textContent==='' && document.getElementById('tool-status').textContent.startsWith('Partagé par le Desktop')`)
	page.until("the checklist is shared with its local item", `document.querySelector('[data-xlet-live] .item-text')?.textContent==='Lait'`)
	page.until("the secret left the address bar", `location.hash===''`)
	page.screenshot("pc-relie.png")
	devices, err := f.store.Devices()
	if err != nil || len(devices) != 1 || devices[0].Name != "PC" || devices[0].Status != "approved" {
		t.Fatalf("the Desktop must hold one approved device named PC: %+v %v", devices, err)
	}
}

// TestOwnerPageButtonLinksThisPC drives the real owner page: the owner presses "Relier ce PC", and the IxForge of the
// same browser profile ends up connected as an approved device, with no code typed and no approval step.
func TestOwnerPageButtonLinksThisPC(t *testing.T) {
	f := newBrowserFixture(t)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	owner := newOwnerFaceAt(t, f.store, f.origin, "http://"+listener.Addr().String())
	server := &http.Server{Handler: owner.handler}
	go server.Serve(listener)
	t.Cleanup(func() { server.Close() })

	page, _ := startChromeIn(t, f.chrome, f.pin, t.TempDir())
	page.navigate(owner.origin + "/workshop-owner/#session=" + owner.session())
	page.until("owner page ready", `document.getElementById('status').textContent==='Desktop connecté' && document.getElementById('link-pc').textContent==='Relier ce PC'`)
	page.click("link-pc")
	page.until("link offered", `document.getElementById('link-view').hidden===false && document.getElementById('link-open').href.startsWith(`+strconv.Quote(f.origin+"/workshop/#pair=")+`) && document.getElementById('link-open').href.endsWith('&link=1')`)
	page.screenshot("proprietaire-relier-ce-pc.png")
	// The button opens IxForge in a new window of this profile. If the browser blocked that window, the owner uses the
	// link shown under the button: do the same here.
	linked := func() bool {
		devices, err := f.store.Devices()
		return err == nil && len(devices) == 1 && devices[0].Name == "PC" && devices[0].Status == "approved"
	}
	opened := false
	for deadline := time.Now().Add(8 * time.Second); time.Now().Before(deadline) && !opened; time.Sleep(200 * time.Millisecond) {
		opened = linked()
	}
	t.Logf("the button's own window linked the PC: %v", opened)
	if !opened {
		var href string
		page.evalInto(`document.getElementById('link-open').href`, &href)
		page.navigate(href)
	} else {
		page.navigate(f.origin + "/workshop/")
	}
	page.until("IxForge connected", `document.getElementById('connection').textContent==='Desktop connecté'`)
	if !linked() {
		devices, _ := f.store.Devices()
		t.Fatalf("the Desktop must hold one approved device named PC: %+v", devices)
	}
}
