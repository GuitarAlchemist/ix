package main

import (
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkshopOwnerBrowserIntegration(t *testing.T) {
	if os.Getenv("WORKSHOP_OWNER_BROWSER") != "1" {
		t.Skip("opt-in fresh-profile owner UI browser test")
	}
	var proxy http.Handler
	bridge := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { proxy.ServeHTTP(w, r) }))
	defer bridge.Close()
	u, _ := url.Parse(bridge.URL)
	origin := "https://workshop.invalid:" + u.Port()
	dir := t.TempDir()
	prepareWorkshopFixture(t, dir, "fixture")
	os.WriteFile(filepath.Join(dir, "workshop", "fixture.workshop-owner.key"), []byte(strings.Repeat("A", 64)), 0600)
	runtime, err := startWorkshop(workshopSettings{Version: 1, Enabled: true, Origin: origin, Listen: "127.0.0.1:0"}, dir, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.close()
	target, _ := url.Parse("http://" + runtime.addr)
	proxy = httputil.NewSingleHostReverseProxy(target)
	pin := sha256.Sum256(bridge.Certificate().RawSubjectPublicKeyInfo)
	session := fixtureOwnerSession(t, "http://"+runtime.addr, runtime.ownerKey)
	cmd := exec.Command("node", "internal/workshop/e2e/owner-browser.mjs")
	cmd.Env = append(os.Environ(), "WORKSHOP_OWNER_ORIGIN=http://"+runtime.addr, "WORKSHOP_OWNER_SESSION="+session, "WORKSHOP_TABLET_ORIGIN="+origin, "WORKSHOP_FIXTURE_PIN="+base64.StdEncoding.EncodeToString(pin[:]), "WORKSHOP_ARTIFACT_DIR="+os.Getenv("WORKSHOP_ARTIFACT_DIR"))
	output, err := cmd.CombinedOutput()
	t.Log(string(output))
	if err != nil {
		t.Fatal(err)
	}
}
