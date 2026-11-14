package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestWorkshopHostDisabledAndExplicitMount(t *testing.T) {
	dir := t.TempDir()
	settings := workshopSettings{Version: 1}
	if runtime, err := startWorkshop(settings, dir, "fixture"); err != nil || runtime != nil {
		t.Fatalf("disabled: %v %v", runtime, err)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Fatal("disabled mode wrote data")
	}
	settings = workshopSettings{Version: 1, Enabled: true, Origin: "https://workshop.invalid", Listen: "127.0.0.1:0"}
	prepareWorkshopFixture(t, dir, "fixture")
	runtime, err := startWorkshop(settings, dir, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.close()
	base := "http://" + runtime.addr
	ownerToken := fixtureOwnerSession(t, base, runtime.ownerKey)
	call := func(method, path, host, origin, token, body string) (int, []byte) {
		t.Helper()
		r, _ := http.NewRequest(method, base+path, strings.NewReader(body))
		r.Host = host
		if origin != "" {
			r.Header.Set("Origin", origin)
		}
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Authorization", "Bearer "+token)
		res, err := http.DefaultClient.Do(r)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		b, _ := io.ReadAll(res.Body)
		return res.StatusCode, b
	}
	if status, _ := call("GET", "/workshop/", "workshop.invalid", "", "", ""); status != 200 {
		t.Fatal(status)
	}
	for _, path := range []string{"/v1/status", "/v1/apply", "/workshop-owner/api/state"} {
		if status, _ := call("GET", path, "workshop.invalid", "https://workshop.invalid", runtime.ownerKey, ""); status == 200 {
			t.Fatalf("tablet reached %s", path)
		}
	}
	if status, _ := call("GET", "/workshop-owner/api/state", runtime.addr, "", "global-ixctl-fixture", ""); status != 401 {
		t.Fatal(status)
	}
	status, body := call("POST", "/workshop-owner/api/invite", runtime.addr, base, ownerToken, "{}")
	if status != 200 {
		t.Fatalf("invite %d %s", status, body)
	}
	var invite struct {
		URL string `json:"url"`
	}
	json.Unmarshal(body, &invite)
	parts := strings.Split(invite.URL, "#pair=")
	if len(parts) != 2 {
		t.Fatal(invite.URL)
	}
	payload, _ := json.Marshal(map[string]string{"name": "Fixture tablet", "code": parts[1]})
	status, body = call("POST", "/workshop/api/pair", "workshop.invalid", "https://workshop.invalid", "", string(payload))
	if status != 200 {
		t.Fatalf("pair: %d %s", status, body)
	}
	var device struct {
		Token  string `json:"token"`
		Device struct {
			ID string `json:"id"`
		} `json:"device"`
	}
	json.Unmarshal(body, &device)
	if status, _ = call("GET", "/workshop/api/me", "workshop.invalid", "https://workshop.invalid", device.Token, ""); status != 403 {
		t.Fatal(status)
	}
	if status, _ = call("POST", "/workshop-owner/api/devices/"+device.Device.ID+"/approved", "workshop.invalid", "https://workshop.invalid", device.Token, "{}"); status != 403 {
		t.Fatalf("self approve: %d", status)
	}
	if status, body = call("POST", "/workshop-owner/api/devices/"+device.Device.ID+"/approved", runtime.addr, base, ownerToken, "{}"); status != 200 {
		t.Fatalf("approve %d %s", status, body)
	}
	if status, _ = call("GET", "/workshop/api/me", "workshop.invalid", "https://workshop.invalid", device.Token, ""); status != 200 {
		t.Fatal(status)
	}
	runtime.close()
	if _, err := http.Get(base + "/workshop/"); err == nil {
		t.Fatal("listener survived close")
	}
	reopened, err := startWorkshop(settings, dir, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.close()
	if reopened.ownerKey != runtime.ownerKey {
		t.Fatal("owner key rotated across restart")
	}
	if _, err := reopened.store.Authenticate(device.Token); err != nil {
		t.Fatal(err)
	}
}

func fixtureOwnerSession(t *testing.T, origin, key string) string {
	t.Helper()
	raw := make([]byte, 32)
	rand.Read(raw)
	nonce := hex.EncodeToString(raw)
	secret, _ := hex.DecodeString(key)
	sign := func(purpose, epoch string, expires int64, extra ...string) string {
		parts := []string{"ixforge-owner/1", purpose, origin, nonce, epoch, strconv.FormatInt(expires, 10)}
		parts = append(parts, extra...)
		mac := hmac.New(sha256.New, secret)
		mac.Write([]byte(strings.Join(parts, "\n")))
		return hex.EncodeToString(mac.Sum(nil))
	}
	response, err := http.Get(origin + "/workshop-owner/challenge?nonce=" + nonce)
	if err != nil {
		t.Fatal(err)
	}
	var challenge struct {
		Epoch   string `json:"epoch"`
		Expires int64  `json:"expires"`
		Proof   string `json:"proof"`
	}
	json.NewDecoder(response.Body).Decode(&challenge)
	response.Body.Close()
	if challenge.Proof != sign("challenge", challenge.Epoch, challenge.Expires) {
		t.Fatal("owner endpoint identity not proven")
	}
	body, _ := json.Marshal(map[string]any{"nonce": nonce, "epoch": challenge.Epoch, "expires": challenge.Expires, "proof": sign("session", challenge.Epoch, challenge.Expires)})
	request, _ := http.NewRequest("POST", origin+"/workshop-owner/session", bytes.NewReader(body))
	request.Header.Set("Origin", origin)
	request.Header.Set("Content-Type", "application/json")
	response, err = http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var session struct {
		Session string `json:"session"`
		Expires int64  `json:"expires"`
		Proof   string `json:"proof"`
	}
	json.NewDecoder(response.Body).Decode(&session)
	if session.Proof != sign("receipt", challenge.Epoch, challenge.Expires, session.Session, strconv.FormatInt(session.Expires, 10)) {
		t.Fatal("owner session receipt invalid")
	}
	return session.Session
}
func TestRealHostMountKeepsControlPlaneBoundary(t *testing.T) {
	dir := t.TempDir()
	log := &syncBuffer{}
	opts := hostOpts(fakeSpec(t, "fixture", nil), dir, log)
	opts.workshop = workshopSettings{Version: 1, Enabled: true, Origin: "https://workshop.invalid", Listen: "127.0.0.1:0"}
	prepareWorkshopFixture(t, dir, "fixture")
	scm, done := startHost(opts)
	scm.waitState(t, hostRunning, 10*time.Second)
	control := hostClient(t, dir, "fixture")
	if code, _, _ := httpDo(t, control, "GET", "/v1/status", nil, nil); code != 200 {
		t.Fatal(code)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "workshop", "fixture.workshop.json"))
	if err != nil {
		t.Fatal(err)
	}
	var app struct {
		Addr string `json:"addr"`
	}
	if json.Unmarshal(raw, &app) != nil || app.Addr == "" {
		t.Fatal(string(raw))
	}
	r, _ := http.NewRequest("GET", "http://"+app.Addr+"/v1/status", nil)
	r.Header.Set("Authorization", "Bearer "+control.token)
	r.Host = "workshop.invalid"
	res, err := http.DefaultClient.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != 404 {
		t.Fatalf("control via app: %d", res.StatusCode)
	}
	r, _ = http.NewRequest("GET", "http://"+control.addr+"/v1/status", nil)
	r.Host = "workshop.invalid"
	r.Header.Set("Origin", "https://workshop.invalid")
	res, err = http.DefaultClient.Do(r)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != 403 {
		t.Fatalf("tablet at control: %d", res.StatusCode)
	}
	scm.controls <- hostControl{Cmd: controlStop, Source: "fixture"}
	if code := waitExit(t, done, 10*time.Second); code != exitOK {
		t.Fatal(code)
	}
	if _, err := os.Stat(filepath.Join(dir, "workshop", "fixture.workshop.json")); !os.IsNotExist(err) {
		t.Fatal("app state survived stop")
	}
	if _, err := http.Get("http://" + app.Addr + "/workshop/"); err == nil {
		t.Fatal("app listener survived host stop")
	}
	if bytes.Contains([]byte(log.String()), []byte(control.token)) {
		t.Fatal("control token logged")
	}
}

func prepareWorkshopFixture(t *testing.T, dir, name string) {
	t.Helper()
	root := filepath.Join(dir, "workshop")
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, name+".workshop-owner.key"), []byte(strings.Repeat("a", 64)), 0600); err != nil {
		t.Fatal(err)
	}
}
