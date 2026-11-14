package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/workshop"
)

func TestWorkshopConfigurationAndKeyAreExplicit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	for _, data := range []string{`{"version":2,"enabled":true}`, `{"version":1,"enabled":true,"origin":"http://workshop.invalid","listen":"127.0.0.1:8761"}`, `{"version":1,"enabled":true,"origin":"https://workshop.invalid","listen":"0.0.0.0:8761"}`, `{"version":1,"enabled":true,"origin":"https://workshop.invalid","listen":"127.0.0.1:8739"}`, `{"version":1,"enabled":true,"origin":"https://workshop.invalid","listen":"127.0.0.1:0"}`, `{"version":1,"enabled":false,"shell":"cmd.exe"}`} {
		os.WriteFile(path, []byte(data), 0600)
		if _, err := readWorkshopSettings(path); err == nil {
			t.Fatalf("accepted %s", data)
		}
	}
	if _, err := startWorkshop(workshopSettings{Version: 1, Enabled: true, Origin: "https://workshop.invalid", Listen: "127.0.0.1:0"}, dir, "fixture"); err == nil {
		t.Fatal("owner key silently created")
	}
	if _, err := os.Stat(filepath.Join(dir, "workshop")); !os.IsNotExist(err) {
		t.Fatal("activation without provisioning changed data")
	}
	prepareWorkshopFixture(t, dir, "fixture")
	settings := workshopSettings{Version: 1, Enabled: true, Origin: "https://workshop.invalid", Listen: "127.0.0.1:0"}
	one, err := startWorkshop(settings, dir, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer one.close()
	if two, err := startWorkshop(settings, dir, "fixture"); err == nil {
		two.close()
		t.Fatal("two owners for one SQLite file")
	}
}

func TestRealHostWorkshopFailureDoesNotLeaveAControlListenerOrLock(t *testing.T) {
	dir := t.TempDir()
	log := &syncBuffer{}
	options := hostOpts(fakeSpec(t, "fixture", nil), dir, log)
	options.workshop = workshopSettings{Version: 1, Enabled: true, Origin: "https://workshop.invalid", Listen: "127.0.0.1:0"}
	_, done := startHost(options)
	if code := waitExit(t, done, 10*time.Second); code != exitStartFailed {
		t.Fatal(code)
	}
	if _, err := os.Stat(filepath.Join(dir, "fixture.json")); !os.IsNotExist(err) {
		t.Fatal("control state survived failed Workshop activation")
	}
	if strings.Contains(log.String(), `"event":"host.ready"`) {
		t.Fatal("host declared ready before Workshop started")
	}
	options.workshop = workshopSettings{}
	manager, next := startHost(options)
	manager.waitState(t, hostRunning, 10*time.Second)
	manager.controls <- hostControl{Cmd: controlStop, Source: "fixture"}
	if code := waitExit(t, next, 10*time.Second); code != exitOK {
		t.Fatal(code)
	}
}
func TestWorkshopStopDrainsAnAdmittedCommitBeforeClosingDatabase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.sqlite")
	store, err := workshop.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	invite, _ := store.NewInvite("https://workshop.invalid")
	device, token, err := store.Claim(invite.Code, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	store.Decide(device.ID, "approved")
	app, _ := workshop.Handler(store, "https://workshop.invalid")
	h := &workshopHost{store: store}
	entered, release := make(chan struct{}), make(chan struct{})
	testServer := httptest.NewUnstartedServer(h.admit(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { close(entered); <-release; app.ServeHTTP(w, r) })))
	h.server = testServer.Config
	testServer.Start()
	defer testServer.Close()
	payload := `{"op_id":"drain-op","base_revision":0,"project":{"id":"p","title":"drained","workspaces":[]}}`
	result := make(chan int, 1)
	go func() {
		r, _ := http.NewRequest("POST", testServer.URL+"/workshop/api/operations", strings.NewReader(payload))
		r.Host = "workshop.invalid"
		r.Header.Set("Origin", "https://workshop.invalid")
		r.Header.Set("Content-Type", "application/json")
		r.Header.Set("Authorization", "Bearer "+token)
		response, e := http.DefaultClient.Do(r)
		if e != nil {
			result <- 0
			return
		}
		io.Copy(io.Discard, response.Body)
		response.Body.Close()
		result <- response.StatusCode
	}()
	<-entered
	h.stopAccepting()
	denied := httptest.NewRecorder()
	h.server.Handler.ServeHTTP(denied, httptest.NewRequest("POST", "/workshop/api/operations", strings.NewReader(payload)))
	if denied.Code != 503 {
		t.Fatalf("admitted after stop: %d", denied.Code)
	}
	closed := make(chan struct{})
	go func() { h.close(); close(closed) }()
	select {
	case <-closed:
		t.Fatal("close skipped admitted operation")
	default:
	}
	close(release)
	if code := <-result; code != 200 {
		t.Fatalf("admitted commit: %d", code)
	}
	<-closed
	reopened, err := workshop.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	changes, err := reopened.Changes(token, 0)
	if err != nil || len(changes.Events) != 1 {
		t.Fatalf("commit lost: %v %+v", err, changes)
	}
	var op workshop.Operation
	json.Unmarshal([]byte(payload), &op)
	receipt, err := reopened.Save(token, op)
	if err != nil || receipt.Revision != 1 {
		t.Fatalf("replayed twice: %+v %v", receipt, err)
	}
}
