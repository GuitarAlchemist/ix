package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestOwnerLauncherProvesRuntimeWithoutSendingPersistentKey(t *testing.T) {
	dir := t.TempDir()
	prepareWorkshopFixture(t, dir, "fixture")
	runtime, err := startWorkshop(workshopSettings{Version: 1, Enabled: true, Origin: "https://workshop.invalid", Listen: "127.0.0.1:0"}, dir, "fixture")
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.close()
	configPath := filepath.Join(dir, "workshop", "config.json")
	writeConfig := func(addr string) {
		t.Helper()
		data, _ := json.Marshal(workshopSettings{Version: 1, Enabled: true, Origin: "https://workshop.invalid", Listen: addr})
		if err := os.WriteFile(configPath, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	invoke := func() ([]byte, error) {
		return exec.Command("pwsh", "-NoProfile", "-File", "deploy/workshop/Open-IxForgeOwner.ps1", "-StateDir", dir, "-InstanceName", "fixture", "-VerifyOnly").CombinedOutput()
	}
	writeConfig(runtime.addr)
	output, err := invoke()
	if err != nil {
		t.Fatalf("launcher: %v %s", err, output)
	}
	if strings.Contains(string(output), runtime.ownerKey) {
		t.Fatal("persistent key printed")
	}
	var captured string
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured = r.URL.String() + r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"epoch":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","expires":9999999999,"proof":"0000000000000000000000000000000000000000000000000000000000000000"}`))
	}))
	defer fake.Close()
	u, _ := url.Parse(fake.URL)
	writeConfig(u.Host)
	output, err = invoke()
	if err == nil {
		t.Fatal("launcher trusted replacement port owner")
	}
	if strings.Contains(captured, runtime.ownerKey) || strings.Contains(string(output), runtime.ownerKey) {
		t.Fatal("persistent key reached fake server/output")
	}
}
