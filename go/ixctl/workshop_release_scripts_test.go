package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/workshop"
)

// TestAReleaseBuiltHereIsOfferedByTheDesktop: Publish-IxForgeRelease.ps1 puts a build of ixctl in the releases folder
// with its manifest (version, hash, commit, client version, notes), and the Desktop then offers it to its admin
// devices. A build whose version is not the one named, or a version already published, is refused.
func TestAReleaseBuiltHereIsOfferedByTheDesktop(t *testing.T) {
	if _, err := exec.LookPath("pwsh"); err != nil {
		t.Skip("pwsh 7.4+ not installed")
	}
	dir := t.TempDir()
	exe := filepath.Join(dir, "ixctl.exe")
	build := exec.Command("go", "build", "-trimpath", "-ldflags", "-X main.version=0.1.0-ixforge.17", "-o", exe, ".")
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	releases := filepath.Join(dir, "releases")
	publish := func(version string) (string, error) {
		out, err := exec.Command("pwsh", "-NoProfile", "-NonInteractive", "-File", "deploy/workshop/Publish-IxForgeRelease.ps1",
			"-Version", version, "-Exe", exe, "-Commit", "aca2e116", "-Notes", "Mise à jour depuis la PWA", "-Releases", releases).CombinedOutput()
		return string(out), err
	}
	if out, err := publish("0.1.0-ixforge.18"); err == nil || !strings.Contains(out, "version") {
		t.Fatalf("a build that says another version is refused: %v\n%s", err, out)
	}
	if out, err := publish("0.1.0-ixforge.17"); err != nil {
		t.Fatalf("publish: %v\n%s", err, out)
	}
	var manifest map[string]string
	raw, err := os.ReadFile(filepath.Join(releases, "0.1.0-ixforge.17", "release.json"))
	if err != nil || json.Unmarshal(raw, &manifest) != nil {
		t.Fatalf("manifest: %s %v", raw, err)
	}
	var branding struct{ Version string }
	brand, _ := os.ReadFile(filepath.Join("internal", "workshop", "web", "branding.json"))
	_ = json.Unmarshal(brand, &branding)
	if manifest["sha256"] != upperSHA(mustRead(t, exe)) || manifest["commit"] != "aca2e116" || manifest["client_version"] != branding.Version ||
		manifest["notes"] != "Mise à jour depuis la PWA" || manifest["built_at"] == "" {
		t.Errorf("the manifest says what the release is: %v", manifest)
	}
	if out, err := publish("0.1.0-ixforge.17"); err == nil {
		t.Errorf("a published version is never overwritten:\n%s", out)
	}

	u := &workshop.Updates{Installed: "0.1.0-ixforge.16", Releases: releases}
	offered, err := u.Offered()
	if err != nil || len(offered) != 1 || offered[0].Version != "0.1.0-ixforge.17" || offered[0].SHA256 != manifest["sha256"] {
		t.Fatalf("the Desktop offers the published release: %v %v", offered, err)
	}
}

// TestInstallingTheUpdaterShowsItsPlanFirst: without -Apply, Install-IxForgeUpdater.ps1 changes nothing and says what
// it would do: a task run as SYSTEM every minute, the runner and the reviewed updater copied where only administrators
// write (with the hashes of the files reviewed here), and who may write releases.
func TestInstallingTheUpdaterShowsItsPlanFirst(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the updater is a Windows task run as SYSTEM; its plan reads ProgramData and ProgramFiles")
	}
	if _, err := exec.LookPath("pwsh"); err != nil {
		t.Skip("pwsh 7.4+ not installed")
	}
	target := filepath.Join(t.TempDir(), "updater")
	out, err := exec.Command("pwsh", "-NoProfile", "-NonInteractive", "-File", "deploy/workshop/Install-IxForgeUpdater.ps1",
		"-OwnerSid", "S-1-5-21-1-2-3-1001", "-UpdaterDir", target).CombinedOutput()
	if err != nil {
		t.Fatalf("plan: %v\n%s", err, out)
	}
	var plan struct {
		Task, Principal, Every, Pwsh, UpdaterDir, Releases, Staging, ReleaseWriters, Apply string
		Files                                                                              map[string]string
	}
	if err := json.Unmarshal(out, &plan); err != nil {
		t.Fatalf("the plan is JSON: %v\n%s", err, out)
	}
	if plan.Task != "IxForge Updater" || plan.Principal != "SYSTEM" || plan.Every != "PT1M" || plan.UpdaterDir != target || plan.Apply != "false" ||
		!strings.Contains(plan.ReleaseWriters, "S-1-5-21-1-2-3-1001") || !strings.HasPrefix(strings.ToLower(plan.Pwsh), strings.ToLower(os.Getenv("ProgramFiles"))) {
		t.Errorf("the plan: %+v", plan)
	}
	for _, name := range []string{"Run-IxForgeUpdate.ps1", "Update-IxForge.ps1", "Update-IxForge.Core.ps1"} {
		if plan.Files[name] != upperSHA(mustRead(t, filepath.Join("deploy", "workshop", name))) {
			t.Errorf("%s: the plan names the hash of the reviewed file: %v", name, plan.Files)
		}
	}
	if _, err := os.Stat(target); !os.IsNotExist(err) {
		t.Errorf("the plan changes nothing: %v", err)
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
