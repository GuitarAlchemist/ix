package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The updater task (Run-IxForgeUpdate.ps1, run as SYSTEM every minute once the owner installed it) takes the request
// an admin device left through the app, checks the release again, copies it where only SYSTEM and administrators
// write, and hands it to the reviewed Update-IxForge.ps1 with the PC's current hashes. These tests run the real script
// against temporary folders and a stub of Update-IxForge.ps1 that records how it was called: no service is touched.

// stubUpdater records its parameters in $env:STUB_OUT\called-<n>.json and leaves a receipt as Update-IxForge.ps1 does;
// STUB_MODE=fail makes it throw before any receipt.
const stubUpdater = `#requires -Version 7.4
[CmdletBinding(SupportsShouldProcess,ConfirmImpact='High')]
param([string]$Candidate,[string]$CandidateSHA256,[string]$ExpectedInstalledSHA256,[string]$ExpectedImagePath,
 [string]$StateDir,[string]$InstanceName,[string]$ExpectedClientVersion,[string]$InstalledExe,[string]$ExpectedSpecSHA256,
 [switch]$Apply,[switch]$Rollback,[string]$BackupDirectory)
$n = @(Get-ChildItem -LiteralPath $env:STUB_OUT -Filter 'called-*.json').Count + 1
[ordered]@{Candidate=$Candidate;CandidateSHA256=$CandidateSHA256;ExpectedInstalledSHA256=$ExpectedInstalledSHA256;
 ExpectedImagePath=$ExpectedImagePath;ExpectedClientVersion=$ExpectedClientVersion;InstalledExe=$InstalledExe;
 ExpectedSpecSHA256=$ExpectedSpecSHA256;Apply=[bool]$Apply;CandidateBytes=[IO.File]::ReadAllText($Candidate)} |
 ConvertTo-Json | Set-Content -LiteralPath (Join-Path $env:STUB_OUT "called-$n.json")
if ($env:STUB_MODE -eq 'fail') { throw 'stub: the candidate did not start' }
$run = Join-Path $env:STUB_RECEIPTS 'update-0123456789abcdef0123456789abcdef'
New-Item -ItemType Directory -Force $run | Out-Null
@{State='completed'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'receipt.json')
`

type runnerFixture struct {
	t                                         *testing.T
	releases, requests, staging, out, receipt string
	installed, spec, updater                  string
}

func newRunnerFixture(t *testing.T) *runnerFixture {
	t.Helper()
	if _, err := exec.LookPath("pwsh"); err != nil {
		t.Skip("pwsh 7.4+ not installed")
	}
	dir := t.TempDir()
	f := &runnerFixture{t: t}
	for _, p := range []*string{&f.releases, &f.requests, &f.staging, &f.out, &f.receipt} {
		*p = t.TempDir()
	}
	f.installed = filepath.Join(dir, "installed-ixctl.exe")
	f.spec = filepath.Join(dir, "spec.json")
	f.updater = filepath.Join(dir, "Update-IxForge.ps1")
	for path, body := range map[string]string{f.installed: "sixteen", f.spec: `{"spec":1}`, f.updater: stubUpdater} {
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return f
}

func upperSHA(b []byte) string {
	sum := sha256.Sum256(b)
	return strings.ToUpper(hex.EncodeToString(sum[:]))
}

// release puts a release as Publish-IxForgeRelease.ps1 does and returns its hash.
func (f *runnerFixture) release(version, body string) string {
	f.t.Helper()
	dir := filepath.Join(f.releases, version)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		f.t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ixctl.exe"), []byte(body), 0o644); err != nil {
		f.t.Fatal(err)
	}
	manifest, _ := json.Marshal(map[string]string{"version": version, "sha256": upperSHA([]byte(body)), "client_version": "0.1.2", "notes": "n"})
	if err := os.WriteFile(filepath.Join(dir, "release.json"), manifest, 0o644); err != nil {
		f.t.Fatal(err)
	}
	return upperSHA([]byte(body))
}

// ask writes the request as the Desktop does.
func (f *runnerFixture) ask(version, hash string) {
	f.t.Helper()
	raw, _ := json.Marshal(map[string]string{"version": version, "sha256": hash, "requested_at": time.Now().UTC().Format(time.RFC3339), "device_id": "pc", "device_name": "PC"})
	if err := os.WriteFile(filepath.Join(f.requests, "update-request.json"), raw, 0o644); err != nil {
		f.t.Fatal(err)
	}
}

func (f *runnerFixture) run(mode string) string {
	f.t.Helper()
	cmd := exec.Command("pwsh", "-NoProfile", "-NonInteractive", "-File", "deploy/workshop/Run-IxForgeUpdate.ps1",
		"-Requests", f.requests, "-Releases", f.releases, "-Staging", f.staging, "-Updater", f.updater,
		"-InstalledExe", f.installed, "-SpecPath", f.spec, "-ReceiptsDir", f.receipt,
		"-ExpectedImagePath", `"C:\Program Files\ixctl\ixctl.exe" service run "C:\ProgramData\ixctl\spec.json"`)
	cmd.Env = append(os.Environ(), "STUB_OUT="+f.out, "STUB_RECEIPTS="+f.receipt, "STUB_MODE="+mode)
	out, err := cmd.CombinedOutput()
	if err != nil {
		f.t.Fatalf("the runner itself must not fail: %v\n%s", err, out)
	}
	return string(out)
}

func (f *runnerFixture) calls() []map[string]any {
	f.t.Helper()
	files, _ := filepath.Glob(filepath.Join(f.out, "called-*.json"))
	calls := []map[string]any{}
	for _, file := range files {
		raw, _ := os.ReadFile(file)
		var c map[string]any
		if err := json.Unmarshal(raw, &c); err != nil {
			f.t.Fatal(err)
		}
		calls = append(calls, c)
	}
	return calls
}

func (f *runnerFixture) result() map[string]any {
	f.t.Helper()
	var r map[string]any
	raw, err := os.ReadFile(filepath.Join(f.requests, "update-result.json"))
	if err != nil || json.Unmarshal(raw, &r) != nil {
		f.t.Fatalf("no readable result: %s %v", raw, err)
	}
	return r
}

func (f *runnerFixture) seen() time.Time {
	f.t.Helper()
	var s struct {
		SeenAt string `json:"seen_at"`
	}
	raw, err := os.ReadFile(filepath.Join(f.requests, "updater-seen.json"))
	if err != nil || json.Unmarshal(raw, &s) != nil {
		f.t.Fatalf("no heartbeat: %s %v", raw, err)
	}
	at, err := time.Parse(time.RFC3339, s.SeenAt)
	if err != nil {
		f.t.Fatalf("heartbeat time: %v", err)
	}
	return at
}

// TestTheUpdaterRunnerInstallsTheRequestedRelease: with no request, the runner only says it is there. With one, it
// takes it (so it is never run twice), copies the release where only SYSTEM writes, and calls the reviewed updater
// with the staged copy, the hashes of the PC as it is now and the release's client version; then it writes the
// receipt's state as the result.
func TestTheUpdaterRunnerInstallsTheRequestedRelease(t *testing.T) {
	f := newRunnerFixture(t)
	f.run("")
	if time.Since(f.seen()) > time.Minute || len(f.calls()) != 0 {
		t.Fatalf("without a request: a heartbeat and nothing else (%d calls)", len(f.calls()))
	}

	hash := f.release("0.1.0-ixforge.17", "seventeen")
	f.ask("0.1.0-ixforge.17", hash)
	f.run("")
	calls := f.calls()
	if len(calls) != 1 {
		t.Fatalf("one call of the reviewed updater: %v", calls)
	}
	c := calls[0]
	staged, _ := c["Candidate"].(string)
	if !strings.HasPrefix(staged, f.staging) || c["CandidateBytes"] != "seventeen" || c["CandidateSHA256"] != hash || c["Apply"] != true {
		t.Errorf("the staged copy of the release, applied: %v", c)
	}
	if c["ExpectedInstalledSHA256"] != upperSHA([]byte("sixteen")) || c["ExpectedSpecSHA256"] != upperSHA([]byte(`{"spec":1}`)) ||
		c["InstalledExe"] != f.installed || c["ExpectedClientVersion"] != "0.1.2" || !strings.Contains(c["ExpectedImagePath"].(string), "service run") {
		t.Errorf("the PC's current hashes, command and the release's client version: %v", c)
	}
	if _, err := os.Stat(filepath.Join(f.requests, "update-request.json")); !os.IsNotExist(err) {
		t.Errorf("the request is taken: %v", err)
	}
	r := f.result()
	if r["version"] != "0.1.0-ixforge.17" || r["state"] != "completed" || r["receipt"] != "update-0123456789abcdef0123456789abcdef" || r["finished_at"] == "" {
		t.Errorf("the receipt's state as the result: %v", r)
	}
	f.run("")
	if len(f.calls()) != 1 {
		t.Fatal("a taken request is never run again")
	}
}

// TestTheUpdaterRunnerRefusesAChangedRelease: a release whose file no longer has the hash the admin device was shown
// is not handed to the updater; the result says why, and the request is not kept for another try.
func TestTheUpdaterRunnerRefusesAChangedRelease(t *testing.T) {
	f := newRunnerFixture(t)
	hash := f.release("0.1.0-ixforge.17", "seventeen")
	f.ask("0.1.0-ixforge.17", hash)
	if err := os.WriteFile(filepath.Join(f.releases, "0.1.0-ixforge.17", "ixctl.exe"), []byte("tampered"), 0o644); err != nil {
		t.Fatal(err)
	}
	f.run("")
	r := f.result()
	if len(f.calls()) != 0 || r["state"] != "failed_before_replacement" || !strings.Contains(r["error"].(string), "hash") {
		t.Fatalf("a changed release is refused before the updater: %d calls, %v", len(f.calls()), r)
	}
	if _, err := os.Stat(filepath.Join(f.requests, "update-request.json")); !os.IsNotExist(err) {
		t.Errorf("no retry: the request is gone: %v", err)
	}
	f.ask("0.1.0-ixforge.18", hash)
	f.run("")
	if r := f.result(); len(f.calls()) != 0 || r["version"] != "0.1.0-ixforge.18" || r["state"] != "failed_before_replacement" {
		t.Fatalf("a release that is not there is refused: %v", r)
	}
}

// TestTheUpdaterRunnerReportsAFailedUpdate: when the reviewed updater fails, the result says so with its message,
// and the runner does not try again on its next run.
func TestTheUpdaterRunnerReportsAFailedUpdate(t *testing.T) {
	f := newRunnerFixture(t)
	hash := f.release("0.1.0-ixforge.17", "seventeen")
	f.ask("0.1.0-ixforge.17", hash)
	f.run("fail")
	r := f.result()
	if len(f.calls()) != 1 || r["state"] != "failed" || !strings.Contains(r["error"].(string), "did not start") {
		t.Fatalf("the updater's failure is the result: %v", r)
	}
	f.run("fail")
	if len(f.calls()) != 1 {
		t.Fatal("a failed update is not retried by itself")
	}
}
