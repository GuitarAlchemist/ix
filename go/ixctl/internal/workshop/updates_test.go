package workshop

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The Desktop runs 0.1.0-ixforge.16 in these tests; the clock is fixed.
var updatesNow = time.Date(2026, 11, 15, 12, 0, 0, 0, time.UTC)

// addRelease puts a release in the releases folder as the owner's account does: <version>/ixctl.exe and release.json.
// A manifest may name another version or another hash than the file's, to be refused.
func addRelease(t *testing.T, dir, folder, manifestVersion string, exe []byte, manifestHash string) string {
	t.Helper()
	sum := sha256.Sum256(exe)
	hash := strings.ToUpper(hex.EncodeToString(sum[:]))
	if manifestHash == "" {
		manifestHash = hash
	}
	path := filepath.Join(dir, folder)
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "ixctl.exe"), exe, 0o644); err != nil {
		t.Fatal(err)
	}
	manifest, _ := json.Marshal(map[string]string{"version": manifestVersion, "sha256": manifestHash, "notes": "Notes of " + manifestVersion,
		"commit": "aca2e116", "built_at": "2026-11-15T11:00:00Z"})
	if err := os.WriteFile(filepath.Join(path, "release.json"), manifest, 0o644); err != nil {
		t.Fatal(err)
	}
	return hash
}

// updatesFixture is a Desktop at ixforge.16 with its releases and requests folders, a linked PC (an admin) and a
// tablet (not an admin).
func updatesFixture(t *testing.T) (u *Updates, h func(method, path, token, body string) (int, map[string]any), admin, tablet string) {
	t.Helper()
	s, tabletToken, _ := fixture(t)
	u = &Updates{Installed: "0.1.0-ixforge.16", Releases: t.TempDir(), Requests: t.TempDir(), Now: func() time.Time { return updatesNow }}
	handler, err := Handler(s, "https://workshop.invalid", WithUpdates(u))
	if err != nil {
		t.Fatal(err)
	}
	link, err := s.NewLinkInvite("https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	if _, admin, err = s.Claim(link.Code, "PC"); err != nil {
		t.Fatal(err)
	}
	h = func(method, path, token, body string) (int, map[string]any) {
		t.Helper()
		w := request(t, handler, method, path, token, body)
		var reply map[string]any
		_ = json.Unmarshal(w.Body.Bytes(), &reply)
		return w.Code, reply
	}
	return u, h, admin, tabletToken
}

// seenUpdater is the updater task's heartbeat: it writes it each time it runs.
func seenUpdater(t *testing.T, u *Updates, at time.Time) {
	t.Helper()
	raw, _ := json.Marshal(map[string]string{"seen_at": at.UTC().Format(time.RFC3339)})
	if err := os.WriteFile(filepath.Join(u.Requests, "updater-seen.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestTheDesktopOffersNewerReleasesToItsAdmins: an admin device sees the version the Desktop runs and the releases
// waiting in the releases folder that are newer, each with its hash and notes. A release whose file does not match its
// manifest, whose manifest names another version than its folder, or that is not newer, is not offered. Other devices
// see nothing of it.
func TestTheDesktopOffersNewerReleasesToItsAdmins(t *testing.T) {
	u, h, admin, tablet := updatesFixture(t)
	seventeen := addRelease(t, u.Releases, "0.1.0-ixforge.17", "0.1.0-ixforge.17", []byte("seventeen"), "")
	addRelease(t, u.Releases, "0.1.0-ixforge.16", "0.1.0-ixforge.16", []byte("sixteen"), "")
	addRelease(t, u.Releases, "0.1.0-ixforge.9", "0.1.0-ixforge.9", []byte("nine"), "")
	addRelease(t, u.Releases, "0.1.0-ixforge.18", "0.1.0-ixforge.18", []byte("eighteen"), strings.Repeat("A", 64))
	addRelease(t, u.Releases, "0.1.0-ixforge.19", "0.1.0-ixforge.20", []byte("nineteen"), "")
	addRelease(t, u.Releases, "junk", "junk", []byte("junk"), "")

	code, reply := h("GET", "/workshop/api/updates", admin, "")
	if code != 200 || reply["installed"] != "0.1.0-ixforge.16" || reply["updater"] != nil || reply["request"] != nil || reply["result"] != nil {
		t.Fatalf("an admin sees the installed version and no updater yet: %d %v", code, reply)
	}
	releases, _ := reply["releases"].([]any)
	if len(releases) != 1 {
		t.Fatalf("only ixforge.17 is offered: %v", releases)
	}
	r := releases[0].(map[string]any)
	if r["version"] != "0.1.0-ixforge.17" || r["sha256"] != seventeen || r["notes"] != "Notes of 0.1.0-ixforge.17" || r["commit"] != "aca2e116" {
		t.Errorf("the release as its manifest says: %v", r)
	}
	if code, _ := h("GET", "/workshop/api/updates", tablet, ""); code != 403 {
		t.Errorf("a device that does not administer sees no update: %d", code)
	}
	seenUpdater(t, u, updatesNow.Add(-time.Minute))
	if _, reply = h("GET", "/workshop/api/updates", admin, ""); reply["updater"] == nil {
		t.Errorf("the updater seen a minute ago is there: %v", reply)
	}
}

// TestAnAdminAsksTheUpdaterToInstallARelease: the install button writes a request for the updater task (installed
// once, run as SYSTEM), naming the release and its hash. It is refused when the updater has not been seen lately, for
// a release that is not offered or whose hash differs, and while another request waits. The updater's result, once
// written, is shown.
func TestAnAdminAsksTheUpdaterToInstallARelease(t *testing.T) {
	u, h, admin, tablet := updatesFixture(t)
	seventeen := addRelease(t, u.Releases, "0.1.0-ixforge.17", "0.1.0-ixforge.17", []byte("seventeen"), "")
	addRelease(t, u.Releases, "0.1.0-ixforge.9", "0.1.0-ixforge.9", []byte("nine"), "")
	ask := func(version, hash string) string { return `{"version":"` + version + `","sha256":"` + hash + `"}` }
	requested := filepath.Join(u.Requests, "update-request.json")

	if code, reply := h("POST", "/workshop/api/updates", admin, ask("0.1.0-ixforge.17", seventeen)); code != 503 || reply["code"] != "updater_absent" {
		t.Fatalf("no updater seen: the request would wait forever: %d %v", code, reply)
	}
	seenUpdater(t, u, updatesNow.Add(-10*time.Minute))
	if code, _ := h("POST", "/workshop/api/updates", admin, ask("0.1.0-ixforge.17", seventeen)); code != 503 {
		t.Fatalf("an updater last seen ten minutes ago is not running: %d", code)
	}
	seenUpdater(t, u, updatesNow.Add(-30*time.Second))
	for name, body := range map[string]string{
		"another hash":       ask("0.1.0-ixforge.17", strings.Repeat("B", 64)),
		"a release not here": ask("0.1.0-ixforge.21", seventeen),
		"an older release":   ask("0.1.0-ixforge.9", seventeen),
	} {
		if code, _ := h("POST", "/workshop/api/updates", admin, body); code != 400 {
			t.Errorf("%s: want 400, got %d", name, code)
		}
	}
	if code, _ := h("POST", "/workshop/api/updates", tablet, ask("0.1.0-ixforge.17", seventeen)); code != 403 {
		t.Errorf("a device that does not administer cannot install: %d", code)
	}
	if _, err := os.Stat(requested); !os.IsNotExist(err) {
		t.Fatalf("nothing refused may leave a request: %v", err)
	}

	code, reply := h("POST", "/workshop/api/updates", admin, ask("0.1.0-ixforge.17", strings.ToLower(seventeen)))
	if code != 202 {
		t.Fatalf("the admin's request is accepted: %d %v", code, reply)
	}
	raw, err := os.ReadFile(requested)
	if err != nil {
		t.Fatal(err)
	}
	var req struct{ Version, SHA256, RequestedAt, DeviceName string }
	var fields map[string]any
	_ = json.Unmarshal(raw, &fields)
	req.Version, _ = fields["version"].(string)
	req.SHA256, _ = fields["sha256"].(string)
	req.RequestedAt, _ = fields["requested_at"].(string)
	req.DeviceName, _ = fields["device_name"].(string)
	if req.Version != "0.1.0-ixforge.17" || req.SHA256 != seventeen || req.RequestedAt != "2026-11-15T12:00:00Z" || req.DeviceName != "PC" {
		t.Fatalf("the request names the release, its hash, when and who: %s", raw)
	}
	if _, reply = h("GET", "/workshop/api/updates", admin, ""); reply["request"] == nil {
		t.Errorf("the waiting request is shown: %v", reply)
	}
	if code, reply := h("POST", "/workshop/api/updates", admin, ask("0.1.0-ixforge.17", seventeen)); code != 409 || reply["code"] != "update_pending" {
		t.Errorf("one request at a time: %d %v", code, reply)
	}

	// The updater takes the request and writes its result.
	if err := os.Remove(requested); err != nil {
		t.Fatal(err)
	}
	result, _ := json.Marshal(map[string]string{"version": "0.1.0-ixforge.17", "state": "completed", "receipt": "update-0123456789abcdef0123456789abcdef", "finished_at": "2026-11-15T12:01:10Z"})
	if err := os.WriteFile(filepath.Join(u.Requests, "update-result.json"), result, 0o644); err != nil {
		t.Fatal(err)
	}
	_, reply = h("GET", "/workshop/api/updates", admin, "")
	last, _ := reply["result"].(map[string]any)
	if reply["request"] != nil || last["state"] != "completed" || last["version"] != "0.1.0-ixforge.17" || last["receipt"] != "update-0123456789abcdef0123456789abcdef" {
		t.Fatalf("the updater's result is shown: %v", reply)
	}
}

// TestTheDesktopInstallsAPublishedReleaseByItself: the owner chose (2026-11-16) that a published release installs by
// itself, publishing being the owner's approval. The Desktop asks the updater for the newest offered release, under the
// name "Desktop (automatique)", once per version ever: never while the updater is absent or a request waits, and never
// again for a version already asked or whose installation failed (a failed install is not retried by itself).
func TestTheDesktopInstallsAPublishedReleaseByItself(t *testing.T) {
	u, _, _, _ := updatesFixture(t)
	requested := filepath.Join(u.Requests, "update-request.json")
	asked := func(want string) {
		t.Helper()
		got, err := u.AutoRequest()
		if err != nil || got != want {
			t.Fatalf("auto request: got %q %v, want %q", got, err, want)
		}
	}
	read := func() map[string]string {
		t.Helper()
		var r map[string]string
		raw, err := os.ReadFile(requested)
		if err != nil || json.Unmarshal(raw, &r) != nil {
			t.Fatalf("request: %v", err)
		}
		return r
	}

	asked("") // nothing published
	seventeen := addRelease(t, u.Releases, "0.1.0-ixforge.17", "0.1.0-ixforge.17", []byte("seventeen"), "")
	eighteen := addRelease(t, u.Releases, "0.1.0-ixforge.18", "0.1.0-ixforge.18", []byte("eighteen"), "")
	asked("") // no updater seen: nobody would take it
	if _, err := os.Stat(requested); err == nil {
		t.Fatal("no request without an updater")
	}
	seenUpdater(t, u, updatesNow.Add(-30*time.Second))
	asked("0.1.0-ixforge.18") // the newest, not each one
	if r := read(); r["version"] != "0.1.0-ixforge.18" || r["sha256"] != eighteen || r["device_name"] != "Desktop (automatique)" || r["device_id"] != "auto" {
		t.Fatalf("the request names the newest release and who asked: %v", r)
	}
	asked("") // it waits for the updater
	_ = seventeen

	// The updater took it and failed: the Desktop does not ask again.
	if err := os.Remove(requested); err != nil {
		t.Fatal(err)
	}
	result, _ := json.Marshal(map[string]string{"version": "0.1.0-ixforge.18", "state": "rolled_back", "error": "health check failed"})
	if err := os.WriteFile(filepath.Join(u.Requests, "update-result.json"), result, 0o644); err != nil {
		t.Fatal(err)
	}
	asked("")
	if _, err := os.Stat(requested); err == nil {
		t.Fatal("a failed version is not asked again")
	}

	// A version an admin asked by hand and that failed is not retried by the Desktop either.
	addRelease(t, u.Releases, "0.1.0-ixforge.19", "0.1.0-ixforge.19", []byte("nineteen"), "")
	result, _ = json.Marshal(map[string]string{"version": "0.1.0-ixforge.19", "state": "failed"})
	if err := os.WriteFile(filepath.Join(u.Requests, "update-result.json"), result, 0o644); err != nil {
		t.Fatal(err)
	}
	asked("")

	// A newer release is asked once the updater is back, and only once.
	addRelease(t, u.Releases, "0.1.0-ixforge.20", "0.1.0-ixforge.20", []byte("twenty"), "")
	seenUpdater(t, u, updatesNow.Add(-10*time.Minute))
	asked("")
	seenUpdater(t, u, updatesNow.Add(-30*time.Second))
	asked("0.1.0-ixforge.20")
	if err := os.Remove(requested); err != nil {
		t.Fatal(err)
	}
	asked("") // taken, no result yet: still never twice
}
