package workshop

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestTheStandaloneSiteHoldsTheClientOnly: the installable site a friend opens is the IxForge client alone, as static
// files: no server code, no owner page, no database. It runs in standalone mode, keeps the Desktop's security
// headers, and has its own cache name so its updates are detected.
func TestTheStandaloneSiteHoldsTheClientOnly(t *testing.T) {
	dir := t.TempDir()
	if err := WriteSite(dir); err != nil {
		t.Fatal(err)
	}
	files := map[string][]byte{}
	if err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, path)
		body, err := os.ReadFile(path)
		files[filepath.ToSlash(rel)] = body
		return err
	}); err != nil {
		t.Fatal(err)
	}
	for _, need := range []string{"index.html", "_headers", "_redirects", "workshop/index.html", "workshop/app.js", "workshop/sync.js", "workshop/widgets.js", "workshop/ai.js",
		"workshop/sw.js", "workshop/branding.json", "workshop/manifest.webmanifest", "workshop/vendor/lit-all.min.js", "workshop/vendor/LICENSE-lit", "workshop/vendor/i18next.min.js", "workshop/vendor/LICENSE"} {
		if files[need] == nil {
			t.Errorf("the site lacks %s", need)
		}
	}
	for name := range files {
		if strings.HasSuffix(name, ".go") || strings.Contains(name, "owner") || strings.Contains(name, "sqlite") || strings.Contains(name, "stackdump") {
			t.Errorf("the site must hold the client only: %s", name)
		}
	}

	var branding struct {
		Name       string
		Version    string
		Standalone bool
	}
	if err := json.Unmarshal(files["workshop/branding.json"], &branding); err != nil || !branding.Standalone || branding.Name != "IxForge" {
		t.Fatalf("the site must run in standalone mode: %s %v", files["workshop/branding.json"], err)
	}
	if !strings.Contains(string(files["_redirects"]), "/ /workshop/ 302") {
		t.Errorf("the site root must lead to the app: %q", files["_redirects"])
	}
	if !strings.Contains(string(files["_headers"]), "Content-Security-Policy: "+clientCSP) || !strings.Contains(string(files["_headers"]), "X-Content-Type-Options: nosniff") {
		t.Errorf("the site must keep the Desktop's security headers:\n%s", files["_headers"])
	}

	// Every file the service worker keeps for offline use is there, and its cache name is the site's own.
	sw := string(files["workshop/sw.js"])
	shell := regexp.MustCompile(`const SHELL = \[([^\]]*)\]`).FindStringSubmatch(sw)
	if shell == nil {
		t.Fatal("no SHELL in the site's service worker")
	}
	for _, quoted := range strings.Split(shell[1], ",") {
		path := strings.Trim(strings.TrimSpace(quoted), "'")
		name := "workshop/" + strings.TrimPrefix(path, "./")
		if path == "./" {
			name = "workshop/index.html"
		}
		if files[name] == nil {
			t.Errorf("the service worker keeps %s, which the site lacks", path)
		}
	}
	names := []string{}
	for name := range files {
		if strings.HasPrefix(name, "workshop/") && name != "workshop/sw.js" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	hash := sha256.New()
	for _, name := range names {
		hash.Write([]byte(strings.TrimPrefix(name, "workshop/") + "\x00"))
		hash.Write(files[name])
		hash.Write([]byte{0})
	}
	want := "ix-workshop-shell-" + hex.EncodeToString(hash.Sum(nil))[:16]
	if !strings.Contains(sw, "const CACHE = '"+want+"'") {
		t.Errorf("the site's service worker must name its own cache %s", want)
	}
	if strings.Contains(sw, "const CACHE = '"+fixtureCacheName(nil)+"'") {
		t.Error("the site must not reuse the Desktop client's cache name")
	}
}

// TestTheAppRunsFromAnyFolder: the client alone can live in any folder of a static host that sets no headers, such
// as GitHub Pages (spareilleux.github.io/apps/ixforge/): every path it uses is relative to its own folder, and its
// page carries the client's policy itself.
func TestTheAppRunsFromAnyFolder(t *testing.T) {
	dir := t.TempDir()
	if err := WriteApp(dir); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"index.html", "app.js", "sync.js", "updates.js", "widgets.js", "ai.js", "sw.js", "manifest.webmanifest", "branding.json"} {
		body, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(body), "/workshop") {
			t.Errorf("%s names /workshop: the app must work from any folder", name)
		}
	}
	page, _ := os.ReadFile(filepath.Join(dir, "index.html"))
	policy := strings.Replace(clientCSP, "; frame-ancestors 'none'", "", 1) // ignored in a meta element
	if !strings.Contains(string(page), `<meta http-equiv="Content-Security-Policy" content="`+policy+`">`) {
		t.Errorf("the page must carry the client's policy itself:\n%s", page)
	}
	sw, _ := os.ReadFile(filepath.Join(dir, "sw.js"))
	shell := regexp.MustCompile(`const SHELL = \[([^\]]*)\]`).FindStringSubmatch(string(sw))
	if shell == nil {
		t.Fatal("no SHELL")
	}
	for _, quoted := range strings.Split(shell[1], ",") {
		path := strings.Trim(strings.TrimSpace(quoted), "'")
		if !strings.HasPrefix(path, "./") {
			t.Errorf("the service worker must keep relative paths: %s", path)
			continue
		}
		name := strings.TrimPrefix(path, "./")
		if name == "" {
			name = "index.html"
		}
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(name))); err != nil {
			t.Errorf("the service worker keeps %s, which the app lacks", path)
		}
	}
}

// TestTheClientMayReachClaudeOnlyForTheOptionalAI:the Desktop's client gets the same policy as the site: its own
// origin, plus Claude's API for the optional AI, and nothing else.
func TestTheClientMayReachClaudeOnlyForTheOptionalAI(t *testing.T) {
	s, _, _ := fixture(t)
	h, err := Handler(s, "https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	w := request(t, h, "GET", "/workshop/", "", "")
	if got := w.Header().Get("Content-Security-Policy"); got != clientCSP || !strings.Contains(got, "connect-src 'self' https://api.anthropic.com;") {
		t.Fatalf("client policy: %q", got)
	}
	if w := request(t, h, "GET", "/workshop/ai.js", "", ""); w.Code != 200 || !strings.Contains(w.Header().Get("Content-Type"), "javascript") {
		t.Fatalf("ai.js: %d", w.Code)
	}
}
