package workshop

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// clientCSP is the IxForge client's policy, on the Desktop and on the standalone site alike: its own origin, plus
// Claude's API for the optional AI, which the browser calls with the person's own key.
const clientCSP = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self' https://api.anthropic.com; img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"

var cacheLine = regexp.MustCompile(`const CACHE = '[^']*';`)

// WriteSite writes the standalone IxForge into dir for a static host that reads _headers and _redirects (Cloudflare
// Pages, Netlify): the app under workshop/ (see WriteApp), the client's security headers, and / leading to it.
func WriteSite(dir string) error {
	if err := WriteApp(filepath.Join(dir, "workshop")); err != nil {
		return err
	}
	host := map[string]string{
		"_redirects": "/ /workshop/ 302\n",
		"_headers": strings.Join([]string{
			"/*",
			"  Content-Security-Policy: " + clientCSP,
			"  Referrer-Policy: no-referrer",
			"  X-Content-Type-Options: nosniff",
			"  Cache-Control: no-cache",
			"/workshop/manifest.webmanifest",
			"  Content-Type: application/manifest+json",
			""}, "\n"),
		"index.html": `<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/workshop/"><title>IxForge</title></head>` +
			`<body><a href="/workshop/">IxForge</a></body></html>` + "\n",
	}
	for name, body := range host {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			return err
		}
	}
	return nil
}

// WriteApp writes the IxForge client alone into dir, in standalone mode (no Desktop, no pairing, data in the
// browser), to be served from any folder of any static host, even one that sets no headers (GitHub Pages): its paths
// are relative, and its page carries the client's policy in a meta element. Its service worker names its own cache,
// so an installed copy picks up the app's updates.
func WriteApp(dir string) error {
	web, err := fs.Sub(assets, "web")
	if err != nil {
		return err
	}
	files := map[string][]byte{}
	if err := fs.WalkDir(web, ".", func(name string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		body, err := fs.ReadFile(web, name)
		files[name] = body
		return err
	}); err != nil {
		return err
	}

	var brand map[string]any
	if err := json.Unmarshal(files["branding.json"], &brand); err != nil {
		return err
	}
	brand["standalone"] = true
	if files["branding.json"], err = json.Marshal(brand); err != nil {
		return err
	}
	var manifest map[string]any
	if err := json.Unmarshal(files["manifest.webmanifest"], &manifest); err != nil {
		return err
	}
	manifest["name"], manifest["short_name"] = brand["name"], brand["short_name"]
	if files["manifest.webmanifest"], err = json.Marshal(manifest); err != nil {
		return err
	}
	// A meta element cannot carry frame-ancestors; the rest of the policy applies as on the Desktop.
	policy := strings.Replace(clientCSP, "; frame-ancestors 'none'", "", 1)
	page := string(files["index.html"])
	if !strings.Contains(page, `<meta charset="utf-8">`) {
		return errors.New("workshop: the client page has no charset to place its policy after")
	}
	files["index.html"] = []byte(strings.Replace(page, `<meta charset="utf-8">`,
		`<meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="`+policy+`"><meta name="referrer" content="no-referrer">`, 1))

	names := make([]string, 0, len(files))
	for name := range files {
		if name != "sw.js" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	hash := sha256.New()
	for _, name := range names {
		hash.Write([]byte(name + "\x00"))
		hash.Write(files[name])
		hash.Write([]byte{0})
	}
	sw := string(files["sw.js"])
	if !cacheLine.MatchString(sw) {
		return errors.New("workshop: the service worker names no cache")
	}
	files["sw.js"] = []byte(cacheLine.ReplaceAllString(sw, "const CACHE = 'ix-workshop-shell-"+hex.EncodeToString(hash.Sum(nil))[:16]+"';"))

	for name, body := range files {
		target := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(target, body, 0o644); err != nil {
			return err
		}
	}
	return nil
}
