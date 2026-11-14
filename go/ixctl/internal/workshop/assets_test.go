package workshop

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"strings"
	"testing"
)

func fixtureCacheName(overrides map[string][]byte) string {
	hash := sha256.New()
	_ = fs.WalkDir(assets, "web", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || path == "web/sw.js" {
			return nil
		}
		name := strings.TrimPrefix(path, "web/")
		body, ok := overrides[name]
		if !ok {
			body, _ = assets.ReadFile(path)
		}
		hash.Write([]byte(name + "\x00"))
		hash.Write(body)
		hash.Write([]byte{0})
		return nil
	})
	return "ix-workshop-shell-" + hex.EncodeToString(hash.Sum(nil))[:16]
}
func TestClientReleaseIncludesEveryAsset(t *testing.T) {
	sw, err := assets.ReadFile("web/sw.js")
	if err != nil {
		t.Fatal(err)
	}
	expected := fixtureCacheName(nil)
	if !strings.Contains(string(sw), "const CACHE = '"+expected+"';") {
		t.Fatal("client assets changed without rebuilding sw.js: run node e2e/build-client.mjs")
	}
	old, _ := assets.ReadFile("web/branding.json")
	changed := []byte(strings.ReplaceAll(string(old), "0.1.1", "0.2.0"))
	if fixtureCacheName(map[string][]byte{"branding.json": changed}) == expected {
		t.Fatal("branding change is not included in release digest")
	}
}
