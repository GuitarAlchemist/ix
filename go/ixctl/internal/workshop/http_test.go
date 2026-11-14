package workshop

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func request(t *testing.T, h http.Handler, method, path, token, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, "https://workshop.invalid"+path, strings.NewReader(body))
	r.Header.Set("Origin", "https://workshop.invalid")
	r.Header.Set("Content-Type", "application/json")
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w
}
func TestHTTPAuthOriginsScopeAndCommit(t *testing.T) {
	s, token, _ := fixture(t)
	h, err := Handler(s, "https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{"/workshop/api/me", "/workshop/api/changes?since=0"} {
		if w := request(t, h, "GET", p, "ixctl-global-fixture", ""); w.Code != 401 {
			t.Fatalf("global token accepted: %d", w.Code)
		}
	}
	for _, scenario := range []struct{ host, origin string }{{"evil.invalid", "https://workshop.invalid"}, {"workshop.invalid", "https://evil.invalid"}, {"workshop.invalid", "null"}} {
		r := httptest.NewRequest("GET", "https://"+scenario.host+"/workshop/api/me", nil)
		r.Header.Set("Origin", scenario.origin)
		r.Header.Set("Authorization", "Bearer "+token)
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 403 {
			t.Fatal(w.Code)
		}
	}
	for _, body := range []string{`{"op_id":"one","base_revision":0,"project":{},"shell":"cmd.exe"}`, `{} {}`, `{"op_id":"one","project":{"id":"scribe","title":"Read books","manuscript":"secret"}}`} {
		if w := request(t, h, "POST", "/workshop/api/operations", token, body); w.Code != 400 {
			t.Fatalf("scope: %d %s", w.Code, w.Body)
		}
	}
	for _, p := range []string{"/v1/status", "/workshop/api/shell", "/workshop/api/books", "/workshop/store.sqlite", "/workshop/api/owner/approve"} {
		if w := request(t, h, "GET", p, token, ""); w.Code != 404 {
			t.Fatalf("%s: %d", p, w.Code)
		}
	}
	payload, _ := json.Marshal(operation("http-save", 0, draft()))
	w := request(t, h, "POST", "/workshop/api/operations", token, string(payload))
	if w.Code != 200 {
		t.Fatalf("save %d %s", w.Code, w.Body)
	}
	// Discarding the first response models a client timeout after COMMIT; replay must return the exact same receipt.
	retry := request(t, h, "POST", "/workshop/api/operations", token, string(payload))
	if retry.Code != 200 || retry.Body.String() != w.Body.String() {
		t.Fatalf("retry receipt: %s vs %s", retry.Body, w.Body)
	}
	w = request(t, h, "GET", "/workshop/api/changes?since=0", token, "")
	var changes ChangeSet
	if json.Unmarshal(w.Body.Bytes(), &changes) != nil || len(changes.Events) != 1 {
		t.Fatalf("reopen: %s", w.Body)
	}
	if w.Header().Get("Cache-Control") != "no-store" {
		t.Fatal("business data cached")
	}
	for _, p := range []string{"/workshop/", "/workshop/manifest.webmanifest", "/workshop/icon-192.png", "/workshop/sw.js"} {
		w = request(t, h, "GET", p, "", "")
		if w.Code != 200 {
			t.Fatalf("asset %s: %d", p, w.Code)
		}
	}
	w = request(t, h, "GET", "/workshop/sw.js", "", "")
	if w.Header().Get("Service-Worker-Allowed") != "/workshop/" {
		t.Fatal("wrong scope")
	}
}
