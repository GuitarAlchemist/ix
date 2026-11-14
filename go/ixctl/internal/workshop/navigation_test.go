package workshop

import (
	"net/http/httptest"
	"testing"
)

func TestQRPublicNavigationKeepsAPIsAndOriginsProtected(t *testing.T) {
	store, token, _ := fixture(t)
	app, err := Handler(store, "https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name, method, path, host, origin, mode, dest, user string
		status                                             int
	}{
		{"QR document", "GET", "/workshop/", "workshop.invalid", "", "navigate", "document", "?1", 200},
		{"noncanonical HTML", "GET", "/workshop/index.html", "workshop.invalid", "", "navigate", "document", "?1", 403},
		{"HEAD document", "HEAD", "/workshop/", "workshop.invalid", "", "navigate", "document", "?1", 200},
		{"foreign Host", "GET", "/workshop/", "foreign.invalid", "", "navigate", "document", "?1", 403},
		{"foreign Origin", "GET", "/workshop/", "workshop.invalid", "https://foreign.invalid", "navigate", "document", "?1", 403},
		{"opaque Origin", "GET", "/workshop/", "workshop.invalid", "null", "navigate", "document", "?1", 403},
		{"cross-site explicit Origin", "GET", "/workshop/", "workshop.invalid", "https://workshop.invalid", "navigate", "document", "?1", 403},
		{"iframe", "GET", "/workshop/", "workshop.invalid", "", "navigate", "iframe", "?1", 403},
		{"background", "GET", "/workshop/", "workshop.invalid", "", "navigate", "document", "", 403},
		{"fetch", "GET", "/workshop/", "workshop.invalid", "", "cors", "", "?1", 403},
		{"query", "GET", "/workshop/?other=1", "workshop.invalid", "", "navigate", "document", "?1", 403},
		{"public POST", "POST", "/workshop/", "workshop.invalid", "", "navigate", "document", "?1", 403},
		{"API read", "GET", "/workshop/api/me", "workshop.invalid", "", "navigate", "document", "?1", 403},
		{"API write", "POST", "/workshop/api/operations", "workshop.invalid", "", "navigate", "document", "?1", 403},
		{"pair", "POST", "/workshop/api/pair", "workshop.invalid", "", "navigate", "document", "?1", 403},
		{"owner", "GET", "/workshop-owner/", "workshop.invalid", "", "navigate", "document", "?1", 403},
		{"control", "GET", "/v1/status", "workshop.invalid", "", "navigate", "document", "?1", 403},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(test.method, "https://"+test.host+test.path, nil)
			for k, v := range map[string]string{"Origin": test.origin, "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": test.mode, "Sec-Fetch-Dest": test.dest, "Sec-Fetch-User": test.user, "Authorization": "Bearer " + token} {
				req.Header.Set(k, v)
			}
			result := httptest.NewRecorder()
			app.ServeHTTP(result, req)
			if result.Code != test.status {
				t.Fatalf("status %d, want %d: %s", result.Code, test.status, result.Body.String())
			}
		})
	}
}
