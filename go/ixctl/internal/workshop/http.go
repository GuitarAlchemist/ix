package workshop

import (
	"embed"
	"encoding/json"
	"errors"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

//go:embed web/* web/vendor/* owner/*
var assets embed.FS

//go:generate node e2e/build-client.mjs

// Handler is a separate application face for an explicitly configured same HTTPS origin.
// Mount only /workshop/ on the future app-facing mux. Do not mount this behind or instead of ixctl's /v1 guard.
func Handler(s *Store, origin string) (http.Handler, error) {
	if !validOrigin(origin) {
		return nil, ErrInvalid
	}
	u, _ := url.Parse(origin)
	web, _ := fs.Sub(assets, "web")
	files := http.StripPrefix("/workshop/", http.FileServer(http.FS(web)))
	mux := http.NewServeMux()
	mux.HandleFunc("GET /workshop", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/workshop/", http.StatusPermanentRedirect)
	})
	mux.HandleFunc("GET /workshop/manifest.webmanifest", func(w http.ResponseWriter, r *http.Request) {
		var manifest map[string]any
		var brand map[string]string
		raw, _ := assets.ReadFile("web/manifest.webmanifest")
		_ = json.Unmarshal(raw, &manifest)
		raw, _ = assets.ReadFile("web/branding.json")
		_ = json.Unmarshal(raw, &brand)
		manifest["name"] = brand["name"]
		manifest["short_name"] = brand["short_name"]
		w.Header().Set("Content-Type", "application/manifest+json")
		_ = json.NewEncoder(w).Encode(manifest)
	})
	mux.HandleFunc("POST /workshop/api/pair", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Code    string `json:"code"`
			Name    string `json:"name"`
			StoreID string `json:"store_id"`
		}
		if err := decode(w, r, &body); err != nil {
			failure(w, ErrInvalid)
			return
		}
		if body.StoreID != "" && body.StoreID != s.id {
			failure(w, ErrStoreChanged)
			return
		}
		device, token, err := s.Claim(body.Code, body.Name)
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, http.StatusOK, map[string]any{"device": device, "token": token, "store_id": s.id})
	})
	mux.HandleFunc("GET /workshop/api/me", func(w http.ResponseWriter, r *http.Request) {
		d, err := s.Authenticate(bearer(r))
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, http.StatusOK, map[string]any{"device": d, "store_id": s.id, "protocol": "workshop/1", "runtime": "not_connected"})
	})
	mux.HandleFunc("POST /workshop/api/operations", func(w http.ResponseWriter, r *http.Request) {
		var op Operation
		if err := decode(w, r, &op); err != nil {
			failure(w, ErrInvalid)
			return
		}
		receipt, err := s.Save(bearer(r), op)
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, http.StatusOK, receipt)
	})
	mux.HandleFunc("GET /workshop/api/changes", func(w http.ResponseWriter, r *http.Request) {
		since, err := strconv.ParseInt(r.URL.Query().Get("since"), 10, 64)
		if err != nil {
			failure(w, ErrInvalid)
			return
		}
		changes, err := s.Changes(bearer(r), since)
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, http.StatusOK, changes)
	})
	mux.HandleFunc("/workshop/", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/workshop/api/") {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.WriteHeader(405)
			return
		}
		switch r.URL.Path {
		case "/workshop/", "/workshop/index.html", "/workshop/sw.js", "/workshop/manifest.webmanifest", "/workshop/app.js", "/workshop/sync.js", "/workshop/voice.js", "/workshop/updates.js", "/workshop/branding.json", "/workshop/style.css", "/workshop/icon-192.png", "/workshop/icon-512.png", "/workshop/vendor/i18next.min.js":
			files.ServeHTTP(w, r)
		default:
			http.NotFound(w, r)
		}
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
		if r.Host != u.Host {
			failureStatus(w, 403, "foreign_host")
			return
		}
		from := r.Header.Get("Origin")
		// A user-followed external link may open only the public HTML entry.
		// API requests, embedded/background loads and explicit foreign Origins stay refused.
		publicEntry := from == "" && (r.Method == http.MethodGet || r.Method == http.MethodHead) &&
			r.URL.Path == "/workshop/" && r.URL.RawQuery == "" &&
			r.Header.Get("Sec-Fetch-Mode") == "navigate" && r.Header.Get("Sec-Fetch-Dest") == "document" &&
			r.Header.Get("Sec-Fetch-User") == "?1"
		if (from != "" && from != origin) || (r.Header.Get("Sec-Fetch-Site") == "cross-site" && !publicEntry) {
			failureStatus(w, 403, "foreign_origin")
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead && from != origin {
			failureStatus(w, 403, "origin_required")
			return
		}
		if r.URL.Path == "/workshop/sw.js" {
			w.Header().Set("Service-Worker-Allowed", "/workshop/")
		}
		mux.ServeHTTP(w, r)
	}), nil
}
func bearer(r *http.Request) string {
	v := r.Header.Get("Authorization")
	if !strings.HasPrefix(v, "Bearer ") {
		return ""
	}
	return strings.TrimPrefix(v, "Bearer ")
}
func decode(w http.ResponseWriter, r *http.Request, v any) error {
	if r.Header.Get("Content-Type") != "application/json" {
		return ErrInvalid
	}
	d := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return ErrInvalid
	}
	return nil
}
func respond(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func failureStatus(w http.ResponseWriter, status int, code string) {
	respond(w, status, map[string]string{"code": code})
}
func failure(w http.ResponseWriter, err error) {
	status := 500
	code := "storage_unavailable"
	switch {
	case errors.Is(err, ErrUnauthorized):
		status = 401
		code = err.Error()
	case errors.Is(err, ErrPending), errors.Is(err, ErrRefused), errors.Is(err, ErrRevoked):
		status = 403
		code = err.Error()
	case errors.Is(err, ErrConflict), errors.Is(err, ErrOpReuse):
		status = 409
		code = err.Error()
	case errors.Is(err, ErrStoreChanged):
		status = 409
		code = err.Error()
	case errors.Is(err, ErrInvalid):
		status = 400
		code = err.Error()
	case errors.Is(err, ErrInvite):
		status = 403
		code = err.Error()
	}
	failureStatus(w, status, code)
}
