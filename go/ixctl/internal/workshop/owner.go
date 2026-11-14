package workshop

import (
	"encoding/hex"
	"net"
	"net/http"
	"net/url"
	"strings"
)

// OwnerHandler is a loopback-only administration face with its own key. Tablet/device and ixctl tokens have no role here.
func OwnerHandler(s *Store, key, localOrigin, appOrigin string) (http.Handler, error) {
	u, err := url.Parse(localOrigin)
	if err != nil || u.Scheme != "http" || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || u.User != nil {
		return nil, ErrInvalid
	}
	host, port, err := net.SplitHostPort(u.Host)
	if err != nil || host != "127.0.0.1" || port == "" || !validOrigin(appOrigin) {
		return nil, ErrInvalid
	}
	decoded, err := hex.DecodeString(key)
	if err != nil || len(decoded) != 32 {
		return nil, ErrInvalid
	}
	authority, err := newOwnerAuthority(decoded, localOrigin)
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	serve := func(path, kind string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			data, err := assets.ReadFile(path)
			if err != nil {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", kind)
			w.Write(data)
		}
	}
	mux.HandleFunc("GET /workshop-owner/{$}", serve("owner/index.html", "text/html; charset=utf-8"))
	mux.HandleFunc("GET /workshop-owner/app.js", serve("owner/app.js", "text/javascript; charset=utf-8"))
	mux.HandleFunc("GET /workshop-owner/style.css", serve("owner/style.css", "text/css; charset=utf-8"))
	mux.HandleFunc("GET /workshop-owner/i18next.min.js", serve("web/vendor/i18next.min.js", "text/javascript; charset=utf-8"))
	authed := func(handler http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if !authority.valid(bearer(r)) {
				failure(w, ErrUnauthorized)
				return
			}
			handler(w, r)
		}
	}
	mux.HandleFunc("GET /workshop-owner/challenge", func(w http.ResponseWriter, r *http.Request) {
		result, err := authority.challenge(r.URL.Query().Get("nonce"))
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, 200, result)
	})
	mux.HandleFunc("POST /workshop-owner/session", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Nonce   string `json:"nonce"`
			Epoch   string `json:"epoch"`
			Expires int64  `json:"expires"`
			Proof   string `json:"proof"`
		}
		if err := decode(w, r, &body); err != nil {
			failure(w, ErrInvalid)
			return
		}
		result, err := authority.session(body.Nonce, body.Epoch, body.Expires, body.Proof)
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, 200, result)
	})
	mux.HandleFunc("GET /workshop-owner/api/state", authed(func(w http.ResponseWriter, r *http.Request) {
		devices, err := s.Devices()
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, 200, map[string]any{"devices": devices, "app_url": appOrigin + "/workshop/", "schema_version": 1})
	}))
	mux.HandleFunc("POST /workshop-owner/api/invite", authed(func(w http.ResponseWriter, r *http.Request) {
		var body struct{}
		if err := decode(w, r, &body); err != nil {
			failure(w, ErrInvalid)
			return
		}
		invite, err := s.NewInvite(appOrigin)
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, 200, invite)
	}))
	mux.HandleFunc("POST /workshop-owner/api/devices/{id}/{decision}", authed(func(w http.ResponseWriter, r *http.Request) {
		var body struct{}
		if err := decode(w, r, &body); err != nil {
			failure(w, ErrInvalid)
			return
		}
		if !identifier.MatchString(r.PathValue("id")) {
			failure(w, ErrInvalid)
			return
		}
		if err := s.Decide(r.PathValue("id"), r.PathValue("decision")); err != nil {
			failure(w, err)
			return
		}
		respond(w, 200, map[string]bool{"ok": true})
	}))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
		if r.Host != u.Host {
			failureStatus(w, 403, "foreign_host")
			return
		}
		origin := r.Header.Get("Origin")
		if (origin != "" && origin != localOrigin) || r.Header.Get("Sec-Fetch-Site") == "cross-site" {
			failureStatus(w, 403, "foreign_origin")
			return
		}
		if r.Method != "GET" && r.Method != "HEAD" && origin != localOrigin {
			failureStatus(w, 403, "origin_required")
			return
		}
		if !strings.HasPrefix(r.URL.Path, "/workshop-owner/") {
			http.NotFound(w, r)
			return
		}
		mux.ServeHTTP(w, r)
	}), nil
}
