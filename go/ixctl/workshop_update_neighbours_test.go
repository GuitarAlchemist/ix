package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// The IxForge updater must leave Tailscale Serve's other routes ("neighbours") as it found them, without claiming to
// know what they are from their content. testdata/update-neighbours.ps1 runs the real PowerShell checks; this test owns
// every observation: the Serve JSON and the port owners per stage (before/after the service stop), and the loopback
// HTTP servers that the real probe reaches. Unknown must never be reported as healthy.

type neighbourOwner struct {
	Listening bool    `json:"listening"`
	PID       *int    `json:"pid"`
	Image     *string `json:"image"`
}

type neighbourReply struct {
	status   int
	body     string
	redirect bool
}

type neighbourStage struct {
	routes  map[string]string // Serve path -> proxy target on the own host
	raw     string            // when set, written as the Serve file instead of routes (unreadable)
	owners  map[string]neighbourOwner
	wmux    neighbourReply
	notes   neighbourReply
	noServe bool
}

func TestUpdateKeepsNeighboursWithoutIdentifyingThem(t *testing.T) {
	if _, err := exec.LookPath("pwsh"); err != nil {
		t.Skip("pwsh 7.4+ not installed")
	}
	var mu sync.Mutex
	var dir string
	var current struct{ before, after neighbourStage }
	var outside atomic.Int32
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { outside.Add(1) }))
	defer elsewhere.Close()
	neighbour := func(pick func(neighbourStage) neighbourReply) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			name, _ := os.ReadFile(filepath.Join(dir, "stage"))
			s := current.before
			if strings.TrimSpace(string(name)) == "after" {
				s = current.after
			}
			mu.Unlock()
			rp := pick(s)
			if rp.redirect {
				w.Header().Set("Location", elsewhere.URL+"/followed")
			}
			w.WriteHeader(rp.status)
			io.WriteString(w, rp.body)
		}))
	}
	wmuxSrv := neighbour(func(s neighbourStage) neighbourReply { return s.wmux })
	defer wmuxSrv.Close()
	notesSrv := neighbour(func(s neighbourStage) neighbourReply { return s.notes })
	defer notesSrv.Close()
	brand := "0.1.0"
	workshop := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/workshop/":
			io.WriteString(w, "<!doctype html><title>IxForge</title>")
		case "/workshop/api/me":
			http.Error(w, "device credential required", http.StatusUnauthorized)
		case "/workshop/branding.json":
			fmt.Fprintf(w, `{"name":"IxForge","version":%q}`, brand)
		default:
			http.NotFound(w, r)
		}
	}))
	defer workshop.Close()
	own := strings.TrimPrefix(workshop.URL, "http://")
	wmuxPort, notesPort := wmuxSrv.URL[strings.LastIndex(wmuxSrv.URL, ":")+1:], notesSrv.URL[strings.LastIndex(notesSrv.URL, ":")+1:]

	base := func() neighbourStage {
		return neighbourStage{
			routes: map[string]string{"/": wmuxSrv.URL, "/notes": notesSrv.URL, "/workshop": "http://127.0.0.1:8761/workshop"},
			owners: map[string]neighbourOwner{
				wmuxPort:  {true, ptr(5744), ptr(`C:\Program Files\wmux\wmux.exe`)},
				notesPort: {true, ptr(30040), ptr(`C:\Program Files\Scribe\scribe.exe`)},
			},
			wmux:  neighbourReply{status: 200, body: "<!doctype html><title>wmux</title>"},
			notes: neighbourReply{status: 200, body: "<!doctype html><title>Scribe</title><p>release 1</p>"},
		}
	}
	with := func(edit func(*neighbourStage)) neighbourStage { s := base(); edit(&s); return s }
	write := func(t *testing.T, name string, s neighbourStage) {
		t.Helper()
		if s.raw != "" {
			os.WriteFile(filepath.Join(dir, name+"-serve.json"), []byte(s.raw), 0o600)
		} else if !s.noServe {
			handlers := map[string]any{}
			for path, target := range s.routes {
				handlers[path] = map[string]string{"Proxy": target}
			}
			serve, _ := json.Marshal(map[string]any{"TCP": map[string]any{"443": map[string]bool{"HTTPS": true}}, "Web": map[string]any{own: map[string]any{"Handlers": handlers}}})
			os.WriteFile(filepath.Join(dir, name+"-serve.json"), serve, 0o600)
		}
		owners, _ := json.Marshal(s.owners)
		os.WriteFile(filepath.Join(dir, name+"-owners.json"), owners, 0o600)
	}
	run := func(t *testing.T, mode string, before, after neighbourStage, extra ...string) map[string]any {
		t.Helper()
		mu.Lock()
		dir = t.TempDir()
		current.before, current.after = before, after
		mu.Unlock()
		os.WriteFile(filepath.Join(dir, "ixforge-neighbours-fixture"), nil, 0o600)
		write(t, "before", before)
		write(t, "after", after)
		outside.Store(0)
		args := append([]string{"-NoProfile", "-NonInteractive", "-File", "testdata/update-neighbours.ps1",
			"-Mode", mode, "-Origin", workshop.URL, "-Listen", own, "-Dir", dir}, extra...)
		out, err := exec.Command("pwsh", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("harness: %v\n%s", err, out)
		}
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		var result map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(lines[len(lines)-1])), &result); err != nil {
			t.Fatalf("harness output: %v\n%s", err, out)
		}
		if n := outside.Load(); n != 0 {
			t.Fatalf("a redirect was followed %d time(s)", n)
		}
		return result
	}

	cases := []struct {
		name          string
		before, after neighbourStage
		want, reason  string
	}{
		{"WMUX baseline preserved", base(), base(), "healthy", ""},
		{"Scribe-like content modified", base(), with(func(s *neighbourStage) {
			s.notes.body = "<!doctype html><title>Scribe</title><p>release 2</p>"
			s.wmux.body = "<!doctype html><title>wmux</title><p>another session</p>"
		}), "healthy", ""},
		{"wrong process on the port", base(), with(func(s *neighbourStage) {
			s.owners[wmuxPort] = neighbourOwner{true, ptr(9999), ptr(`C:\Users\Public\other.exe`)}
		}), "unhealthy", "another executable"},
		{"PID recycled by another executable", base(), with(func(s *neighbourStage) {
			s.owners[wmuxPort] = neighbourOwner{true, ptr(5744), ptr(`C:\Windows\System32\notepad.exe`)}
		}), "unhealthy", "another executable"},
		{"same executable restarted with a new PID", base(), with(func(s *neighbourStage) {
			s.owners[wmuxPort] = neighbourOwner{true, ptr(6100), ptr(`C:\Program Files\wmux\wmux.exe`)}
		}), "healthy", ""},
		{"route changed", base(), with(func(s *neighbourStage) { s.routes["/"] = notesSrv.URL }), "unhealthy", "route changed"},
		{"route removed", base(), with(func(s *neighbourStage) { delete(s.routes, "/notes") }), "unhealthy", "route removed"},
		{"route added", base(), with(func(s *neighbourStage) { s.routes["/extra"] = notesSrv.URL }), "unhealthy", "route added"},
		{"only the Workshop's own route changed", base(), with(func(s *neighbourStage) { s.routes["/workshop"] = "http://127.0.0.1:8762/workshop" }), "healthy", ""},
		{"outage: 502", base(), with(func(s *neighbourStage) { s.wmux = neighbourReply{status: 502, body: "bad gateway"} }), "unhealthy", "unavailable"},
		{"outage: no longer listening", base(), with(func(s *neighbourStage) { s.owners[wmuxPort] = neighbourOwner{} }), "unhealthy", "no longer listening"},
		{"401 is alive", with(func(s *neighbourStage) { s.wmux = neighbourReply{status: 401} }), with(func(s *neighbourStage) { s.wmux = neighbourReply{status: 401} }), "healthy", ""},
		{"302 is alive and not followed", with(func(s *neighbourStage) { s.wmux = neighbourReply{status: 302, redirect: true} }), with(func(s *neighbourStage) { s.wmux = neighbourReply{status: 302, redirect: true} }), "healthy", ""},
		{"owner not observable after", base(), with(func(s *neighbourStage) { s.owners[wmuxPort] = neighbourOwner{true, ptr(5744), nil} }), "unknown", "not observable"},
		{"two processes share the port", base(), with(func(s *neighbourStage) { s.owners[wmuxPort] = neighbourOwner{true, nil, nil} }), "unknown", "not observable"},
		{"Serve status unavailable after", base(), with(func(s *neighbourStage) { s.noServe = true }), "unknown", "unavailable"},
		{"Serve status unreadable after", base(), with(func(s *neighbourStage) { s.raw = "not json" }), "unknown", "unavailable"},
		{"owner not observable before", with(func(s *neighbourStage) { s.owners[notesPort] = neighbourOwner{true, ptr(30040), nil} }), base(), "unknown", "baseline"},
		{"neighbour already down before", with(func(s *neighbourStage) { s.wmux = neighbourReply{status: 503} }), base(), "unknown", "baseline"},
		{"target off loopback", with(func(s *neighbourStage) { s.routes["/remote"] = "http://10.0.0.5:80" }), with(func(s *neighbourStage) { s.routes["/remote"] = "http://10.0.0.5:80" }), "unknown", "baseline"},
		{"TLS target not probed", with(func(s *neighbourStage) { s.routes["/tls"] = "https+insecure://127.0.0.1:8443" }), with(func(s *neighbourStage) { s.routes["/tls"] = "https+insecure://127.0.0.1:8443" }), "unknown", "baseline"},
		{"unhealthy wins over unknown", base(), with(func(s *neighbourStage) {
			s.owners[wmuxPort] = neighbourOwner{true, ptr(9999), ptr(`C:\Users\Public\other.exe`)}
			s.owners[notesPort] = neighbourOwner{true, ptr(30040), nil}
		}), "unhealthy", "another executable"},
	}
	for _, c := range cases {
		t.Run("verdict/"+c.name, func(t *testing.T) {
			got := run(t, "verdict", c.before, c.after)
			reasons, _ := json.Marshal(got["reasons"])
			all := string(reasons) + " " + fmt.Sprint(got["baseline_reason"])
			if got["state"] != c.want || (c.reason != "" && !strings.Contains(all, c.reason) && !(c.reason == "baseline" && got["baseline_known"] == false)) {
				t.Fatalf("state %v (want %s), baseline_known %v, reasons %s, baseline_reason %v", got["state"], c.want, got["baseline_known"], reasons, got["baseline_reason"])
			}
			if c.want == "healthy" && (got["baseline_known"] != true || string(reasons) != "[]") {
				t.Fatalf("healthy with gaps: %v", got)
			}
			if strings.Contains(strings.ToLower(fmt.Sprint(got["reasons"])), "scribe") {
				t.Fatalf("verdict claims an identity: %v", got["reasons"])
			}
		})
	}

	t.Run("rollback/neighbours preserved, Scribe-like content changed", func(t *testing.T) {
		got := run(t, "rollback", base(), with(func(s *neighbourStage) { s.notes.body = "<p>release 2</p>" }))
		if got["error"] != "" || got["receipt_state"] != "rolled_back" || got["installed_is_previous"] != true || got["verdict"] != "healthy" || got["stops"] != float64(1) {
			t.Fatalf("rollback not verified: %v", got)
		}
	})
	t.Run("rollback/route changed during rollback is not verified", func(t *testing.T) {
		got := run(t, "rollback", base(), with(func(s *neighbourStage) { s.routes["/"] = notesSrv.URL }))
		if !strings.Contains(fmt.Sprint(got["error"]), "Rollback application health was not verified") || got["receipt_state"] == "rolled_back" || got["verdict"] != "unhealthy" {
			t.Fatalf("broken neighbour accepted: %v", got)
		}
	})
	t.Run("rollback/unobservable neighbours refuse before any stop", func(t *testing.T) {
		got := run(t, "rollback", with(func(s *neighbourStage) { s.noServe = true }), base())
		if !strings.Contains(fmt.Sprint(got["error"]), "no stop performed") || got["stops"] != float64(0) || got["installed_is_previous"] != false {
			t.Fatalf("rollback went ahead blind: %v", got)
		}
	})

	// Receipts written before HEALTH-R4 (ScribeSHA256, no Neighbours), new ones (Neighbours recorded) and damaged ones,
	// replayed by a rollback that may resume an interrupted one. A damaged or incomplete receipt must be refused before
	// any stop or start; "rolled_back" may only be written after the restored service and its neighbours were verified.
	refused := func(t *testing.T, got map[string]any) {
		t.Helper()
		if got["error"] == "" || got["stops"] != float64(0) || got["starts"] != float64(0) || got["installed_is_previous"] != false || got["receipt_state"] == "rolled_back" {
			t.Fatalf("receipt not refused before any stop: %v", got)
		}
	}
	verified := func(t *testing.T, got map[string]any) {
		t.Helper()
		if got["error"] != "" || got["receipt_state"] != "rolled_back" || got["installed_is_previous"] != true || got["service"] != "Running" || got["verdict"] != "healthy" {
			t.Fatalf("rollback not verified: %v", got)
		}
	}
	notVerified := func(t *testing.T, got map[string]any) {
		t.Helper()
		if got["error"] == "" || got["receipt_state"] == "rolled_back" || got["verdict"] == "healthy" {
			t.Fatalf("false healthy rollback: %v", got)
		}
	}
	unobservable := with(func(s *neighbourStage) { s.noServe = true })
	receiptCases := []struct {
		name          string
		before, after neighbourStage
		args          []string
		check         func(*testing.T, map[string]any)
	}{
		{"old ScribeSHA256 receipt", base(), base(), []string{"-Receipt", "old"}, verified},
		{"old receipt, Scribe-like content changed", base(), with(func(s *neighbourStage) { s.notes.body = "<p>release 2</p>" }), []string{"-Receipt", "old"}, verified},
		{"old receipt, neighbour broken by the rollback", base(), with(func(s *neighbourStage) { s.routes["/"] = notesSrv.URL }), []string{"-Receipt", "old"}, notVerified},
		{"old receipt, neighbours unobservable", unobservable, base(), []string{"-Receipt", "old"}, refused},
		{"new receipt, recorded baseline is not reused", base(), base(), []string{"-Receipt", "stale"}, verified},
		{"new receipt, neighbours unobservable", unobservable, base(), []string{"-Receipt", "stale"}, refused},
		{"truncated receipt", base(), base(), []string{"-Receipt", "truncated"}, refused},
		{"empty receipt", base(), base(), []string{"-Receipt", "empty"}, refused},
		{"unknown schema", base(), base(), []string{"-Receipt", "schema2"}, refused},
		{"unknown state", base(), base(), []string{"-Receipt", "bogus-state"}, refused},
		{"partial: no Origin", base(), base(), []string{"-Receipt", "partial:Origin"}, refused},
		{"partial: no Listen", base(), base(), []string{"-Receipt", "partial:Listen"}, refused},
		{"partial: no PreviousClientVersion", base(), base(), []string{"-Receipt", "partial:PreviousClientVersion"}, refused},
		{"partial: no ConfigPath", base(), base(), []string{"-Receipt", "partial:ConfigPath"}, refused},
		{"partial: no InstalledExe", base(), base(), []string{"-Receipt", "partial:InstalledExe"}, refused},
		{"partial: no ExpectedImagePath", base(), base(), []string{"-Receipt", "partial:ExpectedImagePath"}, refused},
		{"interrupted after stop, old receipt", base(), base(), []string{"-Receipt", "old", "-ServiceState", "Stopped"}, verified},
		{"interrupted after replacement, old receipt", base(), base(), []string{"-Receipt", "old", "-Binary", "previous", "-ServiceState", "Stopped"}, verified},
		{"interrupted after replacement, new receipt", base(), base(), []string{"-Receipt", "stale", "-Binary", "previous", "-ServiceState", "Stopped"}, verified},
		{"interrupted after replacement, neighbours unobservable", unobservable, unobservable, []string{"-Receipt", "old", "-Binary", "previous", "-ServiceState", "Stopped"}, notVerified},
		{"interrupted after replacement, neighbour down", with(func(s *neighbourStage) { s.wmux = neighbourReply{status: 502} }), base(), []string{"-Receipt", "old", "-Binary", "previous", "-ServiceState", "Stopped"}, notVerified},
	}
	for _, c := range receiptCases {
		t.Run("receipt/"+c.name, func(t *testing.T) { c.check(t, run(t, "rollback", c.before, c.after, c.args...)) })
	}

	// The rollback's file replacement fails after the service was stopped. The service may only be restarted on the
	// verified previous executable; anything else is left stopped and recorded for reconciliation, never as rolled back.
	t.Run("replace/executable held by another process", func(t *testing.T) {
		got := run(t, "rollback", base(), base(), "-OnStop", "hold")
		if !strings.Contains(fmt.Sprint(got["error"]), "left stopped") || got["starts"] != float64(0) || got["service"] != "Stopped" ||
			got["receipt_state"] != "reconciliation_required" || got["installed_is_previous"] != false {
			t.Fatalf("failed replacement not recorded for reconciliation: %v", got)
		}
	})
	t.Run("replace/executable removed: verified previous restored and checked", func(t *testing.T) {
		verified(t, run(t, "rollback", base(), base(), "-OnStop", "remove"))
	})
	t.Run("replace/executable removed, neighbour broken: not rolled back", func(t *testing.T) {
		got := run(t, "rollback", base(), with(func(s *neighbourStage) { s.routes["/"] = notesSrv.URL }), "-OnStop", "remove")
		notVerified(t, got)
		if got["installed_is_previous"] != true {
			t.Fatalf("verified previous executable not restored: %v", got)
		}
	})
}
