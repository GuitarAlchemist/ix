package main

import (
	"bytes"
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func mustInventory(t *testing.T, js string) *resumeInventory {
	t.Helper()
	inv, err := parseResumeInventory([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	return inv
}

// The start order puts each component after the ones it names; ties keep inventory order.
func TestResumeOrderFollowsAfter(t *testing.T) {
	inv := mustInventory(t, `{"components":[
		{"name":"relay","after":["plane","voice"],"listen":"127.0.0.1:1"},
		{"name":"voice","listen":"127.0.0.1:2"},
		{"name":"plane","service":"ixctl"}]}`)
	order, _ := resumeOrder(inv.Components)
	var got []string
	for _, c := range order {
		got = append(got, c.Name)
	}
	if strings.Join(got, " ") != "voice plane relay" {
		t.Fatalf("order %v", got)
	}
}

func TestResumeInventoryRefusals(t *testing.T) {
	for _, c := range []struct{ js, want string }{
		{`{"components":[]}`, "no components"},
		{`{"components":[{"name":"a","listen":"127.0.0.1:1"},{"name":"a","listen":"127.0.0.1:2"}]}`, "declared twice"},
		{`{"components":[{"name":"a"}]}`, "nothing to observe"},
		{`{"components":[{"name":"a","after":["b"],"listen":"127.0.0.1:1"}]}`, "not declared"},
		{`{"components":[{"name":"a","after":["b"],"listen":"127.0.0.1:1"},{"name":"b","after":["a"],"listen":"127.0.0.1:2"}]}`, "cycle"},
		{`{"components":[{"name":"a","health":"http://example.com/health"}]}`, "not a loopback address"},
		{`{"components":[{"name":"a","health":"file:///etc/passwd"}]}`, "not http(s)"},
		{`{"components":[{"name":"a","listen":"127.0.0.1"}]}`, "listen"},
		{`{"components":[{"name":"a","listen":"127.0.0.1:1","health_host":"h"}]}`, "without health"},
		{`{"components":[{"name":"a","listen":"127.0.0.1:1","start":"x.exe"}]}`, "unknown field"},
	} {
		if _, err := parseResumeInventory([]byte(c.js)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want %q", c.js, err, c.want)
		}
	}
}

func fakeProbes(services map[string][2]string, listening map[string]bool, healthy map[string]error) resumeProbes {
	return resumeProbes{
		service: func(name string) (string, string, string, error) {
			s, ok := services[name]
			if !ok {
				return "", "", "", errors.New("access is denied")
			}
			return s[0], s[1], "", nil
		},
		health:    func(u, _ string) ([]byte, error) { return nil, healthy[u] },
		listening: func(a string) (bool, error) { return listening[a], nil },
		exists: func(p string) error {
			if strings.Contains(p, "missing") {
				return os.ErrNotExist
			}
			return nil
		},
	}
}

func TestResumeVerdicts(t *testing.T) {
	inv := mustInventory(t, `{"components":[
		{"name":"plane","service":"ixctl","health":"http://127.0.0.1:8761/health","durable":["C:/state/plane.json","C:/state/missing.json"],
		 "owner":"IxForge","remedy":"sc.exe config ixctl start= delayed-auto"},
		{"name":"auto","service":"auto-svc"},
		{"name":"stopped","service":"stopped-svc"},
		{"name":"denied","service":"secret-svc"},
		{"name":"voice","listen":"127.0.0.1:59126","owner":"voice lane","remedy":"start piper-voice serve"},
		{"name":"relay","after":["voice"],"health":"http://localhost:9/health"}]}`)
	p := fakeProbes(
		map[string][2]string{"ixctl": {"running", "manual"}, "auto-svc": {"running", "automatic (delayed)"}, "stopped-svc": {"stopped", "automatic"}},
		map[string]bool{},
		map[string]error{"http://localhost:9/health": errors.New("connection refused")},
	)
	got := map[string]resumeResult{}
	for _, r := range checkResume(inv, p) {
		got[r.Name] = r
	}
	expect := func(name, verdict, line string) {
		t.Helper()
		r := got[name]
		if r.Verdict != verdict || !strings.Contains(strings.Join(r.Checks, "\n"), line) {
			t.Errorf("%s: %s %q, want %s with %q", name, r.Verdict, r.Checks, verdict, line)
		}
	}
	expect("plane", verdictDegraded, "start type manual: it will not start after a reboot")
	expect("plane", verdictDegraded, "durable state C:/state/missing.json is missing")
	expect("auto", verdictOK, "running, start type automatic (delayed)")
	expect("stopped", verdictDown, "is stopped (start type automatic)")
	expect("denied", verdictUnknown, "access is denied")
	expect("voice", verdictDown, "nothing listens on 127.0.0.1:59126 (a client gets connection refused)")
	expect("relay", verdictDown, "connection refused")
	if b := got["relay"].BlockedBy; len(b) != 1 || b[0] != "voice" {
		t.Errorf("relay blocked by %v, want voice", b)
	}

	var out bytes.Buffer
	writeResume(&out, checkResume(inv, p), false)
	text := out.String()
	for _, want := range []string{"6 components, 5 not ready", "owner: IxForge", "remedy (run by a human): start piper-voice serve", "starts after voice, which is not up"} {
		if !strings.Contains(text, want) {
			t.Errorf("text lacks %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "\nready") {
		t.Errorf("printed ready:\n%s", text)
	}
}

// The check observes only: no file of it starts a process.
func TestResumeStartsNothing(t *testing.T) {
	for _, name := range []string{"resume.go", "resume_windows.go", "resume_other.go"} {
		f, err := parser.ParseFile(token.NewFileSet(), name, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imp := range f.Imports {
			if p, _ := strconv.Unquote(imp.Path.Value); p == "os/exec" || p == "syscall" {
				t.Errorf("%s imports %s", name, p)
			}
		}
	}
}

// The real listener probe sees a real listener, then its absence, without connecting.
func TestTCPListeningSeesARealListener(t *testing.T) {
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	if on, err := tcpListening(addr); err != nil || !on {
		t.Fatalf("listening %s: %v %v", addr, on, err)
	}
	ln.Close()
	if on, err := tcpListening(addr); err != nil || on {
		t.Fatalf("closed %s still listening: %v %v", addr, on, err)
	}
}

func TestQueryServiceReadsARealService(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the Windows service manager")
	}
	state, start, image, err := queryService("EventLog")
	if err != nil || state != "running" || !strings.HasPrefix(start, "automatic") || !strings.Contains(strings.ToLower(image), "svchost.exe") {
		t.Fatalf("EventLog: %q %q %q %v", state, start, image, err)
	}
	if state, _, _, err := queryService("ixctl-resume-check-no-such-service"); err != nil || state != "not installed" {
		t.Fatalf("missing service: %q %v", state, err)
	}
}

// A server that gates on the Host header (IxForge's workshop) is healthy only with health_host.
func TestResumeHealthCarriesTheHostHeader(t *testing.T) {
	gated := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Host != "aw.example.ts.net" {
			w.WriteHeader(http.StatusForbidden)
		}
	}))
	defer gated.Close()
	health := realResumeProbes().health
	if _, err := health(gated.URL+"/workshop/branding.json", ""); err == nil || !strings.Contains(err.Error(), "403") {
		t.Fatalf("without the host: %v", err)
	}
	if _, err := health(gated.URL+"/workshop/branding.json", "aw.example.ts.net"); err != nil {
		t.Fatalf("with the host: %v", err)
	}
}

// A 2xx from the wrong program is not health: after a reboot another process can hold the port.
// health_expect names what the body must contain (IxForge's branding.json: "name":"IxForge").
func TestResumeHealthChecksTheIdentity(t *testing.T) {
	body := `{"name":"Impostor","version":"9.9"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { fmt.Fprint(w, body) }))
	defer srv.Close()
	inv := mustInventory(t, fmt.Sprintf(`{"components":[{"name":"workshop","health":%q,"health_expect":"\"name\":\"IxForge\""}]}`,
		srv.URL+"/workshop/branding.json"))
	r := checkResume(inv, realResumeProbes())[0]
	if r.Verdict != verdictDown || !strings.Contains(strings.Join(r.Checks, "\n"), `lacks "\"name\":\"IxForge\""`) {
		t.Fatalf("impostor: %s %q", r.Verdict, r.Checks)
	}
	body = `{"name":"IxForge","short_name":"IxForge","version":"0.1.1"}`
	if r := checkResume(inv, realResumeProbes())[0]; r.Verdict != verdictOK {
		t.Fatalf("IxForge: %s %q", r.Verdict, r.Checks)
	}
	if _, err := parseResumeInventory([]byte(`{"components":[{"name":"a","listen":"127.0.0.1:1","health_expect":"x"}]}`)); err == nil {
		t.Fatal("health_expect without health accepted")
	}
}

// roundTrip is an http.RoundTripper made of a function: no socket, no DNS.
type roundTrip func(*http.Request) (*http.Response, error)

func (f roundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// A loopback health URL that redirects off the machine is not followed: the check stays on
// loopback, as the inventory promises. The transport is a function, so the test makes no request
// at all; it only records where the client tried to go.
func TestResumeHealthRefusesARedirectOffLoopback(t *testing.T) {
	for _, target := range []string{"http://outside.example/x", "http://192.0.2.1/x", "https://[2001:db8::1]/x"} {
		var tried []string
		rt := roundTrip(func(r *http.Request) (*http.Response, error) {
			tried = append(tried, r.URL.Host)
			if r.URL.Hostname() == "127.0.0.1" {
				h := http.Header{}
				h.Set("Location", target)
				return &http.Response{StatusCode: http.StatusFound, Header: h, Body: http.NoBody, Request: r}, nil
			}
			return nil, errors.New("left the machine")
		})
		_, err := healthProbe(rt, resumeHealthTimeout)("http://127.0.0.1:8761/health", "")
		if len(tried) != 1 || err == nil || !strings.Contains(err.Error(), "leaves the loopback") {
			t.Errorf("redirect to %s: tried %v, err %v", target, tried, err)
		}
	}
	// A redirect that stays on loopback is followed.
	var tried []string
	rt := roundTrip(func(r *http.Request) (*http.Response, error) {
		tried = append(tried, r.URL.Path)
		if r.URL.Path == "/old" {
			h := http.Header{}
			h.Set("Location", "http://localhost:8761/new")
			return &http.Response{StatusCode: http.StatusFound, Header: h, Body: http.NoBody, Request: r}, nil
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("ok")), Request: r}, nil
	})
	if body, err := healthProbe(rt, resumeHealthTimeout)("http://127.0.0.1:8761/old", ""); err != nil || string(body) != "ok" || len(tried) != 2 {
		t.Errorf("loopback redirect: %q %v %v", body, err, tried)
	}
}

// listen_image: which build holds the port, from the PID in the TCP table and the process's image.
func TestResumeListenImage(t *testing.T) {
	const want = "delta-remote-v0.6.0-21-g2ca5e6c.exe"
	inv := mustInventory(t, `{"components":[{"name":"remote","listen":"192.168.86.250:8810","listen_image":"`+want+`"}]}`)
	run := func(owners []uint32, image string, openErr error) (resumeResult, int) {
		released, reads := 0, 0
		p := resumeProbes{
			listening: func(string) (bool, error) { return true, nil },
			listenOwner: func(string) (uint32, error) {
				pid := owners[min(reads, len(owners)-1)]
				reads++
				return pid, nil
			},
			imageOf: func(uint32) (string, func(), error) {
				if openErr != nil {
					return "", nil, openErr
				}
				return image, func() { released++ }, nil
			},
		}
		return checkResume(inv, p)[0], released
	}
	expect := func(what string, r resumeResult, verdict, line string) {
		t.Helper()
		if r.Verdict != verdict || !strings.Contains(strings.Join(r.Checks, "\n"), line) {
			t.Errorf("%s: %s %q, want %s with %q", what, r.Verdict, r.Checks, verdict, line)
		}
	}
	r, released := run([]uint32{4242}, `C:\Users\x\Temp\delta-remote\DELTA-REMOTE-v0.6.0-21-g2ca5e6c.EXE`, nil)
	expect("right image", r, verdictOK, "is held by DELTA-REMOTE-v0.6.0-21-g2ca5e6c.EXE")
	if released != 1 {
		t.Errorf("handle released %d times", released)
	}
	r, _ = run([]uint32{4242}, `C:\tools\delta-remote-v0.5.0.exe`, nil)
	expect("wrong image", r, verdictDown, "held by delta-remote-v0.5.0.exe, not "+want)
	r, released = run([]uint32{4242, 5151}, `C:\other\impostor.exe`, nil)
	expect("PID reused", r, verdictUnknown, "changed from process 4242 to 5151")
	if released != 1 {
		t.Errorf("PID reused: handle released %d times", released)
	}
	r, _ = run([]uint32{0}, "", nil)
	expect("listener gone", r, verdictUnknown, "was gone when its image was read")
	r, _ = run([]uint32{4}, "", errors.New("Access is denied."))
	expect("query refused", r, verdictUnknown, "process 4: Access is denied.")

	for _, js := range []string{
		`{"components":[{"name":"a","service":"s","listen_image":"x.exe"}]}`,
		`{"components":[{"name":"a","listen":"127.0.0.1:1","listen_image":"C:\\tools\\x.exe"}]}`,
	} {
		if _, err := parseResumeInventory([]byte(js)); err == nil {
			t.Errorf("accepted %s", js)
		}
	}
}

// On Windows the real probes name this test binary as the holder of its own listener.
func TestResumeListenImageReadsARealProcess(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the Windows TCP table")
	}
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	self, _ := os.Executable()
	got, err := listenerImage(realResumeProbes(), ln.Addr().String())
	if err != nil || !strings.EqualFold(got, filepath.Base(self)) {
		t.Fatalf("image %q %v, want %s", got, err, filepath.Base(self))
	}
}

// End to end with the real probes on stand-in components: ready exits 0, a component down
// exits 1 and names it.
func TestResumeCheckCommand(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	defer up.Close()
	ln, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	gone, _ := net.Listen("tcp4", "127.0.0.1:0")
	goneAddr := gone.Addr().String()
	gone.Close()

	dir := t.TempDir()
	write := func(js string) string {
		p := filepath.Join(dir, fmt.Sprintf("inv%d.json", len(js)))
		os.WriteFile(p, []byte(js), 0o644)
		return p
	}
	ready := write(fmt.Sprintf(`{"components":[{"name":"plane","health":%q},{"name":"voice","listen":%q,"after":["plane"]}]}`,
		up.URL+"/health", ln.Addr().String()))
	var out, errb bytes.Buffer
	if code := cli([]string{"resume", "check", ready}, &out, &errb); code != 0 || !strings.HasSuffix(out.String(), "ready\n") {
		t.Fatalf("ready: exit %d\n%s%s", code, out.String(), errb.String())
	}
	down := write(fmt.Sprintf(`{"components":[{"name":"voice","listen":%q,"remedy":"start it"},{"name":"hook","health":"http://%s/v1/hook","after":["voice"]}]}`,
		goneAddr, goneAddr))
	out.Reset()
	if code := cli([]string{"resume", "check", down}, &out, &errb); code != 1 ||
		!strings.Contains(out.String(), "nothing listens on "+goneAddr) || !strings.Contains(out.String(), "remedy (run by a human): start it") {
		t.Fatalf("down: exit %d\n%s", code, out.String())
	}
	out.Reset()
	if code := cli([]string{"resume", "check", down, "--json"}, &out, &errb); code != 1 || !strings.Contains(out.String(), `"ready": false`) {
		t.Fatalf("json: exit %d\n%s", code, out.String())
	}
	if code := cli([]string{"resume", "check", filepath.Join(dir, "none.json")}, &out, &errb); code != 1 {
		t.Fatalf("missing inventory: exit %d", code)
	}
}
