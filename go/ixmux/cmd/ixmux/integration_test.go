package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// The integration tests build the real binary once and run it as a process; no shell.
var binary, fakeClaude string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "ixmux-bin")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	binary = filepath.Join(dir, "ixmux"+exeSuffix())
	gobin := filepath.Join(runtime.GOROOT(), "bin", "go"+exeSuffix())
	cmd := exec.Command(gobin, "build", "-o", binary, ".")
	if out, err := cmd.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "go build: %v\n%s", err, out)
		os.Exit(1)
	}
	fakeClaude = filepath.Join(dir, "claude"+exeSuffix())
	if out, err := exec.Command(gobin, "build", "-o", fakeClaude, "../../claudestream/internal/fakeclaude").CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "go build fakeclaude: %v\n%s", err, out)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

type proc struct {
	t      *testing.T
	cmd    *exec.Cmd
	state  string
	addr   string
	token  string
	info   server.Info
	exited chan struct{}
	err    error
	out    *bytes.Buffer
}

// start runs ixmux serve on state and waits for its ixmux.json.
func start(t *testing.T, state string, env []string, args ...string) *proc {
	t.Helper()
	p := &proc{t: t, state: state, exited: make(chan struct{}), out: &bytes.Buffer{}}
	p.cmd = exec.Command(binary, append([]string{"serve", "--addr", "127.0.0.1:0", "--state", state}, args...)...)
	p.cmd.Env = append(os.Environ(), env...)
	p.cmd.Stdout, p.cmd.Stderr = p.out, p.out
	if err := p.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() { p.err = p.cmd.Wait(); close(p.exited) }()
	t.Cleanup(func() {
		p.cmd.Process.Kill()
		<-p.exited
	})
	deadline := time.Now().Add(15 * time.Second)
	for {
		b, err := os.ReadFile(filepath.Join(state, server.InfoFile))
		if err == nil && json.Unmarshal(b, &p.info) == nil && p.info.PID == p.cmd.Process.Pid && p.info.State == "running" {
			break
		}
		select {
		case <-p.exited:
			t.Fatalf("ixmux exited at start: %v\n%s", p.err, p.out)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatalf("no ixmux.json of pid %d\n%s", p.cmd.Process.Pid, p.out)
		}
		time.Sleep(20 * time.Millisecond)
	}
	tok, err := os.ReadFile(filepath.Join(state, server.TokenFile))
	if err != nil {
		t.Fatal(err)
	}
	p.addr, p.token = p.info.Addr, string(tok)
	return p
}

// do sends one authenticated request and decodes a JSON answer into out (when not nil).
func (p *proc) do(method, path string, body any, out any) int {
	p.t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, "http://"+p.addr+path, rd)
	req.Header.Set("Authorization", "Bearer "+p.token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		p.t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if out != nil {
		if err := json.Unmarshal(b, out); err != nil {
			p.t.Fatalf("%s %s: %d %s: %v", method, path, resp.StatusCode, b, err)
		}
	}
	return resp.StatusCode
}

func (p *proc) dial(query string, client *http.Client) (*websocket.Conn, *http.Response, error) {
	h := http.Header{"Authorization": {"Bearer " + p.token}}
	c, resp, err := websocket.Dial(context.Background(), "ws://"+p.addr+"/v1/stream"+query,
		&websocket.DialOptions{HTTPHeader: h, HTTPClient: client, CompressionMode: websocket.CompressionDisabled})
	if c != nil {
		c.SetReadLimit(server.MaxStreamMessage) // the library's default (32 KiB) is below a journal page
	}
	return c, resp, err
}

type msg struct {
	Type  string          `json:"type"`
	Epoch string          `json:"epoch"`
	Seq   uint64          `json:"seq"`
	Data  json.RawMessage `json:"data"`
}

type transition struct {
	MandateID string `json:"mandate_id"`
	State     string `json:"state"`
}

func read(t *testing.T, c *websocket.Conn, within time.Duration) msg {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), within)
	defer cancel()
	_, b, err := c.Read(ctx)
	if err != nil {
		t.Fatalf("stream read: %v", err)
	}
	var m msg
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

// until reads mandate events until id reaches state; it returns every event read.
func until(t *testing.T, c *websocket.Conn, id, state string) []msg {
	t.Helper()
	var got []msg
	for {
		m := read(t, c, 10*time.Second)
		got = append(got, m)
		var tr transition
		if m.Type == "mandate" && json.Unmarshal(m.Data, &tr) == nil && tr.MandateID == id && tr.State == state {
			return got
		}
	}
}

func lines(t *testing.T, path string) int {
	t.Helper()
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Count(b, []byte("\n"))
}

type result struct {
	State    string `json:"state"`
	Replayed bool   `json:"replayed"`
	Error    string `json:"error"`
}

type statusView struct {
	Epoch   string `json:"epoch"`
	Targets []struct {
		Name        string `json:"name"`
		Incarnation string `json:"incarnation"`
	} `json:"targets"`
}

func incarnation(p *proc, target string) string {
	var st statusView
	p.do("GET", "/v1/status", nil, &st)
	for _, tg := range st.Targets {
		if tg.Name == target {
			return tg.Incarnation
		}
	}
	p.t.Fatalf("no target %s in status", target)
	return ""
}

func mandate(id, inc string) map[string]string {
	return map[string]string{"mandate_id": id, "target": "s", "incarnation": inc, "text": "hello"}
}

// Oracle 1: Hello.
func TestHello(t *testing.T) {
	dir := t.TempDir()
	session := filepath.Join(dir, "session.jsonl")
	p := start(t, filepath.Join(dir, "state"), nil, "--target", "s="+session)
	c, _, err := p.dial("", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.CloseNow()
	if m := read(t, c, 5*time.Second); m.Type != "hello" || m.Epoch != p.info.Epoch {
		t.Fatalf("first message %+v, want hello of epoch %s", m, p.info.Epoch)
	}
	var res result
	if code := p.do("POST", "/v1/mandates", mandate("hello", incarnation(p, "s")), &res); code != 200 || res.State != "delivered" {
		t.Fatalf("POST hello: %d %+v", code, res)
	}
	until(t, c, "hello", "delivered")
	if n := lines(t, session); n != 1 {
		t.Errorf("the fake session holds %d lines, want 1", n)
	}
	var e result
	if p.do("GET", "/v1/mandates/hello", nil, &e); e.State != "delivered" {
		t.Errorf("GET hello: %+v", e)
	}
}

// Oracle 2: the client hangs up before reading the answer, then sends the mandate again.
func TestLostHTTPAck(t *testing.T) {
	dir := t.TempDir()
	session, hold := filepath.Join(dir, "session.jsonl"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600) // the session holds its answer until the client has hung up
	p := start(t, filepath.Join(dir, "state"), []string{server.HoldEnv + "=" + hold}, "--target", "s="+session)
	b, _ := json.Marshal(mandate("m1", incarnation(p, "s")))
	conn, err := net.Dial("tcp", p.addr)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(conn, "POST /v1/mandates HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", p.addr, p.token, len(b), b)
	conn.Close() // the answer is never read
	deadline := time.Now().Add(10 * time.Second)
	for lines(t, session) == 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	// The client is gone; the delivery must not depend on its connection.
	for range 15 {
		var e result
		if p.do("GET", "/v1/mandates/m1", nil, &e); e.State != "delivering" {
			t.Fatalf("after the client hung up: %+v, want still delivering", e)
		}
		time.Sleep(20 * time.Millisecond)
	}
	os.Remove(hold)
	for e := (result{}); e.State != "delivered"; {
		if time.Now().After(deadline) {
			t.Fatalf("never delivered: %+v", e)
		}
		p.do("GET", "/v1/mandates/m1", nil, &e)
		time.Sleep(20 * time.Millisecond)
	}
	var res result
	code := p.do("POST", "/v1/mandates", mandate("m1", incarnation(p, "s")), &res)
	if code != 200 || !res.Replayed || res.State != "delivered" {
		t.Errorf("resend after the lost answer: %d %+v, want delivered and replayed", code, res)
	}
	if n := lines(t, session); n != 1 {
		t.Errorf("the fake session holds %d lines, want 1", n)
	}
}

// Oracle 3: the stream resumes with since, without losing or duplicating an event; another
// epoch gives a typed gap.
func TestStreamResume(t *testing.T) {
	dir := t.TempDir()
	p := start(t, filepath.Join(dir, "state"), nil, "--target", "s="+filepath.Join(dir, "session.jsonl"))
	inc := incarnation(p, "s")
	c, _, err := p.dial("", nil)
	if err != nil {
		t.Fatal(err)
	}
	hello := read(t, c, 5*time.Second)
	p.do("POST", "/v1/mandates", mandate("m1", inc), nil)
	got := until(t, c, "m1", "delivered")
	last := got[len(got)-1].Seq
	c.Close(websocket.StatusNormalClosure, "")

	p.do("POST", "/v1/mandates", mandate("m2", inc), nil) // while disconnected
	p.do("POST", "/v1/mandates", mandate("m3", inc), nil)
	c, _, err = p.dial(fmt.Sprintf("?since=%d&epoch=%s", last, hello.Epoch), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.CloseNow()
	if m := read(t, c, 5*time.Second); m.Type != "hello" {
		t.Fatalf("first message %+v", m)
	}
	want := last + 1
	for _, m := range until(t, c, "m3", "delivered") {
		if m.Type != "mandate" || m.Seq != want {
			t.Fatalf("after resuming at %d: %s seq %d, want mandate seq %d (no loss, no duplicate)", last, m.Type, m.Seq, want)
		}
		want++
	}

	bad, _, err := p.dial(fmt.Sprintf("?since=%d&epoch=another", last), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer bad.CloseNow()
	read(t, bad, 5*time.Second) // hello
	gap := read(t, bad, 5*time.Second)
	var g server.Gap
	json.Unmarshal(gap.Data, &g)
	if gap.Type != "gap" || g.Reason != "epoch" {
		t.Errorf("resume with another epoch: %+v, want a gap of reason epoch", gap)
	}
	if st := read(t, bad, 5*time.Second); st.Type != "state-page" && st.Type != "state-end" {
		t.Errorf("after the gap: %s, want the state (state-page or state-end)", st.Type)
	}
}

// Oracle 4: a real kill of the server during a held delivery, then a restart on the same state.
func TestKillDuringDelivery(t *testing.T) {
	dir := t.TempDir()
	session, state, hold := filepath.Join(dir, "session.jsonl"), filepath.Join(dir, "state"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	p := start(t, state, []string{server.HoldEnv + "=" + hold}, "--target", "s="+session, "--timeout", "1h")
	inc := incarnation(p, "s")
	go func() {
		defer func() { recover() }() // the request fails when the server is killed
		req, _ := http.NewRequest("POST", "http://"+p.addr+"/v1/mandates", strings.NewReader(fmt.Sprintf(`{"mandate_id":"m1","target":"s","incarnation":%q,"text":"hello"}`, inc)))
		req.Header.Set("Authorization", "Bearer "+p.token)
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
		}
	}()
	deadline := time.Now().Add(10 * time.Second)
	for lines(t, session) == 0 {
		if time.Now().After(deadline) {
			t.Fatalf("the held delivery never took effect\n%s", p.out)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err := p.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	<-p.exited
	os.Remove(hold)

	q := start(t, state, nil, "--target", "s="+session)
	var e result
	if q.do("GET", "/v1/mandates/m1", nil, &e); e.State != "uncertain" {
		t.Errorf("after the kill: %+v, want uncertain", e)
	}
	var res result
	if code := q.do("POST", "/v1/mandates", mandate("m1", inc), &res); code != http.StatusConflict || res.Error != "uncertain-needs-reconcile" {
		t.Errorf("resend before reconcile: %d %+v, want 409 uncertain-needs-reconcile", code, res)
	}
	if code := q.do("POST", "/v1/mandates/m1/reconcile", nil, &res); code != 200 || res.State != "delivered" {
		t.Errorf("reconcile: %d %+v, want delivered", code, res)
	}
	if n := lines(t, session); n != 1 {
		t.Errorf("%d effects, want 1", n)
	}
}

// Oracle 5: a clean stop by POST /v1/shutdown.
func TestShutdown(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state")
	p := start(t, state, nil, "--target", "s="+filepath.Join(dir, "session.jsonl"))
	c, _, err := p.dial("", nil)
	if err != nil {
		t.Fatal(err)
	}
	read(t, c, 5*time.Second)
	begun := time.Now()
	if code := p.do("POST", "/v1/shutdown", nil, nil); code != http.StatusAccepted {
		t.Fatalf("shutdown: %d", code)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, _, err := c.Read(ctx); websocket.CloseStatus(err) != websocket.StatusGoingAway {
		t.Errorf("stream at the stop: %v, want close 1001", err)
	}
	select {
	case <-p.exited:
	case <-time.After(15 * time.Second):
		t.Fatal("ixmux did not exit within 15s")
	}
	if p.cmd.ProcessState.ExitCode() != 0 {
		t.Errorf("exit code %d, want 0\n%s", p.cmd.ProcessState.ExitCode(), p.out)
	}
	t.Logf("stopped in %s", time.Since(begun))
	var info server.Info
	b, _ := os.ReadFile(filepath.Join(state, server.InfoFile))
	if json.Unmarshal(b, &info); info.State != "stopped" || info.ExitCode == nil || *info.ExitCode != 0 {
		t.Errorf("ixmux.json after the stop: %s", b)
	}
	if _, err := os.Stat(filepath.Join(state, server.TokenFile)); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("the token file is left after the stop (%v)", err)
	}
	q := start(t, state, nil) // the lock is free
	q.do("POST", "/v1/shutdown", nil, nil)
	<-q.exited
}

// Oracle 6: ix#511 through the transport, on the fixture of a 1,114,550-byte record.
func TestIssue511OverHTTP(t *testing.T) {
	const bigRecord = 1_114_550
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	big := `{"pad":"` + strings.Repeat("x", bigRecord-len(`{"pad":""}`)-1) + `"}` + "\n"
	os.WriteFile(path, []byte(`{"n":1}`+"\n"+big+`{"n":3}`+"\n"), 0o600)
	p := start(t, filepath.Join(dir, "state"), nil, "--journal", "j="+path)
	for _, budget := range []string{"&max_bytes=16384", ""} {
		var events []map[string]any
		q := "?from=0" + budget
		for calls := 0; len(events) < 3; calls++ {
			if calls > 20 {
				t.Fatalf("budget %q: %d events after 20 calls: %v", budget, len(events), events)
			}
			var page struct {
				Events []map[string]any `json:"events"`
				Cursor string           `json:"cursor"`
				Error  string           `json:"error"`
			}
			if code := p.do("GET", "/v1/journal/j"+q, nil, &page); code != 200 {
				t.Fatalf("budget %q: %d %s", budget, code, page.Error)
			}
			events = append(events, page.Events...)
			q = "?cursor=" + url.QueryEscape(page.Cursor) + budget
		}
		if events[1]["invalid"] != "line-too-long" || events[1]["offset"].(float64) != 8 || events[1]["end"].(float64) != 8+bigRecord || events[2]["line"] != `{"n":3}` {
			t.Errorf("budget %q: events %v", budget, events)
		}
	}
}

// Oracle 7: security.
func TestSecurity(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	os.WriteFile(path, []byte(`{"n":1}`+"\n"+`{"n":2}`+"\n"), 0o600)
	p := start(t, filepath.Join(dir, "state"), nil, "--journal", "j="+path)
	send := func(mod func(*http.Request)) int {
		req, _ := http.NewRequest("GET", "http://"+p.addr+"/v1/status", nil)
		req.Header.Set("Authorization", "Bearer "+p.token)
		mod(req)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if code := send(func(r *http.Request) { r.Header.Del("Authorization") }); code != 401 {
		t.Errorf("no token: %d, want 401", code)
	}
	if code := send(func(r *http.Request) { r.Header.Set("Authorization", "Bearer "+strings.Repeat("0", 64)) }); code != 401 {
		t.Errorf("wrong token: %d, want 401", code)
	}
	if code := send(func(r *http.Request) { r.Host = "evil.example:80" }); code != 403 {
		t.Errorf("Host evil.example: %d, want 403", code)
	}
	if code := send(func(r *http.Request) { r.Header.Set("Origin", "http://127.0.0.1") }); code != 403 {
		t.Errorf("an Origin: %d, want 403", code)
	}
	if strings.Contains(p.out.String(), p.token) {
		t.Error("the token appears in ixmux's output")
	}

	type page struct {
		Events        []map[string]any `json:"events"`
		Cursor        string           `json:"cursor"`
		CursorIgnored bool             `json:"cursor_ignored"`
	}
	var first page
	p.do("GET", "/v1/journal/j?from=0&max_bytes=8", nil, &first)
	if len(first.Events) != 1 || first.Cursor == "" {
		t.Fatalf("first page %+v", first)
	}
	payload, sig, _ := strings.Cut(first.Cursor, ".")
	forged := []string{
		payload,                                 // unsigned
		payload + "." + sig[:len(sig)-2] + "AA", // badly signed
		strings.ToUpper(payload[:4]) + payload[4:] + "." + sig, // payload changed
	}
	for _, f := range forged {
		var got page
		p.do("GET", "/v1/journal/j?from=0&cursor="+url.QueryEscape(f), nil, &got)
		if !got.CursorIgnored || len(got.Events) != 2 || got.Events[0]["line"] != `{"n":1}` {
			t.Errorf("forged cursor %q: %+v, want it ignored and the read from from=0", f, got)
		}
	}

	refused := exec.Command(binary, "serve", "--addr", "0.0.0.0:0", "--state", filepath.Join(dir, "other"))
	if err := refused.Run(); refused.ProcessState.ExitCode() != server.ExitRefused {
		t.Errorf("--addr 0.0.0.0:0: %v, want exit code %d", err, server.ExitRefused)
	}
}

// The token file is the current user's only (Windows: icacls lists one entry).
func TestTokenFileIsTheUsersOnly(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the DACL is Windows'; elsewhere the mode is 0600")
	}
	dir := t.TempDir()
	p := start(t, filepath.Join(dir, "state"), nil)
	out, err := exec.Command(filepath.Join(os.Getenv("SystemRoot"), "System32", "icacls.exe"), filepath.Join(p.state, server.TokenFile)).CombinedOutput()
	if err != nil {
		t.Fatalf("icacls: %v\n%s", err, out)
	}
	aces := 0
	sc := bufio.NewScanner(bytes.NewReader(out))
	for sc.Scan() {
		if strings.Contains(sc.Text(), ":(") {
			aces++
		}
	}
	if aces != 1 || strings.Contains(string(out), "(I)") {
		t.Errorf("token DACL, want one explicit entry:\n%s", out)
	}
}

// Oracle 8: bounds.
func TestBounds(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	p := start(t, filepath.Join(dir, "state"), nil, "--max-conns", "3", "--queue", "4", "--poll", "20ms",
		"--journal", "j="+path, "--target", "s="+filepath.Join(dir, "session.jsonl"))

	// A message over MaxClientMessage closes the connection with 1009.
	big, _, err := p.dial("", nil)
	if err != nil {
		t.Fatal(err)
	}
	read(t, big, 5*time.Second)
	big.Write(context.Background(), websocket.MessageText, bytes.Repeat([]byte("x"), server.MaxClientMessage+1))
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	_, _, err = big.Read(ctx)
	cancel()
	if websocket.CloseStatus(err) != websocket.StatusMessageTooBig {
		t.Errorf("a message of %d bytes: %v, want close 1009", server.MaxClientMessage+1, err)
	}
	big.CloseNow()

	// A slow client (small receive buffer, not reading) and a fast one.
	slowClient := &http.Client{Transport: &http.Transport{DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
		c, err := (&net.Dialer{}).DialContext(ctx, network, addr)
		if err == nil {
			c.(*net.TCPConn).SetReadBuffer(4096)
		}
		return c, err
	}}}
	var slow, fast *websocket.Conn
	deadline := time.Now().Add(5 * time.Second)
	for slow == nil || fast == nil { // wait for the closed connection to be counted out
		if time.Now().After(deadline) {
			t.Fatal("could not open the slow and fast connections")
		}
		if slow == nil {
			slow, _, _ = p.dial("", slowClient)
		}
		if fast == nil {
			fast, _, _ = p.dial("", nil)
		}
		time.Sleep(20 * time.Millisecond)
	}
	defer slow.CloseNow()
	defer fast.CloseNow()
	read(t, fast, 5*time.Second)

	// One more connection than --max-conns 3 (slow, fast, and this one is the third: allowed;
	// the fourth is refused).
	third, _, err := p.dial("", nil)
	if err != nil {
		t.Fatalf("third connection: %v", err)
	}
	defer third.CloseNow()
	if c, resp, err := p.dial("", nil); err == nil || resp == nil || resp.StatusCode != http.StatusServiceUnavailable {
		if c != nil {
			c.CloseNow()
		}
		t.Errorf("a fourth connection over --max-conns 3: %v (%v), want 503", err, resp)
	}
	third.CloseNow()

	// 8 MiB of records appear in the followed journal.
	fh, _ := os.Create(path)
	rec := `{"pad":"` + strings.Repeat("y", 4000) + `"}` + "\n"
	for range 2048 {
		fh.WriteString(rec)
	}
	fh.Close()
	size := int64(2048 * len(rec))

	var fastErr error
	var fastGap bool
	done := make(chan struct{})
	go func() { // the fast client reads every page, to the end of the journal
		defer close(done)
		for {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			_, b, err := fast.Read(ctx)
			cancel()
			if err != nil {
				fastErr = err
				return
			}
			var m struct {
				Type string `json:"type"`
				Data struct {
					Next int64 `json:"next"`
				} `json:"data"`
			}
			json.Unmarshal(b, &m)
			if m.Type == "gap" {
				fastGap = true
				return
			}
			if m.Type == "journal" && m.Data.Next == size {
				return
			}
		}
	}()
	// The engine is not blocked by the slow client.
	time.Sleep(300 * time.Millisecond)
	begun := time.Now()
	var res result
	if code := p.do("POST", "/v1/mandates", mandate("m1", incarnation(p, "s")), &res); code != 200 || time.Since(begun) > 2*time.Second {
		t.Errorf("a delivery beside the slow client: %d %+v in %s", code, res, time.Since(begun))
	}
	<-done
	if fastErr != nil || fastGap {
		t.Errorf("the fast client: error %v, gap %v; want every page", fastErr, fastGap)
	}
	// The slow client now reads: its backlog, then a gap of reason slow, then the close.
	var gap *server.Gap
	for gap == nil {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		_, b, err := slow.Read(ctx)
		cancel()
		if err != nil {
			t.Fatalf("slow client: %v before any gap", err)
		}
		var m msg
		json.Unmarshal(b, &m)
		if m.Type == "gap" {
			gap = &server.Gap{}
			json.Unmarshal(m.Data, gap)
		}
	}
	ctx, cancel = context.WithTimeout(context.Background(), 10*time.Second)
	_, _, err = slow.Read(ctx)
	cancel()
	if gap.Reason != "slow" || websocket.CloseStatus(err) != websocket.StatusPolicyViolation {
		t.Errorf("slow client: gap %+v then %v, want reason slow then close 1008", gap, err)
	}
}
