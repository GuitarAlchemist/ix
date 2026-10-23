package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func TestSecurityRefusals(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil, fakeSvc{name: "pod"}), "")
	port := in.addr[strings.LastIndexByte(in.addr, ':')+1:]
	cases := []struct {
		what   string
		path   string
		hdr    map[string]string
		status int
	}{
		{"health needs no token", "/health", map[string]string{"Authorization": ""}, 200},
		{"no token", "/v1/status", map[string]string{"Authorization": ""}, 401},
		{"wrong token", "/v1/status", map[string]string{"Authorization": "Bearer " + strings.Repeat("0", 64)}, 401},
		{"not bearer", "/v1/status", map[string]string{"Authorization": "Basic " + in.token}, 401},
		{"token in the URL", "/v1/status?token=" + in.token, map[string]string{"Authorization": ""}, 401},
		{"good token", "/v1/status", nil, 200},
		{"localhost host", "/v1/status", map[string]string{"Host": "localhost:" + port}, 200},
		{"foreign host", "/v1/status", map[string]string{"Host": "evil.example:" + port}, 403},
		{"foreign host on health", "/health", map[string]string{"Host": "evil.example"}, 403},
		{"other port", "/v1/status", map[string]string{"Host": "127.0.0.1:1"}, 403},
		{"foreign origin", "/v1/status", map[string]string{"Origin": "http://evil.example"}, 403},
		{"null origin", "/v1/status", map[string]string{"Origin": "null"}, 403},
		{"https origin", "/v1/status", map[string]string{"Origin": "https://127.0.0.1:" + port}, 403},
		{"own origin", "/v1/status", map[string]string{"Origin": "http://127.0.0.1:" + port}, 200},
		{"localhost origin", "/v1/status", map[string]string{"Origin": "http://localhost:" + port}, 200},
	}
	for _, c := range cases {
		code, hdr, body := httpDo(t, in, "GET", c.path, nil, c.hdr)
		if code != c.status {
			t.Errorf("%s: %d %s", c.what, code, body)
		}
		for k := range hdr {
			if strings.HasPrefix(strings.ToLower(k), "access-control-") {
				t.Errorf("%s: CORS header %s sent", c.what, k)
			}
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	wsURL := "ws://" + in.addr + "/v1/ws"
	dialWith := func(h http.Header, protos ...string) (*websocket.Conn, *http.Response, error) {
		return websocket.Dial(ctx, wsURL, &websocket.DialOptions{HTTPClient: loopbackHTTP, HTTPHeader: h, Subprotocols: protos})
	}
	if _, resp, err := dialWith(http.Header{"Authorization": {"Bearer nope"}}, subprotocol); err == nil || resp == nil || resp.StatusCode != 401 {
		t.Errorf("WS with a wrong token: %v", err)
	}
	if _, resp, err := dialWith(http.Header{"Authorization": {"Bearer " + in.token}, "Origin": {"http://evil.example"}}, subprotocol); err == nil || resp == nil || resp.StatusCode != 403 {
		t.Errorf("WS from a foreign origin: %v", err)
	}
	conn, _, err := dialWith(http.Header{"Authorization": {"Bearer " + in.token}})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := conn.Read(ctx); websocket.CloseStatus(err) != websocket.StatusPolicyViolation {
		t.Errorf("WS without ixctl.v1: %v", err)
	}
	conn.CloseNow()
}

func TestLoopbackOnly(t *testing.T) {
	for addr, ok := range map[string]bool{
		"127.0.0.1:0": true, "127.0.0.2:0": true, "[::1]:0": true,
		"0.0.0.0:0": false, "[::]:0": false, "192.168.1.10:0": false, "localhost:0": false, "example.com:80": false,
	} {
		if err := checkLoopback(addr); (err == nil) != ok {
			t.Errorf("%s: %v", addr, err)
		}
	}
	spec := filepath.Join(t.TempDir(), "spec.json")
	os.WriteFile(spec, []byte(`{"name":"x","services":[]}`), 0o600)
	var out, errb bytes.Buffer
	if code := cli([]string{"up", spec, "--addr", "0.0.0.0:0", "--state", t.TempDir()}, &out, &errb); code == 0 ||
		!strings.Contains(errb.String(), "not a loopback") {
		t.Fatalf("up on 0.0.0.0: code %d, %s", code, errb.String())
	}
}

// SSE replays from since, then Last-Event-ID resumes after the last one seen.
func TestSSEReplayWithLastEventID(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	base := in.plane.hub.Head()
	publishN(in.plane, "svc/x", 5)
	in.plane.Publish("ctl", event("noise", map[string]any{}))

	var gaps []Gap
	read := func(hdr map[string]string, query string, n int, during func()) []uint64 {
		t.Helper()
		gaps = nil
		req, _ := http.NewRequest("GET", "http://"+in.addr+"/v1/events?ch=svc/"+query, nil)
		req.Header.Set("Authorization", "Bearer "+in.token)
		for k, v := range hdr {
			req.Header.Set(k, v)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		resp, err := loopbackHTTP.Do(req.WithContext(ctx))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
			t.Fatalf("content type %q", ct)
		}
		if during != nil {
			go during()
		}
		var ids []uint64
		sc := bufio.NewScanner(resp.Body)
		var id uint64
		var kind string
		for len(ids) < n && sc.Scan() {
			line := sc.Text()
			switch {
			case strings.HasPrefix(line, "id: "):
				// id is epoch:seq, so a Last-Event-ID sent back names the boot too.
				epoch, seq, ok := strings.Cut(line[4:], ":")
				if !ok || epoch != in.plane.Epoch() {
					t.Fatalf("id %q is not <epoch>:<seq> of this boot", line[4:])
				}
				id, _ = strconv.ParseUint(seq, 10, 64)
			case strings.HasPrefix(line, "event: "):
				kind = line[7:]
			case strings.HasPrefix(line, "data: ") && kind == "gap":
				var gap Gap
				json.Unmarshal([]byte(line[6:]), &gap)
				gaps = append(gaps, gap)
			case strings.HasPrefix(line, "data: "):
				var env Envelope
				if err := json.Unmarshal([]byte(line[6:]), &env); err != nil || env.Seq != id || env.Ev.Kind != kind {
					t.Fatalf("frame id %d event %s data %s", id, kind, line)
				}
				ids = append(ids, id)
			}
		}
		return ids
	}
	got := read(nil, "&since="+strconv.FormatUint(base, 10), 5, nil)
	if len(got) != 5 || got[0] != base+1 || got[4] != base+5 || len(gaps) != 0 {
		t.Fatalf("replay from since: %v gaps %+v", got, gaps)
	}
	lastID := in.plane.Epoch() + ":" + strconv.FormatUint(base+3, 10)
	got = read(map[string]string{"Last-Event-ID": lastID}, "&since=0", 3, func() {
		time.Sleep(50 * time.Millisecond)
		publishN(in.plane, "svc/x", 1)
	})
	if len(got) != 3 || got[0] != base+4 || got[1] != base+5 || got[2] != base+7 || len(gaps) != 0 {
		t.Fatalf("Last-Event-ID resume then live: %v gaps %+v", got, gaps)
	}
	// F1.1: a Last-Event-ID from before a restart of the plane is told so, by a gap that
	// names the new epoch, and then gets everything this boot still holds.
	got = read(map[string]string{"Last-Event-ID": "0ddb00700000:" + strconv.FormatUint(base+3, 10)}, "", 6, nil)
	if len(gaps) != 1 || gaps[0].Reason != "epoch-changed" || gaps[0].From != 1 || gaps[0].Epoch != in.plane.Epoch() {
		t.Fatalf("stale epoch: gaps %+v", gaps)
	}
	if len(got) != 6 || got[0] != base+1 || got[5] != base+7 {
		t.Fatalf("stale epoch replays this boot: %v", got)
	}
}

// A resume point the ring no longer holds gets a since-too-old gap naming the oldest kept, and
// the stream goes on from there: only a reader that falls behind while reading (overflow) is
// cut off. The README's /v1/events row says both.
func TestSSEGoesOnAfterSinceTooOld(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	publishN(in.plane, "svc/x", ringSize+10)
	_, oldest := in.plane.hub.Span()
	req, _ := http.NewRequest("GET", "http://"+in.addr+"/v1/events?ch=svc/&since=1&epoch="+in.plane.Epoch(), nil)
	req.Header.Set("Authorization", "Bearer "+in.token)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := loopbackHTTP.Do(req.WithContext(ctx))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var gap *Gap
	var kind string
	var seqs []uint64
	sc := bufio.NewScanner(resp.Body)
	for len(seqs) < 3 && sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			kind = line[7:]
		case strings.HasPrefix(line, "data: ") && kind == "gap":
			gap = &Gap{}
			json.Unmarshal([]byte(line[6:]), gap)
		case strings.HasPrefix(line, "data: "):
			if gap == nil {
				t.Fatalf("an envelope before the gap: %s", line)
			}
			var env Envelope
			json.Unmarshal([]byte(line[6:]), &env)
			seqs = append(seqs, env.Seq)
		}
	}
	if gap == nil || gap.Reason != "since-too-old" || gap.From != 2 || gap.Oldest != oldest {
		t.Fatalf("gap %+v, want since-too-old from 2 with the oldest kept %d", gap, oldest)
	}
	if len(seqs) != 3 || seqs[0] != oldest || seqs[2] != oldest+2 {
		t.Fatalf("after the gap: %v, want the stream to go on from the oldest kept %d", seqs, oldest)
	}
}

// /health counts the JSON ops served, whatever their outcome, and not itself: a
// test outside the process can see a client poll the plane.
func TestHealthCountsOps(t *testing.T) {
	in := newTestInstance(t, fakeSpec(t, "local", nil), "")
	ops := func() int64 {
		_, _, body := httpDo(t, in, "GET", "/health", nil, map[string]string{"Authorization": ""})
		var h struct {
			Ops int64 `json:"ops"`
		}
		if err := json.Unmarshal(body, &h); err != nil {
			t.Fatalf("health %s: %v", body, err)
		}
		return h.Ops
	}
	n0 := ops()
	httpDo(t, in, "GET", "/v1/status", nil, nil)
	httpDo(t, in, "GET", "/v1/status?peer=", nil, nil) // refused, counted all the same
	httpDo(t, in, "GET", "/health", nil, map[string]string{"Authorization": ""})
	if n := ops(); n != n0+2 {
		t.Fatalf("ops %d after two, want %d", n, n0+2)
	}
}

// ixctl up as its own process: ready line, state file with a token, health,
// and a clean shutdown that removes the state file.
func TestUpReadyAndShutdown(t *testing.T) {
	dir := t.TempDir()
	spec := filepath.Join(dir, "spec.json")
	os.WriteFile(spec, []byte(`{"name":"solo","services":[]}`), 0o600)
	state := filepath.Join(dir, "state")
	cmd := exec.Command(testExe(t), "up", spec, "--state", state, "--name", "solo-1")
	cmd.Env = append(os.Environ(), "IXCTL_TEST_ROLE=main")
	out, _ := cmd.StdoutPipe()
	t0 := time.Now()
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer cmd.Process.Kill()
	line, err := bufio.NewReader(out).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	wall := time.Since(t0)
	var r readyLine
	if err := json.Unmarshal([]byte(line), &r); err != nil || r.Event != "ready" || r.Name != "solo-1" || r.Pid != cmd.Process.Pid {
		t.Fatalf("ready line %q", line)
	}
	if checkLoopback(r.Addr) != nil || r.MsToReady <= 0 {
		t.Fatalf("ready line %+v", r)
	}
	t.Logf("ms_to_ready %.2f (spawn to ready line, seen by the parent: %s)", r.MsToReady, wall)
	st, err := readStateFile(r.StateFile)
	if err != nil || st.Name != "solo-1" || st.Addr != r.Addr || st.Pid != r.Pid || len(st.Token) != 64 {
		t.Fatalf("state file %+v %v", st, err)
	}
	in := &instance{addr: st.Addr, token: st.Token}
	if code, _, body := httpDo(t, in, "GET", "/health", nil, map[string]string{"Authorization": ""}); code != 200 ||
		!strings.Contains(string(body), `"ok":true`) || !strings.Contains(string(body), `"uptime_ms"`) {
		t.Fatalf("health %d %s", code, body)
	}
	if code, _, _ := httpDo(t, in, "POST", "/v1/shutdown", nil, nil); code != 200 {
		t.Fatalf("shutdown %d", code)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("exit: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("did not exit")
	}
	if _, err := os.Stat(r.StateFile); !os.IsNotExist(err) {
		t.Fatalf("state file left: %v", err)
	}
	j, _ := os.ReadFile(filepath.Join(state, "solo-1.journal.jsonl"))
	if strings.Contains(line, st.Token) || strings.Contains(string(j), st.Token) {
		t.Fatal("the token was printed or journaled")
	}
}
