package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// S3B-2 (Codex round 6, P1): session.jsonl is replaced between the status and the delivery. The
// mandate is refused stale-incarnation and nothing is written to the new session.
func TestFakeSessionReplacedIsStale(t *testing.T) {
	dir := t.TempDir()
	session := filepath.Join(dir, "session.jsonl")
	p := start(t, filepath.Join(dir, "state"), nil, "--target", "s="+session)
	inc := incarnation(p, "s")
	const other = `{"mandate_id":"x","incarnation":"another","text":"another session"}` + "\n"
	if err := os.WriteFile(session+".new", []byte(other), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(session+".new", session); err != nil {
		t.Fatal(err)
	}
	var res result
	if code := p.do("POST", "/v1/mandates", mandate("m1", inc), &res); code != http.StatusConflict || res.Error != "stale-incarnation" {
		t.Errorf("deliver to the replaced session: %d %+v, want 409 stale-incarnation", code, res)
	}
	if b, _ := os.ReadFile(session); string(b) != other {
		t.Errorf("the replacement session was written to:\n%s", b)
	}
	if now := incarnation(p, "s"); now == inc {
		t.Errorf("status still gives incarnation %s after the replacement", now)
	}
}

// S3B-3 (Codex round 6, P1): a client that drip-feeds its body is cut at the read deadline, and
// nothing is delivered.
func TestHTTPBodyDripIsCut(t *testing.T) {
	dir := t.TempDir()
	session := filepath.Join(dir, "session.jsonl")
	p := start(t, filepath.Join(dir, "state"), nil, "--target", "s="+session, "--read-timeout", "1s")
	body, _ := json.Marshal(mandate("drip", incarnation(p, "s")))
	conn, err := net.Dial("tcp", p.addr)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	fmt.Fprintf(conn, "POST /v1/mandates HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n", p.addr, p.token, len(body))
	begun := time.Now()
	cut := make(chan time.Duration, 1)
	go func() { io.Copy(io.Discard, conn); cut <- time.Since(begun) }() // until the server closes
	go func() {
		for i := range body { // one byte every 100 ms
			if _, err := conn.Write(body[i : i+1]); err != nil {
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()
	drip := time.Duration(len(body)) * 100 * time.Millisecond
	select {
	case d := <-cut:
		if d > 3*time.Second {
			t.Errorf("the dripping client was cut after %s, want about the 1s read deadline", d)
		}
	case <-time.After(drip + 3*time.Second):
		t.Errorf("the dripping client was never cut (its body takes %s)", drip)
	}
	if n := lines(t, session); n != 0 {
		t.Errorf("%d deliveries from a cut request, want 0", n)
	}
}

// S3B-3 (Codex round 6, P1): one TCP connection over --max-http-conns is answered 503 at once,
// and a slot is free again once a connection closes.
func TestHTTPConnectionCap(t *testing.T) {
	dir := t.TempDir()
	p := start(t, filepath.Join(dir, "state"), nil, "--max-http-conns", "2")
	var idle []net.Conn
	for range 2 { // two idle connections hold both slots
		c, err := net.Dial("tcp", p.addr)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		idle = append(idle, c)
	}
	third, err := net.Dial("tcp", p.addr)
	if err != nil {
		t.Fatal(err)
	}
	defer third.Close()
	fmt.Fprintf(third, "GET /v1/status HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\n\r\n", p.addr, p.token)
	third.SetReadDeadline(time.Now().Add(5 * time.Second))
	status, err := bufio.NewReader(third).ReadString('\n')
	if !strings.Contains(status, " 503 ") {
		t.Errorf("a third connection over --max-http-conns 2: %q %v, want 503", status, err)
	}
	idle[0].Close()
	deadline := time.Now().Add(5 * time.Second)
	for {
		req, _ := http.NewRequest("GET", "http://"+p.addr+"/v1/status", nil)
		req.Header.Set("Authorization", "Bearer "+p.token)
		resp, err := (&http.Client{Transport: &http.Transport{DisableKeepAlives: true}}).Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("no slot freed after a connection closed: %v %v", resp, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// S3B-4 (Codex round 6, P1): with 5,000 deliveries, a resume on a stale epoch gets the whole
// state through a client that follows the 1 MiB read limit, without 1009; the events after it
// start at the seq the state covers plus one.
func TestStateOfThousandsFitsTheReadLimit(t *testing.T) {
	dir := t.TempDir()
	state := filepath.Join(dir, "state")
	os.MkdirAll(state, 0o700)
	var log strings.Builder
	log.WriteString(`{"format":"ixmux.cmdlog.v1"}` + "\n")
	reason := strings.Repeat("r", 300)
	for i := range 5000 {
		fmt.Fprintf(&log, `{"at":"2026-11-06T00:00:00Z","target":"s","mandate_id":"old-%04d","incarnation":"i","state":"delivered","reason":%q,"fp":"x"}`+"\n", i, reason)
	}
	if err := os.WriteFile(filepath.Join(state, "commands.jsonl"), []byte(log.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	p := start(t, state, nil, "--target", "s="+filepath.Join(dir, "session.jsonl"))
	c, _, err := p.dial("?since=1&epoch=stale", nil) // read limit: server.MaxStreamMessage
	if err != nil {
		t.Fatal(err)
	}
	defer c.CloseNow()
	seen := map[string]bool{}
	var covered uint64
	for done := false; !done; {
		m := read(t, c, 10*time.Second)
		switch m.Type {
		case "state", "state-page":
			var d struct {
				Deliveries []transition `json:"deliveries"`
			}
			json.Unmarshal(m.Data, &d)
			for _, tr := range d.Deliveries {
				seen[tr.MandateID] = true
			}
			done, covered = m.Type == "state", m.Seq
		case "state-end":
			done, covered = true, m.Seq
		}
	}
	if len(seen) != 5000 {
		t.Errorf("the state holds %d deliveries, want 5000", len(seen))
	}
	p.do("POST", "/v1/mandates", mandate("after", incarnation(p, "s")), nil)
	want := covered + 1
	for _, m := range until(t, c, "after", "delivered") {
		var tr transition
		json.Unmarshal(m.Data, &tr)
		if m.Type != "mandate" || m.Seq != want || seen[tr.MandateID] {
			t.Fatalf("after the state covering %d: %s seq %d %+v, want mandate seq %d not in the state", covered, m.Type, m.Seq, tr, want)
		}
		want++
	}
}
