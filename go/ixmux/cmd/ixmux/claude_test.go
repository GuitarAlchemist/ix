package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The claude oracles drive the fake claude (claudestream/internal/fakeclaude): no model call.

type claudeTarget struct {
	Name        string `json:"name"`
	Incarnation string `json:"incarnation"`
	PID         int    `json:"pid"`
	Kind        string `json:"kind"`
}

func claudeOf(p *proc) claudeTarget {
	var st struct {
		Targets []claudeTarget `json:"targets"`
	}
	p.do("GET", "/v1/status", nil, &st)
	for _, tg := range st.Targets {
		if tg.Name == "c" {
			return tg
		}
	}
	p.t.Fatalf("no claude target c in status")
	return claudeTarget{}
}

func startClaude(t *testing.T, state, effects string, env ...string) *proc {
	t.Helper()
	return start(t, state, append([]string{"FAKE_EFFECTS=" + effects}, env...), "--claude-target", "c", "--claude-bin", fakeClaude)
}

func claudeMandate(id, inc, text string) map[string]string {
	return map[string]string{"mandate_id": id, "target": "c", "incarnation": inc, "text": text}
}

type claudeResult struct {
	State    string `json:"state"`
	Replayed bool   `json:"replayed"`
	Error    string `json:"error"`
	Ack      *struct {
		Text    string `json:"text"`
		AckSeen bool   `json:"ack_seen"`
	} `json:"ack"`
}

func results(t *testing.T, state string) int {
	t.Helper()
	b, _ := os.ReadFile(filepath.Join(state, "claude-c", "stream.jsonl"))
	return bytes.Count(b, []byte(`"type":"result"`))
}

// Oracle S4-1: Hello through a claude target; the stream journal is served as a journal.
func TestClaudeHello(t *testing.T) {
	dir := t.TempDir()
	state, eff := filepath.Join(dir, "state"), filepath.Join(dir, "effects")
	p := startClaude(t, state, eff)
	c := claudeOf(p)
	if c.Kind != "claude" || len(c.Incarnation) != 36 || c.PID == 0 {
		t.Fatalf("claude target in status: %+v", c)
	}
	var res claudeResult
	code := p.do("POST", "/v1/mandates", claudeMandate("hello-1", c.Incarnation, "Réponds exactement : ACK hello-1, puis le mot BONJOUR."), &res)
	if code != 200 || res.State != "delivered" || res.Ack == nil || !res.Ack.AckSeen || !strings.Contains(res.Ack.Text, "BONJOUR") {
		t.Fatalf("hello-1: %d %+v", code, res)
	}
	var page struct {
		Events []map[string]any `json:"events"`
	}
	if code := p.do("GET", "/v1/journal/claude-c?from=0", nil, &page); code != 200 || len(page.Events) == 0 {
		t.Errorf("the stream journal over HTTP: %d, %d events", code, len(page.Events))
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}

// Oracle S4-5: the HTTP answer is lost, the mandate is sent again: replayed, still one turn.
func TestClaudeLostHTTPAckReplays(t *testing.T) {
	dir := t.TempDir()
	state, eff := filepath.Join(dir, "state"), filepath.Join(dir, "effects")
	p := startClaude(t, state, eff)
	inc := claudeOf(p).Incarnation
	b, _ := json.Marshal(claudeMandate("m1", inc, "Dis bonjour."))
	conn, err := net.Dial("tcp", p.addr)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(conn, "POST /v1/mandates HTTP/1.1\r\nHost: %s\r\nAuthorization: Bearer %s\r\nContent-Length: %d\r\n\r\n%s", p.addr, p.token, len(b), b)
	conn.Close()
	deadline := time.Now().Add(10 * time.Second)
	for e := (result{}); e.State != "delivered"; p.do("GET", "/v1/mandates/m1", nil, &e) {
		if time.Now().After(deadline) {
			t.Fatalf("m1 never delivered: %+v", e)
		}
		time.Sleep(20 * time.Millisecond)
	}
	before := results(t, state)
	var res claudeResult
	if code := p.do("POST", "/v1/mandates", claudeMandate("m1", inc, "Dis bonjour."), &res); code != 200 || !res.Replayed {
		t.Errorf("resend: %d %+v, want replayed", code, res)
	}
	if n, r := lines(t, eff), results(t, state); n != 1 || r != before || r != 1 {
		t.Errorf("%d turns and %d results (before the resend %d), want 1 and 1", n, r, before)
	}
}

// Oracle S4-9: a mandate text over the bound is refused, nothing is sent.
func TestClaudeTextTooLong(t *testing.T) {
	dir := t.TempDir()
	state, eff := filepath.Join(dir, "state"), filepath.Join(dir, "effects")
	p := startClaude(t, state, eff)
	var res claudeResult
	code := p.do("POST", "/v1/mandates", claudeMandate("big", claudeOf(p).Incarnation, strings.Repeat("x", 64<<10+1)), &res)
	if code != http.StatusRequestEntityTooLarge || res.Error != "text-too-long" {
		t.Errorf("a text of 64 KiB + 1: %d %+v, want 413 text-too-long", code, res)
	}
	if n := lines(t, eff); n != 0 {
		t.Errorf("%d turns, want 0", n)
	}
}

// Oracle S4-8: a clean stop ends the child and exits 0.
func TestClaudeShutdown(t *testing.T) {
	dir := t.TempDir()
	p := startClaude(t, filepath.Join(dir, "state"), filepath.Join(dir, "effects"))
	pid := claudeOf(p).PID
	p.do("POST", "/v1/shutdown", nil, nil)
	select {
	case <-p.exited:
	case <-time.After(30 * time.Second):
		t.Fatal("ixmux did not exit")
	}
	if code := p.cmd.ProcessState.ExitCode(); code != 0 {
		t.Errorf("exit code %d\n%s", code, p.out)
	}
	if alive(pid) {
		t.Errorf("claude child %d alive after the stop", pid)
	}
}
