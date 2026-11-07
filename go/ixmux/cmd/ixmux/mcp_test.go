package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The S6 oracles drive `ixmux mcp` as a subprocess over its pipes, against a real `ixmux serve`
// with a fake claude target. No model is called.

type mcpProc struct {
	t       *testing.T
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	lines   chan []byte
	out     *syncBuffer // everything it wrote, stdout and stderr: no secret may be in it
	exited  chan struct{}
	nextID  int
	secrets []string
}

// startMCP runs `ixmux mcp --state state`. Each secret is looked for in all its output when the
// test ends.
func startMCP(t *testing.T, state string, secrets ...string) *mcpProc {
	t.Helper()
	m := &mcpProc{t: t, lines: make(chan []byte, 64), out: &syncBuffer{}, exited: make(chan struct{}), secrets: secrets}
	m.cmd = exec.Command(binary, "mcp", "--state", state)
	m.cmd.Stderr = m.out
	stdout, err := m.cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if m.stdin, err = m.cmd.StdinPipe(); err != nil {
		t.Fatal(err)
	}
	if err := m.cmd.Start(); err != nil {
		t.Fatal(err)
	}
	go func() {
		br := bufio.NewReaderSize(stdout, 1<<20)
		for {
			line, err := br.ReadBytes('\n')
			if len(line) > 0 {
				m.out.Write(line)
				m.lines <- line
			}
			if err != nil {
				close(m.lines)
				return
			}
		}
	}()
	go func() { m.cmd.Wait(); close(m.exited) }()
	t.Cleanup(func() {
		m.cmd.Process.Kill()
		<-m.exited
		m.checkSecrets()
	})
	return m
}

func (m *mcpProc) checkSecrets() {
	out := m.out.String()
	for _, s := range m.secrets {
		if s != "" && strings.Contains(out, s) {
			m.t.Errorf("the adapter's output holds a secret (%d bytes)", len(s))
		}
	}
}

type rpcResponse struct {
	ID     json.RawMessage `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  *struct {
		Code    int            `json:"code"`
		Message string         `json:"message"`
		Data    map[string]any `json:"data"`
	} `json:"error"`
}

// send writes one raw line.
func (m *mcpProc) send(line string) {
	m.t.Helper()
	if _, err := io.WriteString(m.stdin, line+"\n"); err != nil {
		m.t.Fatalf("write to the adapter: %v", err)
	}
}

// next reads the next response line.
func (m *mcpProc) next() rpcResponse {
	m.t.Helper()
	select {
	case line, ok := <-m.lines:
		if !ok {
			m.t.Fatalf("the adapter closed its stdout\n%s", m.out)
		}
		var r rpcResponse
		if err := json.Unmarshal(line, &r); err != nil {
			m.t.Fatalf("not JSON-RPC: %q: %v", line, err)
		}
		return r
	case <-time.After(20 * time.Second): // a guard against a hung adapter, not a timing bound
		m.t.Fatalf("no answer from the adapter\n%s", m.out)
	}
	return rpcResponse{}
}

// call sends a request and returns its response.
func (m *mcpProc) call(method string, params any) rpcResponse {
	m.t.Helper()
	m.nextID++
	b, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": m.nextID, "method": method, "params": params})
	m.send(string(b))
	r := m.next()
	if string(r.ID) != fmt.Sprint(m.nextID) {
		m.t.Fatalf("%s: answer for id %s, want %d", method, r.ID, m.nextID)
	}
	return r
}

type toolResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Structured map[string]any `json:"structuredContent"`
	IsError    bool           `json:"isError"`
}

// tool calls a tool; it fails the test on a JSON-RPC error.
func (m *mcpProc) tool(name string, args map[string]any) toolResult {
	m.t.Helper()
	r := m.call("tools/call", map[string]any{"name": name, "arguments": args})
	if r.Error != nil {
		m.t.Fatalf("%s: JSON-RPC error %+v", name, r.Error)
	}
	var tr toolResult
	if err := json.Unmarshal(r.Result, &tr); err != nil {
		m.t.Fatalf("%s: %s: %v", name, r.Result, err)
	}
	if len(tr.Content) != 1 || tr.Content[0].Type != "text" {
		m.t.Fatalf("%s: content %+v, want one text block", name, tr.Content)
	}
	var fromText map[string]any // the text block holds the same JSON as structuredContent
	if json.Unmarshal([]byte(tr.Content[0].Text), &fromText) != nil || fmt.Sprint(fromText) != fmt.Sprint(tr.Structured) {
		m.t.Fatalf("%s: text %q and structuredContent %v differ", name, tr.Content[0].Text, tr.Structured)
	}
	return tr
}

func (m *mcpProc) initialize() map[string]any {
	m.t.Helper()
	r := m.call("initialize", map[string]any{"protocolVersion": "2025-06-18", "capabilities": map[string]any{},
		"clientInfo": map[string]any{"name": "test", "version": "0"}})
	if r.Error != nil {
		m.t.Fatalf("initialize: %+v", r.Error)
	}
	var v map[string]any
	json.Unmarshal(r.Result, &v)
	m.send(`{"jsonrpc":"2.0","method":"notifications/initialized"}`) // a notification: no answer
	return v
}

// resultOf polls ixmux_result until the mandate reaches state; the deadline is a guard.
func (m *mcpProc) resultOf(id, state string) map[string]any {
	m.t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for {
		r := m.tool("ixmux_result", map[string]any{"mandate_id": id})
		if !r.IsError && r.Structured["state"] == state {
			return r.Structured
		}
		if time.Now().After(deadline) {
			m.t.Fatalf("%s never %s: %v", id, state, r.Structured)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func submit(id, inc, text string) map[string]any {
	a := map[string]any{"target": "c", "mandate_id": id, "text": text}
	if inc != "" {
		a["incarnation"] = inc
	}
	return a
}

// Oracles S6-1 to S6-4: submit then result, correlated by mandate id; the same submission again
// makes no new turn; the same id with another text is refused; a stale incarnation is refused.
func TestMCPMandates(t *testing.T) {
	dir := t.TempDir()
	state, eff := filepath.Join(dir, "state"), filepath.Join(dir, "effects")
	p := startClaude(t, state, eff)
	m := startMCP(t, state, p.token)
	init := m.initialize()
	if init["protocolVersion"] != "2025-06-18" {
		t.Errorf("initialize: protocol %v, want the one asked, 2025-06-18", init["protocolVersion"])
	}
	var list struct {
		Tools []struct {
			Name        string         `json:"name"`
			InputSchema map[string]any `json:"inputSchema"`
		} `json:"tools"`
	}
	json.Unmarshal(m.call("tools/list", map[string]any{}).Result, &list)
	var names []string
	for _, tl := range list.Tools {
		names = append(names, tl.Name)
		if tl.InputSchema["type"] != "object" {
			t.Errorf("%s: input schema %v", tl.Name, tl.InputSchema)
		}
	}
	if fmt.Sprint(names) != "[ixmux_status ixmux_submit ixmux_result]" {
		t.Errorf("tools %v", names)
	}

	st := m.tool("ixmux_status", map[string]any{})
	targets, _ := st.Structured["targets"].([]any)
	if st.IsError || len(targets) != 1 {
		t.Fatalf("status: %v", st.Structured)
	}
	inc := targets[0].(map[string]any)["incarnation"].(string)

	// 1. submit, then result.
	rc := m.tool("ixmux_submit", submit("m1", "", "Réponds : ACK m1, puis BONJOUR."))
	if rc.IsError || rc.Structured["accepted"] != true || rc.Structured["mandate_id"] != "m1" || rc.Structured["incarnation"] != inc {
		t.Fatalf("submit m1: %v", rc.Structured)
	}
	res := m.resultOf("m1", "delivered")
	if res["mandate_id"] != "m1" || res["ack_seen"] != true || !strings.Contains(fmt.Sprint(res["text"]), "BONJOUR") {
		t.Errorf("result m1: %v", res)
	}

	// 2. the same submission again: replayed, no new turn.
	rc = m.tool("ixmux_submit", submit("m1", inc, "Réponds : ACK m1, puis BONJOUR."))
	if rc.IsError || rc.Structured["replayed"] != true || rc.Structured["state"] != "delivered" {
		t.Errorf("submit m1 again: %v, want replayed delivered", rc.Structured)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns after the replay, want 1", n)
	}

	// 3. the same id, another text: refused, typed.
	rc = m.tool("ixmux_submit", submit("m1", inc, "Un autre texte."))
	if !rc.IsError || rc.Structured["error"] != "mandate-id-reuse" {
		t.Errorf("m1 with another text: %v, want isError mandate-id-reuse", rc.Structured)
	}

	// 4. a stale incarnation: refused.
	rc = m.tool("ixmux_submit", submit("m4", "00000000-0000-0000-0000-000000000000", "Bonjour."))
	if !rc.IsError || rc.Structured["error"] != "stale-incarnation" {
		t.Errorf("a stale incarnation: %v, want isError stale-incarnation", rc.Structured)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}

// Oracle S6-5a: the receipt comes within 2 s while the turn is held; the adapter is killed during
// the turn and started again: the result is read, and the same submission makes no new turn.
func TestMCPAdapterKilledDuringATurn(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	p := startClaude(t, state, eff, "FAKE_HOLD="+hold)
	m := startMCP(t, state, p.token)
	m.initialize()
	inc := claudeOf(p).Incarnation

	began := time.Now()
	rc := m.tool("ixmux_submit", submit("s1", inc, "A slow one."))
	if took := time.Since(began); took > 2*time.Second {
		t.Errorf("the receipt took %s, want 2 s at most", took)
	}
	if rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("receipt of a held turn: %v, want accepted", rc.Structured)
	}
	if st := rc.Structured["state"]; st != "accepted" && st != "delivering" {
		t.Errorf("receipt state %v, want accepted or delivering", st)
	}

	m.cmd.Process.Kill()
	<-m.exited
	os.Remove(hold) // the turn ends with no adapter to hear it

	m2 := startMCP(t, state, p.token)
	m2.initialize()
	res := m2.resultOf("s1", "delivered")
	if res["ack_seen"] != true {
		t.Errorf("result after the adapter restart: %v", res)
	}
	rc = m2.tool("ixmux_submit", submit("s1", inc, "A slow one."))
	if rc.IsError || rc.Structured["replayed"] != true {
		t.Errorf("the same submission: %v, want replayed", rc.Structured)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1", n)
	}
}

// Oracle S6-5b and S6-6: ixmux killed during a turn is unavailable to the adapter, which stays up;
// ixmux started again finds the mandate uncertain, and a new submission does not replay it.
func TestMCPIxmuxKilledDuringATurn(t *testing.T) {
	dir := t.TempDir()
	state, eff, hold := filepath.Join(dir, "state"), filepath.Join(dir, "effects"), filepath.Join(dir, "hold")
	os.WriteFile(hold, nil, 0o600)
	defer os.Remove(hold)
	p := startClaude(t, state, eff, "FAKE_HOLD="+hold)
	m := startMCP(t, state, p.token)
	m.initialize()
	inc := claudeOf(p).Incarnation
	if rc := m.tool("ixmux_submit", submit("k1", inc, "A slow one.")); rc.IsError || rc.Structured["accepted"] != true {
		t.Fatalf("submit k1: %v", rc.Structured)
	}
	deadline := time.Now().Add(10 * time.Second)
	for lines(t, eff) == 0 { // the turn reached the child; the deadline is a guard
		if time.Now().After(deadline) {
			t.Fatalf("the held turn never reached the fake claude\n%s", p.out)
		}
		time.Sleep(20 * time.Millisecond)
	}
	p.cmd.Process.Kill()
	<-p.exited

	if st := m.tool("ixmux_status", map[string]any{}); !st.IsError || st.Structured["error"] != "ixmux-unavailable" {
		t.Errorf("status with ixmux killed: %v, want isError ixmux-unavailable", st.Structured)
	}
	os.Remove(hold)
	q := startClaude(t, state, eff)
	m.secrets = append(m.secrets, q.token)
	res := m.resultOf("k1", "uncertain")
	if res["mandate_id"] != "k1" {
		t.Errorf("result after the ixmux restart: %v", res)
	}
	rc := m.tool("ixmux_submit", submit("k1", inc, "A slow one."))
	if !rc.IsError || rc.Structured["error"] != "uncertain-needs-reconcile" {
		t.Errorf("a new submission of the uncertain k1: %v, want isError uncertain-needs-reconcile", rc.Structured)
	}
	if n := lines(t, eff); n != 1 {
		t.Errorf("%d turns, want 1: the uncertain mandate was replayed", n)
	}
}

// Oracle S6-6: no ixmux (an empty state directory, or one stopped): every tool is ixmux-unavailable
// and initialize works.
func TestMCPWithoutIxmux(t *testing.T) {
	dir := t.TempDir()
	empty := filepath.Join(dir, "empty")
	os.MkdirAll(empty, 0o700)
	m := startMCP(t, empty)
	if v := m.initialize(); v["serverInfo"] == nil {
		t.Errorf("initialize without ixmux: %v", v)
	}
	for _, c := range []struct {
		name string
		args map[string]any
	}{{"ixmux_status", map[string]any{}}, {"ixmux_submit", submit("m1", "", "x")}, {"ixmux_result", map[string]any{"mandate_id": "m1"}}} {
		if r := m.tool(c.name, c.args); !r.IsError || r.Structured["error"] != "ixmux-unavailable" {
			t.Errorf("%s on an empty state: %v, want isError ixmux-unavailable", c.name, r.Structured)
		}
	}

	state := filepath.Join(dir, "state")
	p := start(t, state, nil, "--target", "s="+filepath.Join(dir, "session.jsonl"))
	m2 := startMCP(t, state, p.token)
	m2.initialize()
	if r := m2.tool("ixmux_status", map[string]any{}); r.IsError {
		t.Fatalf("status with ixmux running: %v", r.Structured)
	}
	p.do("POST", "/v1/shutdown", nil, nil)
	<-p.exited
	if r := m2.tool("ixmux_status", map[string]any{}); !r.IsError || r.Structured["error"] != "ixmux-unavailable" {
		t.Errorf("status with ixmux stopped: %v, want isError ixmux-unavailable", r.Structured)
	}
}

// Oracle S6-7: each bound is a typed refusal, and the adapter goes on serving.
func TestMCPBounds(t *testing.T) {
	dir := t.TempDir()
	state, eff := filepath.Join(dir, "state"), filepath.Join(dir, "effects")
	p := startClaude(t, state, eff, "FAKE_REPLY=10000")
	m := startMCP(t, state, p.token)
	m.initialize()
	alive := func(after string) {
		t.Helper()
		if r := m.call("ping", nil); r.Error != nil || string(r.Result) != "{}" {
			t.Fatalf("ping after %s: %+v %s", after, r.Error, r.Result)
		}
	}

	m.send(`{"jsonrpc":"2.0","id":"big","method":"ping","params":{"pad":"` + strings.Repeat("x", 1<<20) + `"}}`)
	if r := m.next(); r.Error == nil || r.Error.Code != -32600 || r.Error.Data["reason"] != "line-too-long" {
		t.Errorf("a line over 1 MiB: %+v, want -32600 line-too-long", r.Error)
	}
	alive("a line too long")

	m.send(`not json`)
	if r := m.next(); r.Error == nil || r.Error.Code != -32700 {
		t.Errorf("a line that is not JSON: %+v, want -32700", r.Error)
	}
	if r := m.call("tools/nope", nil); r.Error == nil || r.Error.Code != -32601 {
		t.Errorf("an unknown method: %+v, want -32601", r.Error)
	}
	if r := m.call("tools/call", map[string]any{"name": "nope", "arguments": map[string]any{}}); r.Error == nil || r.Error.Code != -32602 {
		t.Errorf("an unknown tool: %+v, want -32602", r.Error)
	}
	alive("the protocol errors")

	inc := claudeOf(p).Incarnation
	for _, c := range []struct {
		why  string
		args map[string]any
		want string
	}{
		{"a text over MaxBody", submit("t1", inc, strings.Repeat("x", 128<<10+1)), "text-too-long"},
		{"a mandate id with a slash", submit("a/b", inc, "x"), "bad-mandate-id"},
		{"an empty mandate id", submit("", inc, "x"), "bad-mandate-id"},
		{"a mandate id of 129 bytes", submit(strings.Repeat("a", 129), inc, "x"), "bad-mandate-id"},
		{"no target", map[string]any{"mandate_id": "t2", "text": "x"}, "bad-arguments"},
		{"an unknown argument", map[string]any{"target": "c", "mandate_id": "t3", "text": "x", "extra": 1}, "bad-arguments"},
	} {
		if r := m.tool("ixmux_submit", c.args); !r.IsError || r.Structured["error"] != c.want {
			t.Errorf("%s: %v, want isError %s", c.why, r.Structured, c.want)
		}
	}
	if r := m.tool("ixmux_result", map[string]any{"mandate_id": "../v1/status"}); !r.IsError || r.Structured["error"] != "bad-mandate-id" {
		t.Errorf("a result id with a path: %v, want isError bad-mandate-id", r.Structured)
	}
	if r := m.tool("ixmux_result", map[string]any{"mandate_id": "never"}); !r.IsError || r.Structured["error"] != "unknown-mandate" {
		t.Errorf("an unknown mandate: %v, want isError unknown-mandate", r.Structured)
	}
	alive("the tool refusals")
	if n := lines(t, eff); n != 0 {
		t.Errorf("%d turns from refused submissions, want 0", n)
	}

	// A long answer is cut at the bound, and says so.
	if r := m.tool("ixmux_submit", submit("long", inc, "Long.")); r.IsError {
		t.Fatalf("submit long: %v", r.Structured)
	}
	res := m.resultOf("long", "delivered")
	if text := fmt.Sprint(res["text"]); len(text) != 4096 || res["truncated"] != true {
		t.Errorf("a 10000-byte answer: %d bytes of text, truncated %v; want 4096 and true", len(text), res["truncated"])
	}
}

// Oracle S6-8: the token is in no output of the adapter: initialize, every tool, every error,
// stdout and stderr (checked by startMCP's cleanup on every test, and here over all the paths).
func TestMCPNeverShowsTheToken(t *testing.T) {
	dir := t.TempDir()
	state, eff := filepath.Join(dir, "state"), filepath.Join(dir, "effects")
	p := startClaude(t, state, eff)
	m := startMCP(t, state, p.token)
	m.initialize()
	m.call("tools/list", nil)
	m.tool("ixmux_status", map[string]any{})
	inc := claudeOf(p).Incarnation
	m.tool("ixmux_submit", submit("x1", inc, "Bonjour."))
	m.resultOf("x1", "delivered")
	m.tool("ixmux_submit", submit("x1", inc, "Autre."))
	m.tool("ixmux_result", map[string]any{"mandate_id": "never"})
	m.call("nope", nil)
	m.stdin.Close()
	<-m.exited
	m.checkSecrets()
	if bytes.Contains([]byte(m.out.String()), []byte(p.token)) {
		t.Error("the token is in the adapter's output")
	}
}

// Oracle S6-9: stdin closed, the adapter exits 0 within 2 s; ixmux goes on serving.
func TestMCPStopsWhenStdinCloses(t *testing.T) {
	dir := t.TempDir()
	p := start(t, filepath.Join(dir, "state"), nil, "--target", "s="+filepath.Join(dir, "session.jsonl"))
	m := startMCP(t, p.state, p.token)
	m.initialize()
	m.tool("ixmux_status", map[string]any{})
	m.stdin.Close()
	select {
	case <-m.exited:
	case <-time.After(2 * time.Second):
		t.Fatal("the adapter was still running 2 s after its stdin closed")
	}
	if code := m.cmd.ProcessState.ExitCode(); code != 0 {
		t.Errorf("exit code %d\n%s", code, m.out)
	}
	var st statusView
	if code := p.do("GET", "/v1/status", nil, &st); code != 200 {
		t.Errorf("ixmux after the adapter's stop: %d", code)
	}
}
