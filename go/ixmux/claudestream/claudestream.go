// Package claudestream is the adapter to one real Claude session: a single `claude -p` child in
// stream-json, kept open for the life of ixmux, with no tools. One mandate is one user message;
// the result that follows it is its answer. The child is never relaunched: when it dies, its
// incarnation ends with it.
package claudestream

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// Named bounds.
const (
	MaxReply        = 4 << 10          // bytes of the answer kept in the Ack
	MaxStdoutLine   = 8 << 20          // bytes of one stdout line kept in the stream journal
	DefaultInitWait = 90 * time.Second // for the child's system/init
	DefaultStopWait = 10 * time.Second // for the child to exit once its stdin is closed
	StreamFile      = "stream.jsonl"
)

// Args are the child's flags. Never --bare, never a permission bypass, --setting-sources left
// alone (the user's hooks apply); no built-in tool (--tools "") and no MCP server
// (--strict-mcp-config without --mcp-config); the default permission mode.
func Args(session string) []string {
	return []string{"-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
		"--tools", "", "--strict-mcp-config", "--session-id", session}
}

// Config of one Claude target.
type Config struct {
	Name       string
	Dir        string   // throwaway working directory: <state>\claude-<name>
	Bin        string   // the claude executable
	Env        []string // added to the child's environment
	InitWait   time.Duration
	PrimeAfter time.Duration // with no init by then, an opening message asks for it (one turn)
	StopWait   time.Duration
}

// Adapter drives one child.
type Adapter struct {
	cfg     Config
	session string
	cmd     *exec.Cmd
	job     *job
	stdin   io.WriteCloser
	stream  *os.File
	logMu   sync.Mutex
	turn    sync.Mutex // one delivery at a time: a conversation, its results come in order
	initCh  chan struct{}
	exited  chan struct{}
	stop    sync.Once

	mu      sync.Mutex
	inited  bool
	initSID string
	tools   []string
	dead    error // why the child ended, once it has
	pending *turn
}

type turn struct {
	id, inc string
	done    chan turnResult // buffered 1
}

type turnResult struct {
	text  string
	isErr bool
	err   error
}

func uuid() string {
	b := make([]byte, 16)
	rand.Read(b)
	b[6], b[8] = b[6]&0x0f|0x40, b[8]&0x3f|0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// Start launches the child in its throwaway directory, in a kill-on-close job, and waits for its
// system/init. An init announcing another session than the one asked for is refused. An init
// that lists tools is accepted but every delivery is then refused.
func Start(cfg Config) (*Adapter, error) {
	if cfg.InitWait <= 0 {
		cfg.InitWait = DefaultInitWait
	}
	if cfg.PrimeAfter <= 0 {
		cfg.PrimeAfter = cfg.InitWait / 4
	}
	if cfg.StopWait <= 0 {
		cfg.StopWait = DefaultStopWait
	}
	if err := os.MkdirAll(cfg.Dir, 0o700); err != nil {
		return nil, err
	}
	stream, err := os.OpenFile(filepath.Join(cfg.Dir, StreamFile), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	stderr, err := os.OpenFile(filepath.Join(cfg.Dir, "stderr.log"), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		stream.Close()
		return nil, err
	}
	a := &Adapter{cfg: cfg, session: uuid(), stream: stream, initCh: make(chan struct{}), exited: make(chan struct{})}
	a.cmd = exec.Command(cfg.Bin, Args(a.session)...)
	a.cmd.Dir = cfg.Dir
	a.cmd.Env = append(os.Environ(), cfg.Env...)
	a.cmd.Stderr = stderr
	if a.stdin, err = a.cmd.StdinPipe(); err != nil {
		return nil, err
	}
	stdout, err := a.cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if a.job, err = startInJob(a.cmd); err != nil {
		stream.Close()
		stderr.Close()
		return nil, err
	}
	a.note(map[string]any{"ixmux": "start", "incarnation": a.session, "pid": a.cmd.Process.Pid})
	go a.read(stdout, stderr)

	var primed *turn
	deadline := time.After(cfg.InitWait)
	prime := time.After(cfg.PrimeAfter)
	for done := false; !done; {
		select {
		case <-a.initCh:
			done = true
		case <-prime:
			// No init without a first message: an opening turn, not a mandate.
			p, err := a.send("", "ixmux : ouverture de la session. Réponds seulement : OK.")
			if err != nil {
				a.Stop()
				return nil, fmt.Errorf("opening message: %w", err)
			}
			primed = p
			prime = nil
		case <-a.exited:
			a.Stop()
			return nil, fmt.Errorf("claude exited before its init: %v (see %s)", a.deadErr(), filepath.Join(cfg.Dir, "stderr.log"))
		case <-deadline:
			a.Stop()
			return nil, fmt.Errorf("no system/init from claude within %s", cfg.InitWait)
		}
	}
	a.mu.Lock()
	sid := a.initSID
	a.mu.Unlock()
	if sid != a.session {
		a.Stop()
		return nil, fmt.Errorf("claude's init announces session %q, not the %q it was given", sid, a.session)
	}
	if primed != nil {
		if r := <-primed.done; r.err != nil {
			a.Stop()
			return nil, fmt.Errorf("opening turn: %w", r.err)
		}
	}
	return a, nil
}

func (a *Adapter) deadErr() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.dead
}

// note appends one ixmux line to the stream journal and syncs it.
func (a *Adapter) note(v any) error {
	b, _ := json.Marshal(v)
	a.logMu.Lock()
	defer a.logMu.Unlock()
	if _, err := a.stream.Write(append(b, '\n')); err != nil {
		return err
	}
	return a.stream.Sync()
}

func (a *Adapter) raw(line []byte) {
	a.logMu.Lock()
	defer a.logMu.Unlock()
	a.stream.Write(line)
}

// read journals every stdout line and routes init and results.
func (a *Adapter) read(stdout io.Reader, stderr io.Closer) {
	br := bufio.NewReaderSize(stdout, 64<<10)
	for {
		line, err := readLine(br)
		if len(line) > 0 {
			a.raw(line)
			a.route(line)
		}
		if err != nil {
			break
		}
	}
	werr := a.cmd.Wait()
	stderr.Close()
	if werr == nil {
		werr = errors.New("claude exited")
	} else {
		werr = fmt.Errorf("claude exited: %w", werr)
	}
	a.mu.Lock()
	a.dead = werr
	p := a.pending
	a.pending = nil
	a.mu.Unlock()
	if p != nil {
		p.done <- turnResult{err: werr}
	}
	a.note(map[string]any{"ixmux": "exit", "incarnation": a.session, "error": werr.Error()})
	close(a.exited)
}

// readLine reads one line with its '\n'; a line over MaxStdoutLine is cut and closed by '\n'.
func readLine(br *bufio.Reader) ([]byte, error) {
	var out []byte
	for {
		chunk, err := br.ReadSlice('\n')
		if len(out)+len(chunk) <= MaxStdoutLine {
			out = append(out, chunk...)
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		if len(out) > 0 && out[len(out)-1] != '\n' {
			out = append(out, '\n')
		}
		return out, err
	}
}

func (a *Adapter) route(line []byte) {
	var msg struct {
		Type      string   `json:"type"`
		Subtype   string   `json:"subtype"`
		SessionID string   `json:"session_id"`
		Tools     []string `json:"tools"`
		Result    string   `json:"result"`
		IsError   bool     `json:"is_error"`
	}
	if json.Unmarshal(bytes.TrimSpace(line), &msg) != nil {
		return
	}
	switch {
	case msg.Type == "system" && msg.Subtype == "init":
		a.mu.Lock()
		first := !a.inited
		if first {
			a.inited, a.initSID, a.tools = true, msg.SessionID, msg.Tools
		} else if msg.SessionID != a.session || len(msg.Tools) > 0 {
			a.tools = append(a.tools, msg.Tools...) // a later init can only narrow what is allowed
			if msg.SessionID != a.session {
				a.initSID = msg.SessionID
			}
		}
		a.mu.Unlock()
		if first {
			close(a.initCh)
		}
	case msg.Type == "result":
		a.mu.Lock()
		p := a.pending
		a.pending = nil
		a.mu.Unlock()
		if p == nil {
			return
		}
		ackSeen := strings.HasPrefix(strings.TrimSpace(msg.Result), "ACK "+p.id)
		a.note(map[string]any{"ixmux": "result", "mandate_id": p.id, "incarnation": p.inc, "ack_seen": ackSeen, "is_error": msg.IsError})
		p.done <- turnResult{text: msg.Result, isErr: msg.IsError}
	}
}

// send journals the send, then writes one user message on stdin. The caller holds the turn.
func (a *Adapter) send(id, text string) (*turn, error) {
	b, err := json.Marshal(map[string]any{"type": "user", "message": map[string]any{"role": "user",
		"content": []any{map[string]any{"type": "text", "text": text}}}})
	if err != nil {
		return nil, err
	}
	p := &turn{id: id, inc: a.session, done: make(chan turnResult, 1)}
	a.mu.Lock()
	if a.dead != nil {
		a.mu.Unlock()
		return nil, a.dead
	}
	a.pending = p
	a.mu.Unlock()
	if err := a.note(map[string]any{"ixmux": "send", "mandate_id": id, "incarnation": a.session, "at": time.Now().UTC()}); err != nil {
		return nil, err
	}
	if _, err := a.stdin.Write(append(b, '\n')); err != nil {
		return nil, err
	}
	return p, nil
}

// Incarnation is the session_id of the live child's init: an error once the child has ended.
func (a *Adapter) Incarnation() (string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch {
	case a.dead != nil:
		return "", fmt.Errorf("claude child of %s ended (%v); ixmux does not relaunch it", a.cfg.Name, a.dead)
	case !a.inited:
		return "", fmt.Errorf("claude child of %s has not sent its init", a.cfg.Name)
	}
	return a.initSID, nil
}

func (a *Adapter) sessionID() string { return a.session }

// PID of the child.
func (a *Adapter) PID() int { return a.cmd.Process.Pid }

// Alive says whether the child still runs.
func (a *Adapter) Alive() bool { return a.deadErr() == nil }

// StreamPath is the journal of everything read on the child's stdout, with ixmux's own lines
// (start, send, result, exit).
func (a *Adapter) StreamPath() string { return filepath.Join(a.cfg.Dir, StreamFile) }

// Tools lists the tools the child's init announced (empty is the only acceptable answer).
func (a *Adapter) Tools() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]string(nil), a.tools...)
}

// prompt frames a mandate as the user message.
func prompt(m deliver.Mandate) string {
	return "ixmux mandate_id: " + m.MandateID + "\nCommence ta réponse exactement par « ACK " + m.MandateID + " ».\n\n" + m.Text
}

// Deliver sends m as one user message and returns with its result, or when the child ends; it
// does not return at ctx's deadline, so the next mandate never meets this one's result (the
// engine records uncertain at its own deadline, then this late answer).
//
// Under the turn, and the same lock as the pending send, it checks that the live child is the
// incarnation asked for: this Adapter owns one child for its whole life (no relaunch), so the
// session_id of its init and the process it writes to cannot change between the check and the
// write; a dead child refuses.
// @ai:invariant a mandate is written to the child only if its incarnation is the live child's init session_id and that init listed no tool [T:test conf:0.85 src:go/ixmux/claudestream/claudestream_test.go:TestToolsListedIsRefused]
func (a *Adapter) Deliver(ctx context.Context, m deliver.Mandate) (deliver.Ack, error) {
	a.turn.Lock()
	defer a.turn.Unlock()
	a.mu.Lock()
	var refuse string
	switch {
	case a.dead != nil:
		refuse = "the claude child has ended: " + a.dead.Error()
	case !a.inited:
		refuse = "the claude child has not sent its init"
	case len(a.tools) > 0:
		refuse = fmt.Sprintf("the claude child announces tools %v; it must have none", a.tools)
	case a.initSID != a.session || m.Incarnation != a.initSID:
		refuse = fmt.Sprintf("the mandate is for incarnation %q, the claude child is %q", m.Incarnation, a.initSID)
	}
	a.mu.Unlock()
	if refuse != "" {
		return deliver.Ack{}, fmt.Errorf("%w: %s", deliver.ErrRefused, refuse)
	}
	p, err := a.send(m.MandateID, prompt(m))
	if err != nil {
		return deliver.Ack{}, fmt.Errorf("send to claude: %w", err)
	}
	r := <-p.done
	if r.err != nil {
		return deliver.Ack{}, r.err
	}
	text := r.text
	if len(text) > MaxReply {
		text = text[:MaxReply]
	}
	ack := deliver.Ack{Receipt: a.session + ":" + m.MandateID, Text: text, AckSeen: strings.HasPrefix(strings.TrimSpace(r.text), "ACK "+m.MandateID)}
	if r.isErr {
		return ack, fmt.Errorf("claude answered with an error result: %.200s", r.text)
	}
	return ack, nil
}

// Status answers for incarnation from the stream journal: no send of mandateID is not
// received; a send followed by its result is received; a send without result is unknown (the
// child may still answer, or died mid-turn). Another incarnation than the live child's is
// unknown (the engine refuses it stale-incarnation before asking).
func (a *Adapter) Status(ctx context.Context, mandateID, incarnation string) (deliver.Outcome, error) {
	if incarnation != a.session {
		return deliver.Unknown, nil
	}
	f, err := os.Open(a.StreamPath())
	if err != nil {
		return deliver.Unknown, err
	}
	defer f.Close()
	br := bufio.NewReaderSize(f, 64<<10)
	sent, answered := false, false
	for {
		line, err := readLine(br)
		if bytes.Contains(line, []byte(`"ixmux":`)) { // ixmux's own lines; claude's have a type
			var n struct {
				Type        string `json:"type"`
				Ixmux       string `json:"ixmux"`
				MandateID   string `json:"mandate_id"`
				Incarnation string `json:"incarnation"`
			}
			if json.Unmarshal(line, &n) == nil && n.Type == "" && n.MandateID == mandateID && n.Incarnation == incarnation {
				switch n.Ixmux {
				case "send":
					sent = true
				case "result":
					answered = sent
				}
			}
		}
		if err != nil {
			break
		}
	}
	switch {
	case answered:
		return deliver.Received, nil
	case sent:
		return deliver.Unknown, nil
	}
	return deliver.NotReceived, nil
}

// Stop closes the child's stdin, waits up to StopWait for it to exit, then kills its job.
func (a *Adapter) Stop() error {
	var err error
	a.stop.Do(func() {
		a.stdin.Close()
		select {
		case <-a.exited:
		case <-time.After(a.cfg.StopWait):
			err = a.job.kill()
			<-a.exited
		}
		a.job.Close()
		a.stream.Close()
	})
	return err
}

// Target registers a with an engine.
func (a *Adapter) Target() deliver.Target {
	return deliver.Target{Adapter: a, Incarnation: a.Incarnation}
}
