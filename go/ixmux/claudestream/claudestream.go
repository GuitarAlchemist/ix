// Package claudestream is the adapter to one real Claude session: a `claude -p` child in
// stream-json, with no tools. One mandate is one user message; the result that follows it, of the
// same session and acknowledging that mandate, is its answer. A child that dies ends its
// incarnation and is not relaunched; a child that holds a turn past TurnMax is killed and
// replaced by a new one, under a new incarnation.
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
	"unicode"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
	"github.com/GuitarAlchemist/ix/go/ixmux/journal"
)

// Named bounds.
const (
	MaxReply        = 4 << 10          // bytes of the answer kept in the Ack
	MaxStdoutLine   = 8 << 20          // bytes of one stdout line; a longer one is cut
	MaxStreamBytes  = 256 << 20        // bytes of stdout one incarnation may add to stream.jsonl
	MaxStderrBytes  = 16 << 20         // bytes one incarnation may add to stderr.log
	DefaultInitWait = 90 * time.Second // for the child's system/init and its opening turn
	DefaultStopWait = 10 * time.Second // for the child to exit once its stdin is closed
	DefaultTurnMax  = 10 * time.Minute // a turn without result stays reserved this long at most
	StatusPage      = int64(1 << 20)   // bytes of stream.jsonl one Status read takes at a time
	StreamFile      = "stream.jsonl"
	StderrFile      = "stderr.log"
	lineQueue       = 256 // stdout lines read ahead of the router
)

// BaseEnv names the parent's variables every child gets, when set: what Windows, a Unix system
// and claude need to start. Nothing else is inherited, and nothing by pattern: another variable
// is passed by naming it (Config.PassEnv, --claude-env).
var BaseEnv = []string{
	"SystemRoot", "SystemDrive", "windir", "ComSpec", "PATH", "PATHEXT", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
	"APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "HOME", "USERNAME", "ProgramData", "ProgramFiles", "OS",
	"PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG",
}

// Args are the child's flags. Never --bare, never a permission bypass, --setting-sources left
// alone (the user's hooks apply); no built-in tool (--tools "") and no MCP server
// (--strict-mcp-config without --mcp-config); the default permission mode.
func Args(session string) []string {
	return []string{"-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
		"--tools", "", "--strict-mcp-config", "--session-id", session}
}

// Config of one Claude target. A zero bound takes its default.
type Config struct {
	Name       string
	Dir        string   // throwaway working directory: <state>\claude-<name>
	Bin        string   // the claude executable
	Env        []string // NAME=value added to the child's environment
	InitWait   time.Duration
	PrimeAfter time.Duration // with no init by then, an opening message asks for it (one turn)
	StopWait   time.Duration
	PassEnv    []string      // names of the parent's variables the child also gets
	TurnMax    time.Duration // how long a turn stays reserved without a result (DefaultTurnMax)
	MaxStream  int64         // bytes of stdout one incarnation may add to stream.jsonl (MaxStreamBytes)
	MaxStderr  int64         // bytes one incarnation may add to stderr.log (MaxStderrBytes)
	MaxLine    int           // bytes of one stdout line (MaxStdoutLine)
}

// Test hooks, set by a test only: testHookAfterCheck runs in Deliver between the check of the
// child and the write on its stdin, with the router of the child's lines; testHookStatusFrom gets
// the offset each Status read starts at.
var (
	testHookAfterCheck func(route func([]byte))
	testHookStatusFrom func(int64)
	// testHookBeforeRelaunch runs in watch between its read of stopped and the relaunch.
	testHookBeforeRelaunch func()
	// killJob kills a child's job; a test replaces it to make the kill fail.
	killJob = func(j *job) error { return j.kill() }
)

// childEnv builds the child's environment: the variables of BaseEnv and pass that the parent
// has, then extra.
func childEnv(pass, extra []string) []string {
	var env []string
	seen := map[string]bool{}
	for _, name := range append(append([]string(nil), BaseEnv...), pass...) {
		key := strings.ToUpper(name)
		if seen[key] {
			continue
		}
		if v, ok := os.LookupEnv(name); ok {
			seen[key] = true
			env = append(env, name+"="+v)
		}
	}
	return append(env, extra...)
}

// Adapter is one Claude target: its live child, and the stream journal all its children write.
type Adapter struct {
	cfg    Config
	logMu  sync.Mutex
	stream *os.File

	mu       sync.Mutex
	cur      *child
	reserved bool // a turn was sent and may still answer: no other turn starts
	stopped  bool
	quit     chan struct{}  // closed with stopped set: a relaunch under way gives up
	watchers sync.WaitGroup // one per reserved turn, relaunch included

	statusMu sync.Mutex
	cursor   journal.Cursor
	marks    map[[2]string]*mark // by incarnation and mandate id
}

type mark struct{ sent, answered, failed bool } // failed: its result had is_error

// child is one claude process: one incarnation.
type child struct {
	a       *Adapter
	session string
	cmd     *exec.Cmd
	job     *job
	stdin   io.WriteCloser
	initCh  chan struct{}
	exited  chan struct{}
	stopper sync.Once
	overOne sync.Once

	written int64 // bytes of stdout written to stream.jsonl (under a.logMu)
	over    bool  // past MaxStream: nothing more is written (under a.logMu)

	mu      sync.Mutex
	inited  bool
	initSID string
	tools   []string
	toolsOK bool // every init had a tools field that is an array
	gen     int  // inits routed so far
	dead    error
	pending *turn
}

type turn struct {
	id   string
	done chan struct{}
	once sync.Once
	res  turnResult
}

func newTurn(id string) *turn { return &turn{id: id, done: make(chan struct{})} }

func (p *turn) resolve(r turnResult) { p.once.Do(func() { p.res = r; close(p.done) }) }

type turnResult struct {
	text    string
	ackSeen bool
	isErr   bool
	err     error
}

type lineMsg struct {
	line []byte
	cut  bool
}

func uuid() string {
	b := make([]byte, 16)
	rand.Read(b)
	b[6], b[8] = b[6]&0x0f|0x40, b[8]&0x3f|0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// Start launches the first child in its throwaway directory, in a kill-on-close job, and waits
// for its system/init (and its opening turn, if it needed one) within InitWait. An init
// announcing another session than the one asked for is refused. An init that lists tools, or has
// no tools array, is accepted but every delivery is then refused.
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
	if cfg.TurnMax <= 0 {
		cfg.TurnMax = DefaultTurnMax
	}
	if cfg.MaxStream <= 0 {
		cfg.MaxStream = MaxStreamBytes
	}
	if cfg.MaxStderr <= 0 {
		cfg.MaxStderr = MaxStderrBytes
	}
	if cfg.MaxLine <= 0 {
		cfg.MaxLine = MaxStdoutLine
	}
	if err := os.MkdirAll(cfg.Dir, 0o700); err != nil {
		return nil, err
	}
	stream, err := os.OpenFile(filepath.Join(cfg.Dir, StreamFile), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	a := &Adapter{cfg: cfg, stream: stream, marks: map[[2]string]*mark{}, quit: make(chan struct{})}
	c, err := a.startChild()
	if err != nil {
		stream.Close()
		return nil, err
	}
	a.cur = c
	return a, nil
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

// cappedWriter writes at most max bytes to f; past it, it drops what it gets and calls over once.
type cappedWriter struct {
	f       io.Writer
	max, n  int64
	overOne sync.Once
	over    func()
}

func (w *cappedWriter) Write(p []byte) (int, error) {
	if w.n+int64(len(p)) > w.max {
		w.overOne.Do(w.over)
		return len(p), nil // drained, not kept
	}
	w.n += int64(len(p))
	return w.f.Write(p)
}

// ReasonKillFailed: a child's job could not be killed, or the child did not exit within StopWait
// after the kill; its incarnation is lost.
const ReasonKillFailed = "kill-failed"

// errStopping: a start that began, or would begin, once Stop was called.
var errStopping = errors.New("the claude target is stopping")

// startChild starts one child and waits for its init (and opening turn). Once Stop is called,
// none starts, and one under way is killed: no opening turn is sent past that point.
// @ai:invariant no child starts and no opening turn is sent once stopped is set [T:test conf:0.85 src:go/ixmux/claudestream/s6c_test.go:TestStopDuringTheRelaunch]
func (a *Adapter) startChild() (*child, error) {
	cfg := a.cfg
	select {
	case <-a.quit:
		return nil, errStopping
	default:
	}
	stderr, err := os.OpenFile(filepath.Join(cfg.Dir, StderrFile), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	c := &child{a: a, session: uuid(), initCh: make(chan struct{}), exited: make(chan struct{})}
	c.cmd = exec.Command(cfg.Bin, Args(c.session)...)
	c.cmd.Dir = cfg.Dir
	c.cmd.Env = childEnv(cfg.PassEnv, cfg.Env)
	c.cmd.Stderr = &cappedWriter{f: stderr, max: cfg.MaxStderr, over: func() { go c.overflow("stderr-too-large") }}
	if c.stdin, err = c.cmd.StdinPipe(); err != nil {
		stderr.Close()
		return nil, err
	}
	stdout, err := c.cmd.StdoutPipe()
	if err != nil {
		stderr.Close()
		return nil, err
	}
	if c.job, err = startInJob(c.cmd); err != nil {
		stderr.Close()
		return nil, err
	}
	a.note(map[string]any{"ixmux": "start", "incarnation": c.session, "pid": c.cmd.Process.Pid, "at": time.Now().UTC()})
	lines := make(chan lineMsg, lineQueue)
	go c.read(stdout, lines)
	go c.router(lines, stderr)

	fail := func(err error) (*child, error) {
		if errors.Is(err, errStopping) {
			c.kill() // the same Stop ends it: no wait for a child that is not needed
		}
		c.stop()
		return nil, err
	}
	var primed *turn
	deadline := time.After(cfg.InitWait)
	prime := time.After(cfg.PrimeAfter)
	for done := false; !done; {
		select {
		case <-c.initCh:
			done = true
		case <-a.quit:
			return fail(errStopping)
		case <-prime:
			// No init without a first message: an opening turn, not a mandate. It is sent under
			// a.mu, where Stop sets stopped: none leaves once the stop has begun.
			a.mu.Lock()
			var p *turn
			if a.stopped {
				err = errStopping
			} else {
				p, err = c.send("", "ixmux : ouverture de la session. Réponds seulement : OK.", -1, "")
			}
			a.mu.Unlock()
			if errors.Is(err, errStopping) {
				return fail(err)
			}
			if err != nil {
				return fail(fmt.Errorf("opening message: %w", err))
			}
			primed, prime = p, nil
		case <-c.exited:
			return fail(fmt.Errorf("claude exited before its init: %v (see %s)", c.deadErr(), filepath.Join(cfg.Dir, StderrFile)))
		case <-deadline:
			return fail(fmt.Errorf("no system/init from claude within %s", cfg.InitWait))
		}
	}
	c.mu.Lock()
	sid := c.initSID
	c.mu.Unlock()
	if sid != c.session {
		return fail(fmt.Errorf("claude's init announces session %q, not the %q it was given", sid, c.session))
	}
	if primed != nil {
		select {
		case <-primed.done:
			if primed.res.err != nil {
				return fail(fmt.Errorf("opening turn: %w", primed.res.err))
			}
		case <-a.quit:
			return fail(errStopping)
		case <-deadline:
			return fail(fmt.Errorf("no answer to the opening message within %s", cfg.InitWait))
		}
	}
	return c, nil
}

func (c *child) deadErr() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.dead
}

// journal writes one stdout line to stream.jsonl, within the incarnation's MaxStream. Past it,
// nothing more is written or routed, and the child is killed (overflow).
func (c *child) journal(line []byte) bool {
	c.a.logMu.Lock()
	defer c.a.logMu.Unlock()
	if c.over {
		return false
	}
	if c.written+int64(len(line)) > c.a.cfg.MaxStream {
		c.over = true
		go c.overflow("stream-too-large")
		return false
	}
	c.written += int64(len(line))
	c.a.stream.Write(line)
	return true
}

// overflow ends an incarnation whose output passed its bound: the pending turn is uncertain for
// reason, and the job is killed.
// @ai:invariant one incarnation adds at most MaxStream bytes of stdout to stream.jsonl and MaxStderr to stderr.log; past it the child is killed and its turn is uncertain [T:test conf:0.85 src:go/ixmux/claudestream/s4b5_test.go:TestFloodIsCapped]
func (c *child) overflow(reason string) {
	c.overOne.Do(func() {
		killJob(c.job)
		err := fmt.Errorf("%s: the claude child's output passed its bound; it was killed", reason)
		c.mu.Lock()
		p := c.pending
		c.pending = nil
		if c.dead == nil {
			c.dead = err // the incarnation ends now, not when its process is reaped
		}
		c.mu.Unlock()
		c.a.note(map[string]any{"ixmux": reason, "incarnation": c.session, "at": time.Now().UTC()})
		if p != nil {
			p.resolve(turnResult{err: err})
		}
	})
}

// read journals every stdout line and hands it to the router. It never waits on the child's
// state, so stdout is drained even while a send holds that state.
func (c *child) read(stdout io.Reader, lines chan<- lineMsg) {
	br := bufio.NewReaderSize(stdout, 64<<10)
	for {
		line, cut, err := readLine(br, c.a.cfg.MaxLine)
		if len(line) > 0 && c.journal(line) {
			lines <- lineMsg{line, cut}
		}
		if err != nil {
			break
		}
	}
	close(lines)
}

// router routes the lines in order, then records the child's end.
func (c *child) router(lines <-chan lineMsg, stderr io.Closer) {
	for l := range lines {
		c.route(l)
	}
	werr := c.cmd.Wait()
	stderr.Close()
	if werr == nil {
		werr = errors.New("claude exited")
	} else {
		werr = fmt.Errorf("claude exited: %w", werr)
	}
	c.mu.Lock()
	if c.dead == nil {
		c.dead = werr
	}
	p := c.pending
	c.pending = nil
	c.mu.Unlock()
	if p != nil {
		p.resolve(turnResult{err: werr})
	}
	c.a.note(map[string]any{"ixmux": "exit", "incarnation": c.session, "error": werr.Error(), "at": time.Now().UTC()})
	close(c.exited)
}

// readLine reads one line with its '\n'; a line over max is cut there, closed by '\n', and
// reported cut.
func readLine(br *bufio.Reader, max int) ([]byte, bool, error) {
	var out []byte
	cut := false
	for {
		chunk, err := br.ReadSlice('\n')
		if len(out)+len(chunk) <= max {
			out = append(out, chunk...)
		} else {
			cut = true
		}
		if err == bufio.ErrBufferFull {
			continue
		}
		if len(out) > 0 && out[len(out)-1] != '\n' {
			out = append(out, '\n')
		}
		return out, cut, err
	}
}

// ackOf is the mandate id an answer acknowledges: the word after a leading "ACK ", up to a space
// or the end; "" if the answer does not begin with "ACK ".
func ackOf(text string) string {
	t, ok := strings.CutPrefix(strings.TrimSpace(text), "ACK ")
	if !ok {
		return ""
	}
	if i := strings.IndexFunc(t, unicode.IsSpace); i >= 0 {
		t = t[:i]
	}
	return t
}

// route takes one stdout line: an init sets (or narrows) what the child may be sent; a result
// answers the pending turn only if it is of the init's session and a turn is pending.
// @ai:invariant a result answers the pending turn only if its session_id is the child's init session, and is delivered only if it acknowledges that mandate or no mandate [T:test conf:0.85 src:go/ixmux/claudestream/s4b1_test.go:TestResultsAreCorrelated]
func (c *child) route(l lineMsg) {
	if l.cut {
		// A line over the bound cannot be read; a turn waiting for it could wait forever.
		c.mu.Lock()
		p := c.pending
		c.pending = nil
		c.mu.Unlock()
		c.a.note(map[string]any{"ixmux": "line-too-long", "incarnation": c.session, "bytes_kept": len(l.line), "at": time.Now().UTC()})
		if p != nil {
			p.resolve(turnResult{err: fmt.Errorf("a stdout line of claude passed %d bytes: its answer cannot be read", c.a.cfg.MaxLine)})
		}
		return
	}
	var msg struct {
		Type      string          `json:"type"`
		Subtype   string          `json:"subtype"`
		SessionID string          `json:"session_id"`
		Tools     json.RawMessage `json:"tools"`
		Result    string          `json:"result"`
		IsError   bool            `json:"is_error"`
	}
	if json.Unmarshal(bytes.TrimSpace(l.line), &msg) != nil {
		return
	}
	switch {
	case msg.Type == "system" && msg.Subtype == "init":
		var tools []string
		ok := len(msg.Tools) > 0 && string(msg.Tools) != "null" && json.Unmarshal(msg.Tools, &tools) == nil
		c.mu.Lock()
		c.gen++
		first := !c.inited
		if first {
			c.inited, c.initSID, c.tools, c.toolsOK = true, msg.SessionID, tools, ok
		} else {
			c.tools = append(c.tools, tools...) // a later init can only narrow what is allowed
			c.toolsOK = c.toolsOK && ok
			if msg.SessionID != c.session {
				c.initSID = msg.SessionID
			}
		}
		c.mu.Unlock()
		if first {
			close(c.initCh)
		}
	case msg.Type == "result":
		c.mu.Lock()
		p, sid := c.pending, c.initSID
		why := ""
		switch {
		case msg.SessionID != sid:
			why = "result of session " + msg.SessionID + ", not " + sid
		case p == nil:
			why = "result with no turn pending"
		default:
			c.pending = nil
		}
		c.mu.Unlock()
		if why != "" {
			c.a.note(map[string]any{"ixmux": "ignored-result", "incarnation": c.session, "why": why, "at": time.Now().UTC()})
			return
		}
		ack := ackOf(msg.Result)
		if p.id != "" && ack != "" && ack != p.id {
			// The answer acknowledges another mandate: what it answers is unknown.
			c.a.note(map[string]any{"ixmux": "ack-mismatch", "mandate_id": p.id, "acknowledged": ack, "incarnation": c.session, "at": time.Now().UTC()})
			p.resolve(turnResult{err: fmt.Errorf("the answer acknowledges %q, not %q", ack, p.id)})
			return
		}
		seen := p.id != "" && ack == p.id
		c.a.note(map[string]any{"ixmux": "result", "mandate_id": p.id, "incarnation": c.session, "ack_seen": seen, "is_error": msg.IsError})
		p.resolve(turnResult{text: msg.Result, ackSeen: seen, isErr: msg.IsError})
	}
}

// refusal says why this child may not take a mandate for incarnation inc (under c.mu), or "".
func (c *child) refusal(inc string) string {
	switch {
	case c.dead != nil:
		return "the claude child has ended: " + c.dead.Error()
	case !c.inited:
		return "the claude child has not sent its init"
	case !c.toolsOK:
		return "the claude child's init has no tools array; it must have an empty one"
	case len(c.tools) > 0:
		return fmt.Sprintf("the claude child announces tools %v; it must have none", c.tools)
	case c.initSID != c.session || inc != c.initSID:
		return fmt.Sprintf("the mandate is for incarnation %q, the claude child is %q", inc, c.initSID)
	}
	return ""
}

// check is the first look at the child for a mandate of incarnation inc: the init generation it
// saw, or why it refuses.
func (c *child) check(inc string) (int, string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.gen, c.refusal(inc)
}

// send writes one user message on stdin. For a mandate (gen >= 0) it checks again, under the
// lock it writes under, that the child still takes incarnation inc and has routed no init since
// the check at gen; the router takes that lock to route an init, so none is taken in between.
// The send is journaled (synced) before the write.
// @ai:invariant a mandate is written to the child only if, under the lock of the write, its incarnation is the child's init session_id, the init had an empty tools array, and no init came since the check [T:test conf:0.85 src:go/ixmux/claudestream/s4b2_test.go:TestInitBetweenCheckAndWrite]
func (c *child) send(id, text string, gen int, inc string) (*turn, error) {
	b, err := json.Marshal(map[string]any{"type": "user", "message": map[string]any{"role": "user",
		"content": []any{map[string]any{"type": "text", "text": text}}}})
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if gen >= 0 {
		if why := c.refusal(inc); why != "" {
			return nil, fmt.Errorf("%w: %s", deliver.ErrRefused, why)
		}
		if c.gen != gen {
			return nil, fmt.Errorf("%w: the claude child sent another init since it was checked", deliver.ErrRefused)
		}
	}
	if c.dead != nil {
		return nil, c.dead
	}
	if err := c.a.note(map[string]any{"ixmux": "send", "mandate_id": id, "incarnation": c.session, "at": time.Now().UTC()}); err != nil {
		return nil, err
	}
	p := newTurn(id)
	c.pending = p
	if _, err := c.stdin.Write(append(b, '\n')); err != nil {
		c.pending = nil
		return nil, err
	}
	return p, nil
}

// kill kills the child's job and waits StopWait at most for it to exit. Past it, whether the kill
// failed or the child did not die, the incarnation is lost: the child is dead with kill-failed,
// its pending turn resolves with that error, and the line is noted. Nothing waits on it longer.
// @ai:invariant every wait after a kill is bounded by StopWait; past it the incarnation is lost (kill-failed) and its turn resolves in error [T:test conf:0.85 src:go/ixmux/claudestream/s6c_kill_test.go:TestKillThatFailsAtTurnMax]
func (c *child) kill() error {
	kerr := killJob(c.job)
	select {
	case <-c.exited:
		return nil
	case <-time.After(c.a.cfg.StopWait):
	}
	if kerr == nil {
		kerr = fmt.Errorf("the child did not exit within %s of the kill", c.a.cfg.StopWait)
	}
	lost := fmt.Errorf("%s: the claude child could not be killed (%v); its incarnation is lost", ReasonKillFailed, kerr)
	c.mu.Lock()
	p := c.pending
	c.pending = nil
	if c.dead == nil {
		c.dead = lost
	}
	c.mu.Unlock()
	c.a.note(map[string]any{"ixmux": ReasonKillFailed, "incarnation": c.session, "error": kerr.Error(), "at": time.Now().UTC()})
	if p != nil {
		p.resolve(turnResult{err: lost})
	}
	return lost
}

// stop closes the child's stdin, waits up to StopWait for it to exit, then kills its job. Each
// step is noted in stream.jsonl with its time ({"ixmux":"stop","step":...}): stdin-closed, then
// exited or killed with the milliseconds since stdin was closed.
// @ai:invariant the stop of a claude child is traced: stdin closed, then exited or killed past StopWait [T:test conf:0.85 src:go/ixmux/claudestream/claudestream_test.go:TestStopIsTraced]
func (c *child) stop() error {
	var err error
	c.stopper.Do(func() {
		begun := time.Now()
		c.a.note(map[string]any{"ixmux": "stop", "step": "stdin-closed", "incarnation": c.session, "at": begun.UTC()})
		c.stdin.Close()
		step := "exited"
		select {
		case <-c.exited:
		case <-time.After(c.a.cfg.StopWait):
			step = "killed"
			if err = c.kill(); err != nil {
				step = ReasonKillFailed
			}
		}
		c.a.note(map[string]any{"ixmux": "stop", "step": step, "incarnation": c.session, "at": time.Now().UTC(), "after_ms": time.Since(begun).Milliseconds()})
		c.job.Close()
	})
	return err
}

func (a *Adapter) live() *child {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.cur
}

// Incarnation is the session_id of the live child's init: an error once the child has ended.
func (a *Adapter) Incarnation() (string, error) {
	c := a.live()
	c.mu.Lock()
	defer c.mu.Unlock()
	switch {
	case c.dead != nil:
		return "", fmt.Errorf("claude child of %s ended (%v); ixmux relaunches it only past TurnMax", a.cfg.Name, c.dead)
	case !c.inited:
		return "", fmt.Errorf("claude child of %s has not sent its init", a.cfg.Name)
	}
	return c.initSID, nil
}

func (a *Adapter) sessionID() string { return a.live().session }

// PID of the live child.
func (a *Adapter) PID() int { return a.live().cmd.Process.Pid }

// Alive says whether the live child still runs.
func (a *Adapter) Alive() bool { return a.live().deadErr() == nil }

// StreamPath is the journal of everything read on the children's stdout, with ixmux's own lines.
func (a *Adapter) StreamPath() string { return filepath.Join(a.cfg.Dir, StreamFile) }

// Tools lists the tools the live child's init announced (empty is the only acceptable answer).
func (a *Adapter) Tools() []string {
	c := a.live()
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.tools...)
}

// prompt frames a mandate as the user message.
func prompt(m deliver.Mandate) string {
	return "ixmux mandate_id: " + m.MandateID + "\nCommence ta réponse exactement par « ACK " + m.MandateID + " ».\n\n" + m.Text
}

// Deliver sends m as one user message and returns with its result. ctx is not consulted once the
// message is written: past the engine's deadline the call goes on, and its answer comes late. A
// turn stays reserved until its result: no other mandate is sent while it may still answer. Past
// TurnMax, its child is killed and replaced (a new incarnation), and the target is free again.
// An answer that acknowledges another mandate is an error (the delivery is uncertain); an answer
// with no ACK is delivered with AckSeen false.
// @ai:invariant one turn at a time per target: a turn past its deadline stays reserved until its result, its child's death or TurnMax [T:test conf:0.85 src:go/ixmux/claudestream/s4b4_test.go:TestTurnMaxFreesTheTarget]
func (a *Adapter) Deliver(ctx context.Context, m deliver.Mandate) (deliver.Ack, error) {
	a.mu.Lock()
	c := a.cur
	refuse := ""
	switch {
	case a.stopped:
		refuse = "the claude target is stopping"
	case a.reserved:
		refuse = "a previous turn may still answer (TurnMax " + a.cfg.TurnMax.String() + ")"
	default:
		a.reserved = true
		a.watchers.Add(1)
	}
	a.mu.Unlock()
	if refuse != "" {
		return deliver.Ack{}, fmt.Errorf("%w: %s", deliver.ErrRefused, refuse)
	}
	gen, why := c.check(m.Incarnation)
	if why != "" {
		a.release()
		return deliver.Ack{}, fmt.Errorf("%w: %s", deliver.ErrRefused, why)
	}
	if testHookAfterCheck != nil {
		testHookAfterCheck(func(b []byte) { c.route(lineMsg{line: b}) })
	}
	if err := ctx.Err(); err != nil { // before the send, the deadline still refuses: nothing was written
		a.release()
		return deliver.Ack{}, fmt.Errorf("%w: the delivery deadline passed before the send (%v)", deliver.ErrRefused, err)
	}
	p, err := c.send(m.MandateID, prompt(m), gen, m.Incarnation)
	if err != nil {
		a.release()
		if errors.Is(err, deliver.ErrRefused) {
			return deliver.Ack{}, err
		}
		return deliver.Ack{}, fmt.Errorf("send to claude: %w", err)
	}
	go a.watch(c, p)
	// Once the message is written, the call waits for the turn's end, not for ctx: the engine
	// records uncertain at its own deadline, keeps the call (still_running) and settles the
	// delivery with this late answer. The wait is bounded all the same: watch kills the child at
	// TurnMax, and Stop past StopWait; either resolves p with the child's death.
	// @ai:invariant after the send, Deliver returns with the turn's result or the child's death, never at ctx's end [T:test conf:0.85 src:go/ixmux/claudestream/s6b_test.go:TestLateAnswerIsDelivered]
	<-p.done
	r := p.res
	if r.err != nil {
		return deliver.Ack{}, r.err
	}
	text, cut := r.text, false
	if len(text) > MaxReply {
		text, cut = text[:MaxReply], true
	}
	ack := deliver.Ack{Receipt: c.session + ":" + m.MandateID, Text: text, AckSeen: r.ackSeen, Truncated: cut}
	if r.isErr {
		return ack, fmt.Errorf("claude answered with an error result: %.200s", r.text)
	}
	return ack, nil
}

// release ends a reservation.
func (a *Adapter) release() {
	a.mu.Lock()
	a.reserved = false
	a.mu.Unlock()
	a.watchers.Done()
}

// watch holds the reservation until p is resolved, or TurnMax: then the child is killed (its
// turn resolves with its death) and a new one starts, unless the target is stopping.
func (a *Adapter) watch(c *child, p *turn) {
	defer a.release()
	t := time.NewTimer(a.cfg.TurnMax)
	defer t.Stop()
	select {
	case <-p.done:
		return
	case <-t.C:
	}
	a.note(map[string]any{"ixmux": "turn-max", "mandate_id": p.id, "incarnation": c.session, "turn_max": a.cfg.TurnMax.String(), "at": time.Now().UTC()})
	if c.kill() != nil {
		return // lost: its turn resolved in error; no new child beside one that may still run
	}
	c.stop()
	a.mu.Lock()
	stopped := a.stopped
	a.mu.Unlock()
	if stopped {
		return
	}
	if testHookBeforeRelaunch != nil {
		testHookBeforeRelaunch()
	}
	nc, err := a.startChild()
	if errors.Is(err, errStopping) {
		return
	}
	if err != nil {
		a.note(map[string]any{"ixmux": "relaunch-failed", "error": err.Error(), "at": time.Now().UTC()})
		return
	}
	a.mu.Lock()
	stopped = a.stopped
	if !stopped {
		a.cur = nc
	}
	a.mu.Unlock()
	if stopped {
		nc.stop()
	}
}

// Status answers for incarnation from the stream journal: no send of mandateID is not
// received; a send followed by its result is received; a send without result is unknown (the
// child may still answer, or died mid-turn). Another incarnation than the live child's is
// unknown. Each call reads only what was appended since the last one, a page at a time, and
// stops at ctx's end.
func (a *Adapter) Status(ctx context.Context, mandateID, incarnation string) (deliver.Outcome, error) {
	if incarnation != a.sessionID() {
		return deliver.Unknown, nil
	}
	a.statusMu.Lock()
	defer a.statusMu.Unlock()
	if err := a.catchUp(ctx); err != nil {
		return deliver.Unknown, err
	}
	switch mk := a.marks[[2]string{incarnation, mandateID}]; {
	case mk == nil:
		return deliver.NotReceived, nil
	case mk.answered:
		return deliver.Received, nil
	case mk.failed:
		// The message reached the session and its turn ended in error: neither received (no
		// answer) nor not received (it was sent). It stays uncertain, and the reason says why.
		return deliver.Unknown, fmt.Errorf("the result of %s had is_error: the session took it, its answer is an error", mandateID)
	}
	return deliver.Unknown, nil
}

// catchUp folds the stream's new ixmux lines into marks (under statusMu).
// @ai:invariant Status reads stream.jsonl from where its last read stopped, never again from byte 0 [T:test conf:0.85 src:go/ixmux/claudestream/s4b5_test.go:TestStatusHonoursItsContextAndCursor]
func (a *Adapter) catchUp(ctx context.Context) error {
	path := a.StreamPath()
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if a.cursor.Incarnation == "" {
			inc, _, err := journal.Identify(path)
			if err != nil {
				return err
			}
			a.cursor = journal.Cursor{Incarnation: inc}
		}
		if testHookStatusFrom != nil {
			testHookStatusFrom(a.cursor.From)
		}
		page, err := journal.ReadCursor(path, a.cursor, StatusPage)
		if err != nil {
			return err
		}
		for _, ev := range page.Events {
			a.index(ev.Line)
		}
		if page.Pending != nil {
			a.cursor = page.Pending.Resume
		} else {
			a.cursor = journal.Cursor{Incarnation: page.Incarnation, From: page.Next}
		}
		if len(page.Events) == 0 && (page.Pending == nil || page.Pending.Scanned == page.Pending.At+page.Pending.Bytes) {
			return nil // caught up, or a record not complete yet
		}
	}
}

func (a *Adapter) index(line []byte) {
	if !bytes.Contains(line, []byte(`"ixmux":`)) { // ixmux's own lines; claude's have a type
		return
	}
	var n struct {
		Type        string `json:"type"`
		Ixmux       string `json:"ixmux"`
		MandateID   string `json:"mandate_id"`
		Incarnation string `json:"incarnation"`
		IsError     bool   `json:"is_error"`
	}
	if json.Unmarshal(line, &n) != nil || n.Type != "" || n.MandateID == "" {
		return
	}
	k := [2]string{n.Incarnation, n.MandateID}
	switch n.Ixmux {
	case "send":
		if a.marks[k] == nil {
			a.marks[k] = &mark{}
		}
		a.marks[k].sent = true
	case "result":
		// A result with is_error is not an answer: Deliver treats it as an error, so Status must
		// not call it received.
		// @ai:invariant Status never answers received for a mandate whose result had is_error [T:test conf:0.85 src:go/ixmux/claudestream/s6b_test.go:TestErrorResultIsNotReceived]
		if mk := a.marks[k]; mk != nil && mk.sent {
			if n.IsError {
				mk.failed = true
			} else {
				mk.answered = true
			}
		}
	}
}

// Stop stops the live child (see child.stop), waits for any reserved turn to end (a relaunch
// under way gives up, its child killed), then closes the stream journal.
func (a *Adapter) Stop() error {
	a.mu.Lock()
	if !a.stopped {
		a.stopped = true
		close(a.quit)
	}
	c := a.cur
	a.mu.Unlock()
	err := c.stop()
	a.watchers.Wait()
	a.logMu.Lock()
	a.stream.Close()
	a.logMu.Unlock()
	return err
}

// Target registers a with an engine.
func (a *Adapter) Target() deliver.Target {
	return deliver.Target{Adapter: a, Incarnation: a.Incarnation}
}
