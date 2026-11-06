// Package server is the durable ixmux process: one Go process, no shell and no child process per
// action, that serves the journal and deliver packages over HTTP and WebSocket on loopback.
package server

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"

	"github.com/GuitarAlchemist/ix/go/ixmux/claudestream"
	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
	"github.com/GuitarAlchemist/ix/go/ixmux/journal"
)

// Named bounds. The ones a flag can lower for a test are defaults of Config.
const (
	MaxConns         = 16                     // simultaneous WebSocket connections
	MaxHTTPConns     = 64                     // simultaneous TCP connections, WebSockets included
	MaxClientMessage = 4 << 10                // bytes of one message a client sends on the stream
	MaxStreamMessage = 1 << 20                // bytes of one message the stream sends
	MaxBody          = 128 << 10              // bytes of an HTTP request body
	ReadTimeout      = 10 * time.Second       // an HTTP request's headers and body
	WriteTimeout     = 10 * time.Second       // one stream message
	RingSize         = 1024                   // events kept for a resuming client
	QueueSize        = 256                    // messages queued per connection before it is slow
	StreamPageBytes  = 64 << 10               // budget of one followed journal page
	StatePageBytes   = 256 << 10              // budget of the deliveries of one state-page message
	PagesPerTick     = 4                      // pages one followed journal publishes per poll
	PollInterval     = 200 * time.Millisecond // between two reads of a followed journal
	ShutdownGrace    = 5 * time.Second        // for HTTP handlers and WebSocket closes, beyond the engine's deadline
)

// File names in the state directory.
const (
	InfoFile    = "ixmux.json"
	TokenFile   = "ixmux.token"
	CommandFile = "commands.jsonl"
)

// Exit codes of ixmux serve.
const (
	ExitOK             = 0 // stopped cleanly: no adapter call left, log closed, lock released
	ExitError          = 1 // serving failed
	ExitUsage          = 2 // bad command line
	ExitRefused        = 3 // start refused: address not loopback, state held by another ixmux, token file
	ExitCallsInFlight  = 4 // stopped with adapter calls still running (the engine's Close refused)
	ExitStopIncomplete = 5 // stopped, but ixmux.json could not be marked stopped or the token removed
)

// Named is a name=path flag.
type Named struct{ Name, Path string }

// Config of one ixmux serve.
type Config struct {
	Addr      string
	State     string
	Journals  []Named
	Targets   []Named
	Claude    []string      // claude targets: one stream-json child each, in <state>\claude-<name>
	ClaudeBin string        // the claude executable (default: claude on PATH)
	Timeout   time.Duration // of one adapter call (deliver.Options.Timeout)
	MaxConns  int
	MaxHTTP   int           // simultaneous TCP connections (MaxHTTPConns)
	ReadLimit time.Duration // to read one request's headers and body (ReadTimeout)
	Queue     int
	Poll      time.Duration
	Version   string
}

// StartError refuses a start (ExitRefused).
type StartError struct{ Reason string }

func (e *StartError) Error() string { return "start refused: " + e.Reason }

// Info is <state>/ixmux.json.
type Info struct {
	Addr      string     `json:"addr"`
	PID       int        `json:"pid"`
	Started   time.Time  `json:"started"`
	Version   string     `json:"version"`
	ExeSHA256 string     `json:"exe_sha256"`
	Epoch     string     `json:"epoch"`
	State     string     `json:"state"` // running or stopped
	Stopped   *time.Time `json:"stopped,omitempty"`
	ExitCode  *int       `json:"exit_code,omitempty"`
}

type targetInfo struct {
	Name        string   `json:"name"`
	Kind        string   `json:"kind"`
	Path        string   `json:"path,omitempty"`
	Incarnation string   `json:"incarnation"`
	PID         int      `json:"pid,omitempty"`
	Alive       bool     `json:"alive,omitempty"`
	Tools       []string `json:"tools,omitempty"`
	Error       string   `json:"error,omitempty"`
}

// Server is one running ixmux.
type Server struct {
	cfg      Config
	eng      *deliver.Engine
	hub      *hub
	token    []byte
	key      []byte // signs the journal cursors given to clients
	info     Info
	fakes    []*FakeSession
	journals map[string]string
	claude   map[string]*claudestream.Adapter
	ln       net.Listener
	srv      *http.Server
	conns    atomic.Int64
	ws       sync.WaitGroup
	stopping chan struct{} // closed when the stop begins: stream handlers close their connections
	stopReq  chan struct{}
	stopOnce sync.Once
	cancel   context.CancelFunc // of the journal followers
	served   chan error
}

func random(n int) []byte {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic(err) // crypto/rand does not fail on supported systems
	}
	return b
}

// Start opens the state, the engine and the listener, writes the token and ixmux.json, and
// serves. It refuses (StartError) an address that is not a loopback IP, a state held by
// another ixmux, and a token file it cannot make the user's only.
func Start(cfg Config) (*Server, error) {
	if cfg.MaxConns <= 0 {
		cfg.MaxConns = MaxConns
	}
	if cfg.MaxHTTP <= 0 {
		cfg.MaxHTTP = MaxHTTPConns
	}
	if cfg.ReadLimit <= 0 {
		cfg.ReadLimit = ReadTimeout
	}
	if cfg.Queue <= 0 {
		cfg.Queue = QueueSize
	}
	if cfg.Poll <= 0 {
		cfg.Poll = PollInterval
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = deliver.DefaultTimeout
	}
	host, _, err := net.SplitHostPort(cfg.Addr)
	if ip := net.ParseIP(host); err != nil || ip == nil || !ip.IsLoopback() {
		return nil, &StartError{fmt.Sprintf("--addr %q is not a loopback IP address and port", cfg.Addr)}
	}
	if cfg.State == "" {
		return nil, &StartError{"--state is required"}
	}
	if err := os.MkdirAll(cfg.State, 0o700); err != nil {
		return nil, err
	}
	s := &Server{cfg: cfg, journals: map[string]string{}, claude: map[string]*claudestream.Adapter{}, stopping: make(chan struct{}), stopReq: make(chan struct{}), served: make(chan error, 1)}
	epoch := hex.EncodeToString(random(6))
	s.hub = newHub(epoch, RingSize)
	eng, _, err := deliver.Open(filepath.Join(cfg.State, CommandFile), deliver.Options{Timeout: cfg.Timeout, Notify: s.onTransition})
	if r := (*deliver.Refusal)(nil); errors.As(err, &r) && (r.Code == deliver.JournalLocked || r.Code == deliver.LockUnsupported) {
		return nil, &StartError{r.Error()}
	}
	if err != nil {
		return nil, err
	}
	s.eng = eng
	s.hub.seed(eng.Status()) // the deliveries replayed from the log, before any event
	tokenPath := filepath.Join(cfg.State, TokenFile)
	tokenWritten := false
	fail := func(err error) (*Server, error) {
		s.stopClaude()
		eng.Close()
		if tokenWritten {
			os.Remove(tokenPath) // a failed start leaves no token
		}
		return nil, err
	}
	for _, t := range cfg.Targets {
		f, err := NewFakeSession(t.Name, t.Path)
		if err != nil {
			return fail(err)
		}
		eng.Register(t.Name, f.Target())
		s.fakes = append(s.fakes, f)
	}
	for _, name := range cfg.Claude {
		bin := cfg.ClaudeBin
		if bin == "" {
			if bin, err = exec.LookPath("claude"); err != nil {
				return fail(&StartError{"--claude-target needs claude on PATH or --claude-bin: " + err.Error()})
			}
		}
		a, err := claudestream.Start(claudestream.Config{Name: name, Dir: filepath.Join(cfg.State, "claude-"+name), Bin: bin})
		if err != nil {
			return fail(&StartError{"claude target " + name + ": " + err.Error()})
		}
		s.claude[name] = a
		eng.Register(name, a.Target())
		s.journals["claude-"+name] = a.StreamPath()
	}
	for _, j := range cfg.Journals {
		abs, err := filepath.Abs(j.Path)
		if err != nil {
			return fail(err)
		}
		s.journals[j.Name] = abs
	}
	s.token = []byte(hex.EncodeToString(random(32)))
	s.key = random(32)
	if err := writeSecret(tokenPath, s.token); err != nil {
		return fail(&StartError{"token file: " + err.Error()})
	}
	tokenWritten = true
	ln, err := net.Listen("tcp", cfg.Addr)
	if err != nil {
		return fail(err)
	}
	ln = &limitListener{Listener: ln, slots: make(chan struct{}, cfg.MaxHTTP)}
	s.ln = ln
	exe, sum := "", ""
	if exe, err = os.Executable(); err == nil {
		if b, err := os.ReadFile(exe); err == nil {
			h := sha256.Sum256(b)
			sum = hex.EncodeToString(h[:])
		}
	}
	s.info = Info{Addr: ln.Addr().String(), PID: os.Getpid(), Started: time.Now().UTC(), Version: cfg.Version, ExeSHA256: sum, Epoch: epoch, State: "running"}
	if err := s.writeInfo(); err != nil {
		ln.Close()
		return fail(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/mandates", s.guard(s.postMandate))
	mux.HandleFunc("GET /v1/mandates/{id}", s.guard(s.getMandate))
	mux.HandleFunc("POST /v1/mandates/{id}/reconcile", s.guard(s.reconcile))
	mux.HandleFunc("GET /v1/status", s.guard(s.getStatus))
	mux.HandleFunc("GET /v1/journal/{name}", s.guard(s.getJournal))
	mux.HandleFunc("GET /v1/stream", s.guard(s.stream))
	mux.HandleFunc("POST /v1/shutdown", s.guard(s.shutdown))
	// Every request is bounded: its headers and body within ReadLimit, its answer within
	// WriteTimeout (a delivery or a reconcile extends it by its own deadline; the stream clears
	// both and bounds each message itself).
	s.srv = &http.Server{Handler: mux, ReadHeaderTimeout: cfg.ReadLimit, ReadTimeout: cfg.ReadLimit, WriteTimeout: WriteTimeout,
		IdleTimeout: cfg.ReadLimit, MaxHeaderBytes: 16 << 10}
	ctx, cancel := context.WithCancel(context.Background())
	s.cancel = cancel
	for name, path := range s.journals {
		go s.follow(ctx, name, path)
	}
	go func() { s.served <- s.srv.Serve(ln) }()
	return s, nil
}

// Addr is the address actually listened on.
func (s *Server) Addr() string { return s.info.Addr }

func (s *Server) writeInfo() error {
	b, err := json.MarshalIndent(s.info, "", "  ")
	if err != nil {
		return err
	}
	path := filepath.Join(s.cfg.State, InfoFile)
	tmp := path + ".tmp"
	fh, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	_, err = fh.Write(append(b, '\n'))
	if err == nil {
		err = fh.Sync() // the content is on disk before the rename makes it ixmux.json
	}
	if cerr := fh.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	// On Windows, replacing a file that a reader holds open fails for as long as it holds it:
	// readers of ixmux.json hold it briefly, so the rename is retried for a bounded time.
	for tries := 0; ; tries++ {
		err = os.Rename(tmp, path)
		if err == nil || tries == 100 {
			return err
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// RequestStop asks the server to stop (a signal, or POST /v1/shutdown).
func (s *Server) RequestStop() { s.stopOnce.Do(func() { close(s.stopReq) }) }

// Wait serves until a stop is requested or serving fails, then stops and returns the exit code.
func (s *Server) Wait() int {
	code := ExitOK
	select {
	case <-s.stopReq:
	case err := <-s.served:
		if !errors.Is(err, http.ErrServerClosed) {
			code = ExitError
		}
	}
	return s.stop(code)
}

// stop: no new request; stream connections closed with 1001 going away; HTTP handlers given
// ShutdownGrace; followers stopped; the engine closed (it waits up to its deadline for adapter
// calls, else calls-in-flight and ExitCallsInFlight); ixmux.json marked stopped; token removed.
// The whole stop takes at most the engine's deadline plus 2*ShutdownGrace.
func (s *Server) stop(code int) int {
	close(s.stopping)
	ctx, cancel := context.WithTimeout(context.Background(), ShutdownGrace)
	s.srv.Shutdown(ctx) // stream connections are hijacked: Shutdown does not wait for them
	cancel()
	wsDone := make(chan struct{})
	go func() { s.ws.Wait(); close(wsDone) }()
	select {
	case <-wsDone:
	case <-time.After(ShutdownGrace):
	}
	// The claude children are stopped before the engine is closed: closing stdin ends their turn
	// or kills them, so no adapter call is left running (Close is not relied on for that).
	s.stopClaude()
	s.cancel()
	if err := s.eng.Close(); err != nil {
		if code == ExitOK {
			code = ExitError
			if r := (*deliver.Refusal)(nil); errors.As(err, &r) && r.Code == deliver.CallsInFlight {
				code = ExitCallsInFlight
			}
		}
	}
	// The token is removed first, so ixmux.json records the final exit code. Neither failure is
	// left silent: the stop then exits ExitStopIncomplete (unless it already failed), with the
	// error on stderr; the error names the file, never the token.
	// @ai:invariant a stop exits 0 only if ixmux.json says stopped and the token is removed [T:test conf:0.85 src:go/ixmux/cmd/ixmux/s3b_windows_test.go:TestStopWithTheTokenHeldIsNotZero]
	if err := os.Remove(filepath.Join(s.cfg.State, TokenFile)); err != nil && !errors.Is(err, os.ErrNotExist) {
		fmt.Fprintln(os.Stderr, "ixmux: the token file was not removed:", err)
		if code == ExitOK {
			code = ExitStopIncomplete
		}
	}
	now := time.Now().UTC()
	s.info.State, s.info.Stopped, s.info.ExitCode = "stopped", &now, &code
	if err := s.writeInfo(); err != nil {
		fmt.Fprintln(os.Stderr, "ixmux: ixmux.json was not marked stopped:", err)
		if code == ExitOK {
			code = ExitStopIncomplete
		}
	}
	return code
}

func (s *Server) stopClaude() {
	var wg sync.WaitGroup
	for _, a := range s.claude {
		wg.Add(1)
		go func() { defer wg.Done(); a.Stop() }()
	}
	wg.Wait()
}

// --- guard -----------------------------------------------------------------------------------

type apiError struct {
	Error  string `json:"error"`
	Detail string `json:"detail,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func refuse(w http.ResponseWriter, status int, code, detail string) {
	writeJSON(w, status, apiError{code, detail})
}

// loopbackHost: the Host header names a loopback IP or localhost, so a page of another site,
// resolved to 127.0.0.1 by DNS rebinding, is refused.
func loopbackHost(hostport string) bool {
	host := hostport
	if h, _, err := net.SplitHostPort(hostport); err == nil {
		host = h
	}
	host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// guard: Host on loopback (403), no Origin (403: no browser page may call ixmux), the Bearer
// token (401), then a bounded body.
// @ai:invariant every route needs a loopback Host, no Origin and the Bearer token [T:test conf:0.85 src:go/ixmux/cmd/ixmux/integration_test.go:TestSecurity]
func (s *Server) guard(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case !loopbackHost(r.Host):
			refuse(w, http.StatusForbidden, "host-not-loopback", "")
			return
		case r.Header.Get("Origin") != "":
			refuse(w, http.StatusForbidden, "origin-refused", "")
			return
		}
		got, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || subtle.ConstantTimeCompare([]byte(got), s.token) != 1 {
			refuse(w, http.StatusUnauthorized, "unauthorized", "")
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, MaxBody)
		h(w, r)
	}
}

func refusalStatus(code string) int {
	switch code {
	case deliver.BadMandate:
		return http.StatusBadRequest
	case deliver.TextTooLong:
		return http.StatusRequestEntityTooLarge
	case deliver.UnknownTarget, deliver.UnknownMandate:
		return http.StatusNotFound
	case deliver.Busy, deliver.EngineClosed, deliver.LogBroken, deliver.CallsInFlight:
		return http.StatusServiceUnavailable
	case deliver.ReconcileUnsupported:
		return http.StatusNotImplemented
	}
	return http.StatusConflict
}

func writeErr(w http.ResponseWriter, err error) {
	if r := (*deliver.Refusal)(nil); errors.As(err, &r) {
		refuse(w, refusalStatus(r.Code), r.Code, r.Detail)
		return
	}
	refuse(w, http.StatusInternalServerError, "internal", err.Error())
}

// --- mandates --------------------------------------------------------------------------------

type mandateRequest struct {
	MandateID   string `json:"mandate_id"`
	Target      string `json:"target"`
	Incarnation string `json:"incarnation"`
	Text        string `json:"text"`
}

// postMandate delivers at most once. The delivery does not depend on the HTTP connection
// (WithoutCancel): a client that hangs up mid-request finds it done, replayed, when it sends the
// same mandate again.
func (s *Server) postMandate(w http.ResponseWriter, r *http.Request) {
	var m mandateRequest
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		if me := (*http.MaxBytesError)(nil); errors.As(err, &me) {
			refuse(w, http.StatusRequestEntityTooLarge, "body-too-large", fmt.Sprintf("bound %d bytes", MaxBody))
			return
		}
		refuse(w, http.StatusBadRequest, "bad-request", err.Error())
		return
	}
	s.answerWithin(w)
	res, err := s.eng.Deliver(context.WithoutCancel(r.Context()), deliver.Mandate{MandateID: m.MandateID, Target: m.Target, Incarnation: m.Incarnation, Text: m.Text})
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// answerWithin gives a request that waits for the engine its delivery deadline, plus WriteTimeout
// for the answer itself.
func (s *Server) answerWithin(w http.ResponseWriter) {
	http.NewResponseController(w).SetWriteDeadline(time.Now().Add(s.cfg.Timeout + WriteTimeout))
}

func (s *Server) entry(id string) (deliver.Entry, bool) {
	for _, e := range s.eng.Status() {
		if e.MandateID == id {
			return e, true
		}
	}
	return deliver.Entry{}, false
}

func (s *Server) getMandate(w http.ResponseWriter, r *http.Request) {
	e, ok := s.entry(r.PathValue("id"))
	if !ok {
		refuse(w, http.StatusNotFound, deliver.UnknownMandate, "")
		return
	}
	writeJSON(w, http.StatusOK, e)
}

func (s *Server) reconcile(w http.ResponseWriter, r *http.Request) {
	e, ok := s.entry(r.PathValue("id"))
	if !ok {
		refuse(w, http.StatusNotFound, deliver.UnknownMandate, "")
		return
	}
	s.answerWithin(w)
	res, err := s.eng.Reconcile(context.WithoutCancel(r.Context()), e.Target, e.MandateID)
	if err != nil {
		writeErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// --- status ----------------------------------------------------------------------------------

type statusView struct {
	Version     string            `json:"version"`
	Started     time.Time         `json:"started"`
	Epoch       string            `json:"epoch"`
	Seq         uint64            `json:"seq"`
	Targets     []targetInfo      `json:"targets"`
	Journals    map[string]string `json:"journals"`
	Deliveries  []deliver.Entry   `json:"deliveries"`
	Connections int64             `json:"connections"`
	Bounds      map[string]any    `json:"bounds"`
}

func (s *Server) status(epoch string, seq uint64) any {
	targets := []targetInfo{}
	for _, f := range s.fakes { // read now: the file at the path may have been replaced
		ti := targetInfo{Name: f.Name, Kind: "fake", Path: f.Path}
		if inc, err := f.Incarnation(); err != nil {
			ti.Error = err.Error()
		} else {
			ti.Incarnation = inc
		}
		targets = append(targets, ti)
	}
	for name, a := range s.claude {
		ti := targetInfo{Name: name, Kind: "claude", PID: a.PID(), Alive: a.Alive(), Tools: a.Tools()}
		if inc, err := a.Incarnation(); err != nil {
			ti.Error = err.Error()
		} else {
			ti.Incarnation = inc
		}
		targets = append(targets, ti)
	}
	return statusView{Version: s.info.Version, Started: s.info.Started, Epoch: epoch, Seq: seq, Targets: targets,
		Journals: s.journals, Deliveries: s.eng.Status(), Connections: s.conns.Load(),
		Bounds: map[string]any{
			"max_conns": s.cfg.MaxConns, "max_http_conns": s.cfg.MaxHTTP, "max_client_message": MaxClientMessage, "max_stream_message": MaxStreamMessage,
			"max_body": MaxBody, "read_timeout": s.cfg.ReadLimit.String(), "write_timeout": WriteTimeout.String(),
			"ring_size": RingSize, "queue": s.cfg.Queue, "delivery_timeout": s.cfg.Timeout.String(),
			"max_in_flight": deliver.MaxInFlight, "max_text": deliver.MaxText,
		}}
}

// stateEnd ends the state sent after a gap: how many state-page messages and deliveries it held,
// and the targets and journals (the bounds are in /v1/status).
type stateEnd struct {
	Pages      int               `json:"pages"`
	Deliveries int               `json:"deliveries"`
	Targets    []targetInfo      `json:"targets"`
	Journals   map[string]string `json:"journals"`
}

func (s *Server) getStatus(w http.ResponseWriter, r *http.Request) {
	epoch, seq := s.hub.current()
	writeJSON(w, http.StatusOK, s.status(epoch, seq))
}

// --- journals --------------------------------------------------------------------------------

type wireEvent struct {
	Offset  int64  `json:"offset"`
	End     int64  `json:"end"`
	Line    string `json:"line,omitempty"`
	Blank   bool   `json:"blank,omitempty"`
	Invalid string `json:"invalid,omitempty"`
}

type wirePending struct {
	At      int64 `json:"at"`
	Bytes   int64 `json:"bytes"`
	Scanned int64 `json:"scanned"`
}

type journalPage struct {
	Journal       string       `json:"journal"`
	Incarnation   string       `json:"incarnation"`
	From          int64        `json:"from"`
	Events        []wireEvent  `json:"events"`
	LinesOmitted  bool         `json:"lines_omitted,omitempty"`
	Next          int64        `json:"next"`
	Size          int64        `json:"size"`
	Pending       *wirePending `json:"pending,omitempty"`
	AtEnd         bool         `json:"at_end,omitempty"`
	Provisional   bool         `json:"provisional,omitempty"`
	PromotedFrom  string       `json:"promoted_from,omitempty"`
	Cursor        string       `json:"cursor,omitempty"`
	CursorIgnored bool         `json:"cursor_ignored,omitempty"`
}

func wire(name string, from int64, p journal.Page, lines bool) journalPage {
	out := journalPage{Journal: name, Incarnation: p.Incarnation, From: from, Events: make([]wireEvent, 0, len(p.Events)), Next: p.Next,
		Size: p.Size, AtEnd: p.AtEnd, Provisional: p.Provisional, PromotedFrom: p.PromotedFrom, LinesOmitted: !lines}
	for _, e := range p.Events {
		we := wireEvent{Offset: e.Offset, End: e.End, Blank: e.Blank, Invalid: e.Invalid}
		if lines {
			we.Line = string(e.Line)
		}
		out.Events = append(out.Events, we)
	}
	if p.Pending != nil {
		out.Pending = &wirePending{p.Pending.At, p.Pending.Bytes, p.Pending.Scanned}
	}
	return out
}

// next is where a reader of p continues: its resume cursor while a record is pending.
func next(p journal.Page) journal.Cursor {
	if p.Pending != nil {
		return p.Pending.Resume
	}
	return journal.Cursor{Incarnation: p.Incarnation, From: p.Next}
}

type signedCursor struct {
	Journal string         `json:"j"`
	Cursor  journal.Cursor `json:"c"`
}

// sign makes the cursor given to a client: base64url(JSON) "." base64url(HMAC-SHA256), under a
// key of this process. Its Scanned hint is trusted only under a valid signature for the same
// journal; any other cursor is ignored and the read scans again.
// @ai:invariant a resume hint is used only when signed by this process for this journal [T:test conf:0.85 src:go/ixmux/cmd/ixmux/integration_test.go:TestSecurity]
func (s *Server) sign(name string, c journal.Cursor) string {
	b, _ := json.Marshal(signedCursor{name, c})
	p := base64.RawURLEncoding.EncodeToString(b)
	m := hmac.New(sha256.New, s.key)
	m.Write([]byte(p))
	return p + "." + base64.RawURLEncoding.EncodeToString(m.Sum(nil))
}

func (s *Server) open(name, raw string) (journal.Cursor, bool) {
	p, sig, ok := strings.Cut(raw, ".")
	if !ok {
		return journal.Cursor{}, false
	}
	got, err := base64.RawURLEncoding.DecodeString(sig)
	if err != nil {
		return journal.Cursor{}, false
	}
	m := hmac.New(sha256.New, s.key)
	m.Write([]byte(p))
	if !hmac.Equal(got, m.Sum(nil)) {
		return journal.Cursor{}, false
	}
	b, err := base64.RawURLEncoding.DecodeString(p)
	var sc signedCursor
	if err != nil || json.Unmarshal(b, &sc) != nil || sc.Journal != name {
		return journal.Cursor{}, false
	}
	return sc.Cursor, true
}

// getJournal returns one page of journal.ReadCursor. The checkpoint is cursor (if validly
// signed), else from and incarnation (the journal's current one when empty).
func (s *Server) getJournal(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	path, ok := s.journals[name]
	if !ok {
		refuse(w, http.StatusNotFound, "unknown-journal", name)
		return
	}
	q := r.URL.Query()
	var c journal.Cursor
	var err error
	if v := q.Get("from"); v != "" {
		if c.From, err = strconv.ParseInt(v, 10, 64); err != nil {
			refuse(w, http.StatusBadRequest, "bad-request", "from: "+err.Error())
			return
		}
	}
	var budget int64
	if v := q.Get("max_bytes"); v != "" {
		if budget, err = strconv.ParseInt(v, 10, 64); err != nil {
			refuse(w, http.StatusBadRequest, "bad-request", "max_bytes: "+err.Error())
			return
		}
	}
	c.Incarnation = q.Get("incarnation")
	ignored := false
	if raw := q.Get("cursor"); raw != "" {
		if sc, ok := s.open(name, raw); ok {
			c = sc
		} else {
			ignored = true
		}
	}
	if c.Incarnation == "" {
		if c.Incarnation, _, err = journal.Identify(path); err != nil {
			writeJournalErr(w, err)
			return
		}
	}
	page, err := journal.ReadCursor(path, c, budget)
	if err != nil {
		writeJournalErr(w, err)
		return
	}
	out := wire(name, c.From, page, true)
	out.Cursor, out.CursorIgnored = s.sign(name, next(page)), ignored
	writeJSON(w, http.StatusOK, out)
}

func writeJournalErr(w http.ResponseWriter, err error) {
	if r := (*journal.Refusal)(nil); errors.As(err, &r) {
		refuse(w, http.StatusConflict, r.Code, r.Detail)
		return
	}
	if errors.Is(err, os.ErrNotExist) {
		refuse(w, http.StatusNotFound, "journal-missing", "")
		return
	}
	refuse(w, http.StatusInternalServerError, "internal", err.Error())
}

// follow reads a followed journal every poll, at most PagesPerTick pages, and publishes each
// page that has events. A refused checkpoint (another incarnation, a truncation) restarts it at
// 0 and publishes journal-reset.
func (s *Server) follow(ctx context.Context, name, path string) {
	t := time.NewTicker(s.cfg.Poll)
	defer t.Stop()
	var c journal.Cursor
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		for range PagesPerTick {
			if c.Incarnation == "" {
				inc, _, err := journal.Identify(path)
				if err != nil {
					break
				}
				c = journal.Cursor{Incarnation: inc}
			}
			page, err := journal.ReadCursor(path, c, StreamPageBytes)
			if r := (*journal.Refusal)(nil); errors.As(err, &r) {
				s.hub.publish("journal-reset", map[string]string{"journal": name, "reason": r.Code})
				c = journal.Cursor{}
				break
			}
			if err != nil {
				break
			}
			if len(page.Events) > 0 && !s.hub.publish("journal", wire(name, c.From, page, true)) {
				s.hub.publish("journal", wire(name, c.From, page, false))
			}
			c = next(page)
			if len(page.Events) == 0 && (page.Pending == nil || page.Pending.Scanned == page.Pending.At+page.Pending.Bytes) {
				break // caught up, or a record not complete yet
			}
		}
	}
}

// --- stream ----------------------------------------------------------------------------------

func (s *Server) onTransition(t deliver.Transition) {
	if testHookBeforePublish != nil {
		testHookBeforePublish(t)
	}
	s.hub.publish("mandate", t)
}

// testHookBeforePublish, set by a test only, runs once a transition is applied by the engine and
// before the hub numbers it.
var testHookBeforePublish func(deliver.Transition)

// stream serves GET /v1/stream[?since=<seq>&epoch=<epoch>]: hello, then the missed events or a
// gap and the state, then live events. A connection over MaxConns is refused (503); a client
// message over MaxClientMessage closes it (1009); a full queue sends a gap and closes it (1008);
// a stop closes it (1001).
func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	if s.conns.Add(1) > int64(s.cfg.MaxConns) {
		s.conns.Add(-1)
		refuse(w, http.StatusServiceUnavailable, "too-many-connections", fmt.Sprintf("bound %d", s.cfg.MaxConns))
		return
	}
	defer s.conns.Add(-1)
	q := r.URL.Query()
	resume := q.Has("since")
	since, err := strconv.ParseUint(q.Get("since"), 10, 64)
	if resume && err != nil {
		refuse(w, http.StatusBadRequest, "bad-request", "since: "+err.Error())
		return
	}
	select {
	case <-s.stopping:
		refuse(w, http.StatusServiceUnavailable, "stopping", "")
		return
	default:
	}
	// The connection outlives the request: the server's read and write deadlines are cleared, and
	// each stream message has its own WriteTimeout.
	rc := http.NewResponseController(w)
	rc.SetReadDeadline(time.Time{})
	rc.SetWriteDeadline(time.Time{})
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{CompressionMode: websocket.CompressionDisabled})
	if err != nil {
		return
	}
	s.ws.Add(1)
	defer s.ws.Done()
	defer c.CloseNow()
	c.SetReadLimit(MaxClientMessage)
	sub, first, snap := s.hub.subscribe(resume, since, q.Get("epoch"), s.cfg.Queue)
	if snap != nil { // built outside the hub's lock
		first = append(first, s.hub.pages(snap, func(pages, deliveries int) any {
			return stateEnd{Pages: pages, Deliveries: deliveries, Targets: s.status("", 0).(statusView).Targets, Journals: s.journals}
		})...)
	}
	defer s.hub.unsubscribe(sub)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { // clients send nothing the stream uses; reading handles close and the read limit
		for {
			if _, _, err := c.Read(ctx); err != nil {
				cancel()
				return
			}
		}
	}()
	write := func(b []byte) bool {
		wctx, wcancel := context.WithTimeout(ctx, WriteTimeout)
		defer wcancel()
		return c.Write(wctx, websocket.MessageText, b) == nil
	}
	for _, b := range first {
		if !write(b) {
			return
		}
	}
	for {
		select {
		case b := <-sub.q:
			if !write(b) {
				return
			}
		case <-sub.over:
			for drained := false; !drained; {
				select {
				case b := <-sub.q:
					if !write(b) {
						return
					}
				default:
					drained = true
				}
			}
			epoch, seq := s.hub.current()
			b, _ := json.Marshal(Message{Type: "gap", Epoch: epoch, Seq: seq, Data: Gap{Reason: "slow", LastSeq: sub.lastSeq(s.hub)}})
			write(b)
			c.Close(websocket.StatusPolicyViolation, "slow consumer: resume with since")
			return
		case <-s.stopping:
			c.Close(websocket.StatusGoingAway, "ixmux is stopping")
			return
		case <-ctx.Done():
			return
		}
	}
}

func (sb *sub) lastSeq(h *hub) uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return sb.last
}

// --- shutdown --------------------------------------------------------------------------------

func (s *Server) shutdown(w http.ResponseWriter, r *http.Request) {
	io.Copy(io.Discard, r.Body)
	writeJSON(w, http.StatusAccepted, map[string]string{"state": "stopping"})
	s.RequestStop()
}
