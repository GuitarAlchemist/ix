package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// version is what /health, `ixctl version` and `probe.kinds` report. A release build sets it
// from the tag: `go build -ldflags "-X main.version=0.2.0"`.
var version = "0.1.0"

// server is the HTTP face of a plane: JSON ops, SSE and the WebSocket.
// Every request passes the same checks first, whatever its route.
type server struct {
	b       Broker
	name    string
	token   string
	hosts   map[string]bool
	origins map[string]bool
	started time.Time
	mux     *http.ServeMux
	ops     atomic.Int64 // JSON ops served (any outcome); /health shows it, so a test can see a client poll

	ctx    context.Context // cancelled at shutdown: ends SSE streams and WebSockets
	cancel context.CancelFunc
}

func newServer(b Broker, name, token string, addr *net.TCPAddr) *server {
	port := strconv.Itoa(addr.Port)
	ctx, cancel := context.WithCancel(context.Background())
	s := &server{
		b: b, name: name, token: token, started: time.Now(), ctx: ctx, cancel: cancel,
		hosts: map[string]bool{
			"127.0.0.1:" + port: true, "localhost:" + port: true, "[::1]:" + port: true,
			strings.ToLower(addr.String()): true,
		},
		origins: map[string]bool{"http://127.0.0.1:" + port: true, "http://localhost:" + port: true},
	}
	m := http.NewServeMux()
	m.HandleFunc("GET /health", s.health)
	m.HandleFunc("GET /v1/status", s.op(func(r *http.Request) (Request, error) {
		peer, err := peerQuery(r)
		return Request{Op: "status", Peer: peer}, err
	}))
	m.HandleFunc("POST /v1/plan", s.op(desiredOp("plan")))
	m.HandleFunc("POST /v1/apply", s.op(desiredOp("apply")))
	m.HandleFunc("POST /v1/stop", s.op(func(r *http.Request) (Request, error) {
		var body struct {
			Service string          `json:"service"`
			Peer    json.RawMessage `json:"peer"`
		}
		if err := decodeBody(r, &body); err != nil {
			return Request{}, err
		}
		peer, err := peerField(body.Peer)
		return Request{Op: "stop", Service: body.Service, Peer: peer}, err
	}))
	m.HandleFunc("POST /v1/shutdown", s.op(func(*http.Request) (Request, error) { return Request{Op: "shutdown"}, nil }))
	// Probes: the kinds and limits, the tree, and one route per node op. `?peer=`
	// beside a create, a `peer` member beside a node name.
	m.HandleFunc("GET /v1/probes", s.op(func(r *http.Request) (Request, error) {
		peer, err := peerQuery(r)
		return Request{Op: "probe.kinds", Peer: peer}, err
	}))
	m.HandleFunc("GET /v1/probe", s.op(func(r *http.Request) (Request, error) {
		peer, err := peerQuery(r)
		return Request{Op: "probe.list", Peer: peer}, err
	}))
	m.HandleFunc("POST /v1/probe/create", s.op(func(r *http.Request) (Request, error) {
		peer, err := peerQuery(r)
		if err != nil {
			return Request{}, err
		}
		var spec NodeSpec
		if err := decodeBody(r, &spec); err != nil {
			return Request{}, err
		}
		return Request{Op: "probe.create", Probe: &spec, Peer: peer}, nil
	}))
	for _, op := range []string{"start", "stop", "touch", "remove"} {
		op := op
		m.HandleFunc("POST /v1/probe/"+op, s.op(func(r *http.Request) (Request, error) {
			var body struct {
				Node string          `json:"node"`
				Peer json.RawMessage `json:"peer"`
			}
			if err := decodeBody(r, &body); err != nil {
				return Request{}, err
			}
			peer, err := peerField(body.Peer)
			return Request{Op: "probe." + op, Node: body.Node, Peer: peer}, err
		}))
	}
	m.HandleFunc("GET /v1/events", s.events)
	m.HandleFunc("GET /v1/ws", s.ws)
	s.mux = m
	return s
}

// ServeHTTP refuses foreign Hosts (DNS rebinding), foreign Origins and
// missing or wrong tokens, before any route runs. No CORS header is ever sent.
func (s *server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.hosts[strings.ToLower(r.Host)] {
		httpError(w, 403, "forbidden host")
		return
	}
	if o, ok := r.Header["Origin"]; ok && (len(o) != 1 || !s.origins[o[0]]) {
		httpError(w, 403, "forbidden origin")
		return
	}
	if !(r.Method == http.MethodGet && r.URL.Path == "/health") && !s.authorized(r) {
		httpError(w, 401, "missing or wrong token")
		return
	}
	s.mux.ServeHTTP(w, r)
}

func (s *server) authorized(r *http.Request) bool {
	h := r.Header.Get("Authorization")
	tok, ok := strings.CutPrefix(h, "Bearer ")
	return ok && subtle.ConstantTimeCompare([]byte(tok), []byte(s.token)) == 1
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, 200, map[string]any{
		"ok": true, "name": s.name, "version": version, "epoch": s.b.Epoch(),
		"uptime_ms": time.Since(s.started).Milliseconds(), "ops": s.ops.Load()})
}

// errEmptyPeer: a peer that is named must name someone. An empty one (or a
// null, or a number) is not "local": the op would run on the wrong instance,
// and on every transport alike it is refused instead.
var errEmptyPeer = badRequest("peer must not be empty")

// peerQuery is `?peer=` as the op's peer: absent means local, present means
// that peer, and present but empty (or given twice) is refused. The query is
// parsed strictly: `URL.Query()` drops a malformed pair without a word, and a
// dropped `peer` would read as "local".
func peerQuery(r *http.Request) (string, error) {
	q, err := url.ParseQuery(r.URL.RawQuery)
	if err != nil {
		return "", badRequest("bad query: %v", err)
	}
	v, present := q["peer"]
	switch {
	case !present:
		return "", nil
	case len(v) != 1 || v[0] == "":
		return "", errEmptyPeer
	}
	return v[0], nil
}

// peerField is a JSON `peer` member as the op's peer: absent means local,
// a non-empty string means that peer, and anything else present (an empty
// string, null, another type) is refused rather than read as "local".
func peerField(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var s string
	if json.Unmarshal(raw, &s) != nil || s == "" {
		return "", errEmptyPeer
	}
	return s, nil
}

// desiredOp reads a desired-state document from the body; `?peer=` beside it
// says which peer the op is for, the way the WebSocket frame's peer does.
func desiredOp(op string) func(*http.Request) (Request, error) {
	return func(r *http.Request) (Request, error) {
		peer, err := peerQuery(r)
		if err != nil {
			return Request{}, err
		}
		var d Desired
		if err := decodeBody(r, &d); err != nil {
			return Request{}, err
		}
		return Request{Op: op, Desired: &d, Peer: peer}, nil
	}
}

func (s *server) op(build func(*http.Request) (Request, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.ops.Add(1)
		req, err := build(r)
		if err != nil {
			httpError(w, 400, err.Error())
			return
		}
		data, err := s.b.Request(r.Context(), req)
		if err != nil {
			httpError(w, statusOf(err), err.Error())
			return
		}
		writeJSON(w, 200, data)
	}
}

func decodeBody(r *http.Request, v any) error {
	return decodeStrict(io.LimitReader(r.Body, 1<<20), v)
}

// decodeStrict is the one decoder for what clients send, on HTTP and on
// ixctl.v1 alike: one JSON value, no unknown field, nothing after it.
func decodeStrict(r io.Reader, v any) error {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return badRequest("bad body: %v", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return badRequest("bad body: data after the JSON value")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func httpError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": msg})
}

// events is the SSE stream: id = epoch:seq, event = ev.kind, data = envelope.
// Last-Event-ID works like since, and may carry the epoch the same way
// (`epoch:seq`; a bare seq is read with the `epoch` query parameter, if any).
// A slow reader that falls off the ring gets a gap event and the stream ends.
func (s *server) events(w http.ResponseWriter, r *http.Request) {
	prefix := r.URL.Query().Get("ch")
	epoch := r.URL.Query().Get("epoch")
	var since *uint64
	for _, v := range []string{r.Header.Get("Last-Event-ID"), r.URL.Query().Get("since")} {
		if v == "" {
			continue
		}
		if e, seq, ok := strings.Cut(v, ":"); ok {
			epoch, v = e, seq
		}
		n, err := strconv.ParseUint(v, 10, 64)
		if err != nil {
			httpError(w, 400, "bad since: "+v)
			return
		}
		since = &n
		break
	}
	fl, ok := w.(http.Flusher)
	if !ok {
		httpError(w, 500, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	fl.Flush()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	stop := context.AfterFunc(s.ctx, cancel)
	defer stop()
	cur := s.b.Subscribe(prefix, since, epoch)
	for {
		wctx, wcancel := context.WithTimeout(ctx, 15*time.Second)
		envs, gap, err := cur.Next(wctx, 256)
		wcancel()
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			fmt.Fprint(w, ": ping\n\n")
			fl.Flush()
			continue
		}
		if gap != nil {
			b, _ := json.Marshal(gap)
			fmt.Fprintf(w, "event: gap\ndata: %s\n\n", b)
			fl.Flush()
			if gap.Reason == "overflow" {
				return
			}
			continue
		}
		var sb strings.Builder
		for _, env := range envs {
			sb.WriteString("id: ")
			sb.WriteString(env.Epoch)
			sb.WriteByte(':')
			sb.WriteString(strconv.FormatUint(env.Seq, 10))
			sb.WriteString("\nevent: ")
			sb.WriteString(strings.NewReplacer("\r", " ", "\n", " ").Replace(env.Ev.Kind))
			sb.WriteString("\ndata: ")
			sb.Write(env.Raw())
			sb.WriteString("\n\n")
		}
		if _, err := w.Write([]byte(sb.String())); err != nil {
			return
		}
		fl.Flush()
	}
}

// checkLoopback refuses any address that is not a loopback IP and port.
func checkLoopback(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return fmt.Errorf("address %q: %w", addr, err)
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("address %q is not a loopback IP (127.0.0.0/8 or ::1)", addr)
	}
	return nil
}
