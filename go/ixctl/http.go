package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const version = "0.1.0"

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
	m.HandleFunc("GET /v1/status", s.op(func(*http.Request) (Request, error) { return Request{Op: "status"}, nil }))
	m.HandleFunc("POST /v1/plan", s.op(desiredOp("plan")))
	m.HandleFunc("POST /v1/apply", s.op(desiredOp("apply")))
	m.HandleFunc("POST /v1/stop", s.op(func(r *http.Request) (Request, error) {
		var body struct {
			Service string `json:"service"`
			Peer    string `json:"peer"`
		}
		err := decodeBody(r, &body)
		return Request{Op: "stop", Service: body.Service, Peer: body.Peer}, err
	}))
	m.HandleFunc("POST /v1/shutdown", s.op(func(*http.Request) (Request, error) { return Request{Op: "shutdown"}, nil }))
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
		"ok": true, "name": s.name, "version": version, "uptime_ms": time.Since(s.started).Milliseconds()})
}

func desiredOp(op string) func(*http.Request) (Request, error) {
	return func(r *http.Request) (Request, error) {
		var d Desired
		if err := decodeBody(r, &d); err != nil {
			return Request{}, err
		}
		return Request{Op: op, Desired: &d, Peer: d.Peer}, nil
	}
}

func (s *server) op(build func(*http.Request) (Request, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
	dec := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("bad body: %w", err)
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

// events is the SSE stream: id = seq, event = ev.kind, data = envelope.
// Last-Event-ID works like since. A slow reader that falls off the ring gets
// a gap event and the stream ends.
func (s *server) events(w http.ResponseWriter, r *http.Request) {
	prefix := r.URL.Query().Get("ch")
	var since *uint64
	for _, v := range []string{r.Header.Get("Last-Event-ID"), r.URL.Query().Get("since")} {
		if v == "" {
			continue
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
	cur := s.b.Subscribe(prefix, since)
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
