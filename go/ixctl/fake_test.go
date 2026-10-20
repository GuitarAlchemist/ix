package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"testing"
	"time"
)

// The test binary doubles as the fake service, its grandchild and ixctl
// itself, chosen by IXCTL_TEST_ROLE.
func TestMain(m *testing.M) {
	switch os.Getenv("IXCTL_TEST_ROLE") {
	case "fake":
		os.Exit(fakeService(os.Args[1:]))
	case "grandchild":
		time.Sleep(60 * time.Second) // a safety net: never outlive a failed test for long
		os.Exit(0)
	case "main":
		os.Exit(cli(os.Args[1:], os.Stdout, os.Stderr))
	}
	os.Exit(m.Run())
}

// fakeService prints ix-stream-shaped lines: a log line, a stderr line, a
// meta event, then stats at --rate per second. It exits when stdin closes.
func fakeService(args []string) int {
	fs := flag.NewFlagSet("fake", flag.ContinueOnError)
	nodes := fs.Int("nodes", 2, "nodes in the pod")
	rate := fs.Float64("rate", 5, "stats lines per second")
	crashMS := fs.Int("crash-after-ms", 0, "exit with --crash-code after this long")
	crashCode := fs.Int("crash-code", 3, "exit code of a crash")
	spawn := fs.Bool("spawn-child", false, "start a grandchild and report its pid")
	ignoreStdin := fs.Bool("ignore-stdin", false, "keep running after stdin closes")
	hogMB := fs.Int("hog-mb", 0, "hold this much memory, touched, for the whole run")
	burnMS := fs.Int("burn-ms", 0, "spin on one core for this long at start")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	for end := time.Now().Add(time.Duration(*burnMS) * time.Millisecond); time.Now().Before(end); {
	}
	var hog []byte
	if *hogMB > 0 {
		hog = make([]byte, *hogMB<<20)
		for i := range hog {
			hog[i] = 1
		}
		defer func() { hog[0] = 0 }() // kept alive
	}
	var mu sync.Mutex
	emit := func(kind string, data map[string]any) {
		b, _ := json.Marshal(map[string]any{"kind": kind, "t": time.Now().UnixMilli(), "data": data})
		mu.Lock()
		os.Stdout.Write(append(b, '\n'))
		mu.Unlock()
	}
	fmt.Println("fake service up")
	fmt.Fprintln(os.Stderr, "fake stderr line")
	emit("meta", map[string]any{"nodes": *nodes})
	if *spawn {
		cmd := exec.Command(os.Args[0])
		cmd.Env = append(os.Environ(), "IXCTL_TEST_ROLE=grandchild")
		if err := cmd.Start(); err != nil {
			fmt.Fprintln(os.Stderr, "spawn:", err)
			return 4
		}
		emit("child", map[string]any{"pid": cmd.Process.Pid})
	}
	go func() {
		io.Copy(io.Discard, os.Stdin)
		if !*ignoreStdin {
			os.Exit(0)
		}
	}()
	if *crashMS > 0 {
		time.AfterFunc(time.Duration(*crashMS)*time.Millisecond, func() { os.Exit(*crashCode) })
	}
	time.AfterFunc(5*time.Minute, func() { os.Exit(0) })
	tick := time.NewTicker(time.Duration(float64(time.Second) / *rate))
	for n := 0; ; n++ {
		<-tick.C
		ns := make([]map[string]any, *nodes)
		for i := range ns {
			ns[i] = map[string]any{"name": fmt.Sprintf("n%d", i), "queued": 0, "cap": 4,
				"received": n, "emitted": n, "busy_ms": 0, "blocked_ms": 0, "workers": 1}
		}
		emit("stats", map[string]any{"nodes": ns, "hr_ns": hrnow()})
	}
}

type fakeSvc struct {
	name    string
	args    []string
	restart string
	max     int
	wallMs  int64 // limits.wall_ms, when > 0
	memMB   int64 // limits.memory_mb, when > 0
}

func testExe(t testing.TB) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return exe
}

// fakeSpec declares services that all run the fake.
func fakeSpec(t testing.TB, name string, peers []PeerSpec, svcs ...fakeSvc) *Spec {
	t.Helper()
	type svcJSON struct {
		Name        string            `json:"name"`
		Command     string            `json:"command"`
		Args        []string          `json:"args,omitempty"`
		Env         map[string]string `json:"env"`
		Restart     string            `json:"restart,omitempty"`
		MaxRestarts *int              `json:"max_restarts,omitempty"`
		Limits      *Limits           `json:"limits,omitempty"`
	}
	doc := struct {
		Name     string     `json:"name"`
		Services []svcJSON  `json:"services"`
		Peers    []PeerSpec `json:"peers,omitempty"`
	}{Name: name, Peers: peers, Services: []svcJSON{}}
	for _, s := range svcs {
		j := svcJSON{Name: s.name, Command: testExe(t), Args: s.args, Restart: s.restart,
			Env: map[string]string{"IXCTL_TEST_ROLE": "fake"}}
		if s.max > 0 {
			m := s.max
			j.MaxRestarts = &m
		}
		if s.wallMs > 0 || s.memMB > 0 {
			j.Limits = &Limits{WallMs: s.wallMs, MemoryMB: s.memMB}
		}
		doc.Services = append(doc.Services, j)
	}
	b, _ := json.Marshal(doc)
	spec, err := ParseSpec(b, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return spec
}

// newTestInstance serves a plane in process, in its own state dir.
func newTestInstance(t testing.TB, spec *Spec, stateDir string) *instance {
	t.Helper()
	if stateDir == "" {
		stateDir = t.TempDir()
	}
	in, err := startInstance(spec, "", "127.0.0.1:0", stateDir)
	if err != nil {
		t.Fatal(err)
	}
	in.plane.Start()
	t.Cleanup(in.close)
	return in
}

func dial(t testing.TB, in *instance) *Client {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	c, err := DialWS(ctx, in.addr, in.token)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func subscribe(t testing.TB, c *Client, prefix string, since *uint64) *frameWaiter {
	t.Helper()
	_, frames, err := c.Subscribe(context.Background(), prefix, since, 4096)
	if err != nil {
		t.Fatal(err)
	}
	return &frameWaiter{frames: frames}
}

func waitEnv(t testing.TB, w *frameWaiter, d time.Duration, what string, match func(*Envelope) bool) *Envelope {
	t.Helper()
	f, err := w.next(d, match)
	if err != nil {
		t.Fatalf("waiting for %s: %v", what, err)
	}
	return f.Env
}

func do(t testing.TB, c *Client, req Request) json.RawMessage {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	data, err := c.Do(ctx, req)
	if err != nil {
		t.Fatalf("%s: %v", req.Op, err)
	}
	return data
}

func running(names ...string) *Desired {
	return desiredOf(desiredRunning, names...)
}

func desiredOf(state string, names ...string) *Desired {
	d := &Desired{Services: map[string]DesiredService{}}
	for _, n := range names {
		d.Services[n] = DesiredService{State: state}
	}
	return d
}

func waitStatus(t testing.TB, in *instance, d time.Duration, what string, ok func(StatusReply) bool) StatusReply {
	t.Helper()
	deadline := time.Now().Add(d)
	for {
		st := in.plane.Status()
		if ok(st) {
			return st
		}
		if time.Now().After(deadline) {
			b, _ := json.Marshal(st)
			t.Fatalf("waiting for %s; status %s", what, b)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func svcStatus(st StatusReply, name string) ServiceStatus {
	for _, s := range st.Services {
		if s.Name == name {
			return s
		}
	}
	return ServiceStatus{}
}

// httpDo sends a request with the instance's token unless hdr overrides it.
func httpDo(t testing.TB, in *instance, method, path string, body any, hdr map[string]string) (int, http.Header, []byte) {
	t.Helper()
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequest(method, "http://"+in.addr+path, rd)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+in.token)
	for k, v := range hdr {
		switch {
		case k == "Host":
			req.Host = v
		case v == "":
			req.Header.Del(k)
		default:
			req.Header.Set(k, v)
		}
	}
	resp, err := loopbackHTTP.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, resp.Header, b
}

func ctlData(e *Envelope) map[string]any {
	var m map[string]any
	json.Unmarshal(e.Ev.Data, &m)
	return m
}

// shortCtx ends quickly: for draining a cursor until it has nothing more.
func shortCtx(t testing.TB) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	t.Cleanup(cancel)
	return ctx
}

// waitPid polls until pid's liveness matches want.
func waitPid(pid int, want bool, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for {
		if pidAlive(pid) == want {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(10 * time.Millisecond)
	}
}
