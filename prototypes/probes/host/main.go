// Command host runs every probe under -probes as its own process. When a
// probe's Go source changes it rebuilds the probe and swaps the running
// process, without restarting itself. Whatever the probes print, one JSON
// object per line, it appends to out/probes.jsonl for DuckDB to read. Its own
// events (builds, starts, stops, bad lines) go to out/host-events.jsonl.
//
// A probe that no longer builds keeps its last good version running.
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// A probe process, and the source hash it was built from.
type proc struct {
	hash    string
	exe     string
	cmd     *exec.Cmd
	started time.Time
	done    chan struct{}
}

func (p *proc) alive() bool {
	select {
	case <-p.done:
		return false
	default:
		return true
	}
}

type host struct {
	probesDir string
	binDir    string
	interval  time.Duration
	records   chan map[string]any
	events    chan map[string]any
	procs     map[string]*proc
	// The source hash whose build failed, per probe, so a broken source is
	// built once rather than on every poll.
	failed map[string]string
}

// now is fixed width, like the probes' timestamps, so DuckDB reads it as a
// timestamp and the strings sort in time order.
func now() string { return time.Now().UTC().Format("2006-01-02T15:04:05.000000Z07:00") }

func (h *host) event(kind, probe, build string, extra map[string]any) {
	e := map[string]any{"ts": now(), "event": kind}
	if probe != "" {
		e["probe"] = probe
	}
	if build != "" {
		e["build"] = build
	}
	for k, v := range extra {
		e[k] = v
	}
	h.events <- e
}

// sourceHash hashes the names and contents of the .go files under dir.
// Contents, not modification times: an editor that rewrites a file unchanged
// does not trigger a rebuild.
func sourceHash(dir string) (string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(p, ".go") {
			files = append(files, p)
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	if len(files) == 0 {
		return "", fmt.Errorf("no .go files in %s", dir)
	}
	sort.Strings(files)
	h := sha256.New()
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			return "", err
		}
		rel, _ := filepath.Rel(dir, f)
		fmt.Fprintf(h, "%s\x00%d\x00", filepath.ToSlash(rel), len(b))
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil))[:12], nil
}

// build compiles one probe to an executable named after its source hash.
// Windows locks a running executable, so each build gets a path of its own.
func (h *host) build(name, hash string) (string, error) {
	exe := filepath.Join(h.binDir, fmt.Sprintf("%s-%s.exe", name, hash))
	pkg := "./" + filepath.ToSlash(filepath.Join(h.probesDir, name))
	out, err := exec.Command("go", "build", "-o", exe, pkg).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%v: %s", err, strings.TrimSpace(string(out)))
	}
	return exe, nil
}

func (h *host) start(name, hash, exe string) error {
	cmd := exec.Command(exe)
	cmd.Env = append(os.Environ(), fmt.Sprintf("PROBE_INTERVAL_MS=%d", h.interval.Milliseconds()))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	p := &proc{hash: hash, exe: exe, cmd: cmd, started: time.Now(), done: make(chan struct{})}
	h.procs[name] = p
	h.event("started", name, hash, map[string]any{"pid": cmd.Process.Pid})
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); h.pump(name, hash, stdout) }()
	go func() { defer wg.Done(); h.drain(name, hash, stderr) }()
	// Wait closes the pipes, so it runs only once both readers are done.
	go func() {
		wg.Wait()
		err := cmd.Wait()
		extra := map[string]any{}
		if err != nil {
			extra["error"] = err.Error()
		}
		h.event("exited", name, hash, extra)
		close(p.done)
	}()
	return nil
}

func (h *host) stop(name string, p *proc, why string) {
	_ = p.cmd.Process.Kill()
	select {
	case <-p.done:
	case <-time.After(5 * time.Second):
	}
	_ = os.Remove(p.exe)
	h.event("stopped", name, p.hash, map[string]any{"why": why})
}

// pump turns each line a probe prints into a record. A line must be a JSON
// object with a string "metric" and a number "value".
func (h *host) pump(name, hash string, r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		var rec map[string]any
		err := json.Unmarshal(sc.Bytes(), &rec)
		_, isMetric := rec["metric"].(string)
		_, isValue := rec["value"].(float64)
		if err != nil || !isMetric || !isValue {
			line := sc.Text()
			if len(line) > 200 {
				line = line[:200]
			}
			h.event("bad_line", name, hash, map[string]any{"line": line})
			continue
		}
		rec["probe"] = name
		rec["build"] = hash
		if _, ok := rec["ts"].(string); !ok {
			rec["ts"] = now()
		}
		h.records <- rec
	}
}

func (h *host) drain(name, hash string, r io.Reader) {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		h.event("stderr", name, hash, map[string]any{"line": sc.Text()})
	}
}

// reconcile brings the running probes in line with the probes directory:
// builds and starts new probes, swaps changed ones, restarts crashed ones and
// stops removed ones.
func (h *host) reconcile() {
	entries, err := os.ReadDir(h.probesDir)
	if err != nil {
		h.event("scan_failed", "", "", map[string]any{"error": err.Error()})
		return
	}
	seen := map[string]bool{}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		seen[name] = true
		hash, err := sourceHash(filepath.Join(h.probesDir, name))
		if err != nil {
			continue
		}
		cur := h.procs[name]
		if cur != nil && cur.hash == hash {
			// Same source: restart only a crashed probe, at most every 2 s.
			if !cur.alive() && time.Since(cur.started) > 2*time.Second {
				if err := h.start(name, hash, cur.exe); err != nil {
					h.event("start_failed", name, hash, map[string]any{"error": err.Error()})
				}
			}
			continue
		}
		if h.failed[name] == hash {
			continue
		}
		t0 := time.Now()
		exe, err := h.build(name, hash)
		if err != nil {
			h.failed[name] = hash
			kept := ""
			if cur != nil && cur.alive() {
				kept = cur.hash
			}
			h.event("build_failed", name, hash, map[string]any{"error": err.Error(), "kept_running": kept})
			continue
		}
		delete(h.failed, name)
		h.event("built", name, hash, map[string]any{"ms": time.Since(t0).Milliseconds()})
		// The new build starts before the old one stops: starting a fresh
		// executable can take seconds while Windows scans it, and the old one
		// keeps sampling meanwhile, or keeps running if the new one fails.
		if err := h.start(name, hash, exe); err != nil {
			h.failed[name] = hash
			h.event("start_failed", name, hash, map[string]any{"error": err.Error()})
			continue
		}
		if cur != nil {
			h.stop(name, cur, "source changed")
		}
	}
	for name, p := range h.procs {
		if !seen[name] {
			h.stop(name, p, "probe removed")
			delete(h.procs, name)
		}
	}
}

// appendJSONL writes each value from in as one line of path, until in closes.
func appendJSONL(path string, in <-chan map[string]any, done chan<- struct{}) {
	defer close(done)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		for range in {
		}
		return
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for v := range in {
		if err := enc.Encode(v); err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
	}
}

func main() {
	probesDir := flag.String("probes", "probes", "one subdirectory per probe, inside this Go module")
	outDir := flag.String("out", "out", "where probes.jsonl, host-events.jsonl and the probe builds go")
	poll := flag.Duration("poll", 500*time.Millisecond, "how often to look for changed probe source")
	interval := flag.Duration("interval", time.Second, "sampling interval passed to probes as PROBE_INTERVAL_MS")
	runFor := flag.Duration("for", 0, "stop after this long; 0 runs until interrupted")
	flag.Parse()

	h := &host{
		probesDir: filepath.Clean(*probesDir),
		binDir:    filepath.Join(*outDir, "bin"),
		interval:  *interval,
		records:   make(chan map[string]any, 256),
		events:    make(chan map[string]any, 256),
		procs:     map[string]*proc{},
		failed:    map[string]string{},
	}
	if err := os.MkdirAll(h.binDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	recordsDone, eventsDone := make(chan struct{}), make(chan struct{})
	go appendJSONL(filepath.Join(*outDir, "probes.jsonl"), h.records, recordsDone)
	go appendJSONL(filepath.Join(*outDir, "host-events.jsonl"), h.events, eventsDone)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	if *runFor > 0 {
		ctx, cancel = context.WithTimeout(ctx, *runFor)
		defer cancel()
	}
	h.event("host_started", "", "", map[string]any{"pid": os.Getpid()})
	h.reconcile()
	tick := time.NewTicker(*poll)
	defer tick.Stop()
loop:
	for {
		select {
		case <-ctx.Done():
			break loop
		case <-tick.C:
			h.reconcile()
		}
	}
	for name, p := range h.procs {
		h.stop(name, p, "host stopping")
	}
	h.event("host_stopped", "", "", nil)
	close(h.records)
	close(h.events)
	<-recordsDone
	<-eventsDone
}
