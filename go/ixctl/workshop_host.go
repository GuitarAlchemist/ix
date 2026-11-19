package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixctl/internal/workshop"
)

// Opt-in settings have no control-plane token and cannot choose an arbitrary database path.
type workshopSettings struct {
	Version int    `json:"version"`
	Enabled bool   `json:"enabled"`
	Origin  string `json:"origin"`
	Listen  string `json:"listen"`
}

func readWorkshopSettings(path string) (workshopSettings, error) {
	if path == "" {
		return workshopSettings{}, nil
	}
	var settings workshopSettings
	file, err := os.Open(path)
	if err != nil {
		return settings, err
	}
	defer file.Close()
	if err = decodeStrict(io.LimitReader(file, 4096), &settings); err != nil {
		return settings, err
	}
	if err = settings.validate(); err != nil {
		return settings, err
	}
	_, configPort, _ := net.SplitHostPort(settings.Listen)
	number, _ := strconv.Atoi(configPort)
	if settings.Enabled && number == 0 {
		return settings, fmt.Errorf("Workshop config requires a fixed explicit loopback port")
	}
	return settings, nil
}
func (o workshopSettings) validate() error {
	if !o.Enabled {
		if o.Version != 0 && o.Version != 1 {
			return fmt.Errorf("unsupported Workshop config version")
		}
		return nil
	}
	if o.Version != 1 {
		return fmt.Errorf("Workshop config version must be 1")
	}
	if err := workshop.ValidateOrigin(o.Origin); err != nil {
		return fmt.Errorf("Workshop requires an exact HTTPS origin")
	}
	host, port, err := net.SplitHostPort(o.Listen)
	if err != nil || host != "127.0.0.1" {
		return fmt.Errorf("Workshop listen must be 127.0.0.1:port")
	}
	number, err := strconv.Atoi(port)
	if err != nil || number < 0 || number > 65535 || strconv.Itoa(number) != port || (number >= 8737 && number <= 8740) {
		return fmt.Errorf("Workshop port invalid or reserved for Scribe")
	}
	return nil
}

type workshopHost struct {
	store     *workshop.Store
	server    *http.Server
	addr      string
	ownerKey  string
	statePath string
	mu        sync.Mutex
	stopping  bool
	active    sync.WaitGroup
	closeOnce sync.Once
	lock      *os.File
	stopAuto  chan struct{}  // closed by close() to stop autoUpdate
	auto      sync.WaitGroup // autoUpdate, waited for by close()
}

func readWorkshopKey(path string) (string, error) {
	if info, err := os.Lstat(path); err == nil && !info.Mode().IsRegular() {
		return "", fmt.Errorf("Workshop key must be a regular file")
	}
	raw, err := os.ReadFile(path)
	if err == nil {
		key := strings.TrimSpace(string(raw))
		decoded, e := hex.DecodeString(key)
		if e != nil || len(decoded) != 32 {
			return "", fmt.Errorf("invalid Workshop owner key; refusing automatic rotation")
		}
		return key, nil
	}
	return "", fmt.Errorf("Workshop owner key must be explicitly provisioned before activation: %w", err)
}

// workshopUpdates lets admin devices install a newer build from the app: releases are put in
// %ProgramData%\IxForge\releases by Publish-IxForgeRelease.ps1; the requests go to <state>\workshop\updates, which the
// updater task (Install-IxForgeUpdater.ps1, run as SYSTEM) reads. Without ProgramData, nothing is offered.
func workshopUpdates(stateDir string) *workshop.Updates {
	data := os.Getenv("ProgramData")
	if data == "" || !filepath.IsAbs(data) {
		return nil
	}
	requests := filepath.Join(stateDir, "updates")
	if err := os.MkdirAll(requests, 0o755); err != nil {
		return nil
	}
	return &workshop.Updates{Installed: version, Releases: filepath.Join(data, "IxForge", "releases"), Requests: requests, Now: time.Now}
}

// autoUpdateEvery is how often the running Desktop looks for a newly published release to ask for by itself (the
// owner's choice, 2026-11-16: publishing is the approval). The updater task runs every minute too.
var autoUpdateEvery = time.Minute

// autoUpdate asks for the newest published release now and then every autoUpdateEvery, until stop is closed. An error
// is logged and not retried any faster: AutoRequest marks a version before asking, so it is asked at most once.
func autoUpdate(u *workshop.Updates, stop <-chan struct{}) {
	tick := time.NewTicker(autoUpdateEvery)
	defer tick.Stop()
	for {
		if asked, err := u.AutoRequest(); err != nil {
			fmt.Fprintf(os.Stderr, "ixctl: automatic update: %v\n", err)
		} else if asked != "" {
			fmt.Fprintf(os.Stderr, "ixctl: automatic update: asked the updater for %s\n", asked)
		}
		select {
		case <-stop:
			return
		case <-tick.C:
		}
	}
}

// workshopQueries lets every approved device run the listed DuckDB queries on the Desktop, once the owner has named the
// ix repo and the duckdb executable in <data>\IxForge\queries.json ({"repo": "...", "duckdb": "..."}, both absolute and
// present). The service's account must be able to read them. Without the file, or with a path that is not there, the
// Desktop offers no queries. The file may also name the LadybugDB CLI ("lbug", absolute and present): the graph queries
// then run on the builds of the repo's state\graph; without it they answer that the Desktop has no graph engine.
func workshopQueries(data string) *workshop.QueryRunner {
	raw, err := os.ReadFile(filepath.Join(data, "IxForge", "queries.json"))
	if err != nil {
		return nil
	}
	var c struct {
		Repo   string `json:"repo"`
		DuckDB string `json:"duckdb"`
		Lbug   string `json:"lbug"`
	}
	if json.Unmarshal(raw, &c) != nil || !filepath.IsAbs(c.Repo) || !filepath.IsAbs(c.DuckDB) {
		return nil
	}
	if fi, err := os.Stat(c.Repo); err != nil || !fi.IsDir() {
		return nil
	}
	if fi, err := os.Stat(c.DuckDB); err != nil || fi.IsDir() {
		return nil
	}
	r := &workshop.QueryRunner{Repo: c.Repo, DuckDB: c.DuckDB, Files: workshop.DefaultQueryFiles, Graphs: filepath.Join(c.Repo, "state", "graph")}
	if fi, err := os.Stat(c.Lbug); err == nil && !fi.IsDir() && filepath.IsAbs(c.Lbug) {
		r.Lbug = c.Lbug
	}
	return r
}

func startWorkshop(settings workshopSettings, stateDir, name string) (*workshopHost, error) {
	if err := settings.validate(); err != nil {
		return nil, err
	}
	if !settings.Enabled {
		return nil, nil
	}
	if !nameRE.MatchString(name) || !filepath.IsAbs(stateDir) {
		return nil, fmt.Errorf("Workshop requires the host's absolute state directory and instance name")
	}
	stateDir = filepath.Join(stateDir, "workshop")
	key, err := readWorkshopKey(filepath.Join(stateDir, name+".workshop-owner.key"))
	if err != nil {
		return nil, err
	}
	lock, held, err := lockFile(filepath.Join(stateDir, name+".workshop.lock"))
	if err != nil {
		return nil, err
	}
	if !held {
		return nil, fmt.Errorf("Workshop database already owned")
	}
	success := false
	defer func() {
		if !success {
			lock.Close()
		}
	}()
	listener, err := net.Listen("tcp", settings.Listen)
	if err != nil {
		return nil, err
	}
	store, err := workshop.Open(filepath.Join(stateDir, name+".workshop.sqlite"))
	if err != nil {
		listener.Close()
		return nil, err
	}
	h := &workshopHost{store: store, addr: listener.Addr().String(), ownerKey: key, statePath: filepath.Join(stateDir, name+".workshop.json"), lock: lock}
	options := []workshop.Option{workshop.WithVersion(version)}
	updates := workshopUpdates(stateDir)
	if updates != nil {
		options = append(options, workshop.WithUpdates(updates))
	}
	if data := os.Getenv("ProgramData"); filepath.IsAbs(data) {
		if q := workshopQueries(data); q != nil {
			options = append(options, workshop.WithQueries(q))
		}
	}
	app, err := workshop.Handler(store, settings.Origin, options...)
	if err != nil {
		listener.Close()
		store.Close()
		return nil, err
	}
	owner, err := workshop.OwnerHandler(store, key, "http://"+h.addr, settings.Origin)
	if err != nil {
		listener.Close()
		store.Close()
		return nil, err
	}
	routes := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/workshop-owner/") {
			owner.ServeHTTP(w, r)
		} else {
			app.ServeHTTP(w, r)
		}
	})
	h.server = &http.Server{ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, IdleTimeout: 60 * time.Second, Handler: h.admit(routes)}
	state, _ := json.Marshal(map[string]any{"protocol": "workshop/1", "addr": h.addr, "origin": settings.Origin, "pid": os.Getpid(), "schema_version": 1})
	if err := replaceFile(h.statePath, state); err != nil {
		listener.Close()
		store.Close()
		return nil, err
	}
	go h.server.Serve(listener)
	if updates != nil {
		h.stopAuto = make(chan struct{})
		h.auto.Add(1)
		go func() { defer h.auto.Done(); autoUpdate(updates, h.stopAuto) }()
	}
	success = true
	return h, nil
}
func (h *workshopHost) admit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		if h.stopping {
			h.mu.Unlock()
			http.Error(w, "Workshop stopping", 503)
			return
		}
		h.active.Add(1)
		h.mu.Unlock()
		defer h.active.Done()
		next.ServeHTTP(w, r)
	})
}
func (h *workshopHost) stopAccepting() {
	if h == nil {
		return
	}
	h.mu.Lock()
	h.stopping = true
	h.mu.Unlock()
}
func (h *workshopHost) close() {
	if h == nil {
		return
	}
	h.closeOnce.Do(func() {
		h.stopAccepting()
		if h.stopAuto != nil {
			close(h.stopAuto)
			h.auto.Wait()
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_ = h.server.Shutdown(ctx)
		cancel()
		_ = h.server.Close()
		h.active.Wait()
		_ = h.store.Close()
		_ = os.Remove(h.statePath)
		if h.lock != nil {
			_ = h.lock.Close()
		}
	})
}
