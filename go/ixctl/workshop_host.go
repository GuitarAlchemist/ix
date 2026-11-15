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
	app, err := workshop.Handler(store, settings.Origin)
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
	if err := writeFileAtomic(h.statePath, state); err != nil {
		listener.Close()
		store.Close()
		return nil, err
	}
	go h.server.Serve(listener)
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
