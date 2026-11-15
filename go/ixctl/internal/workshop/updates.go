package workshop

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Updating the Desktop from the app. Releases are built on this PC and put in a releases folder, one folder per
// version holding ixctl.exe and release.json. The Desktop offers its admin devices the releases newer than the one it
// runs. Installing one replaces the service's program, which needs the machine's administrator: the service only
// writes a request; an updater task installed once by the owner (run as SYSTEM) takes it, checks the release's hash
// again, runs Update-IxForge.ps1 and writes its result. The task writes a heartbeat each time it runs; without a
// recent one the Desktop refuses the request instead of letting it wait for nobody.

// Updates configures the update routes. Installed is the version the Desktop runs; Releases the releases folder;
// Requests the folder shared with the updater task (update-request.json, update-result.json, updater-seen.json).
type Updates struct {
	Installed string
	Releases  string
	Requests  string
	Now       func() time.Time

	mu     sync.Mutex // guards hashes
	hashes map[string]hashed
	asking sync.Mutex // one request written at a time
}

type hashed struct {
	size int64
	mod  time.Time
	hash string
}

// Release is a release the Desktop offers, as its manifest says, once its file matches the manifest's hash.
type Release struct {
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
	Notes   string `json:"notes,omitempty"`
	Commit  string `json:"commit,omitempty"`
	BuiltAt string `json:"built_at,omitempty"`
}

// The updater counts as running when its heartbeat is at most this old (the task runs every minute).
const updaterFresh = 5 * time.Minute

var (
	errUpdaterAbsent = errors.New("updater_absent")
	errUpdatePending = errors.New("update_pending")
	ixforgeVersion   = regexp.MustCompile(`^(\d+\.\d+\.\d+-ixforge)\.(\d{1,6})$`)
	sha256Hex        = regexp.MustCompile(`^[0-9A-Fa-f]{64}$`)
)

// Option configures Handler.
type Option func(*handlerOptions)

type handlerOptions struct {
	updates *Updates
}

// WithUpdates serves the update routes (GET and POST /workshop/api/updates) to admin devices.
func WithUpdates(u *Updates) Option { return func(o *handlerOptions) { o.updates = u } }

// newer says whether version is a later build of the same line as installed (0.1.0-ixforge.17 after .16).
func newer(version, installed string) bool {
	v, i := ixforgeVersion.FindStringSubmatch(version), ixforgeVersion.FindStringSubmatch(installed)
	if v == nil || i == nil || v[1] != i[1] {
		return false
	}
	a, _ := strconv.Atoi(v[2])
	b, _ := strconv.Atoi(i[2])
	return a > b
}

// readSmall reads a small JSON file (at most 64 KiB) into v. A missing file is (false, nil).
func readSmall(path string, v any) (bool, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, 64<<10))
	if err != nil {
		return false, err
	}
	return true, json.Unmarshal(raw, v)
}

// fileHash is the uppercase SHA-256 of path, kept while the file's size and time do not change.
func (u *Updates) fileHash(path string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	u.mu.Lock()
	c, ok := u.hashes[path]
	u.mu.Unlock()
	if ok && c.size == info.Size() && c.mod.Equal(info.ModTime()) {
		return c.hash, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	sum := strings.ToUpper(hex.EncodeToString(h.Sum(nil)))
	u.mu.Lock()
	if u.hashes == nil {
		u.hashes = map[string]hashed{}
	}
	u.hashes[path] = hashed{info.Size(), info.ModTime(), sum}
	u.mu.Unlock()
	return sum, nil
}

// Offered lists the releases newer than the installed one whose folder, manifest and file agree, newest first.
func (u *Updates) Offered() ([]Release, error) {
	entries, err := os.ReadDir(u.Releases)
	if errors.Is(err, os.ErrNotExist) {
		return []Release{}, nil
	}
	if err != nil {
		return nil, err
	}
	offered := []Release{}
	for _, e := range entries {
		if !e.IsDir() || !newer(e.Name(), u.Installed) {
			continue
		}
		dir := filepath.Join(u.Releases, e.Name())
		var r Release
		if found, err := readSmall(filepath.Join(dir, "release.json"), &r); !found || err != nil {
			continue
		}
		if r.Version != e.Name() || !sha256Hex.MatchString(r.SHA256) {
			continue
		}
		r.SHA256 = strings.ToUpper(r.SHA256)
		if hash, err := u.fileHash(filepath.Join(dir, "ixctl.exe")); err != nil || hash != r.SHA256 {
			continue
		}
		offered = append(offered, r)
	}
	// Newest first: same line, so the build number orders them.
	for i := 1; i < len(offered); i++ {
		for j := i; j > 0 && newer(offered[j].Version, offered[j-1].Version); j-- {
			offered[j], offered[j-1] = offered[j-1], offered[j]
		}
	}
	return offered, nil
}

// updaterSeen is the updater's last heartbeat, if it wrote one.
func (u *Updates) updaterSeen() (time.Time, bool) {
	var seen struct {
		SeenAt string `json:"seen_at"`
	}
	if found, err := readSmall(filepath.Join(u.Requests, "updater-seen.json"), &seen); !found || err != nil {
		return time.Time{}, false
	}
	at, err := time.Parse(time.RFC3339, seen.SeenAt)
	return at, err == nil
}

// State is what an admin device sees: the installed version, the releases offered, the updater's heartbeat, the
// waiting request and the updater's last result.
func (u *Updates) State() (map[string]any, error) {
	offered, err := u.Offered()
	if err != nil {
		return nil, err
	}
	state := map[string]any{"installed": u.Installed, "releases": offered, "updater": nil, "request": nil, "result": nil}
	if at, ok := u.updaterSeen(); ok {
		state["updater"] = map[string]any{"seen_at": at.UTC().Format(time.RFC3339), "running": u.Now().Sub(at) <= updaterFresh}
	}
	for key, file := range map[string]string{"request": "update-request.json", "result": "update-result.json"} {
		var v map[string]any
		if found, err := readSmall(filepath.Join(u.Requests, file), &v); err != nil {
			return nil, err
		} else if found {
			state[key] = v
		}
	}
	return state, nil
}

// Request asks the updater to install version, whose hash the admin device saw. by is the asking device.
func (u *Updates) Request(by Device, version, hash string) error {
	at, ok := u.updaterSeen()
	if !ok || u.Now().Sub(at) > updaterFresh {
		return errUpdaterAbsent
	}
	u.asking.Lock()
	defer u.asking.Unlock()
	path := filepath.Join(u.Requests, "update-request.json")
	if _, err := os.Stat(path); err == nil {
		return errUpdatePending
	}
	offered, err := u.Offered()
	if err != nil {
		return err
	}
	var release *Release
	for i := range offered {
		if offered[i].Version == version {
			release = &offered[i]
		}
	}
	if release == nil || !strings.EqualFold(release.SHA256, hash) {
		return ErrInvalid
	}
	raw, _ := json.Marshal(map[string]string{"version": release.Version, "sha256": release.SHA256,
		"requested_at": u.Now().UTC().Format(time.RFC3339), "device_id": by.ID, "device_name": by.Name})
	// Written whole, then renamed, so the updater never reads half a request.
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Admin authenticates token and requires an administering device.
func (s *Store) Admin(token string) (Device, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.admin(token)
}

func updateRoutes(mux *http.ServeMux, s *Store, u *Updates) {
	mux.HandleFunc("GET /workshop/api/updates", func(w http.ResponseWriter, r *http.Request) {
		if _, err := s.Admin(bearer(r)); err != nil {
			failure(w, err)
			return
		}
		state, err := u.State()
		if err != nil {
			failure(w, err)
			return
		}
		respond(w, http.StatusOK, state)
	})
	mux.HandleFunc("POST /workshop/api/updates", func(w http.ResponseWriter, r *http.Request) {
		d, err := s.Admin(bearer(r))
		if err != nil {
			failure(w, err)
			return
		}
		var body struct {
			Version string `json:"version"`
			SHA256  string `json:"sha256"`
		}
		if err := decode(w, r, &body); err != nil {
			failure(w, ErrInvalid)
			return
		}
		switch err := u.Request(d, body.Version, body.SHA256); {
		case errors.Is(err, errUpdaterAbsent):
			failureStatus(w, http.StatusServiceUnavailable, err.Error())
		case errors.Is(err, errUpdatePending):
			failureStatus(w, http.StatusConflict, err.Error())
		case err != nil:
			failure(w, err)
		default:
			respond(w, http.StatusAccepted, map[string]bool{"ok": true})
		}
	})
}
