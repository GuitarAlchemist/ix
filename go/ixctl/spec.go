package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"time"
)

var nameRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,62}$`)

// Spec is the declared world of one instance: the only services it can ever
// start, and the peers it federates with.
type Spec struct {
	Name     string        `json:"name"`
	Services []ServiceSpec `json:"services"`
	Peers    []PeerSpec    `json:"peers,omitempty"`

	dir string // folder of the spec file; relative paths resolve against it
}

type ServiceSpec struct {
	Name        string            `json:"name"`
	Command     string            `json:"command"`
	Args        []string          `json:"args,omitempty"`
	Dir         string            `json:"dir,omitempty"`
	Env         map[string]string `json:"env,omitempty"`
	Restart     string            `json:"restart,omitempty"`
	MaxRestarts *int              `json:"max_restarts,omitempty"`
	Autostart   bool              `json:"autostart,omitempty"`
	// Limits bounds each run of the service; absent, nothing bounds it.
	Limits *Limits `json:"limits,omitempty"`
}

// Limits bounds one run of a service. A run past a limit is killed, its tree with it, and
// its exit names the limit (`last_exit.reason`, and `reason` on `service.exited`); the restart
// policy then applies as to any failure, so a job so ended stays `stopped` and a service
// restarts with backoff.
type Limits struct {
	// WallMs is the longest a run may last, in milliseconds, from its launch; 0 is no limit.
	WallMs int64 `json:"wall_ms,omitempty"`
	// MemoryMB is the most memory the run's whole tree may hold, in MiB; 0 is no limit. On
	// Windows the job object enforces it (commits past it fail) and reports it; on Linux the
	// plane reads the group's resident set from /proc every 100 ms. Either way the run is
	// killed once past it.
	MemoryMB int64 `json:"memory_mb,omitempty"`
	// CpuMs is the most processor time, user and kernel, the run's whole tree may use, in
	// milliseconds; 0 is no limit. The plane reads it every 100 ms from the same accounting
	// the exit reports as `cpu_ms` (the job's on Windows, the group's /proc stat on Linux), so
	// a run is killed once the figure it would report goes past the limit, at most a poll late.
	CpuMs int64 `json:"cpu_ms,omitempty"`
	// Processes is the most processes the run's tree may hold at once; 0 is no limit. The
	// plane counts them every 100 ms (the job's active processes on Windows, the group's live
	// members on Linux) and kills a run once the count it would report as `peak_processes`
	// goes past the limit. On Windows the job object also refuses a process past a cap an
	// eighth above (at least one), so a fork storm is bounded between two polls.
	Processes int64 `json:"processes,omitempty"`
}

type PeerSpec struct {
	Name      string `json:"name"`
	StateFile string `json:"state_file"`
}

const (
	restartNever     = "never"
	restartOnFailure = "on-failure"
	restartAlways    = "always"

	// The largest limits that still fit a time.Duration (wall and processor time) and a byte count.
	maxWallMs   = math.MaxInt64 / int64(time.Millisecond)
	maxMemoryMB = math.MaxInt64 >> 20
	// The largest process limit whose cap (an eighth above) still fits the job object's DWORD.
	maxProcesses = 1 << 31
)

// LoadSpec reads and validates a spec file. Relative commands, dirs and peer
// state files are made absolute against the spec's folder.
func LoadSpec(path string) (*Spec, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	return ParseSpec(b, filepath.Dir(abs))
}

func ParseSpec(b []byte, dir string) (*Spec, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var s Spec
	if err := dec.Decode(&s); err != nil {
		return nil, fmt.Errorf("spec: %w", err)
	}
	s.dir = dir
	if err := s.validate(); err != nil {
		return nil, fmt.Errorf("spec: %w", err)
	}
	return &s, nil
}

func (s *Spec) validate() error {
	if !nameRE.MatchString(s.Name) {
		return fmt.Errorf("instance name %q must match %s", s.Name, nameRE)
	}
	seen := map[string]bool{}
	for i := range s.Services {
		sv := &s.Services[i]
		if !nameRE.MatchString(sv.Name) {
			return fmt.Errorf("service name %q must match %s", sv.Name, nameRE)
		}
		if seen[sv.Name] {
			return fmt.Errorf("duplicate service %q", sv.Name)
		}
		seen[sv.Name] = true
		if sv.Command == "" {
			return fmt.Errorf("service %q: command is required", sv.Name)
		}
		switch sv.Restart {
		case "":
			sv.Restart = restartOnFailure
		case restartNever, restartOnFailure, restartAlways:
		default:
			return fmt.Errorf("service %q: restart %q is not never|on-failure|always", sv.Name, sv.Restart)
		}
		if sv.MaxRestarts == nil {
			n := 5
			sv.MaxRestarts = &n
		} else if *sv.MaxRestarts < 0 {
			return fmt.Errorf("service %q: max_restarts must be >= 0", sv.Name)
		}
		if l := sv.Limits; l != nil {
			if l.WallMs < 0 || l.WallMs > maxWallMs {
				return fmt.Errorf("service %q: limits.wall_ms must be 0..%d", sv.Name, maxWallMs)
			}
			if l.MemoryMB < 0 || l.MemoryMB > maxMemoryMB {
				return fmt.Errorf("service %q: limits.memory_mb must be 0..%d", sv.Name, int64(maxMemoryMB))
			}
			if l.CpuMs < 0 || l.CpuMs > maxWallMs {
				return fmt.Errorf("service %q: limits.cpu_ms must be 0..%d", sv.Name, maxWallMs)
			}
			if l.Processes < 0 || l.Processes > maxProcesses {
				return fmt.Errorf("service %q: limits.processes must be 0..%d", sv.Name, int64(maxProcesses))
			}
		}
		sv.Command = s.abs(sv.Command)
		if sv.Dir == "" {
			sv.Dir = "."
		}
		sv.Dir = s.abs(sv.Dir)
	}
	seenPeer := map[string]bool{}
	for i := range s.Peers {
		p := &s.Peers[i]
		if !nameRE.MatchString(p.Name) {
			return fmt.Errorf("peer name %q must match %s", p.Name, nameRE)
		}
		if seenPeer[p.Name] {
			return fmt.Errorf("duplicate peer %q", p.Name)
		}
		seenPeer[p.Name] = true
		if p.StateFile == "" {
			return fmt.Errorf("peer %q: state_file is required", p.Name)
		}
		p.StateFile = s.abs(p.StateFile)
	}
	return nil
}

func (s *Spec) abs(p string) string {
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Join(s.dir, p)
}

func (s *Spec) service(name string) *ServiceSpec {
	for i := range s.Services {
		if s.Services[i].Name == name {
			return &s.Services[i]
		}
	}
	return nil
}

// resolveCommand returns the path to launch, trying ".exe" on Windows when the
// declared path has no extension and does not exist as written.
func resolveCommand(path string) string {
	if runtime.GOOS != "windows" || filepath.Ext(path) != "" {
		return path
	}
	if _, err := os.Stat(path); err == nil {
		return path
	}
	if _, err := os.Stat(path + ".exe"); err == nil {
		return path + ".exe"
	}
	return path
}

// desiredSchema names the desired-state document's format. A document may
// carry it as `schema`; one that names another schema is refused, one that
// names none is read as this one.
const desiredSchema = "ixctl-desired/0"

// Desired is a desired-state document: the state each named service should be
// in. Services it does not name are left as they are. It says what is wanted,
// never where: routing (`peer`) travels beside it, in the WebSocket frame or
// as `?peer=` on HTTP, so a journaled document holds no address.
type Desired struct {
	Schema   string                    `json:"schema,omitempty"`
	Services map[string]DesiredService `json:"services"`
}

type DesiredService struct {
	State string `json:"state"`
}

const (
	desiredRunning = "running"
	desiredStopped = "stopped"
)

// names returns the document's service names in a stable order.
func (d *Desired) names() []string {
	out := make([]string, 0, len(d.Services))
	for n := range d.Services {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
