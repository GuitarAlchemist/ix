// Package inventory lists, read-only, what can be found again after a crash, a power loss or a
// Windows restart: the local Claude Code sessions, the ixmux processes and their mandates. Each
// entry keeps what is proven now (a live process whose creation time matches the one its record
// gives) apart from what is only remembered (what a file declares). It sends, starts, replays,
// locks and writes nothing: what to do next is said in Resume, for a person to decide.
package inventory

import (
	"errors"
	"io"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// Format names the report's JSON shape.
const Format = "ixmux.inventory.v1"

// Status is an entry's state, decided from what was proven at ReadAt.
type Status string

const (
	Running     Status = "running"     // proven alive and working: a busy session, a mandate in its turn
	Idle        Status = "idle"        // proven alive, waiting
	Stopped     Status = "stopped"     // an ixmux that recorded a clean stop
	Done        Status = "done"        // a finished mandate: delivered, or nothing was sent
	Uncertain   Status = "uncertain"   // a mandate whose effect may have happened: a person decides
	Unreachable Status = "unreachable" // remembered, but its process is gone or is another one now
	Unknown     Status = "unknown"     // cannot be decided from here
	Unreadable  Status = "unreadable"  // its record could not be read
)

// Liveness says how an entry's process was checked.
type Liveness string

const (
	Proven     Liveness = "proven"     // the pid is live and was created when the record says
	Absent     Liveness = "absent"     // no live process has that pid
	PIDReused  Liveness = "pid-reused" // a live process has that pid, created at another time
	Unverified Liveness = "unverified" // the check could not be made (Problem says why)
)

// The tolerances of check: ExactTol between two reads of one clock, ClockTol else.
const (
	ExactTol = time.Millisecond
	ClockTol = time.Minute
)

// ErrNoProcess: no live process has the pid.
var ErrNoProcess = errors.New("no such process")

// Entry is one thing found: a session, an ixmux, a mandate.
type Entry struct {
	Kind         string            `json:"kind"` // claude-session, ixmux, mandate
	ID           string            `json:"id"`   // stable: a session id, a state directory, a mandate id
	Incarnation  string            `json:"incarnation,omitempty"`
	Name         string            `json:"name,omitempty"`
	Status       Status            `json:"status"`
	Liveness     Liveness          `json:"liveness,omitempty"`
	Proof        string            `json:"proof,omitempty"` // how liveness was decided
	LastActivity *time.Time        `json:"last_activity,omitempty"`
	AgeSeconds   *int64            `json:"age_s,omitempty"`
	Declared     map[string]string `json:"declared,omitempty"` // what the record says: no proof
	Resume       string            `json:"resume"`             // for a person; never done here
	Problem      string            `json:"problem,omitempty"`
	Source       string            `json:"source"` // the file read
}

// Duplicate is one identity found in more than one incarnation.
type Duplicate struct {
	Kind         string   `json:"kind"`
	ID           string   `json:"id"`
	Incarnations []string `json:"incarnations"`
	Live         int      `json:"live"` // how many of them were proven alive
}

// Report is one read of every source.
type Report struct {
	Format     string         `json:"format"`
	ReadAt     time.Time      `json:"read_at"`
	Entries    []Entry        `json:"entries"`
	Duplicates []Duplicate    `json:"duplicates,omitempty"`
	Omitted    map[string]int `json:"omitted,omitempty"` // entries left out by a bound, by reason
	Partial    bool           `json:"partial"`           // a bound left something out or cut a search: see omitted
	Gaps       []string       `json:"gaps"`              // sources this inventory does not read
}

// Gaps lists, in every report, what this inventory does not see, so that nothing missing from it
// is taken for absent or for active.
var Gaps = []string{
	"cloud sessions: no local file lists them; none is shown, and none is presumed active",
	"wmux panes and their blocked state (wmux agent-state) are not read",
	"Windows services and ports: see `ixctl resume check` (the IX Service Windows lane)",
}

// Options name the sources and the bounds. Zero values take the defaults.
type Options struct {
	ClaudeHome   string   // the Claude Code configuration directory (~/.claude); "" skips sessions
	IxmuxStates  []string // ixmux state directories
	Now          func() time.Time
	Created      func(pid int) (time.Time, error) // a process's creation time, or ErrNoProcess
	Hostname     string                           // this machine's name, against a record's pidDomain: os.Hostname
	MaxSessions  int                              // session records read: 256
	MaxFileBytes int64                            // one session or ixmux.json record: 64 KiB
	MaxDone      int                              // finished mandates listed per ixmux, newest first: 50
	MaxLogBytes  int64                            // a command log read: deliver.MaxLogBytes
	// The transcript search, shared by every session of one Collect. It reads no byte of a
	// transcript, its modification time only; once spent, the rest is "search-cut" and the
	// report partial.
	TranscriptFiles int           // paths stat-ed and project directory entries listed: 4096
	TranscriptTime  time.Duration // time spent searching, by Now: 2 s

	transcripts *transcripts // the search's budget, set by Collect
}

func (o *Options) defaults() {
	if o.Now == nil {
		o.Now = time.Now
	}
	if o.Created == nil {
		o.Created = created
	}
	if o.Hostname == "" {
		o.Hostname, _ = os.Hostname()
	}
	if o.MaxSessions <= 0 {
		o.MaxSessions = 256
	}
	if o.MaxFileBytes <= 0 {
		o.MaxFileBytes = 64 << 10
	}
	if o.MaxDone <= 0 {
		o.MaxDone = 50
	}
	if o.MaxLogBytes <= 0 {
		o.MaxLogBytes = deliver.MaxLogBytes
	}
	if o.TranscriptFiles <= 0 {
		o.TranscriptFiles = 4096
	}
	if o.TranscriptTime <= 0 {
		o.TranscriptTime = 2 * time.Second
	}
}

// Collect reads every source once. A source that cannot be read is an unreadable entry, never an
// error: the report always comes back.
func Collect(o Options) Report {
	o.defaults()
	r := Report{Format: Format, ReadAt: o.Now().UTC(), Omitted: map[string]int{}, Gaps: Gaps}
	o.transcripts = &transcripts{files: o.TranscriptFiles, deadline: r.ReadAt.Add(o.TranscriptTime), now: o.Now}
	if o.ClaudeHome != "" {
		r.claudeSessions(o)
	}
	for _, dir := range o.IxmuxStates {
		r.ixmux(o, dir)
	}
	for i := range r.Entries {
		if t := r.Entries[i].LastActivity; t != nil {
			age := int64(r.ReadAt.Sub(*t) / time.Second)
			r.Entries[i].AgeSeconds = &age
		}
	}
	sort.SliceStable(r.Entries, func(i, j int) bool {
		a, b := r.Entries[i], r.Entries[j]
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		if a.ID != b.ID {
			return a.ID < b.ID
		}
		return a.Incarnation < b.Incarnation
	})
	r.Partial = len(r.Omitted) > 0
	if !r.Partial {
		r.Omitted = nil
	}
	return r
}

// check decides a pid's liveness against the creation time its record declares, within tol either
// way. tol is ExactTol for a time read from the same source as the process's (a Windows FILETIME);
// ClockTol for one read from a wall clock (ixmux's Started, a session's startedAt, Linux's btime),
// which a process takes a little after its creation and which a clock step (NTP, a VM resumed)
// moves. A pid reused by a process created within ClockTol of the recorded one is taken for it.
// @ai:invariant a live process whose wall-clock creation time moved less than ClockTol either way is proven, not pid-reused [T:test conf:0.85 src:go/ixmux/inventory/inventory_test.go:TestAClockStepDoesNotKillALiveProcess]
func check(o Options, pid int, declared time.Time, tol time.Duration) (Liveness, string) {
	c, err := o.Created(pid)
	switch {
	case errors.Is(err, ErrNoProcess):
		return Absent, "no live process has pid " + itoa(pid)
	case err != nil:
		return Unverified, "pid " + itoa(pid) + ": " + err.Error()
	}
	d := declared.Sub(c)
	if d < -tol || d > tol {
		return PIDReused, "pid " + itoa(pid) + " is live but was created at " + c.Format(time.RFC3339Nano) + ", not " + declared.Format(time.RFC3339Nano)
	}
	return Proven, "pid " + itoa(pid) + " is live, created at " + c.Format(time.RFC3339Nano)
}

// readBounded reads a whole small regular file, refusing one over max bytes, and never blocks on a
// FIFO put in its place (deliver.OpenRegular).
func readBounded(path string, max int64) ([]byte, error) {
	f, err := deliver.OpenRegular(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, max+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > max {
		return nil, errors.New("over " + itoa(int(max)) + " bytes")
	}
	return b, nil
}

func itoa(n int) string { return strconv.Itoa(n) }

func later(a *time.Time, b time.Time) *time.Time {
	if b.IsZero() {
		return a
	}
	if a == nil || b.After(*a) {
		b = b.UTC()
		return &b
	}
	return a
}
