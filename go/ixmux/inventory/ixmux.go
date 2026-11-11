package inventory

import (
	"bytes"
	"encoding/json"
	"errors"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// ixmuxInfo is the part of <state>/ixmux.json read here (server.Info).
type ixmuxInfo struct {
	Addr     string     `json:"addr"`
	PID      int        `json:"pid"`
	Started  time.Time  `json:"started"`
	Version  string     `json:"version"`
	Epoch    string     `json:"epoch"`
	State    string     `json:"state"` // running or stopped
	Stopped  *time.Time `json:"stopped"`
	ExitCode *int       `json:"exit_code"`
}

// uncertainResume is what an uncertain mandate allows: nothing automatic.
const uncertainResume = "a person decides: the target may have acted. Never send it again, under this id or another; " +
	"settle it with a reconcile while its incarnation lives, else by asking the target's owner"

// ixmux reads one state directory: ixmux.json, then the mandates of its command log.
func (r *Report) ixmux(o Options, dir string) {
	src := filepath.Join(dir, "ixmux.json")
	e := Entry{Kind: "ixmux", ID: dir, Source: src}
	b, err := readBounded(src, o.MaxFileBytes)
	var in ixmuxInfo
	if err == nil {
		err = json.Unmarshal(b, &in)
		if err == nil && (in.PID <= 0 || in.Started.IsZero()) {
			err = errors.New("no pid or start time")
		}
	}
	switch {
	case errors.Is(err, fs.ErrNotExist):
		e.Status, e.Problem = Unknown, "no ixmux.json: no ixmux started on this directory"
		e.Resume = "nothing recorded to resume"
	case err != nil:
		e.Status, e.Problem = Unreadable, err.Error()
		e.Resume = "read again: ixmux rewrites this file at each start and stop"
	default:
		e.Incarnation = "epoch:" + in.Epoch
		e.Name = in.Addr
		e.Declared = map[string]string{"state": in.State, "addr": in.Addr, "version": in.Version, "pid": strconv.Itoa(in.PID)}
		if in.ExitCode != nil {
			e.Declared["exit_code"] = strconv.Itoa(*in.ExitCode)
		}
		e.LastActivity = later(nil, in.Started)
		if in.Stopped != nil {
			e.LastActivity = later(e.LastActivity, *in.Stopped)
		}
		e.Liveness, e.Proof = check(o, in.PID, in.Started, ClockTol)
		switch {
		case in.State == "running" && e.Liveness == Proven:
			e.Status = Running
			e.Resume = "live: reach it with `ixmux mcp --state` on its directory; a second ixmux there is refused (its log is locked)"
		case in.State == "running" && (e.Liveness == Absent || e.Liveness == PIDReused):
			e.Status = Unreachable
			e.Problem = "ixmux.json says running but its process is gone: ixmux did not stop cleanly (crash, kill, power loss or restart)"
			e.Resume = "start ixmux again on this directory: its start records a delivering mandate uncertain (cause restart) and an accepted one refused; it sends nothing again"
		case in.State == "stopped" && e.Liveness == Proven:
			e.Status = Unknown
			e.Problem = "ixmux.json says stopped, but a live process has its pid and creation time"
			e.Resume = "look at that process before starting ixmux here"
		case in.State == "stopped":
			e.Status = Stopped
			e.Resume = "stopped cleanly: start ixmux again on this directory when needed"
		default:
			e.Status = Unknown
			if e.Problem == "" {
				e.Problem = e.Proof
			}
			e.Resume = "cannot tell whether it runs: look for its process before starting ixmux here"
		}
	}
	r.Entries = append(r.Entries, e)
	at := len(r.Entries) - 1
	r.mandates(o, dir, e.Status)
	if testHookAfterLog != nil {
		testHookAfterLog()
	}
	// An ixmux that started (or stopped) between the two reads makes the log's unfinished mandates
	// another host's: a mandate it has just accepted must not be called never sent. ixmux rewrites
	// ixmux.json before it accepts any mandate, so reading it again after the log closes the window.
	// @ai:invariant a mandate under way is never advised resendable when ixmux.json changed during the read [T:test conf:0.85 src:go/ixmux/inventory/inventory_test.go:TestAnIxmuxStartingDuringTheReadIsReadAgain]
	if b2, err2 := readBounded(src, o.MaxFileBytes); (err == nil) != (err2 == nil) || !bytes.Equal(b, b2) {
		const again = "ixmux.json changed during the read (ixmux started or stopped): read again"
		r.Entries[at].Status, r.Entries[at].Problem, r.Entries[at].Resume = Unknown, again, "read again"
		for i := at + 1; i < len(r.Entries); i++ {
			m := &r.Entries[i]
			if m.Kind == "mandate" && (m.Declared["state"] == string(deliver.Accepted) || m.Declared["state"] == string(deliver.Delivering)) {
				m.Status, m.Problem, m.Resume = Unknown, again, "read again; never send it again meanwhile"
			}
		}
	}
}

// testHookAfterLog runs between the read of a command log and the second read of ixmux.json.
var testHookAfterLog func()

// mandates lists the mandates of dir's command log. Whether a mandate under way still runs depends
// on its ixmux (host): only a live ixmux can still be delivering it.
func (r *Report) mandates(o Options, dir string, host Status) {
	path := filepath.Join(dir, "commands.jsonl")
	ms, torn, err := deliver.ReadLog(path, o.MaxLogBytes)
	if errors.Is(err, fs.ErrNotExist) {
		return
	}
	if err != nil {
		r.Entries = append(r.Entries, Entry{Kind: "mandate-log", ID: path, Status: Unreadable, Problem: err.Error(), Source: path,
			Resume: "none of its mandates can be shown here; never delete the log"})
		return
	}
	if torn != nil {
		t := Entry{Kind: "mandate-log", ID: path, Source: path, Declared: map[string]string{"torn_bytes": strconv.FormatInt(torn.Bytes, 10)}}
		if host == Running {
			t.Status, t.Problem = Running, "its last record was being written during the read"
			t.Resume = "read again"
		} else {
			t.Status, t.Problem = Unknown, "a torn last record: the next start of ixmux cuts it"
			t.Resume = "start ixmux again on this directory"
		}
		r.Entries = append(r.Entries, t)
	}
	var done []Entry
	for _, m := range ms {
		since := m.Since.UTC()
		e := Entry{Kind: "mandate", ID: m.MandateID, Name: m.Target, Incarnation: m.Incarnation, Source: path, LastActivity: &since,
			Declared: map[string]string{"state": string(m.State), "ixmux": dir}}
		if m.Cause != "" {
			e.Declared["cause"] = m.Cause
		}
		if m.Reason != "" {
			e.Declared["reason"] = m.Reason
		}
		switch m.State {
		case deliver.Delivered:
			e.Status, e.Resume = Done, "finished: read its answer with ixmux_result; never send it again"
		case deliver.Refused:
			e.Status, e.Resume = Done, "refused: nothing was sent; a person may send it again"
		case deliver.Uncertain:
			e.Status, e.Resume = Uncertain, uncertainResume
		case deliver.Accepted, deliver.Delivering:
			switch {
			case host == Running:
				e.Status, e.Resume = Running, "under way: wait for its final state (ixmux_result)"
			case host == Unreachable || host == Stopped:
				if m.State == deliver.Delivering {
					e.Status = Uncertain
					e.Problem = "delivering while its ixmux is down: the target may have acted; the next start records it uncertain, cause restart"
					e.Resume = uncertainResume
				} else {
					e.Status = Done
					e.Problem = "accepted while its ixmux is down: it was never sent; the next start records it refused"
					e.Resume = "nothing was sent: a person may send it again once ixmux runs"
				}
			default:
				e.Status, e.Problem = Unknown, "its ixmux could not be checked: it may still be under way"
				e.Resume = "check its ixmux first; never send it again meanwhile"
			}
		default:
			e.Status, e.Problem, e.Resume = Unknown, "unknown state "+string(m.State), "read it by hand"
		}
		if m.State == deliver.Delivered || m.State == deliver.Refused { // finished in the log: only these are bounded
			done = append(done, e)
		} else {
			r.Entries = append(r.Entries, e) // an unfinished mandate is never left out
		}
	}
	sort.SliceStable(done, func(i, j int) bool { return done[i].LastActivity.After(*done[j].LastActivity) })
	if len(done) > o.MaxDone {
		r.Omitted["finished mandates past the bound in "+dir] = len(done) - o.MaxDone
		done = done[:o.MaxDone]
	}
	r.Entries = append(r.Entries, done...)
}
