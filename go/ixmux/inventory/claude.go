package inventory

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// sessionRecord is the part of a Claude Code session record (<home>/sessions/<pid>.json) read
// here. Claude Code writes it while the session runs; nothing documents it as a contract, so any
// field may be missing, and a record that does not hold a session id and a pid is unreadable.
type sessionRecord struct {
	PID             int    `json:"pid"`
	SessionID       string `json:"sessionId"`
	Cwd             string `json:"cwd"`
	StartedAt       int64  `json:"startedAt"` // ms since 1970
	ProcStart       string `json:"procStart"` // on Windows the process's creation FILETIME
	Kind            string `json:"kind"`
	Name            string `json:"name"`
	Status          string `json:"status"`          // busy or idle
	UpdatedAt       int64  `json:"updatedAt"`       // ms
	StatusUpdatedAt int64  `json:"statusUpdatedAt"` // ms
	Bridge          string `json:"bridgeSessionId"` // set when the session is also reachable remotely
	PIDDomain       string `json:"pidDomain"`       // where its pid means something: <os>:<host>
}

// filetimeEpoch is 1970-01-01 as a FILETIME (100 ns intervals since 1601-01-01).
const filetimeEpoch = 116444736000000000

// declaredStart gives the creation time a record declares, and how close the live process's must
// be: exact for a FILETIME (both come from the same clock read), else startedAt, which Claude Code
// takes a little after its process was created.
func (s sessionRecord) declaredStart() (time.Time, time.Duration, bool) {
	if ft, err := strconv.ParseInt(s.ProcStart, 10, 64); err == nil && ft > filetimeEpoch {
		return time.Unix(0, (ft-filetimeEpoch)*100).UTC(), ExactTol, true
	}
	if s.StartedAt > 0 {
		return time.UnixMilli(s.StartedAt).UTC(), ClockTol, true
	}
	return time.Time{}, 0, false
}

// safeID: a session id used in a file pattern holds no pattern or path character.
func safeID(id string) bool {
	if id == "" || len(id) > 64 {
		return false
	}
	for _, c := range id {
		if !(c == '-' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
			return false
		}
	}
	return true
}

// claudeSessions reads <home>/sessions/*.json: one record per running Claude Code process, which
// its process removes when it exits; one left behind by a crash names a process that is gone.
func (r *Report) claudeSessions(o Options) {
	dir := filepath.Join(o.ClaudeHome, "sessions")
	d, err := os.Open(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return
	}
	if err != nil {
		r.Entries = append(r.Entries, Entry{Kind: "claude-session", ID: dir, Status: Unreadable, Problem: err.Error(), Source: dir,
			Resume: "no session can be shown: fix the access to this directory"})
		return
	}
	var names []string
	scanned, over := 0, 0
	for scanned < 16*o.MaxSessions { // a directory flooded with files is not read to its end
		batch, err := d.ReadDir(256)
		for _, e := range batch {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
				continue
			}
			if len(names) < o.MaxSessions {
				names = append(names, e.Name())
			} else {
				over++
			}
		}
		scanned += len(batch)
		if err != nil || len(batch) == 0 {
			break
		}
	}
	d.Close()
	if over > 0 {
		r.Omitted["claude session records past the bound"] = over
	}
	if scanned >= 16*o.MaxSessions {
		r.Omitted["claude session directory entries not scanned"] = 1
	}
	sort.Strings(names)
	first := len(r.Entries)
	for _, n := range names {
		r.Entries = append(r.Entries, session(o, filepath.Join(dir, n)))
	}
	r.sessionDuplicates(r.Entries[first:])
}

func session(o Options, path string) Entry {
	e := Entry{Kind: "claude-session", ID: filepath.Base(path), Source: path}
	b, err := readBounded(path, o.MaxFileBytes)
	var s sessionRecord
	if err == nil {
		err = json.Unmarshal(b, &s)
	}
	if err == nil && (!safeID(s.SessionID) || s.PID <= 0) {
		err = errors.New("no usable sessionId and pid")
	}
	if err != nil {
		// A record being rewritten, or torn by a crash: shown, never guessed.
		e.Status, e.Problem = Unreadable, err.Error()
		e.Resume = "read again: a record being rewritten reads whole a moment later"
		return e
	}
	e.ID, e.Name = s.SessionID, s.Name
	e.Declared = map[string]string{"status": s.Status, "cwd": s.Cwd, "kind": s.Kind}
	if s.Bridge != "" {
		e.Declared["remote_control"] = "yes"
	}
	for _, ms := range []int64{s.UpdatedAt, s.StatusUpdatedAt} {
		if ms > 0 {
			e.LastActivity = later(e.LastActivity, time.UnixMilli(ms))
		}
	}
	// The transcript's last write is activity too: its mtime only, never its content.
	if m, _ := filepath.Glob(filepath.Join(o.ClaudeHome, "projects", "*", s.SessionID+".jsonl")); len(m) > 0 {
		for _, p := range m {
			if st, err := os.Stat(p); err == nil {
				e.LastActivity = later(e.LastActivity, st.ModTime())
			}
		}
	}
	if i := strings.LastIndexByte(s.PIDDomain, ':'); i >= 0 && o.Hostname != "" && !strings.EqualFold(s.PIDDomain[i+1:], o.Hostname) {
		e.Status, e.Liveness = Unknown, Unverified
		e.Problem = "recorded on another machine (" + s.PIDDomain + "): its pid says nothing here"
		e.Resume = "look for it on that machine"
		return e
	}
	start, tol, ok := s.declaredStart()
	if !ok {
		e.Status, e.Liveness, e.Problem = Unknown, Unverified, "the record declares no creation time: its pid alone proves nothing"
		e.Resume = "cannot tell whether it runs: look for it in its terminal before resuming it"
		return e
	}
	e.Incarnation = "pid:" + strconv.Itoa(s.PID) + "@" + start.Format(time.RFC3339Nano)
	e.Liveness, e.Proof = check(o, s.PID, start, tol)
	switch e.Liveness {
	case Proven:
		switch s.Status {
		case "busy":
			e.Status = Running
		case "idle":
			e.Status = Idle
		default:
			e.Status = Unknown
		}
		e.Resume = "live: reach it in its terminal (" + s.Cwd + "); never start a second copy of it"
	case Absent, PIDReused:
		e.Status = Unreachable
		e.Resume = "claude --resume " + s.SessionID + " in " + s.Cwd + ", once no other incarnation of it is live"
	default:
		e.Status = Unknown
		e.Problem = e.Proof
		e.Resume = "cannot tell whether it runs: look for it in its terminal before resuming it"
	}
	return e
}

// sessionDuplicates reports a session id held by more than one record, and keeps any of them from
// being resumed while another is live.
func (r *Report) sessionDuplicates(es []Entry) {
	by := map[string][]int{}
	var ids []string
	for i, e := range es {
		if e.Status == Unreadable {
			continue
		}
		if _, ok := by[e.ID]; !ok {
			ids = append(ids, e.ID)
		}
		by[e.ID] = append(by[e.ID], i)
	}
	sort.Strings(ids)
	for _, id := range ids {
		is := by[id]
		if len(is) < 2 {
			continue
		}
		d := Duplicate{Kind: "claude-session", ID: id}
		var live []string
		for _, i := range is {
			d.Incarnations = append(d.Incarnations, es[i].Incarnation)
			if es[i].Liveness == Proven {
				live = append(live, es[i].Incarnation)
			}
		}
		d.Live = len(live)
		sort.Strings(d.Incarnations)
		r.Duplicates = append(r.Duplicates, d)
		for _, i := range is {
			switch {
			case len(live) > 1:
				es[i].Problem = "more than one live process holds this session id: do not resume it, reach each in its terminal"
				if es[i].Liveness != Proven {
					es[i].Resume = "do not resume: it is already live in " + strings.Join(live, " and ")
				}
			case len(live) == 1 && es[i].Liveness != Proven:
				es[i].Resume = "do not resume: it is already live as " + live[0]
			}
		}
	}
}
