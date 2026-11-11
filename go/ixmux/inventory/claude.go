package inventory

import (
	"encoding/json"
	"errors"
	"io"
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

// What the transcript search found for a session, in its Declared["transcript"].
const (
	transcriptFound      = "found"      // its modification time is in the last activity
	transcriptNone       = "none"       // searched to the end: there is none
	transcriptCut        = "search-cut" // the budget ran out first: not known
	transcriptUnreadable = "unreadable" // projects/ or a path in it could not be read: not known
)

// transcripts is the search for the sessions' transcripts, with one budget for a whole Collect: a
// count of paths stat-ed and directory entries listed, and a deadline read on Options.Now. It
// opens no transcript; it lists projects/ at most once.
type transcripts struct {
	files    int
	deadline time.Time
	now      func() time.Time
	projects []string // the directories of projects/, sorted
	listed   bool
	cut      bool  // the listing stopped on the budget
	listErr  error // the listing failed
}

// spend takes one step of the budget, or says it is spent.
func (t *transcripts) spend() bool {
	if t.files <= 0 || !t.now().Before(t.deadline) {
		return false
	}
	t.files--
	return true
}

// projectDir is the directory Claude Code keeps a cwd's transcripts in: every character but an
// ASCII letter or digit made '-' (C:\work is C--work). A cwd it names otherwise (a long one) is
// found by the search of projects/.
func projectDir(cwd string) string {
	b := []byte(cwd)
	for i, c := range b {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
			b[i] = '-'
		}
	}
	return string(b)
}

// find looks for <home>/projects/*/<id>.jsonl: first in its cwd's directory, then in the others,
// one stat each, until the budget is spent.
// @ai:invariant a session's transcript search reads no transcript content and is "search-cut" or "unreadable", never "none", when it did not run to its end [T:test conf:0.85 src:go/ixmux/inventory/transcript_test.go:TestASpentFileBudgetIsPartialNeverComplete]
func (t *transcripts) find(home string, s sessionRecord) (time.Time, string, string) {
	direct := ""
	if s.Cwd != "" {
		direct = projectDir(s.Cwd)
		if !t.spend() {
			return time.Time{}, transcriptCut, ""
		}
		if mt, how, why := statTranscript(filepath.Join(home, "projects", direct, s.SessionID+".jsonl")); how != "" {
			return mt, how, why
		}
	}
	if !t.listed {
		t.list(filepath.Join(home, "projects"))
	}
	if t.listErr != nil {
		return time.Time{}, transcriptUnreadable, t.listErr.Error()
	}
	for _, p := range t.projects {
		if p == direct {
			continue
		}
		if !t.spend() {
			return time.Time{}, transcriptCut, ""
		}
		if mt, how, why := statTranscript(filepath.Join(home, "projects", p, s.SessionID+".jsonl")); how != "" {
			return mt, how, why
		}
	}
	if t.cut {
		return time.Time{}, transcriptCut, ""
	}
	return time.Time{}, transcriptNone, ""
}

// statTranscript: found with its modification time, unreadable, or "" when no transcript is there.
func statTranscript(p string) (time.Time, string, string) {
	st, err := os.Stat(p)
	switch {
	case err == nil && st.Mode().IsRegular():
		return st.ModTime(), transcriptFound, ""
	case err == nil || errors.Is(err, fs.ErrNotExist):
		return time.Time{}, "", ""
	default:
		return time.Time{}, transcriptUnreadable, err.Error()
	}
}

// list reads the directory names of projects/ once, one budget step an entry. A missing projects/
// holds no transcript.
func (t *transcripts) list(dir string) {
	t.listed = true
	d, err := os.Open(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return
	}
	if err != nil {
		t.listErr = err
		return
	}
	defer d.Close()
	defer sort.Strings(t.projects)
	for {
		batch, err := d.ReadDir(256)
		for _, e := range batch {
			if !t.spend() {
				t.cut = true
				return
			}
			if e.IsDir() {
				t.projects = append(t.projects, e.Name())
			}
		}
		if err != nil || len(batch) == 0 {
			if err != nil && !errors.Is(err, io.EOF) {
				t.listErr = err
			}
			return
		}
	}
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
		e := session(o, filepath.Join(dir, n))
		switch e.Declared["transcript"] {
		case transcriptCut:
			r.Omitted["transcript searches cut by the budget"]++
		case transcriptUnreadable:
			r.Omitted["transcript searches that failed"]++
		}
		r.Entries = append(r.Entries, e)
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
	mt, how, why := o.transcripts.find(o.ClaudeHome, s)
	e.Declared["transcript"] = how
	switch how {
	case transcriptFound:
		e.LastActivity = later(e.LastActivity, mt)
	case transcriptCut:
		e.Problem = "the transcript search ran out of its budget: the last activity is the record's, maybe older than the truth"
	case transcriptUnreadable:
		e.Problem = "the transcript search failed (" + why + "): the last activity is the record's, maybe older than the truth"
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
