package inventory

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// private is a transcript's content: no report may carry it, since only a transcript's
// modification time is ever looked at.
const private = "PRIVATE-TRANSCRIPT-TEXT-7f3a"

const sid2 = "9b1e0c2d-0000-4000-8000-000000000002"

// transcript writes a session's transcript in <home>/projects/<dir>, last written at mtime.
func transcript(t *testing.T, home, dir, id string, mtime time.Time) {
	t.Helper()
	p := filepath.Join(home, "projects", dir, id+".jsonl")
	if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(`{"type":"user","message":"`+private+`"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p, mtime, mtime); err != nil {
		t.Fatal(err)
	}
}

// decoys makes n project directories that hold no transcript of the sessions under test, named to
// sort before "zz-elsewhere", where a search outside the cwd's directory finds them last.
func decoys(t *testing.T, home string, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		if err := os.MkdirAll(filepath.Join(home, "projects", fmt.Sprintf("d%04d", i)), 0o700); err != nil {
			t.Fatal(err)
		}
	}
}

func collectT(home string, o Options) Report {
	o.ClaudeHome = home
	if o.Now == nil {
		o.Now = func() time.Time { return t0 }
	}
	if o.Created == nil {
		o.Created = procs{}.created
	}
	return Collect(o)
}

// one finds the single entry of a session, and checks that the report carries no transcript text.
func one(t *testing.T, r Report, id string) Entry {
	t.Helper()
	b, _ := json.Marshal(r)
	if strings.Contains(string(b), private) {
		t.Errorf("the report carries a transcript's content: %s", b)
	}
	es := find(r, "claude-session", id)
	if len(es) != 1 {
		t.Fatalf("%s: %+v", id, es)
	}
	return es[0]
}

// The transcript is where Claude Code writes it, projects/<cwd with every other character than a
// letter or a digit made '-'>/<id>.jsonl: one stat finds it, whatever else projects/ holds.
func TestTheTranscriptAtItsProjectPathIsOneStat(t *testing.T) {
	home := t.TempDir()
	writeSession(t, home, 4242, sid, "busy", t0.Add(-time.Hour)) // cwd C:\work
	transcript(t, home, "C--work", sid, t0.Add(-10*time.Second))
	decoys(t, home, 1000)
	r := collectT(home, Options{TranscriptFiles: 1})
	e := one(t, r, sid)
	if e.Declared["transcript"] != "found" || e.AgeSeconds == nil || *e.AgeSeconds != 10 {
		t.Errorf("transcript %q, age %v: want found, 10 s", e.Declared["transcript"], e.AgeSeconds)
	}
	if r.Partial || r.Omitted != nil {
		t.Errorf("partial %v, omitted %v: nothing was cut", r.Partial, r.Omitted)
	}
}

// A transcript under another project directory (a cwd renamed, a rule changed) is searched for,
// within the budget, and found.
func TestATranscriptElsewhereIsFoundWithinTheBudget(t *testing.T) {
	home := t.TempDir()
	writeSession(t, home, 4242, sid, "busy", t0.Add(-time.Hour))
	transcript(t, home, "zz-elsewhere", sid, t0.Add(-10*time.Second))
	decoys(t, home, 20)
	r := collectT(home, Options{})
	e := one(t, r, sid)
	if e.Declared["transcript"] != "found" || e.AgeSeconds == nil || *e.AgeSeconds != 10 || r.Partial {
		t.Errorf("transcript %q, age %v, partial %v: want found, 10 s, complete", e.Declared["transcript"], e.AgeSeconds, r.Partial)
	}
}

// A file budget spent before the transcript is found is said on the entry and in the report: its
// last activity comes from its record alone, and the report is never complete. The budget is one
// for the whole read, not one per session.
func TestASpentFileBudgetIsPartialNeverComplete(t *testing.T) {
	home := t.TempDir()
	writeSession(t, home, 4242, sid, "busy", t0.Add(-time.Hour))
	writeSession(t, home, 4343, sid2, "idle", t0.Add(-time.Hour))
	transcript(t, home, "zz-elsewhere", sid, t0.Add(-10*time.Second))
	transcript(t, home, "zz-elsewhere", sid2, t0.Add(-10*time.Second))
	decoys(t, home, 1000)
	r := collectT(home, Options{TranscriptFiles: 100})
	for _, id := range []string{sid, sid2} {
		e := one(t, r, id)
		if e.Declared["transcript"] != "search-cut" || !strings.Contains(e.Problem, "budget") || e.AgeSeconds == nil || *e.AgeSeconds != 60 {
			t.Errorf("%s: transcript %q, problem %q, age %v: want search-cut, said, 60 s from the record", id, e.Declared["transcript"], e.Problem, e.AgeSeconds)
		}
	}
	if !r.Partial || r.Omitted["transcript searches cut by the budget"] != 2 {
		t.Errorf("partial %v, omitted %v: want partial, 2 searches cut", r.Partial, r.Omitted)
	}
}

// The time budget is read on Now: a clock moving one second a reading spends a 5 s budget in five
// steps, whatever the machine's speed.
func TestASpentTimeBudgetIsPartial(t *testing.T) {
	home := t.TempDir()
	writeSession(t, home, 4242, sid, "busy", t0.Add(-time.Hour))
	transcript(t, home, "zz-elsewhere", sid, t0.Add(-10*time.Second))
	decoys(t, home, 1000)
	n := 0
	r := collectT(home, Options{TranscriptTime: 5 * time.Second, Now: func() time.Time { n++; return t0.Add(time.Duration(n) * time.Second) }})
	if e := one(t, r, sid); e.Declared["transcript"] != "search-cut" || !r.Partial {
		t.Errorf("transcript %q, partial %v: want search-cut, partial", e.Declared["transcript"], r.Partial)
	}
	if n > 20 {
		t.Errorf("Now read %d times: the search went on past its time", n)
	}
}

// A search that ran to its end without finding a transcript is complete: no transcript is a fact.
func TestNoTranscriptIsNotPartial(t *testing.T) {
	home := t.TempDir()
	writeSession(t, home, 4242, sid, "busy", t0.Add(-time.Hour))
	decoys(t, home, 20)
	r := collectT(home, Options{})
	if e := one(t, r, sid); e.Declared["transcript"] != "none" || e.Problem != "" || r.Partial {
		t.Errorf("transcript %q, problem %q, partial %v: want none, no problem, complete", e.Declared["transcript"], e.Problem, r.Partial)
	}
}

// projects/ that cannot be listed makes the search fail: said, and the report partial.
func TestAnUnreadableProjectsDirectoryIsPartial(t *testing.T) {
	home := t.TempDir()
	writeSession(t, home, 4242, sid, "busy", t0.Add(-time.Hour))
	if err := os.WriteFile(filepath.Join(home, "projects"), []byte("not a directory"), 0o600); err != nil {
		t.Fatal(err)
	}
	r := collectT(home, Options{})
	if e := one(t, r, sid); e.Declared["transcript"] != "unreadable" || !r.Partial || r.Omitted["transcript searches that failed"] != 1 {
		t.Errorf("transcript %q, partial %v, omitted %v: want unreadable, partial", e.Declared["transcript"], r.Partial, r.Omitted)
	}
}

// A listing of projects/ cut by the budget before it met any directory leaves nothing to stat: the
// search is still cut, never "none" (a false complete).
func TestACutListingIsNeverNone(t *testing.T) {
	home := t.TempDir()
	writeSession(t, home, 4242, sid, "busy", t0.Add(-time.Hour))
	os.MkdirAll(filepath.Join(home, "projects"), 0o700)
	for i := 0; i < 300; i++ {
		os.WriteFile(filepath.Join(home, "projects", fmt.Sprintf("f%04d", i)), nil, 0o600)
	}
	transcript(t, home, "zz-elsewhere", sid, t0.Add(-10*time.Second))
	r := collectT(home, Options{TranscriptFiles: 50})
	if e := one(t, r, sid); e.Declared["transcript"] != "search-cut" || !r.Partial {
		t.Errorf("transcript %q, partial %v: want search-cut, partial", e.Declared["transcript"], r.Partial)
	}
}
