package inventory

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

var t0 = time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)

func filetime(t time.Time) string { return strconv.FormatInt(t.UnixNano()/100+filetimeEpoch, 10) }

// procs is a fake process table: pid to creation time.
type procs map[int]time.Time

func (p procs) created(pid int) (time.Time, error) {
	if c, ok := p[pid]; ok {
		return c, nil
	}
	return time.Time{}, ErrNoProcess
}

func writeSession(t *testing.T, home string, pid int, id, status string, created time.Time) {
	t.Helper()
	dir := filepath.Join(home, "sessions")
	os.MkdirAll(dir, 0o700)
	b, _ := json.Marshal(map[string]any{"pid": pid, "sessionId": id, "cwd": `C:\work`, "startedAt": created.UnixMilli() + 400,
		"procStart": filetime(created), "kind": "interactive", "name": "n" + id[:4], "status": status,
		"updatedAt": t0.Add(-time.Minute).UnixMilli(), "statusUpdatedAt": t0.Add(-2 * time.Minute).UnixMilli()})
	if err := os.WriteFile(filepath.Join(dir, strconv.Itoa(pid)+".json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
}

func collect(home string, p procs, states ...string) Report {
	return Collect(Options{ClaudeHome: home, IxmuxStates: states, Now: func() time.Time { return t0 }, Created: p.created})
}

func find(r Report, kind, id string) []Entry {
	var out []Entry
	for _, e := range r.Entries {
		if e.Kind == kind && e.ID == id {
			out = append(out, e)
		}
	}
	return out
}

const sid = "4fa73dfa-a6c0-48cb-8c2d-8fbb519ea0c5"

// After a restart, a session record left behind names a process that is gone: the session is
// unreachable, never shown active, and its resume names the session to resume.
func TestASessionWhoseProcessIsGoneIsUnreachable(t *testing.T) {
	home, c := t.TempDir(), t0.Add(-time.Hour)
	writeSession(t, home, 4242, sid, "busy", c)
	es := find(collect(home, procs{}), "claude-session", sid)
	if len(es) != 1 || es[0].Status != Unreachable || es[0].Liveness != Absent || !strings.Contains(es[0].Resume, "claude --resume "+sid) {
		t.Fatalf("%+v", es)
	}
	if es[0].Declared["status"] != "busy" {
		t.Errorf("the declared status is kept apart as metadata: %v", es[0].Declared)
	}
	if es[0].AgeSeconds == nil || *es[0].AgeSeconds != 60 {
		t.Errorf("last activity: %v age %v, want the latest of its times, 60 s old", es[0].LastActivity, es[0].AgeSeconds)
	}
}

// A live process with the record's pid but another creation time is not the session (pid reuse).
func TestAReusedPIDIsNotTheSession(t *testing.T) {
	home, c := t.TempDir(), t0.Add(-time.Hour)
	writeSession(t, home, 4242, sid, "busy", c)
	// Created after or before the declared time: a FILETIME is exact, so two seconds is another process.
	for _, other := range []time.Time{c.Add(3 * time.Second), c.Add(-2 * time.Second)} {
		es := find(collect(home, procs{4242: other}), "claude-session", sid)
		if len(es) != 1 || es[0].Status != Unreachable || es[0].Liveness != PIDReused {
			t.Fatalf("created at %s: %+v", other, es)
		}
	}
}

// A creation time read from a wall clock (ixmux's Started, a session's startedAt, Linux's btime) moves
// with a clock step (NTP, a VM resumed): within a minute either way, the live process is still the
// one recorded. Review P1-1: a +2 s step made a live ixmux pid-reused, so unreachable.
func TestAClockStepDoesNotKillALiveProcess(t *testing.T) {
	started := t0.Add(-time.Hour)
	dir := ixmuxState(t, started)
	for _, c := range []time.Time{started.Add(-300 * time.Millisecond).Add(2 * time.Second), started.Add(-50 * time.Second), started.Add(50 * time.Second)} {
		r := collect("", procs{777: c}, dir)
		if es := find(r, "ixmux", dir); len(es) != 1 || es[0].Status != Running || es[0].Liveness != Proven {
			t.Errorf("process created at %s, ixmux started at %s: %+v", c, started, es)
		}
	}
	r := collect("", procs{777: started.Add(-2 * time.Minute)}, dir)
	if es := find(r, "ixmux", dir); len(es) != 1 || es[0].Liveness != PIDReused {
		t.Errorf("a process created two minutes off: %+v", es)
	}
	// A session record without procStart falls back on startedAt, with the same tolerance.
	home := t.TempDir()
	os.MkdirAll(filepath.Join(home, "sessions"), 0o700)
	os.WriteFile(filepath.Join(home, "sessions", "4242.json"), []byte(fmt.Sprintf(`{"pid":4242,"sessionId":%q,"startedAt":%d,"status":"idle"}`, sid, started.UnixMilli())), 0o600)
	for c, want := range map[time.Time]Liveness{started.Add(2 * time.Second): Proven, started.Add(-2 * time.Minute): PIDReused} {
		if es := find(collect(home, procs{4242: c}), "claude-session", sid); len(es) != 1 || es[0].Liveness != want {
			t.Errorf("startedAt, process created at %s: %+v, want %s", c, es, want)
		}
	}
}

// A record whose pidDomain names another machine (a shared ~/.claude) is not checked against this
// machine's pids: unknown, never proven (review P3-10).
func TestARecordFromAnotherMachineIsNotProvenHere(t *testing.T) {
	home, c := t.TempDir(), t0.Add(-time.Hour)
	os.MkdirAll(filepath.Join(home, "sessions"), 0o700)
	rec := fmt.Sprintf(`{"pid":4242,"sessionId":%q,"procStart":%q,"pidDomain":"win32:other-host","status":"busy"}`, sid, filetime(c))
	os.WriteFile(filepath.Join(home, "sessions", "4242.json"), []byte(rec), 0o600)
	r := Collect(Options{ClaudeHome: home, Now: func() time.Time { return t0 }, Created: procs{4242: c}.created, Hostname: "this-host"})
	if es := find(r, "claude-session", sid); len(es) != 1 || es[0].Liveness != Unverified || es[0].Status != Unknown {
		t.Fatalf("%+v", es)
	}
	r = Collect(Options{ClaudeHome: home, Now: func() time.Time { return t0 }, Created: procs{4242: c}.created, Hostname: "OTHER-HOST"})
	if es := find(r, "claude-session", sid); len(es) != 1 || es[0].Liveness != Proven {
		t.Fatalf("the same machine, its name in another case: %+v", es)
	}
}

// A live process with the record's pid and creation time proves the session: busy is running,
// idle is idle, and its resume forbids a second copy.
func TestALiveSessionIsProven(t *testing.T) {
	home, c := t.TempDir(), t0.Add(-time.Hour)
	writeSession(t, home, 4242, sid, "busy", c)
	writeSession(t, home, 4343, "1ca11168-7b40-4a96-a6b7-0f26c365415e", "idle", c)
	r := collect(home, procs{4242: c, 4343: c})
	if es := find(r, "claude-session", sid); len(es) != 1 || es[0].Status != Running || es[0].Liveness != Proven || !strings.Contains(es[0].Resume, "never start a second copy") {
		t.Fatalf("%+v", es)
	}
	if es := find(r, "claude-session", "1ca11168-7b40-4a96-a6b7-0f26c365415e"); len(es) != 1 || es[0].Status != Idle {
		t.Fatalf("%+v", es)
	}
}

// Two records of one session id (a session resumed in a second process): both are listed as a
// duplicate; while one is live, the other is never to be resumed; two live ones are a problem.
func TestADuplicateSessionIsNeverResumedTwice(t *testing.T) {
	home, c := t.TempDir(), t0.Add(-time.Hour)
	writeSession(t, home, 4242, sid, "busy", c)
	writeSession(t, home, 5353, sid, "idle", c.Add(time.Minute))
	r := collect(home, procs{4242: c})
	if len(r.Duplicates) != 1 || r.Duplicates[0].ID != sid || r.Duplicates[0].Live != 1 || len(r.Duplicates[0].Incarnations) != 2 {
		t.Fatalf("duplicates: %+v", r.Duplicates)
	}
	for _, e := range find(r, "claude-session", sid) {
		if e.Liveness != Proven && !strings.HasPrefix(e.Resume, "do not resume") {
			t.Errorf("the dead record of a live session: %+v", e)
		}
	}
	r = collect(home, procs{4242: c, 5353: c.Add(time.Minute)})
	for _, e := range find(r, "claude-session", sid) {
		if !strings.Contains(e.Problem, "more than one live process") {
			t.Errorf("two live processes of one session: %+v", e)
		}
	}
}

// A record being rewritten (read torn) or too large is unreadable, never guessed; the bound on
// records read is reported, not silent.
func TestUnreadableAndTooManyRecordsAreSaid(t *testing.T) {
	home, c := t.TempDir(), t0.Add(-time.Hour)
	for i := 0; i < 12; i++ {
		writeSession(t, home, 1000+i, fmt.Sprintf("%08d-0000-0000-0000-000000000000", i), "idle", c)
	}
	os.WriteFile(filepath.Join(home, "sessions", "9.json"), []byte(`{"pid":9,"sessionId":"x`), 0o600)
	os.WriteFile(filepath.Join(home, "sessions", "8.json"), bytes.Repeat([]byte(" "), 70<<10), 0o600)
	r := Collect(Options{ClaudeHome: home, Now: func() time.Time { return t0 }, Created: procs{}.created, MaxSessions: 10})
	if n := len(r.Entries); n != 10 {
		t.Errorf("%d entries, want the bound, 10", n)
	}
	if r.Omitted["claude session records past the bound"] != 4 {
		t.Errorf("omitted: %v, want 4 records past the bound", r.Omitted)
	}
	r = Collect(Options{ClaudeHome: home, Now: func() time.Time { return t0 }, Created: procs{}.created})
	for _, n := range []string{"8.json", "9.json"} {
		if es := find(r, "claude-session", n); len(es) != 1 || es[0].Status != Unreadable || es[0].Problem == "" {
			t.Errorf("%s: %+v", n, es)
		}
	}
}

// Reads run while a record is rewritten in place: each read sees it whole or unreadable, never a
// session made up of two writes. Run with -race.
func TestReadsDuringRewritesSeeAWholeRecordOrNone(t *testing.T) {
	home, c := t.TempDir(), t0.Add(-time.Hour)
	writeSession(t, home, 4242, sid, "busy", c)
	path := filepath.Join(home, "sessions", "4242.json")
	whole, _ := os.ReadFile(path)
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { // rewrites the record in place, in two halves
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			if f, err := os.OpenFile(path, os.O_WRONLY|os.O_TRUNC, 0); err == nil {
				f.Write(whole[:len(whole)/2])
				f.Write(whole[len(whole)/2:])
				f.Close()
			}
		}
	}()
	var bad []string
	var mu sync.Mutex
	var readers sync.WaitGroup
	for g := 0; g < 8; g++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for i := 0; i < 25; i++ {
				for _, e := range collect(home, procs{4242: c}).Entries {
					ok := e.Status == Unreadable || (e.ID == sid && e.Status == Running && e.Liveness == Proven)
					if !ok {
						mu.Lock()
						bad = append(bad, fmt.Sprintf("%+v", e))
						mu.Unlock()
					}
				}
			}
		}()
	}
	readers.Wait()
	close(stop)
	wg.Wait()
	if len(bad) > 0 {
		t.Fatalf("%d reads neither whole nor unreadable, first: %s", len(bad), bad[0])
	}
}

// logLine is one record of an ixmux.cmdlog.v1 log.
func logLine(at time.Time, id, state string, extra string) string {
	return fmt.Sprintf(`{"at":%q,"target":"c","mandate_id":%q,"incarnation":"inc1","state":%q%s}`, at.Format(time.RFC3339Nano), id, state, extra) + "\n"
}

// ixmuxState writes the state directory of an ixmux that said it runs, with one mandate in each
// state: delivered, uncertain, delivering (in its turn), accepted (not yet sent).
func ixmuxState(t *testing.T, started time.Time) string {
	t.Helper()
	dir := t.TempDir()
	info, _ := json.Marshal(map[string]any{"addr": "127.0.0.1:5000", "pid": 777, "started": started, "version": "v", "epoch": "e1", "state": "running"})
	os.WriteFile(filepath.Join(dir, "ixmux.json"), info, 0o600)
	fp := `,"fp":"f"`
	at := started.Add(time.Minute)
	log := `{"format":"ixmux.cmdlog.v1"}` + "\n" +
		logLine(at, "done1", "accepted", fp) + logLine(at, "done1", "delivering", fp) + logLine(at, "done1", "delivered", `,"ack":{"receipt":"r"}`) +
		logLine(at, "unc1", "accepted", fp) + logLine(at, "unc1", "delivering", fp) + logLine(at, "unc1", "uncertain", `,"cause":"deadline"`) +
		logLine(at, "turn1", "accepted", fp) + logLine(at.Add(time.Second), "turn1", "delivering", fp) +
		logLine(at, "acc1", "accepted", fp)
	os.WriteFile(filepath.Join(dir, "commands.jsonl"), []byte(log), 0o600)
	return dir
}

func snapshot(t *testing.T, dir string) map[string]string {
	t.Helper()
	out := map[string]string{}
	filepath.Walk(dir, func(p string, fi os.FileInfo, err error) error {
		if err == nil && !fi.IsDir() {
			b, _ := os.ReadFile(p)
			out[p] = fi.ModTime().String() + string(b)
		}
		return nil
	})
	return out
}

// An ixmux that crashed (ixmux.json says running, its process is gone): it is unreachable, the
// mandate that was in its turn is uncertain for a person to decide, the one only accepted was never
// sent. Nothing is written: every file is byte for byte and time for time as it was.
func TestACrashedIxmuxShowsItsMandatesForAPerson(t *testing.T) {
	started := t0.Add(-time.Hour)
	dir := ixmuxState(t, started)
	before := snapshot(t, dir)
	r := collect("", procs{}, dir)
	want := map[string]Status{"done1": Done, "unc1": Uncertain, "turn1": Uncertain, "acc1": Done}
	for id, st := range want {
		es := find(r, "mandate", id)
		if len(es) != 1 || es[0].Status != st {
			t.Errorf("%s: %+v, want %s", id, es, st)
			continue
		}
		if st == Uncertain && !strings.Contains(es[0].Resume, "Never send it again") {
			t.Errorf("%s: resume %q", id, es[0].Resume)
		}
	}
	if es := find(r, "mandate", "acc1"); len(es) == 1 && !strings.Contains(es[0].Problem, "never sent") {
		t.Errorf("acc1: %+v", es[0])
	}
	if es := find(r, "ixmux", dir); len(es) != 1 || es[0].Status != Unreachable || !strings.Contains(es[0].Problem, "did not stop cleanly") {
		t.Errorf("ixmux: %+v", es)
	}
	if after := snapshot(t, dir); fmt.Sprint(after) != fmt.Sprint(before) {
		t.Error("the inventory changed the state directory")
	}
	// The same directory with its ixmux live: the mandate in its turn is running.
	r = collect("", procs{777: started.Add(-200 * time.Millisecond)}, dir)
	if es := find(r, "mandate", "turn1"); len(es) != 1 || es[0].Status != Running {
		t.Errorf("turn1 under a live ixmux: %+v", es)
	}
	if es := find(r, "ixmux", dir); len(es) != 1 || es[0].Status != Running || es[0].Liveness != Proven {
		t.Errorf("live ixmux: %+v", es)
	}
}

// Past the bound on finished mandates, the oldest finished ones are counted, not listed; an
// uncertain mandate is never left out.
func TestAnUncertainMandateIsNeverLeftOut(t *testing.T) {
	dir := t.TempDir()
	var b strings.Builder
	b.WriteString(`{"format":"ixmux.cmdlog.v1"}` + "\n")
	for i := 0; i < 30; i++ {
		b.WriteString(logLine(t0.Add(time.Duration(i)*time.Second), fmt.Sprintf("d%02d", i), "delivered", ""))
	}
	for i := 0; i < 3; i++ {
		b.WriteString(logLine(t0.Add(-time.Hour), fmt.Sprintf("u%d", i), "uncertain", `,"cause":"restart"`))
	}
	os.WriteFile(filepath.Join(dir, "commands.jsonl"), []byte(b.String()), 0o600)
	r := Collect(Options{IxmuxStates: []string{dir}, Now: func() time.Time { return t0 }, Created: procs{}.created, MaxDone: 5})
	var done, unc int
	for _, e := range r.Entries {
		switch {
		case e.Kind == "mandate" && e.Status == Done:
			done++
			if e.ID < "d25" {
				t.Errorf("an older finished mandate listed before a newer one: %s", e.ID)
			}
		case e.Kind == "mandate" && e.Status == Uncertain:
			unc++
		}
	}
	if done != 5 || unc != 3 || r.Omitted["finished mandates past the bound in "+dir] != 25 {
		t.Errorf("%d done, %d uncertain, omitted %v; want 5, 3 and 25", done, unc, r.Omitted)
	}
}

// A mandate the log leaves accepted or delivering is unfinished, whatever its ixmux: the bound on
// finished mandates never leaves it out (review P2-3: acc1, accepted under a dead ixmux, was).
func TestAnUnfinishedMandateIsNeverLeftOut(t *testing.T) {
	dir := ixmuxState(t, t0.Add(-time.Hour))
	r := Collect(Options{IxmuxStates: []string{dir}, Now: func() time.Time { return t0 }, Created: procs{}.created, MaxDone: 1})
	for _, id := range []string{"acc1", "turn1", "unc1"} {
		if len(find(r, "mandate", id)) != 1 {
			t.Errorf("%s left out: omitted %v", id, r.Omitted)
		}
	}
}

// A new ixmux starting on the directory between the read of ixmux.json (a dead host) and the read
// of the log may have accepted a mandate: it must not be called never sent (review P2-4).
func TestAnIxmuxStartingDuringTheReadIsReadAgain(t *testing.T) {
	dir := ixmuxState(t, t0.Add(-time.Hour))
	testHookAfterLog = func() {
		info, _ := json.Marshal(map[string]any{"addr": "127.0.0.1:5001", "pid": 888, "started": t0, "epoch": "e2", "state": "running"})
		os.WriteFile(filepath.Join(dir, "ixmux.json"), info, 0o600)
	}
	t.Cleanup(func() { testHookAfterLog = nil })
	r := collect("", procs{888: t0}, dir)
	for _, id := range []string{"acc1", "turn1"} {
		if es := find(r, "mandate", id); len(es) != 1 || es[0].Status != Unknown || strings.Contains(es[0].Resume, "send it again once") {
			t.Errorf("%s read across a start: %+v", id, es)
		}
	}
	if es := find(r, "ixmux", dir); len(es) != 1 || es[0].Status != Unknown {
		t.Errorf("ixmux read across a start: %+v", es)
	}
}

// A log that cannot be read (over its bound, corrupt) is one unreadable entry, never a crash.
func TestAnUnreadableLogIsSaid(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "commands.jsonl"), []byte("not a log\n"), 0o600)
	r := collect("", procs{}, dir)
	if es := find(r, "mandate-log", filepath.Join(dir, "commands.jsonl")); len(es) != 1 || es[0].Status != Unreadable {
		t.Fatalf("%+v", r.Entries)
	}
	if len(r.Gaps) == 0 {
		t.Error("the report must say what it does not read")
	}
}

// The real process check: this process is live and was created before now; a process that has
// exited is no longer.
func TestCreatedProvesALiveProcessOnly(t *testing.T) {
	c, err := created(os.Getpid())
	if err != nil {
		t.Skipf("no process start time on this system: %v", err)
	}
	if c.After(time.Now()) || time.Since(c) > 24*time.Hour {
		t.Errorf("this process created at %s", c)
	}
	exe, _ := os.Executable()
	cmd := exec.Command(exe, "-test.run=^$")
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
	if _, err := created(cmd.Process.Pid); err != ErrNoProcess {
		t.Errorf("an exited process: %v, want ErrNoProcess", err)
	}
}
