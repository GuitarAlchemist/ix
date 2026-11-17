package main

import (
	"bytes"
	"regexp"
	"strings"
	"testing"
)

// The host.ready line the resume check needs may have been moved to service.log.1 by the
// rotation (the log went past its size after the host said ready): the check reads it there
// rather than finding no ready line.
func TestTheReadyLineIsFoundAcrossTheRotation(t *testing.T) {
	inv := mustInventory(t, controlInventory)
	j := &journalRun{pid: 4242}
	p := j.probes()
	var read []string
	p.readTail = func(path string, max int64) ([]byte, error) {
		read = append(read, path)
		if path == journalPath+".1" {
			return []byte(journalReady("4242", "127.0.0.1:59762") + "\n" + `{"t":2,"event":"host.stop","pid":1111}`), nil
		}
		return []byte(`{"t":3,"event":"workshop.failed","error":"x"}` + "\n"), nil
	}
	r := checkResume(inv, p)[0]
	if r.Verdict != verdictOK {
		t.Fatalf("verdict %s after a rotation, read %v: %q", r.Verdict, read, r.Checks)
	}
	if !strings.Contains(resumeChecks(r), "host.ready of PID 4242") || !strings.Contains(resumeChecks(r), journalPath+".1:") {
		t.Fatalf("checks %q, want the ready line of PID 4242", r.Checks)
	}
}

// A newer ready line in service.log wins over the rotated file: the rotation is read only when
// the current file holds none.
func TestTheCurrentLogsReadyLineWinsOverTheRotatedOne(t *testing.T) {
	read := func(path string, _ int64) ([]byte, error) {
		if strings.HasSuffix(path, ".1") {
			return []byte(journalReady("1111", "127.0.0.1:50001")), nil
		}
		return []byte(journalReady("4242", "127.0.0.1:59762")), nil
	}
	r, from, found, err := lastReady(read, journalPath, resumeJournalTail)
	if err != nil || !found || r.Pid != 4242 || from != journalPath {
		t.Fatalf("lastReady: %+v from %q found %v err %v, want PID 4242 from the current log", r, from, found, err)
	}
}

// What the host writes is what the check reads back: one record, no second parser.
func TestTheHostsReadyRecordReadsBack(t *testing.T) {
	var buf bytes.Buffer
	l := &hostLog{w: &buf}
	l.ready(hostReady{Name: "n", Addr: "127.0.0.1:59762", Pid: 4242, Epoch: "e", StateFile: "s.json", Version: "v"})
	r, _, found, err := lastReady(func(string, int64) ([]byte, error) { return buf.Bytes(), nil }, journalPath, resumeJournalTail)
	if err != nil || !found || r.Addr != "127.0.0.1:59762" || r.Pid != 4242 || r.Epoch != "e" || r.Event != "host.ready" || r.T == 0 {
		t.Fatalf("read back %+v (found %v, err %v) from %s", r, found, err, buf.String())
	}
}

// A host that ended (its final host.stop: observed or unobserved) after its ready line is not the
// current host: a later host under the same PID (a PID comes back) has not said ready yet, and
// the ended host's port is not this one's. The check answers unknown and probes nothing, in the
// current log as across the rotation. A stop still in progress (requested, received) ends
// nothing.
func TestAReadyLineFollowedByTheHostsEndIsNotTheCurrentHosts(t *testing.T) {
	inv := mustInventory(t, controlInventory)
	ready := journalReady("4242", "127.0.0.1:59762")
	ended := `{"event":"host.stop","ms":12,"phase":"observed","source":"scm","t":5}`
	unobserved := `{"event":"host.stop","ms":30000,"phase":"unobserved","source":"scm","t":5}`
	for name, files := range map[string][2]string{
		"across the rotation":        {ended + "\n", ready + "\n"},
		"in the current log":         {ready + "\n" + ended + "\n", ""},
		"an unobserved end, rotated": {unobserved + "\n", ready + "\n"},
	} {
		j := &journalRun{pid: 4242}
		p := j.probes()
		p.readTail = func(path string, _ int64) ([]byte, error) {
			if strings.HasSuffix(path, ".1") {
				return []byte(files[1]), nil
			}
			return []byte(files[0]), nil
		}
		p.listening = func(string) (bool, error) { return false, nil }
		r := checkResume(inv, p)[0]
		if r.Verdict != verdictUnknown || len(j.healthURL) != 0 || !strings.Contains(resumeChecks(r), "stopped") ||
			!strings.Contains(resumeChecks(r), "the last host in "+journalPath) {
			t.Fatalf("%s: verdict %s, health probed %v: %q", name, r.Verdict, j.healthURL, r.Checks)
		}
	}
	for _, phase := range []string{"requested", "received"} {
		inProgress := ready + "\n" + `{"event":"host.stop","phase":"` + phase + `","source":"scm","t":5}` + "\n"
		read := func(path string, _ int64) ([]byte, error) {
			if strings.HasSuffix(path, ".1") {
				return nil, nil
			}
			return []byte(inProgress), nil
		}
		if _, _, found, err := lastReady(read, journalPath, resumeJournalTail); !found || err != nil {
			t.Fatalf("a stop still in progress (%s) ended the ready line (found %v, err %v)", phase, found, err)
		}
	}
}

// The host.ready line keeps the bytes the map wrote before hostReady: keys in order, every field,
// one line.
func TestTheHostsReadyLineKeepsItsBytes(t *testing.T) {
	var buf bytes.Buffer
	(&hostLog{w: &buf}).ready(hostReady{Name: "n", Addr: "127.0.0.1:59762", Pid: 4242, Epoch: "e", StateFile: "s.json", Version: "v"})
	want := regexp.MustCompile(`^\{"addr":"127\.0\.0\.1:59762","epoch":"e","event":"host\.ready","name":"n","pid":4242,"state_file":"s\.json","t":[0-9]+,"version":"v"\}\n$`)
	if !want.Match(buf.Bytes()) {
		t.Fatalf("host.ready line %q", buf.String())
	}
}

// The stop lines, as the host's own writer emits them, are what the reader of its ready line
// takes as its end: a final phase (observed, unobserved) ends it, a phase in progress does not.
// Each line keeps the bytes hostLog.event wrote before hostLog.stop. The phases are literals
// here, not the constants, so a renamed phase goes red.
func TestTheHostsStopLinesEndItsReadyLineOnlyWhenFinal(t *testing.T) {
	for _, c := range []struct {
		phase string
		final bool
	}{{"requested", false}, {"received", false}, {"observed", true}, {"unobserved", true}} {
		var buf bytes.Buffer
		l := &hostLog{w: &buf}
		l.ready(hostReady{Name: "n", Addr: "127.0.0.1:59762", Pid: 4242, Epoch: "e"})
		mark := buf.Len()
		l.stop(c.phase, "scm", map[string]any{"ms": int64(12)})
		line := regexp.MustCompile(`^\{"event":"host\.stop","ms":12,"phase":"` + c.phase + `","source":"scm","t":[0-9]+\}\n$`)
		if !line.Match(buf.Bytes()[mark:]) {
			t.Fatalf("%s: stop line %q", c.phase, buf.Bytes()[mark:])
		}
		if _, decided, ok := lastReadyIn(buf.Bytes()); !decided || ok == c.final {
			t.Fatalf("%s: decided %v, ready found %v, want found %v", c.phase, decided, ok, !c.final)
		}
	}
}

// The epoch refusal names the file the ready line was read from: after a rotation, the .1.
func TestTheEpochRefusalNamesTheRotatedFile(t *testing.T) {
	inv := mustInventory(t, controlInventory)
	j := &journalRun{pid: 4242, healthBody: `{"ok":true,"epoch":"other"}`}
	p := j.probes()
	p.readTail = func(path string, _ int64) ([]byte, error) {
		if strings.HasSuffix(path, ".1") {
			return []byte(journalReady("4242", "127.0.0.1:59762") + "\n"), nil
		}
		return []byte(`{"t":3,"event":"workshop.failed","error":"x"}` + "\n"), nil
	}
	r := checkResume(inv, p)[0]
	if want := "the host.ready line of PID 4242 in " + journalPath + ".1 is epoch"; r.Verdict != verdictUnknown || !strings.Contains(resumeChecks(r), want) {
		t.Fatalf("verdict %s, checks %q, want %q", r.Verdict, r.Checks, want)
	}
}
