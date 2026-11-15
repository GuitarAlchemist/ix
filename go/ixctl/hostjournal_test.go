package main

import (
	"bytes"
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
	if !strings.Contains(resumeChecks(r), "host.ready of PID 4242") {
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
