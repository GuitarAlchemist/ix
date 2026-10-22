package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The delta@1 kind tails Delta's JSONL journal (one JSON object per line, as brickyard's
// brain writes it: t, t_ms, event, pid, journal_seq and the event's own fields) and emits one
// line per tick as a record, so the declared rate 1000/every_ms stays the bound whatever the
// journal's own pace. The fixture is the writer's real shapes with neutral texts.

func deltaPlane(t *testing.T, journal string) *Plane {
	t.Helper()
	p := probePlane(t)
	p.SetDeltaJournal(journal)
	return p
}

// fixtureJournal copies testdata/delta-events.jsonl into a scratch file the test may append to.
func fixtureJournal(t *testing.T) string {
	t.Helper()
	src, err := os.ReadFile(filepath.Join("testdata", "delta-events.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "delta-events.jsonl")
	if err := os.WriteFile(path, src, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func appendJournal(t *testing.T, path, text string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString(text); err != nil {
		t.Fatal(err)
	}
}

func startDelta(t *testing.T, p *Plane, name string) {
	t.Helper()
	mustCreate(t, p, node(name, "delta@1", 10, nil))
	if _, err := probeOp(t, p, Request{Op: "probe.start", Node: name}); err != nil {
		t.Fatal(err)
	}
}

func TestTheDeltaKindEmitsEachJournalLineInOrderWithItsOffset(t *testing.T) {
	journal := fixtureJournal(t)
	p := deltaPlane(t, journal)
	defer p.Shutdown()

	raw, err := probeOp(t, p, Request{Op: "probe.kinds"})
	if err != nil {
		t.Fatal(err)
	}
	var kinds struct {
		Kinds []struct {
			Kind   string   `json:"kind"`
			Fields []string `json:"fields"`
			Doc    string   `json:"doc"`
		} `json:"kinds"`
	}
	if err := json.Unmarshal(raw, &kinds); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, k := range kinds.Kinds {
		if k.Kind == "delta@1" {
			found = true
			for _, want := range []string{"event", "journal_seq", "t_ms", "pid", "offset", "utterance_id", "say_id", "route", "cost"} {
				if !contains(k.Fields, want) {
					t.Errorf("delta@1 fields lack %q: %v", want, k.Fields)
				}
			}
			if !strings.Contains(k.Doc, "journal") {
				t.Errorf("delta@1 doc should say what it reads: %q", k.Doc)
			}
		}
	}
	if !found {
		t.Fatalf("probe.kinds does not list delta@1: %s", raw)
	}

	startDelta(t, p, "delta")
	recs := readSamples(t, p, "probe/delta", 9, 3*time.Second)
	if len(recs) != 9 {
		t.Fatalf("got %d records, want the 9 lines of the fixture", len(recs))
	}
	wantEvents := []string{"start", "heard", "say", "speaking", "heard", "spoken", "heard", "say", "stop"}
	lastOffset := -1.0
	for i, r := range recs {
		if r["event"] != wantEvents[i] {
			t.Errorf("record %d: event %v, want %s", i, r["event"], wantEvents[i])
		}
		if _, isErr := r["error"]; isErr {
			t.Errorf("record %d is an error: %v", i, r["error"])
		}
		if r["kind"] != "delta@1" || r["node"] != "delta" {
			t.Errorf("record %d not stamped by the tree: %v", i, r)
		}
		if r["journal_seq"] != float64(i+1) {
			t.Errorf("record %d: journal_seq %v, want %d (the writer's own counter passes through)", i, r["journal_seq"], i+1)
		}
		if r["pid"] != float64(98996) {
			t.Errorf("record %d: pid %v", i, r["pid"])
		}
		off, ok := r["offset"].(float64)
		if !ok || off <= lastOffset {
			t.Errorf("record %d: offset %v must be the line's byte offset, increasing (last %v)", i, r["offset"], lastOffset)
		}
		lastOffset = off
	}
	// The event's own fields pass through untouched, in their own types.
	heard := recs[1]
	if heard["route"] != "channel" || heard["triage"] != "claude" || heard["asr_ms"] != float64(2179) || heard["utterance_id"] != "1" {
		t.Errorf("heard fields: %v", heard)
	}
	if recs[6]["parrot"] != true || recs[6]["promise"] != "later" {
		t.Errorf("optional heard fields must pass through: %v", recs[6])
	}
	if recs[8]["replaced_by"] != float64(99120) {
		t.Errorf("stop fields: %v", recs[8])
	}
	if recs[0]["offset"] != float64(0) {
		t.Errorf("first line sits at offset 0, got %v", recs[0]["offset"])
	}

	// Nothing follows the last line: the sampler waits, it does not emit errors or repeats.
	more := readSamples(t, p, "probe/delta", 10, 300*time.Millisecond)
	if len(more) != 9 {
		t.Fatalf("after the journal's end the pipe holds %d records, want still 9", len(more))
	}
}

func TestTheDeltaKindFollowsAppendsAndWaitsForTheNewline(t *testing.T) {
	journal := fixtureJournal(t)
	p := deltaPlane(t, journal)
	defer p.Shutdown()
	startDelta(t, p, "d")
	if got := readSamples(t, p, "probe/d", 9, 3*time.Second); len(got) != 9 {
		t.Fatalf("fixture: %d records", len(got))
	}

	// A line without its newline is a line still being written: nothing is emitted for it.
	appendJournal(t, journal, `{"event":"heard","journal_seq":10,"pid":98996,"route":"channel","t_ms":1761162070000,"text":"Bonjour"`)
	if got := readSamples(t, p, "probe/d", 10, 300*time.Millisecond); len(got) != 9 {
		t.Fatalf("a partial line was emitted: %d records", len(got))
	}
	appendJournal(t, journal, ",\"utterance_id\":\"4\"}\n")
	appendJournal(t, journal, `{"by":"claude","event":"say","journal_seq":11,"pid":98996,"say_id":3,"t_ms":1761162071000,"text":"Bonjour.","utterance_id":"4"}`+"\n")
	got := readSamples(t, p, "probe/d", 11, 3*time.Second)
	if len(got) != 11 {
		t.Fatalf("after two appended lines: %d records, want 11", len(got))
	}
	if got[9]["event"] != "heard" || got[9]["utterance_id"] != "4" || got[9]["text"] != "Bonjour" {
		t.Errorf("the completed line is emitted whole: %v", got[9])
	}
	if got[10]["event"] != "say" || got[10]["say_id"] != float64(3) {
		t.Errorf("the next line follows: %v", got[10])
	}
}

func TestADeltaLineThatIsNotAJSONObjectIsAnErrorRecordNamingItsOffset(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	first := `{"event":"start","journal_seq":1,"pid":1,"t_ms":1,"version":"x"}` + "\n"
	if err := os.WriteFile(path, []byte(first+"not json\n"+`[1,2]`+"\n"+`{"event":"stop","journal_seq":2,"pid":1,"t_ms":2}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	p := deltaPlane(t, path)
	defer p.Shutdown()
	startDelta(t, p, "d")
	got := readSamples(t, p, "probe/d", 4, 3*time.Second)
	if len(got) != 4 {
		t.Fatalf("%d records, want 4 (two good lines, two bad)", len(got))
	}
	for i, off := range []int{len(first), len(first) + len("not json\n")} {
		e, _ := got[i+1]["error"].(string)
		if !strings.Contains(e, "offset") || !strings.Contains(e, itoa(off)) {
			t.Errorf("bad line %d: error %q should name offset %d", i, e, off)
		}
	}
	if got[3]["event"] != "stop" {
		t.Errorf("the line after the bad ones still flows: %v", got[3])
	}
}

func TestAMissingDeltaJournalIsSaidOnceAndATruncatedOneRestartsAtZero(t *testing.T) {
	path := filepath.Join(t.TempDir(), "later.jsonl")
	p := deltaPlane(t, path)
	defer p.Shutdown()
	startDelta(t, p, "d")

	got := readSamples(t, p, "probe/d", 5, 400*time.Millisecond)
	if len(got) != 1 {
		t.Fatalf("a missing journal is said once, got %d records: %v", len(got), got)
	}
	if e, _ := got[0]["error"].(string); !strings.Contains(e, "later.jsonl") {
		t.Errorf("the error names the file: %q", e)
	}

	// The journal appears: its lines flow, with no second error.
	line := func(seq int) string {
		return `{"event":"heard","journal_seq":` + itoa(seq) + `,"pid":7,"t_ms":` + itoa(seq) + `}` + "\n"
	}
	if err := os.WriteFile(path, []byte(line(1)+line(2)), 0o600); err != nil {
		t.Fatal(err)
	}
	got = readSamples(t, p, "probe/d", 3, 3*time.Second)
	if len(got) != 3 || got[1]["journal_seq"] != float64(1) || got[2]["journal_seq"] != float64(2) {
		t.Fatalf("after the file appears: %v", got)
	}

	// The file shrinks (a new journal under the same name): one error record says so, and the
	// tail restarts at offset 0 instead of waiting for the old length.
	if err := os.WriteFile(path, []byte(line(1)), 0o600); err != nil {
		t.Fatal(err)
	}
	got = readSamples(t, p, "probe/d", 5, 3*time.Second)
	if len(got) != 5 {
		t.Fatalf("after truncation: %d records: %v", len(got), got)
	}
	if e, _ := got[3]["error"].(string); !strings.Contains(e, "shrank") {
		t.Errorf("truncation is said: %v", got[3])
	}
	if got[4]["journal_seq"] != float64(1) || got[4]["offset"] != float64(0) {
		t.Errorf("the restart reads from offset 0: %v", got[4])
	}
}

func TestAnOversizedDeltaLineIsAnErrorAndTheTailKeepsItsMemoryBounded(t *testing.T) {
	path := filepath.Join(t.TempDir(), "big.jsonl")
	head := `{"event":"start","journal_seq":1,"pid":1,"t_ms":1}` + "\n"
	big := `{"event":"heard","text":"` + strings.Repeat("x", journalMaxLine) + `"}` + "\n"
	tail := `{"event":"stop","journal_seq":3,"pid":1,"t_ms":3}` + "\n"
	if err := os.WriteFile(path, []byte(head+big+tail), 0o600); err != nil {
		t.Fatal(err)
	}
	p := deltaPlane(t, path)
	defer p.Shutdown()
	startDelta(t, p, "d")
	got := readSamples(t, p, "probe/d", 3, 5*time.Second)
	if len(got) != 3 {
		t.Fatalf("%d records, want 3 (start, the oversized line as one error, stop): %v", len(got), got)
	}
	e, _ := got[1]["error"].(string)
	if !strings.Contains(e, "offset "+itoa(len(head))) || !strings.Contains(e, "max_line") {
		t.Errorf("the oversized line is one error naming its offset and the bound: %q", e)
	}
	if got[2]["event"] != "stop" || got[2]["offset"] != float64(len(head)+len(big)) {
		t.Errorf("the line after it flows at its own offset: %v", got[2])
	}
}

func TestADeltaNodeIsRefusedWhenThePlaneHasNoJournal(t *testing.T) {
	p := probePlane(t)
	defer p.Shutdown()
	refused(t, create(t, p, node("d", "delta@1", 10, nil)), "delta@1", "--delta-journal")
	if nodes := listNodes(t, p); len(nodes) != 0 {
		t.Fatalf("a refused node must not be listed: %v", nodes)
	}
}

func contains(xs []string, x string) bool {
	for _, s := range xs {
		if s == x {
			return true
		}
	}
	return false
}
