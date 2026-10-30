package main

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func TestSpecValidation(t *testing.T) {
	bad := map[string]string{
		"bad instance name":    `{"name":"Local","services":[]}`,
		"bad service name":     `{"name":"x","services":[{"name":"-a","command":"c"}]}`,
		"duplicate":            `{"name":"x","services":[{"name":"a","command":"c"},{"name":"a","command":"c"}]}`,
		"no command":           `{"name":"x","services":[{"name":"a"}]}`,
		"bad restart":          `{"name":"x","services":[{"name":"a","command":"c","restart":"sometimes"}]}`,
		"negative max":         `{"name":"x","services":[{"name":"a","command":"c","max_restarts":-1}]}`,
		"negative wall":        `{"name":"x","services":[{"name":"a","command":"c","limits":{"wall_ms":-1}}]}`,
		"negative memory":      `{"name":"x","services":[{"name":"a","command":"c","limits":{"memory_mb":-1}}]}`,
		"wall past a Duration": `{"name":"x","services":[{"name":"a","command":"c","limits":{"wall_ms":9223372036855}}]}`,
		"memory past int64":    `{"name":"x","services":[{"name":"a","command":"c","limits":{"memory_mb":8796093022208}}]}`,
		"negative cpu":         `{"name":"x","services":[{"name":"a","command":"c","limits":{"cpu_ms":-1}}]}`,
		"cpu past a Duration":  `{"name":"x","services":[{"name":"a","command":"c","limits":{"cpu_ms":9223372036855}}]}`,
		"negative processes":   `{"name":"x","services":[{"name":"a","command":"c","limits":{"processes":-1}}]}`,
		"processes past DWORD": `{"name":"x","services":[{"name":"a","command":"c","limits":{"processes":2147483649}}]}`,
		"unknown limit":        `{"name":"x","services":[{"name":"a","command":"c","limits":{"cpu":1}}]}`,
		"unknown field":        `{"name":"x","services":[{"name":"a","command":"c","shell":"rm -rf /"}]}`,
		"duplicate peer":       `{"name":"x","services":[],"peers":[{"name":"b","state_file":"b.json"},{"name":"b","state_file":"c.json"}]}`,
		"name too long":        `{"name":"x","services":[{"name":"` + strings.Repeat("a", 64) + `","command":"c"}]}`,
	}
	for what, doc := range bad {
		if _, err := ParseSpec([]byte(doc), t.TempDir()); err == nil {
			t.Errorf("%s: accepted", what)
		}
	}

	dir := t.TempDir()
	s, err := ParseSpec([]byte(`{"name":"local","services":[{"name":"pod-a","command":"../bin/pod","dir":"work"}],
		"peers":[{"name":"b","state_file":"b.json"}]}`), dir)
	if err != nil {
		t.Fatal(err)
	}
	sv := s.Services[0]
	if sv.Restart != restartOnFailure || *sv.MaxRestarts != 5 {
		t.Errorf("defaults: restart %q max %d", sv.Restart, *sv.MaxRestarts)
	}
	if sv.Command != filepath.Join(dir, "..", "bin", "pod") || sv.Dir != filepath.Join(dir, "work") {
		t.Errorf("relative paths resolve against the spec folder: %q %q", sv.Command, sv.Dir)
	}
	if s.Peers[0].StateFile != filepath.Join(dir, "b.json") {
		t.Errorf("peer state file: %q", s.Peers[0].StateFile)
	}
}

func TestParseEventLine(t *testing.T) {
	cases := []struct {
		line string
		ok   bool
		kind string
	}{
		{`{"kind":"stats","t":12,"data":{"nodes":[]}}`, true, "stats"},
		{`  {"kind":"meta"}  `, true, "meta"},
		{`{"kind":3,"data":{}}`, false, ""},
		{`{"data":{}}`, false, ""},
		{`["kind"]`, false, ""},
		{`hello`, false, ""},
		{`{"kind":"x"`, false, ""},
	}
	for _, c := range cases {
		ev, ok := parseEventLine([]byte(c.line))
		if ok != c.ok || ev.Kind != c.kind {
			t.Errorf("%s: got %v %q", c.line, ok, ev.Kind)
		}
	}
	ev, _ := parseEventLine([]byte(`{"kind":"stats","t":12,"data":{"nodes":[1]}}`))
	if ev.T != 12 || string(ev.Data) != `{"nodes":[1]}` {
		t.Errorf("t and data kept: %+v", ev)
	}
	// t is milliseconds: a fractional value is rounded, not truncated (F1.2), and a
	// value that is not a finite number is dropped rather than read as 0.
	for raw, want := range map[string]int64{`1760700000000.6`: 1760700000001, `0.4`: 0, `"12"`: 0, `null`: 0, `1e400`: 0} {
		ev, ok := parseEventLine([]byte(`{"kind":"stats","t":` + raw + `}`))
		if !ok || ev.T != want {
			t.Errorf("t %s: got %d ok %v, want %d", raw, ev.T, ok, want)
		}
	}
}

// Only kind, t and data are kept from a service's line: any other top-level field is dropped,
// and a null data is no data. The README's svc/<name> row says so.
func TestParseEventLineKeepsOnlyKindTAndData(t *testing.T) {
	ev, ok := parseEventLine([]byte(`{"kind":"tick","n":7,"t":5,"data":{"n":7},"run":"forged"}`))
	if !ok || ev.Kind != "tick" || ev.T != 5 || string(ev.Data) != `{"n":7}` {
		t.Fatalf("event %+v ok %v, want tick at 5 with its data", ev, ok)
	}
	b, _ := json.Marshal(ev)
	var top map[string]json.RawMessage
	json.Unmarshal(b, &top)
	if len(top) != 3 || top["n"] != nil || top["run"] != nil {
		t.Errorf("event %s, want only kind, t and data", b)
	}
	if ev, _ := parseEventLine([]byte(`{"kind":"tick","data":null}`)); ev.Data != nil {
		t.Errorf("null data kept: %s", ev.Data)
	}
}
