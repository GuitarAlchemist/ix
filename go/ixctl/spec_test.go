package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestSpecValidation(t *testing.T) {
	bad := map[string]string{
		"bad instance name": `{"name":"Local","services":[]}`,
		"bad service name":  `{"name":"x","services":[{"name":"-a","command":"c"}]}`,
		"duplicate":         `{"name":"x","services":[{"name":"a","command":"c"},{"name":"a","command":"c"}]}`,
		"no command":        `{"name":"x","services":[{"name":"a"}]}`,
		"bad restart":       `{"name":"x","services":[{"name":"a","command":"c","restart":"sometimes"}]}`,
		"negative max":      `{"name":"x","services":[{"name":"a","command":"c","max_restarts":-1}]}`,
		"unknown field":     `{"name":"x","services":[{"name":"a","command":"c","shell":"rm -rf /"}]}`,
		"duplicate peer":    `{"name":"x","services":[],"peers":[{"name":"b","state_file":"b.json"},{"name":"b","state_file":"c.json"}]}`,
		"name too long":     `{"name":"x","services":[{"name":"` + strings.Repeat("a", 64) + `","command":"c"}]}`,
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
}
