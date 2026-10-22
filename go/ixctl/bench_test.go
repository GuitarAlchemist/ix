package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The bench end to end, with few iterations: the test binary plays ixctl
// and the fake services.
func TestBenchSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns several instances")
	}
	exe := testExe(t)
	dir := t.TempDir()
	spec := filepath.Join(dir, "spec.json")
	env := map[string]string{"IXCTL_TEST_ROLE": "fake"}
	doc := map[string]any{"name": "bench", "services": []map[string]any{
		{"name": "pod-a", "command": exe, "args": []string{"--nodes", "4", "--rate", "500"}, "env": env},
		{"name": "pod-b", "command": exe, "args": []string{"--crash-after-ms", "100"}, "env": env, "max_restarts": 50},
	}}
	b, _ := json.Marshal(doc)
	os.WriteFile(spec, b, 0o600)
	res, err := runBench(benchConfig{spec: spec, exe: exe, env: []string{"IXCTL_TEST_ROLE=main"},
		n: 2, rtt: 50, fanout: 50, quiet: os.Stderr})
	if err != nil {
		t.Fatal(err)
	}
	out, _ := json.Marshal(res)
	t.Logf("%s", out)
	for name, d := range map[string]*dist{
		"plan": res.PlanUs, "ready": res.ReadyMs, "apply": res.ApplyStartedMs, "first event": res.ApplyFirstEventMs,
		"stop": res.StopMs, "restart": res.RestartMs, "rtt": res.WSStatusRTTUs, "fanout": res.FanoutUs,
		"peer apply": res.PeerApplyConvergedMs,
	} {
		if d == nil || d.N == 0 {
			t.Errorf("%s: no samples (notes %v)", name, res.Notes)
		}
	}
	if res.FanoutClock != "hr_ns" {
		t.Errorf("fan-out clock %s", res.FanoutClock)
	}
}
