package main

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

// S3B-2 (Codex round 6, P1): session.jsonl is replaced between the status and the delivery. The
// mandate is refused stale-incarnation and nothing is written to the new session.
func TestFakeSessionReplacedIsStale(t *testing.T) {
	dir := t.TempDir()
	session := filepath.Join(dir, "session.jsonl")
	p := start(t, filepath.Join(dir, "state"), nil, "--target", "s="+session)
	inc := incarnation(p, "s")
	const other = `{"mandate_id":"x","incarnation":"another","text":"another session"}` + "\n"
	if err := os.WriteFile(session+".new", []byte(other), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(session+".new", session); err != nil {
		t.Fatal(err)
	}
	var res result
	if code := p.do("POST", "/v1/mandates", mandate("m1", inc), &res); code != http.StatusConflict || res.Error != "stale-incarnation" {
		t.Errorf("deliver to the replaced session: %d %+v, want 409 stale-incarnation", code, res)
	}
	if b, _ := os.ReadFile(session); string(b) != other {
		t.Errorf("the replacement session was written to:\n%s", b)
	}
	if now := incarnation(p, "s"); now == inc {
		t.Errorf("status still gives incarnation %s after the replacement", now)
	}
}
