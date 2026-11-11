package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// getMandate reads GET /v1/mandates/{id} of s.
func getMandate(t *testing.T, s *Server, id string) (int, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest("GET", "http://"+s.Addr()+"/v1/mandates/"+id, nil)
	req.Header.Set("Authorization", "Bearer "+string(s.token))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var v map[string]any
	if err := json.Unmarshal(b, &v); err != nil {
		t.Fatalf("GET /v1/mandates/%s: %d %s", id, resp.StatusCode, b)
	}
	return resp.StatusCode, v
}

// S6-1: the result of a delivered mandate is read back by its id, also after ixmux restarts (it
// comes from the command log), and not only in the answer of the POST that delivered it.
func TestMandateResultIsReadBackAfterARestart(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{Addr: "127.0.0.1:0", State: filepath.Join(dir, "state"),
		Targets: []Named{{Name: "s", Path: filepath.Join(dir, "session.jsonl")}}}
	s, err := Start(cfg)
	if err != nil {
		t.Fatal(err)
	}
	inc, err := s.fakes[0].Incarnation()
	if err != nil {
		t.Fatal(err)
	}
	res, err := s.eng.Deliver(context.Background(), deliver.Mandate{MandateID: "m1", Target: "s", Incarnation: inc, Text: "hello"})
	if err != nil || res.Ack == nil || res.Ack.Receipt == "" {
		t.Fatalf("deliver: %+v %v", res, err)
	}
	want := res.Ack.Receipt
	check := func(when string, s *Server) {
		code, v := getMandate(t, s, "m1")
		ack, _ := v["ack"].(map[string]any)
		if code != 200 || v["state"] != "delivered" || ack == nil || ack["receipt"] != want {
			t.Errorf("%s: %d %v, want delivered with the receipt %q", when, code, v, want)
		}
	}
	check("before the restart", s)
	s.RequestStop()
	if code := s.Wait(); code != ExitOK {
		t.Fatalf("stop: exit %d", code)
	}

	s, err = Start(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { s.RequestStop(); s.Wait() }()
	check("after the restart", s)
	if code, v := getMandate(t, s, "nope"); code != http.StatusNotFound || v["error"] != deliver.UnknownMandate {
		t.Errorf("an unknown id: %d %v, want 404 unknown-mandate", code, v)
	}
}
