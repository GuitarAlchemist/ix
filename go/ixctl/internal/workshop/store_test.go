package workshop

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func fixture(t *testing.T) (*Store, string, string) {
	t.Helper()
	s, err := Open(filepath.Join(t.TempDir(), "workshop-test.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	token, id := paired(t, s)
	return s, token, id
}
func paired(t *testing.T, s *Store) (string, string) {
	t.Helper()
	invite, err := s.NewInvite("https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	d, token, err := s.Claim(invite.Code, "Fixture tablet")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.Authenticate(token); !errors.Is(err, ErrPending) {
		t.Fatalf("pending: %v", err)
	}
	if err = s.Decide(d.ID, "approved"); err != nil {
		t.Fatal(err)
	}
	return token, d.ID
}
func draft() Project {
	return Project{ID: "project-one", Title: "Atelier", Workspaces: []Workspace{{ID: "workspace-one", Title: "Mes compteurs", Widgets: []Widget{{ID: "widget-one", Title: "Compteur", Kind: "counter"}}, IXLETDefinitions: []IXLETDefinition{{ID: "ixlet-one", Title: "Compteur", Kind: "counter", Status: "draft"}}}}}
}
func operation(id string, base int64, p Project) Operation {
	return Operation{OpID: id, BaseRevision: base, Project: p}
}

func TestSaveRetryConflictAndRestart(t *testing.T) {
	s, token, _ := fixture(t)
	op := operation("op-one", 0, draft())
	first, err := s.Save(token, op)
	if err != nil {
		t.Fatal(err)
	}
	retry, err := s.Save(token, op)
	if err != nil || retry.Revision != first.Revision {
		t.Fatalf("retry: %+v %v", retry, err)
	}
	changed := op
	changed.Project.Title = "Different"
	if _, err = s.Save(token, changed); !errors.Is(err, ErrOpReuse) {
		t.Fatalf("op reuse: %v", err)
	}
	other, _ := paired(t, s)
	p := draft()
	p.Title = "Device B"
	if _, err = s.Save(other, operation("op-two", 0, p)); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflict: %v", err)
	}
	sync, err := s.Changes(token, 0)
	if err != nil || len(sync.Events) != 1 {
		t.Fatalf("changes: %+v %v", sync, err)
	}
	if sync.Events[0].Project.Workspaces[0].IXLETDefinitions[0].Status != "draft" {
		t.Fatal("not draft")
	}
	path := s.path
	s.Close()
	reopened, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	sync, err = reopened.Changes(token, 0)
	if err != nil || len(sync.Events) != 1 || sync.Events[0].Revision != 1 {
		t.Fatalf("restart: %+v %v", sync, err)
	}
	retry, err = reopened.Save(token, op)
	if err != nil || retry.Revision != 1 {
		t.Fatalf("retry after restart: %+v %v", retry, err)
	}
	empty, err := reopened.Changes(token, sync.Cursor)
	if err != nil || len(empty.Events) != 0 {
		t.Fatalf("resume: %+v %v", empty, err)
	}
	p.Title = "Merged"
	if _, err = reopened.Save(other, operation("op-three", 1, p)); err != nil {
		t.Fatal(err)
	}
	sync, err = reopened.Changes(token, 1)
	if err != nil || len(sync.Events) != 1 || sync.Events[0].Revision != 2 {
		t.Fatalf("resume revision: %+v %v", sync, err)
	}
	var count int
	if err = reopened.db.QueryRow("SELECT count(*) FROM workshop_revisions").Scan(&count); err != nil || count != 2 {
		t.Fatalf("history: %d %v", count, err)
	}
}
func TestPairingExpirationReuseDecisionAndRevocation(t *testing.T) {
	s, _, _ := fixture(t)
	now := time.Date(2026, 11, 5, 0, 0, 0, 0, time.UTC)
	s.now = func() time.Time { return now }
	invite, err := s.NewInvite("https://workshop.invalid")
	if err != nil {
		t.Fatal(err)
	}
	if invite.URL != "https://workshop.invalid/workshop/#pair="+invite.Code {
		t.Fatal(invite.URL)
	}
	if len(invite.QRSVG) == 0 {
		t.Fatal("QR absent")
	}
	now = now.Add(2 * time.Minute)
	if _, _, err = s.Claim(invite.Code, "Expired"); !errors.Is(err, ErrInvite) {
		t.Fatalf("expired: %v", err)
	}
	invite, _ = s.NewInvite("https://workshop.invalid")
	d, token, err := s.Claim(invite.Code, "Pending")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = s.Claim(invite.Code, "Reuse"); !errors.Is(err, ErrInvite) {
		t.Fatalf("reuse: %v", err)
	}
	if _, err = s.Save(token, operation("pending-op", 0, draft())); !errors.Is(err, ErrPending) {
		t.Fatalf("unapproved: %v", err)
	}
	if err = s.Decide(d.ID, "refused"); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Authenticate(token); !errors.Is(err, ErrRefused) {
		t.Fatalf("refused: %v", err)
	}
	if err = s.Decide(d.ID, "approved"); err == nil {
		t.Fatal("refusal resurrected")
	}
	token, id := paired(t, s)
	if err = s.Decide(id, "revoked"); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Save(token, operation("revoked-op", 0, draft())); !errors.Is(err, ErrRevoked) {
		t.Fatalf("revoked: %v", err)
	}
}
func TestOutOfScopeDataRefused(t *testing.T) {
	s, token, _ := fixture(t)
	p := draft()
	p.Workspaces = nil
	if _, err := s.Save(token, operation("null-workspaces", 0, p)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("null workspaces: %v", err)
	}
	p = draft()
	p.Workspaces[0].Widgets = nil
	if _, err := s.Save(token, operation("null-widgets", 0, p)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("null widgets: %v", err)
	}
	p = draft()
	p.Workspaces[0].IXLETDefinitions = nil
	if _, err := s.Save(token, operation("null-definitions", 0, p)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("null definitions: %v", err)
	}
	p = draft()
	p.Workspaces[0].IXLETDefinitions[0].Kind = "shell"
	if _, err := s.Save(token, operation("shell-op", 0, p)); !errors.Is(err, ErrInvalid) {
		t.Fatal(err)
	}
	p = draft()
	p.ID = "../scribe"
	if _, err := s.Save(token, operation("path-op", 0, p)); !errors.Is(err, ErrInvalid) {
		t.Fatal(err)
	}
	if _, err := s.NewInvite("https://evil.invalid/workshop/"); err == nil {
		t.Fatal("ambiguous origin accepted")
	}
	raw, _ := json.Marshal(draft())
	if string(raw) == "" {
		t.Fatal("invalid fixture")
	}
}
