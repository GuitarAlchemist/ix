package workshop

import (
	"database/sql"
	"errors"
	"path/filepath"
	"testing"
)

func act(t *testing.T, s *Store, token, op, action string) XletState {
	t.Helper()
	state, err := s.Act(token, XletAction{OpID: op, ProjectID: "project-one", DefinitionID: "ixlet-one", Action: action})
	if err != nil {
		t.Fatalf("%s %s: %v", op, action, err)
	}
	return state
}

func TestCounterIXLETRunsOnTheDesktopOnlyOnceActivated(t *testing.T) {
	s, token, _ := fixture(t)
	if _, err := s.Act(token, XletAction{OpID: "early", ProjectID: "project-one", DefinitionID: "ixlet-one", Action: "activate"}); !errors.Is(err, ErrNotSaved) {
		t.Fatalf("a definition not yet committed on the Desktop must not activate: %v", err)
	}
	p := draft()
	p.Workspaces[0].Widgets[0].InitialValue = 5
	if _, err := s.Save(token, operation("save", 0, p)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Act(token, XletAction{OpID: "inc-before", ProjectID: "project-one", DefinitionID: "ixlet-one", Action: "increment"}); !errors.Is(err, ErrNotActive) {
		t.Fatalf("an inactive IXLET must not run: %v", err)
	}
	if _, err := s.Act(token, XletAction{OpID: "other", ProjectID: "project-one", DefinitionID: "widget-one", Action: "activate"}); !errors.Is(err, ErrNotSaved) {
		t.Fatalf("only an IXLET definition activates, not a widget id: %v", err)
	}
	if got := act(t, s, token, "on", "activate"); got.Value != 5 || got.Version != 1 || got.Kind != "counter" {
		t.Fatalf("activation starts from the presenting widget's initial value: %+v", got)
	}
	act(t, s, token, "a", "increment")
	act(t, s, token, "b", "increment")
	if got := act(t, s, token, "c", "decrement"); got.Value != 6 || got.Version != 4 {
		t.Fatalf("5 +1 +1 -1 must be 6 at version 4: %+v", got)
	}
	if got := act(t, s, token, "on-again", "activate"); got.Value != 6 {
		t.Fatalf("activating again must not reset a running IXLET: %+v", got)
	}

	// Exactly once: the same op_id replays the original result and does not run again.
	if got := act(t, s, token, "a", "increment"); got.Value != 6 || got.Version != 2 {
		t.Fatalf("a repeated op_id must return its first result (value 6, version 2) without running again: %+v", got)
	}
	if _, err := s.Act(token, XletAction{OpID: "a", ProjectID: "project-one", DefinitionID: "ixlet-one", Action: "reset"}); !errors.Is(err, ErrOpReuse) {
		t.Fatalf("a reused op_id with another action must be refused: %v", err)
	}
	if _, err := s.Act(token, XletAction{OpID: "x", ProjectID: "project-one", DefinitionID: "ixlet-one", Action: "delete"}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("unknown action: %v", err)
	}

	// A second approved device sees and drives the same instance.
	other, _ := paired(t, s)
	states, err := s.Xlets(other)
	if err != nil || len(states) != 1 || states[0].Value != 6 {
		t.Fatalf("second device must read the shared instance: %+v %v", states, err)
	}
	if got := act(t, s, other, "z", "reset"); got.Value != 0 || got.Version != 5 {
		t.Fatalf("reset: %+v", got)
	}

	// Durable across a restart of the Desktop.
	path := s.path
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	if states, err = s2.Xlets(token); err != nil || len(states) != 1 || states[0].Value != 0 || states[0].Version != 5 {
		t.Fatalf("after restart: %+v %v", states, err)
	}
}

func TestIXLETAccessFollowsDeviceApproval(t *testing.T) {
	s, token, id := fixture(t)
	if _, err := s.Save(token, operation("save", 0, draft())); err != nil {
		t.Fatal(err)
	}
	act(t, s, token, "on", "activate")
	if _, err := s.Xlets(""); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("anonymous read: %v", err)
	}
	if err := s.Decide(id, "revoked"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Act(token, XletAction{OpID: "after", ProjectID: "project-one", DefinitionID: "ixlet-one", Action: "increment"}); !errors.Is(err, ErrRevoked) {
		t.Fatalf("a revoked device must not drive an IXLET: %v", err)
	}
	if _, err := s.Xlets(token); !errors.Is(err, ErrRevoked) {
		t.Fatalf("a revoked device must not read IXLETs: %v", err)
	}
}

// A database created by the binary currently installed (schema 1, no workshop_xlets) opens unchanged in version and
// gains only the additive table; restoring that older binary must still accept the file.
func TestIXLETTableIsAdditiveToSchemaOne(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.sqlite")
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	token, _ := paired(t, s)
	if _, err = s.Save(token, operation("save", 0, draft())); err != nil {
		t.Fatal(err)
	}
	s.Close()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = db.Exec(`DROP TABLE workshop_xlets; DROP TABLE workshop_xlet_items`); err != nil { // as ixforge.4 leaves it
		t.Fatal(err)
	}
	db.Close()
	if s, err = Open(path); err != nil {
		t.Fatalf("an installed-binary database must open: %v", err)
	}
	act(t, s, token, "on", "activate")
	s.Close()
	db, _ = sql.Open("sqlite", path)
	defer db.Close()
	var version, projects int
	db.QueryRow(`SELECT version FROM workshop_metadata`).Scan(&version)
	db.QueryRow(`SELECT count(*) FROM workshop_projects`).Scan(&projects)
	if version != 1 || projects != 1 {
		t.Fatalf("schema version must stay 1 and data intact: version=%d projects=%d", version, projects)
	}
	// What the installed binary checks on open: these queries only, and version 1.
	for _, q := range []string{`SELECT id,name,token_hash,status FROM workshop_devices LIMIT 0`, `SELECT id,revision,document FROM workshop_projects LIMIT 0`, `SELECT sequence,project_id,revision,document FROM workshop_revisions LIMIT 0`, `SELECT device_id,op_id,payload_hash,receipt FROM workshop_operations LIMIT 0`} {
		rows, err := db.Query(q)
		if err != nil {
			t.Fatalf("older binary check %q: %v", q, err)
		}
		rows.Close()
	}
}
