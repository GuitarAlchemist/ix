package workshop

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"
)

func TestSchemaBootstrapAndFutureVersionAreNonDestructive(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.sqlite")
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	id := s.id
	s.Close()
	s, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if s.id != id {
		t.Fatal("identity changed")
	}
	s.Close()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = db.Exec(`UPDATE workshop_metadata SET version=99; DROP TABLE workshop_operations`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	before, _ := os.ReadFile(path)
	if s, err := Open(path); err == nil {
		s.Close()
		t.Fatal("future schema accepted")
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("unsupported schema was modified")
	}
	db, _ = sql.Open("sqlite", path)
	defer db.Close()
	var count int
	db.QueryRow(`SELECT count(*) FROM sqlite_master WHERE name='workshop_operations'`).Scan(&count)
	if count != 0 {
		t.Fatal("future schema was silently migrated")
	}
	if _, err = db.Exec(`UPDATE workshop_metadata SET version=1`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	if s, err := Open(path); err == nil {
		s.Close()
		t.Fatal("incomplete version 1 silently repaired")
	}
}
