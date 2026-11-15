package workshop

import (
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

type Store struct {
	mu            sync.Mutex
	db            *sql.DB
	path          string
	id            string
	now           func() time.Time
	inviteHash    string
	inviteExpires time.Time
	inviteErrors  int
}

func randomID() (string, error) {
	var b [24]byte
	_, err := rand.Read(b[:])
	return hex.EncodeToString(b[:]), err
}
func digest(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }

// Open requires an explicitly chosen file. The caller owns lifecycle and a local, non-synchronized data directory.
// It refuses an existing non-Workshop database and never opens Scribe's data.
func Open(path string) (*Store, error) {
	if !filepath.IsAbs(path) {
		return nil, fmt.Errorf("database path must be absolute")
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	fail := func(err error) (*Store, error) { db.Close(); return nil, err }
	var tables, ours int
	if err = db.QueryRow(`SELECT count(*), coalesce(sum(name='workshop_metadata'),0) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'`).Scan(&tables, &ours); err != nil {
		return fail(err)
	}
	if tables > 0 && ours == 0 {
		return fail(fmt.Errorf("database is not a Workshop database"))
	}
	s := &Store{db: db, path: path, now: time.Now}
	if tables > 0 {
		var version, rows int
		if err = db.QueryRow(`SELECT count(*) FROM workshop_metadata`).Scan(&rows); err != nil || rows != 1 {
			return fail(fmt.Errorf("invalid Workshop metadata"))
		}
		if err = db.QueryRow(`SELECT id,version FROM workshop_metadata`).Scan(&s.id, &version); err != nil {
			return fail(err)
		}
		if version != 1 {
			return fail(fmt.Errorf("unsupported Workshop schema %d", version))
		}
		for _, query := range []string{`SELECT id,name,token_hash,status FROM workshop_devices LIMIT 0`, `SELECT id,revision,document FROM workshop_projects LIMIT 0`, `SELECT sequence,project_id,revision,document FROM workshop_revisions LIMIT 0`, `SELECT device_id,op_id,payload_hash,receipt FROM workshop_operations LIMIT 0`} {
			result, e := db.Query(query)
			if e != nil {
				return fail(fmt.Errorf("invalid Workshop schema: %w", e))
			}
			result.Close()
		}
	}
	if _, err = db.Exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;`); err != nil {
		return fail(err)
	}
	if tables == 0 {
		tx, e := db.Begin()
		if e != nil {
			return fail(e)
		}
		if _, err = tx.Exec(`
 CREATE TABLE IF NOT EXISTS workshop_metadata (id TEXT PRIMARY KEY, version INTEGER NOT NULL);
 CREATE TABLE IF NOT EXISTS workshop_devices (id TEXT PRIMARY KEY, name TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE, status TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS workshop_projects (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, document TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS workshop_revisions (sequence INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, revision INTEGER NOT NULL, document TEXT NOT NULL, UNIQUE(project_id,revision));
 CREATE TABLE IF NOT EXISTS workshop_operations (device_id TEXT NOT NULL REFERENCES workshop_devices(id), op_id TEXT NOT NULL, payload_hash TEXT NOT NULL, receipt TEXT NOT NULL, PRIMARY KEY(device_id,op_id));`); err != nil {
			tx.Rollback()
			return fail(err)
		}
		s.id, err = randomID()
		if err == nil {
			_, err = tx.Exec(`INSERT INTO workshop_metadata VALUES(?,1)`, s.id)
		}
		if err != nil {
			tx.Rollback()
			return fail(err)
		}
		if err = tx.Commit(); err != nil {
			return fail(err)
		}
	}
	return s, nil
}
func (s *Store) Close() error { s.mu.Lock(); defer s.mu.Unlock(); return s.db.Close() }

type queryRow interface{ QueryRow(string, ...any) *sql.Row }

func authenticate(q queryRow, token string) (Device, error) {
	var d Device
	if token == "" {
		return d, ErrUnauthorized
	}
	err := q.QueryRow(`SELECT id,name,status FROM workshop_devices WHERE token_hash=?`, digest(token)).Scan(&d.ID, &d.Name, &d.Status)
	if errors.Is(err, sql.ErrNoRows) {
		return d, ErrUnauthorized
	}
	if err != nil {
		return d, err
	}
	switch d.Status {
	case "approved":
		return d, nil
	case "pending":
		return d, ErrPending
	case "refused":
		return d, ErrRefused
	case "revoked":
		return d, ErrRevoked
	}
	return d, ErrUnauthorized
}
func (s *Store) Authenticate(token string) (Device, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return authenticate(s.db, token)
}

func (s *Store) Save(token string, op Operation) (Receipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var result Receipt
	if err := validate(op); err != nil {
		return result, err
	}
	payload, err := json.Marshal(op)
	if err != nil || len(payload) > 1<<20 {
		return result, ErrInvalid
	}
	tx, err := s.db.Begin()
	if err != nil {
		return result, err
	}
	defer tx.Rollback()
	d, err := authenticate(tx, token)
	if err != nil {
		return result, err
	}
	hash := digest(string(payload))
	var previousHash, receipt string
	err = tx.QueryRow(`SELECT payload_hash,receipt FROM workshop_operations WHERE device_id=? AND op_id=?`, d.ID, op.OpID).Scan(&previousHash, &receipt)
	if err == nil {
		if hash != previousHash {
			return result, ErrOpReuse
		}
		err = json.Unmarshal([]byte(receipt), &result)
		return result, err
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return result, err
	}
	var revision int64
	err = tx.QueryRow(`SELECT revision FROM workshop_projects WHERE id=?`, op.Project.ID).Scan(&revision)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return result, err
	}
	if revision != op.BaseRevision {
		return result, ErrConflict
	}
	doc, err := json.Marshal(op.Project)
	if err != nil {
		return result, err
	}
	revision++
	if _, err = tx.Exec(`INSERT INTO workshop_projects VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET revision=excluded.revision,document=excluded.document`, op.Project.ID, revision, string(doc)); err != nil {
		return result, err
	}
	inserted, err := tx.Exec(`INSERT INTO workshop_revisions(project_id,revision,document) VALUES(?,?,?)`, op.Project.ID, revision, string(doc))
	if err != nil {
		return result, err
	}
	seq, err := inserted.LastInsertId()
	if err != nil {
		return result, err
	}
	result = Receipt{op.OpID, op.Project.ID, revision, seq}
	encoded, err := json.Marshal(result)
	if err != nil {
		return result, err
	}
	if _, err = tx.Exec(`INSERT INTO workshop_operations VALUES(?,?,?,?)`, d.ID, op.OpID, hash, string(encoded)); err != nil {
		return result, err
	}
	if err = tx.Commit(); err != nil {
		return Receipt{}, err
	}
	return result, nil
}

// Changes is a durable, ordered journal. The returned cursor covers exactly the returned page.
func (s *Store) Changes(token string, since int64) (ChangeSet, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	changes := ChangeSet{StoreID: s.id, Cursor: since, Events: []Event{}}
	if since < 0 {
		return changes, ErrInvalid
	}
	if _, err := authenticate(s.db, token); err != nil {
		return changes, err
	}
	var head int64
	if err := s.db.QueryRow(`SELECT coalesce(max(sequence),0) FROM workshop_revisions`).Scan(&head); err != nil {
		return changes, err
	}
	if since > head {
		return changes, ErrInvalid
	}
	rows, err := s.db.Query(`SELECT sequence,revision,document FROM workshop_revisions WHERE sequence>? ORDER BY sequence LIMIT 100`, since)
	if err != nil {
		return changes, err
	}
	defer rows.Close()
	for rows.Next() {
		var e Event
		var doc string
		if err = rows.Scan(&e.Sequence, &e.Revision, &doc); err != nil {
			return changes, err
		}
		if err = json.Unmarshal([]byte(doc), &e.Project); err != nil {
			return changes, err
		}
		changes.Events = append(changes.Events, e)
		changes.Cursor = e.Sequence
	}
	return changes, rows.Err()
}
