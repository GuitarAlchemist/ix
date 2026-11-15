package workshop

import (
	"database/sql"
	"encoding/json"
	"errors"
)

// The Desktop hosts one IXLET instance per activated definition: a counter or a checklist. The instance is the authority
// for its state; widgets only present it and request actions. Durability contract: state lives in workshop_xlets (and
// workshop_xlet_items for checklist items) of this Workshop SQLite file, survives a restart, and is owned by the
// Desktop. A definition stays a draft until a device explicitly activates it from a project already committed on the
// Desktop.
//
// Both tables are additive to schema version 1: an older binary ignores them, so restoring that binary keeps working.

var (
	ErrNotSaved  = errors.New("ixlet_not_saved")
	ErrNotActive = errors.New("ixlet_not_active")
	ErrNoItem    = errors.New("ixlet_item_missing")
)

const (
	maxSafe           = 9007199254740991
	maxChecklistItems = 200
)

var xletSchema = []string{
	`CREATE TABLE IF NOT EXISTS workshop_xlets (project_id TEXT NOT NULL, definition_id TEXT NOT NULL, kind TEXT NOT NULL, value INTEGER NOT NULL, version INTEGER NOT NULL, PRIMARY KEY(project_id, definition_id))`,
	`CREATE TABLE IF NOT EXISTS workshop_xlet_items (project_id TEXT NOT NULL, definition_id TEXT NOT NULL, item_id TEXT NOT NULL, position INTEGER NOT NULL, text TEXT NOT NULL, done INTEGER NOT NULL, PRIMARY KEY(project_id, definition_id, item_id))`,
	// Devices that administer the others from the app (admin.go). Additive like the tables above.
	`CREATE TABLE IF NOT EXISTS workshop_admins (device_id TEXT PRIMARY KEY)`,
}
var xletSchemaChecks = []string{
	`SELECT project_id,definition_id,kind,value,version FROM workshop_xlets LIMIT 0`,
	`SELECT project_id,definition_id,item_id,position,text,done FROM workshop_xlet_items LIMIT 0`,
	`SELECT device_id FROM workshop_admins LIMIT 0`,
}

// XletAction is one request to an IXLET. Activate creates the instance. A counter takes increment, decrement and reset;
// a checklist takes add (with text), check, uncheck and remove (with item_id). Any other field is refused.
type XletAction struct {
	OpID         string    `json:"op_id"`
	ProjectID    string    `json:"project_id"`
	DefinitionID string    `json:"definition_id"`
	Action       string    `json:"action"`
	Text         string    `json:"text,omitempty"`
	ItemID       string    `json:"item_id,omitempty"`
	Seed         *XletSeed `json:"seed,omitempty"`
}

// XletSeed is what a device did locally before the Desktop ran the IXLET: checklist items, or a counter value. It
// travels only with activate and is applied only when the instance is created, never over a running one.
type XletSeed struct {
	Value int64          `json:"value,omitempty"`
	Items []XletSeedItem `json:"items,omitempty"`
}
type XletSeedItem struct {
	Text string `json:"text"`
	Done bool   `json:"done"`
}

func validSeed(seed *XletSeed) bool {
	if seed == nil {
		return true
	}
	if seed.Value > maxSafe || seed.Value < -maxSafe || len(seed.Items) > maxChecklistItems {
		return false
	}
	for _, i := range seed.Items {
		if !titleOK(i.Text) {
			return false
		}
	}
	return true
}

// XletItem is one checklist entry, in list order.
type XletItem struct {
	ID   string `json:"id"`
	Text string `json:"text"`
	Done bool   `json:"done"`
}

// XletState is the authoritative state of one instance, as returned after an action or by a query. Value is the counter's
// value; for a checklist it is the number of items. Items is set for a checklist only.
type XletState struct {
	ProjectID    string     `json:"project_id"`
	DefinitionID string     `json:"definition_id"`
	Kind         string     `json:"kind"`
	Value        int64      `json:"value"`
	Version      int64      `json:"version"`
	Items        []XletItem `json:"items,omitempty"`
	Seeded       bool       `json:"seeded,omitempty"` // this activation created the instance from the device's seed
}

var actionKind = map[string]string{"increment": "counter", "decrement": "counter", "reset": "counter", "add": "checklist", "check": "checklist", "uncheck": "checklist", "remove": "checklist"}

func validAction(a XletAction) bool {
	if !identifier.MatchString(a.OpID) || !identifier.MatchString(a.ProjectID) || !identifier.MatchString(a.DefinitionID) {
		return false
	}
	if a.Seed != nil && (a.Action != "activate" || !validSeed(a.Seed)) {
		return false
	}
	switch a.Action {
	case "add":
		return titleOK(a.Text) && a.ItemID == ""
	case "check", "uncheck", "remove":
		return identifier.MatchString(a.ItemID) && a.Text == ""
	case "activate", "increment", "decrement", "reset":
		return a.Text == "" && a.ItemID == ""
	}
	return false
}

// Act runs one action exactly once per (device, op_id). Repeating the same op_id and payload returns the state the
// first execution produced, without running it again; a different payload under that op_id is refused.
func (s *Store) Act(token string, a XletAction) (XletState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var state XletState
	if !validAction(a) {
		return state, ErrInvalid
	}
	payload, _ := json.Marshal(a)
	hash := digest("xlet\x00" + string(payload))
	tx, err := s.db.Begin()
	if err != nil {
		return state, err
	}
	defer tx.Rollback()
	d, err := authenticate(tx, token)
	if err != nil {
		return state, err
	}
	var previousHash, receipt string
	err = tx.QueryRow(`SELECT payload_hash,receipt FROM workshop_operations WHERE device_id=? AND op_id=?`, d.ID, a.OpID).Scan(&previousHash, &receipt)
	if err == nil {
		if hash != previousHash {
			return state, ErrOpReuse
		}
		err = json.Unmarshal([]byte(receipt), &state)
		return state, err
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return state, err
	}
	err = tx.QueryRow(`SELECT project_id,definition_id,kind,value,version FROM workshop_xlets WHERE project_id=? AND definition_id=?`, a.ProjectID, a.DefinitionID).
		Scan(&state.ProjectID, &state.DefinitionID, &state.Kind, &state.Value, &state.Version)
	exists := err == nil
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return state, err
	}
	switch {
	case a.Action == "activate" && exists:
		// Already running: activation is idempotent and does not reset the state; a seed is not applied.
		if a.Seed != nil && ((state.Kind == "counter" && len(a.Seed.Items) > 0) || (state.Kind == "checklist" && a.Seed.Value != 0)) {
			return XletState{}, ErrInvalid
		}
	case a.Action == "activate":
		kind, initial, err := committedDefinition(tx, a.ProjectID, a.DefinitionID)
		if err != nil {
			return state, err
		}
		state = XletState{ProjectID: a.ProjectID, DefinitionID: a.DefinitionID, Kind: kind, Value: initial, Version: 1}
		var seeded []XletSeedItem
		if a.Seed != nil {
			if (kind == "counter" && len(a.Seed.Items) > 0) || (kind == "checklist" && a.Seed.Value != 0) {
				return XletState{}, ErrInvalid
			}
			if kind == "counter" {
				state.Value = a.Seed.Value
			}
			seeded, state.Seeded = a.Seed.Items, true
			if kind == "checklist" {
				state.Value = int64(len(seeded))
			}
		}
		if _, err = tx.Exec(`INSERT INTO workshop_xlets VALUES(?,?,?,?,?)`, state.ProjectID, state.DefinitionID, state.Kind, state.Value, state.Version); err != nil {
			return state, err
		}
		for position, item := range seeded {
			id, err := randomID()
			if err != nil {
				return XletState{}, err
			}
			if _, err = tx.Exec(`INSERT INTO workshop_xlet_items VALUES(?,?,?,?,?,?)`, a.ProjectID, a.DefinitionID, id[:16], position+1, item.Text, item.Done); err != nil {
				return XletState{}, err
			}
		}
	case !exists:
		return XletState{}, ErrNotActive
	case actionKind[a.Action] != state.Kind:
		return XletState{}, ErrInvalid
	case state.Kind == "counter":
		next := map[string]int64{"increment": state.Value + 1, "decrement": state.Value - 1, "reset": 0}[a.Action]
		if next > maxSafe || next < -maxSafe {
			return state, ErrInvalid
		}
		state.Value = next
	default:
		if state.Value, err = changeChecklist(tx, a); err != nil {
			return XletState{}, err
		}
	}
	if a.Action != "activate" {
		state.Version++
		if _, err = tx.Exec(`UPDATE workshop_xlets SET value=?,version=? WHERE project_id=? AND definition_id=?`, state.Value, state.Version, state.ProjectID, state.DefinitionID); err != nil {
			return state, err
		}
	}
	if state.Kind == "checklist" {
		if state.Items, err = checklistItems(tx, state.ProjectID, state.DefinitionID); err != nil {
			return state, err
		}
	}
	encoded, err := json.Marshal(state)
	if err != nil {
		return state, err
	}
	if _, err = tx.Exec(`INSERT INTO workshop_operations VALUES(?,?,?,?)`, d.ID, a.OpID, hash, string(encoded)); err != nil {
		return state, err
	}
	if err = tx.Commit(); err != nil {
		return XletState{}, err
	}
	return state, nil
}

// changeChecklist applies one checklist action and returns the new number of items.
func changeChecklist(tx *sql.Tx, a XletAction) (int64, error) {
	var count, last int64
	if err := tx.QueryRow(`SELECT count(*),coalesce(max(position),0) FROM workshop_xlet_items WHERE project_id=? AND definition_id=?`, a.ProjectID, a.DefinitionID).Scan(&count, &last); err != nil {
		return 0, err
	}
	if a.Action == "add" {
		if count >= maxChecklistItems {
			return 0, ErrInvalid
		}
		id, err := randomID()
		if err != nil {
			return 0, err
		}
		_, err = tx.Exec(`INSERT INTO workshop_xlet_items VALUES(?,?,?,?,?,0)`, a.ProjectID, a.DefinitionID, id[:16], last+1, a.Text)
		return count + 1, err
	}
	query := map[string]string{
		"check":   `UPDATE workshop_xlet_items SET done=1 WHERE project_id=? AND definition_id=? AND item_id=?`,
		"uncheck": `UPDATE workshop_xlet_items SET done=0 WHERE project_id=? AND definition_id=? AND item_id=?`,
		"remove":  `DELETE FROM workshop_xlet_items WHERE project_id=? AND definition_id=? AND item_id=?`,
	}[a.Action]
	result, err := tx.Exec(query, a.ProjectID, a.DefinitionID, a.ItemID)
	if err != nil {
		return 0, err
	}
	if n, err := result.RowsAffected(); err != nil || n != 1 {
		return 0, ErrNoItem
	}
	if a.Action == "remove" {
		count--
	}
	return count, nil
}

func checklistItems(q interface {
	Query(string, ...any) (*sql.Rows, error)
}, projectID, definitionID string) ([]XletItem, error) {
	rows, err := q.Query(`SELECT item_id,text,done FROM workshop_xlet_items WHERE project_id=? AND definition_id=? ORDER BY position`, projectID, definitionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	list := []XletItem{}
	for rows.Next() {
		var i XletItem
		if err = rows.Scan(&i.ID, &i.Text, &i.Done); err != nil {
			return nil, err
		}
		list = append(list, i)
	}
	return list, rows.Err()
}

// committedDefinition finds a definition in the Desktop's committed project. A counter starts from the initial value of
// the counter widget that presents it: the one in the same workspace with the same kind and title (the template's
// pairing), else 0. A checklist starts empty.
func committedDefinition(tx *sql.Tx, projectID, definitionID string) (string, int64, error) {
	var doc string
	err := tx.QueryRow(`SELECT document FROM workshop_projects WHERE id=?`, projectID).Scan(&doc)
	if errors.Is(err, sql.ErrNoRows) {
		return "", 0, ErrNotSaved
	}
	if err != nil {
		return "", 0, err
	}
	var p Project
	if err = json.Unmarshal([]byte(doc), &p); err != nil {
		return "", 0, err
	}
	for _, ws := range p.Workspaces {
		for _, def := range ws.IXLETDefinitions {
			if def.ID != definitionID || !kindOK(def.Kind) {
				continue
			}
			if def.Kind == "checklist" {
				return def.Kind, 0, nil
			}
			for _, w := range ws.Widgets {
				if w.Kind == def.Kind && w.Title == def.Title {
					return def.Kind, w.InitialValue, nil
				}
			}
			return def.Kind, 0, nil
		}
	}
	return "", 0, ErrNotSaved
}

// Xlets returns every instance hosted by this Desktop, for an approved device.
func (s *Store) Xlets(token string) ([]XletState, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	states := []XletState{}
	if _, err := authenticate(s.db, token); err != nil {
		return states, err
	}
	rows, err := s.db.Query(`SELECT project_id,definition_id,kind,value,version FROM workshop_xlets ORDER BY project_id,definition_id`)
	if err != nil {
		return states, err
	}
	for rows.Next() {
		var x XletState
		if err = rows.Scan(&x.ProjectID, &x.DefinitionID, &x.Kind, &x.Value, &x.Version); err != nil {
			rows.Close()
			return states, err
		}
		states = append(states, x)
	}
	rows.Close()
	if err = rows.Err(); err != nil {
		return states, err
	}
	for i := range states {
		if states[i].Kind == "checklist" {
			if states[i].Items, err = checklistItems(s.db, states[i].ProjectID, states[i].DefinitionID); err != nil {
				return states, err
			}
		}
	}
	return states, nil
}
