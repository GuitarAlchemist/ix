// Package workshop owns tablet project drafts. It has no control-plane or process execution dependency.
package workshop

import (
	"errors"
	"regexp"
	"strings"
)

var (
	ErrInvalid      = errors.New("invalid_document")
	ErrUnauthorized = errors.New("unauthorized")
	ErrPending      = errors.New("device_pending")
	ErrRefused      = errors.New("device_refused")
	ErrRevoked      = errors.New("device_revoked")
	ErrInvite       = errors.New("invite_invalid_or_expired")
	ErrConflict     = errors.New("revision_conflict")
	ErrOpReuse      = errors.New("operation_id_reused")
	ErrStoreChanged = errors.New("store_changed")
)

type Project struct {
	ID         string      `json:"id"`
	Title      string      `json:"title"`
	Workspaces []Workspace `json:"workspaces"`
}
type Workspace struct {
	ID               string            `json:"id"`
	Title            string            `json:"title"`
	Widgets          []Widget          `json:"widgets"`
	IXLETDefinitions []IXLETDefinition `json:"ixlet_definitions"`
}
type Widget struct {
	ID           string `json:"id"`
	Title        string `json:"title"`
	Kind         string `json:"kind"`
	InitialValue int64  `json:"initial_value"`
}
type IXLETDefinition struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Kind   string `json:"kind"`
	Status string `json:"status"`
}
type Operation struct {
	OpID         string  `json:"op_id"`
	BaseRevision int64   `json:"base_revision"`
	Project      Project `json:"project"`
}
type Receipt struct {
	OpID      string `json:"op_id"`
	ProjectID string `json:"project_id"`
	Revision  int64  `json:"revision"`
	Sequence  int64  `json:"sequence"`
}
type Event struct {
	Sequence int64   `json:"sequence"`
	Revision int64   `json:"revision"`
	Project  Project `json:"project"`
}
type ChangeSet struct {
	StoreID string  `json:"store_id"`
	Cursor  int64   `json:"cursor"`
	Events  []Event `json:"events"`
}

var identifier = regexp.MustCompile(`^[A-Za-z0-9_-]{1,80}$`)

func titleOK(s string) bool { return strings.TrimSpace(s) != "" && len(s) <= 240 }
func validate(op Operation) error {
	if !identifier.MatchString(op.OpID) || op.BaseRevision < 0 || !identifier.MatchString(op.Project.ID) || !titleOK(op.Project.Title) || op.Project.Workspaces == nil || len(op.Project.Workspaces) > 100 {
		return ErrInvalid
	}
	ids := map[string]bool{op.Project.ID: true}
	add := func(id, title string) bool {
		if !identifier.MatchString(id) || ids[id] || !titleOK(title) {
			return false
		}
		ids[id] = true
		return true
	}
	for _, ws := range op.Project.Workspaces {
		if !add(ws.ID, ws.Title) || ws.Widgets == nil || ws.IXLETDefinitions == nil || len(ws.Widgets) > 100 || len(ws.IXLETDefinitions) > 100 {
			return ErrInvalid
		}
		for _, w := range ws.Widgets {
			if !add(w.ID, w.Title) || w.Kind != "counter" || w.InitialValue > 9007199254740991 || w.InitialValue < -9007199254740991 {
				return ErrInvalid
			}
		}
		for _, d := range ws.IXLETDefinitions {
			if !add(d.ID, d.Title) || d.Kind != "counter" || d.Status != "draft" {
				return ErrInvalid
			}
		}
	}
	return nil
}
