package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

// A bookmark names one point in the instance's sequence. The hub's seq is shared by every
// channel, so one bookmark covers a whole probe tree's pipes, the services' pipes and the
// ctl lifecycle at one instant: a replay from it gives back what every reader saw after
// that point, in the order it happened. The plane owns the bookmarks and keeps them in its
// state dir (`<name>.bookmarks.json`), so they survive a restart; a bookmark of an earlier
// boot replays with an epoch-changed gap first. The ring is the only replay source for now
// (door 3 untouched): a bookmark older than the ring says since-too-old, never a silently
// partial replay.
const (
	maxBookmarks = 64
	maxNoteLen   = 256

	bookmarksSchema = "ixctl-bookmarks/0"
)

type bookmark struct {
	Name  string `json:"name"`
	Epoch string `json:"epoch"`
	Seq   uint64 `json:"seq"`
	T     int64  `json:"t"`
	Note  string `json:"note,omitempty"`
}

type bookmarkFile struct {
	Schema    string     `json:"schema"`
	Bookmarks []bookmark `json:"bookmarks"`
}

type bookmarkStore struct {
	hub  *Hub
	path string // "" keeps the bookmarks in memory only

	mu    sync.Mutex
	marks []bookmark // in creation order; setting an existing name moves its point, not its place
}

// newBookmarkStore opens the store: on disk at path (`ixctl up --bookmarks disk`, the
// default: the file is loaded if there is one and rewritten on every change) or in memory
// when path is empty (`--bookmarks memory`: this boot only). A file that cannot be read
// as bookmarks is refused rather than emptied: the operator decides what to do with it.
func newBookmarkStore(hub *Hub, path string) (*bookmarkStore, error) {
	s := &bookmarkStore{hub: hub, path: path}
	if path == "" {
		return s, nil
	}
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return s, nil
	}
	if err != nil {
		return nil, fmt.Errorf("bookmarks: %w", err)
	}
	var f bookmarkFile
	if err := decodeStrict(strings.NewReader(string(b)), &f); err != nil {
		return nil, fmt.Errorf("bookmarks %s: %v", path, err)
	}
	if f.Schema != bookmarksSchema {
		return nil, fmt.Errorf("bookmarks %s: schema %q, want %q", path, f.Schema, bookmarksSchema)
	}
	s.marks = f.Bookmarks
	return s, nil
}

// save rewrites the file whole: a temporary beside it, then a rename, so a crash leaves
// the previous file or the new one, never half of either.
func (s *bookmarkStore) save() error {
	if s.path == "" {
		return nil
	}
	b, err := json.Marshal(bookmarkFile{Schema: bookmarksSchema, Bookmarks: s.marks})
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("bookmarks: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("bookmarks: %w", err)
	}
	return nil
}

func (s *bookmarkStore) find(name string) int {
	for i := range s.marks {
		if s.marks[i].Name == name {
			return i
		}
	}
	return -1
}

// set puts a bookmark at a point: the head ("here") when the request names no since, or
// the given since, which must be at or below the head and, when the request names an
// epoch, of this boot. An existing name is moved; a new one counts against max_bookmarks.
func (s *bookmarkStore) set(req Request) (any, error) {
	if req.Bookmark == "" {
		return nil, badRequest("bookmark.set needs a `bookmark` name")
	}
	if !nameRE.MatchString(req.Bookmark) {
		return nil, badRequest("bookmark name %q must match %s", req.Bookmark, nameRE)
	}
	if len(req.Note) > maxNoteLen {
		return nil, badRequest("note is %d chars, over max_note_len %d", len(req.Note), maxNoteLen)
	}
	epoch := s.hub.Epoch()
	if req.Epoch != "" && req.Epoch != epoch {
		return nil, badRequest("epoch %q is not this boot's (%s): a bookmark is set in the current sequence", req.Epoch, epoch)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	head := s.hub.Head()
	seq := head
	if req.Since != nil {
		if *req.Since > head {
			return nil, badRequest("since %d is past the head %d", *req.Since, head)
		}
		seq = *req.Since
	}
	i := s.find(req.Bookmark)
	if i < 0 && len(s.marks) >= maxBookmarks {
		return nil, badRequest("%d bookmarks already, max_bookmarks %d: remove one first", len(s.marks), maxBookmarks)
	}
	bm := bookmark{Name: req.Bookmark, Epoch: epoch, Seq: seq, T: time.Now().UnixMilli(), Note: req.Note}
	if i >= 0 {
		if bm.Note == "" {
			bm.Note = s.marks[i].Note
		}
		s.marks[i] = bm
	} else {
		s.marks = append(s.marks, bm)
	}
	if err := s.save(); err != nil {
		return nil, err
	}
	return map[string]any{"bookmark": bm.Name, "epoch": bm.Epoch, "seq": bm.Seq, "t": bm.T, "note": bm.Note, "head": head}, nil
}

func (s *bookmarkStore) get(name string) (bookmark, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if i := s.find(name); i >= 0 {
		return s.marks[i], nil
	}
	return bookmark{}, badRequest("unknown bookmark %q", name)
}

// list reports every bookmark with whether this boot made it (a replay from one that is
// not live starts with an epoch-changed gap), and the bounds a client plans against.
func (s *bookmarkStore) list() any {
	s.mu.Lock()
	defer s.mu.Unlock()
	epoch := s.hub.Epoch()
	out := make([]map[string]any, 0, len(s.marks))
	for _, bm := range s.marks {
		out = append(out, map[string]any{
			"name": bm.Name, "epoch": bm.Epoch, "seq": bm.Seq, "t": bm.T, "note": bm.Note, "live": bm.Epoch == epoch,
		})
	}
	store := "disk"
	if s.path == "" {
		store = "memory"
	}
	return map[string]any{
		"bookmarks": out, "count": len(out), "store": store,
		"max_bookmarks": maxBookmarks, "max_note_len": maxNoteLen, "ring": ringSize,
	}
}

func (s *bookmarkStore) remove(name string) (any, error) {
	if name == "" {
		return nil, badRequest("bookmark.remove needs a `bookmark` name")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.find(name)
	if i < 0 {
		return nil, badRequest("unknown bookmark %q", name)
	}
	s.marks = append(s.marks[:i], s.marks[i+1:]...)
	if err := s.save(); err != nil {
		return nil, err
	}
	return map[string]any{"bookmark": name, "removed": true}, nil
}

// resolveBookmark turns a subscription's `bookmark` into the since and epoch the hub
// understands, through the broker's own request path, so a transport never reaches past
// the Broker interface. A bookmark beside an explicit since or epoch is refused: the
// request would name two points.
func resolveBookmark(ctx context.Context, b Broker, req *Request) error {
	if req.Bookmark == "" {
		return nil
	}
	if req.Since != nil || req.Epoch != "" {
		return badRequest("bookmark or since (with its epoch), not both")
	}
	r, err := b.Request(ctx, Request{Op: "bookmark.get", Bookmark: req.Bookmark})
	if err != nil {
		return err
	}
	bm, ok := r.(bookmark)
	if !ok {
		return fmt.Errorf("bookmark.get answered %T", r)
	}
	seq := bm.Seq
	req.Since, req.Epoch, req.Bookmark = &seq, bm.Epoch, ""
	return nil
}
