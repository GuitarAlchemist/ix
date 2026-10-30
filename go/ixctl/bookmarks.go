package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"
)

// A bookmark names one point in the instance's sequence. The hub's seq is shared by every
// channel, so one bookmark covers a whole probe tree's pipes, the services' pipes and the
// ctl lifecycle at one instant: a replay from it gives back what every reader saw after
// that point, in the order it happened. The plane owns the bookmarks and keeps them in its
// state dir (`<name>.bookmarks.json`) or in memory, as chosen at startup; a bookmark of an
// earlier boot replays with an epoch-changed gap first. The ring is the only replay source
// for now (door 3 untouched): a bookmark older than the ring says since-too-old, never a
// silently partial replay.
const (
	maxBookmarks = 64
	maxNoteLen   = 256 // characters (runes), not bytes

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
// when path is empty (`--bookmarks memory`: this boot only). A file that cannot be read as
// bookmarks, or whose records break the invariants `set` enforces, is refused rather than
// emptied: the operator decides what to do with it.
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
	if err := checkBookmarks(f.Bookmarks); err != nil {
		return nil, fmt.Errorf("bookmarks %s: %v", path, err)
	}
	s.marks = f.Bookmarks
	return s, nil
}

// checkBookmarks holds a loaded list to what set would have accepted: the count, the name
// rule, unique names, the note length, and an epoch (a bookmark with none would replay
// against the current boot without the epoch-changed gap it owes).
func checkBookmarks(list []bookmark) error {
	if len(list) > maxBookmarks {
		return fmt.Errorf("%d bookmarks, over max_bookmarks %d", len(list), maxBookmarks)
	}
	seen := map[string]bool{}
	for i, bm := range list {
		switch {
		case !nameRE.MatchString(bm.Name):
			return fmt.Errorf("bookmarks[%d]: name %q must match %s", i, bm.Name, nameRE)
		case seen[bm.Name]:
			return fmt.Errorf("bookmarks[%d]: name %q appears twice", i, bm.Name)
		case bm.Epoch == "":
			return fmt.Errorf("bookmarks[%d] (%s): no epoch", i, bm.Name)
		case utf8.RuneCountInString(bm.Note) > maxNoteLen:
			return fmt.Errorf("bookmarks[%d] (%s): note over max_note_len %d", i, bm.Name, maxNoteLen)
		}
		seen[bm.Name] = true
	}
	return nil
}

// save writes a candidate list whole: a temporary beside the file, then a rename, so a
// crash leaves the previous file or the new one, never half of either. The caller adopts
// the list only once this returned nil, so a failed write changes nothing a reader sees.
func (s *bookmarkStore) save(list []bookmark) error {
	if s.path == "" {
		return nil
	}
	b, err := json.Marshal(bookmarkFile{Schema: bookmarksSchema, Bookmarks: list})
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o600); err != nil {
		return fmt.Errorf("bookmarks: %w", err)
	}
	if err := renameOver(tmp, s.path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("bookmarks: %w", err)
	}
	return nil
}

// renameOver replaces path with tmp. On Windows the replace is refused (access denied, or
// a sharing violation) for as long as another process holds the file it replaces (an
// indexer or an antivirus reading what the previous save wrote); a few short retries, about
// a second in all, ride that out. Any other error, and a file that stays held, come back at
// once: the caller holds the store's lock meanwhile.
func renameOver(tmp, path string) error {
	const attempts = 20
	for i := 1; ; i++ {
		err := os.Rename(tmp, path)
		if err == nil || i == attempts || !heldByAnother(err) {
			return err
		}
		time.Sleep(time.Duration(5*i) * time.Millisecond)
	}
}

// heldByAnother says whether a rename failed because the target is open elsewhere:
// ERROR_ACCESS_DENIED (os.IsPermission) or ERROR_SHARING_VIOLATION (32) on Windows.
func heldByAnother(err error) bool {
	var errno syscall.Errno
	if runtime.GOOS == "windows" && errors.As(err, &errno) && errno == 32 {
		return true
	}
	return os.IsPermission(err)
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
// A note given replaces the old one (an empty one clears it); no note keeps it.
func (s *bookmarkStore) set(req Request) (any, error) {
	if req.Bookmark == "" {
		return nil, badRequest("bookmark.set needs a `bookmark` name")
	}
	if !nameRE.MatchString(req.Bookmark) {
		return nil, badRequest("bookmark name %q must match %s", req.Bookmark, nameRE)
	}
	if req.Note != nil && utf8.RuneCountInString(*req.Note) > maxNoteLen {
		return nil, badRequest("note is %d chars, over max_note_len %d", utf8.RuneCountInString(*req.Note), maxNoteLen)
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
	bm := bookmark{Name: req.Bookmark, Epoch: epoch, Seq: seq, T: time.Now().UnixMilli()}
	next := append([]bookmark(nil), s.marks...)
	if i >= 0 {
		bm.Note = next[i].Note
		if req.Note != nil {
			bm.Note = *req.Note
		}
		next[i] = bm
	} else {
		if req.Note != nil {
			bm.Note = *req.Note
		}
		next = append(next, bm)
	}
	if err := s.save(next); err != nil {
		return nil, err
	}
	s.marks = next
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

// list reports every bookmark with whether this boot made it (`live`: a replay from one
// that is not live starts with an epoch-changed gap) and whether the ring still holds
// everything after its point (`retained`: of this boot, and seq + 1 at or after the oldest
// seq held; otherwise a replay opens with since-too-old), beside the head, that oldest seq,
// the store, and the bounds a client plans against. The two flags are read at one instant;
// the ring keeps turning after the reply.
func (s *bookmarkStore) list() any {
	s.mu.Lock()
	defer s.mu.Unlock()
	epoch := s.hub.Epoch()
	head, oldest := s.hub.Span()
	out := make([]map[string]any, 0, len(s.marks))
	for _, bm := range s.marks {
		live := bm.Epoch == epoch
		out = append(out, map[string]any{
			"name": bm.Name, "epoch": bm.Epoch, "seq": bm.Seq, "t": bm.T, "note": bm.Note,
			"live": live, "retained": live && bm.Seq+1 >= oldest,
		})
	}
	store := "disk"
	if s.path == "" {
		store = "memory"
	}
	return map[string]any{
		"bookmarks": out, "count": len(out), "store": store, "head": head, "oldest": oldest,
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
	next := append(append([]bookmark(nil), s.marks[:i]...), s.marks[i+1:]...)
	if err := s.save(next); err != nil {
		return nil, err
	}
	s.marks = next
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
