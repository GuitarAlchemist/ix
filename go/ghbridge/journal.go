// Package ghbridge is the offline core of a bridge from GitHub issue comments to a local
// coordinator: it checks a webhook delivery, records it durably before it is acknowledged,
// and tracks each mandate through explicit states. It opens no port and starts no service:
// the HTTP handler is a value a caller may mount, and the delivery is a function the caller
// gives.
package ghbridge

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"time"
)

// State is where a mandate (one issue comment) stands.
type State string

const (
	// Received: recorded durably, not yet handed to a target.
	Received State = "received"
	// Delivering: the intent to deliver is recorded and the hand-over has begun. A journal
	// opened with a mandate still here does not know whether the target got it.
	Delivering State = "delivering"
	// Delivered: the target acknowledged the mandate.
	Delivered State = "delivered"
	// Result: the target's result is recorded.
	Result State = "result"
	// Uncertain: the hand-over began and its outcome is unknown. Never delivered again by
	// the bridge: a person decides.
	Uncertain State = "uncertain"
	// Refused: the comment is not a mandate the bridge takes (author not allowed).
	Refused State = "refused"
	// Withdrawn: the comment was deleted or edited before it was delivered.
	Withdrawn State = "withdrawn"
)

// Entry is one line of the journal. The state of a comment is its last entry.
type Entry struct {
	Seq       int64     `json:"seq"`
	At        time.Time `json:"at"`
	CommentID int64     `json:"comment_id"`
	State     State     `json:"state"`
	// Delivery is GitHub's delivery id, empty for an entry made by a catch-up.
	Delivery string `json:"delivery,omitempty"`
	Source   string `json:"source,omitempty"` // "webhook" or "catch-up"
	Author   string `json:"author,omitempty"`
	Body     string `json:"body,omitempty"`
	// UpdatedAt is the comment's updated_at when it was recorded: the version received.
	UpdatedAt string `json:"updated_at,omitempty"`
	// Note says why: the reason of a refusal, the ack of a delivery, the text of a result,
	// what could not be reconstructed.
	Note string `json:"note,omitempty"`
	// CoordID and From are the id and the declared sender of the request's coord block.
	CoordID string `json:"coord_id,omitempty"`
	From    string `json:"from,omitempty"`
}

// Journal is an append-only file of entries. Every append is on disk when Append returns.
type Journal struct {
	mu    sync.Mutex
	f     *os.File
	seq   int64
	last  map[int64]Entry
	seen  map[string]bool  // delivery ids
	coord map[string]int64 // coord id -> the comment that holds it
	// TornTail is true when Open found a last line that was not whole JSON (a crash during
	// a write) and left it out. The entry it held was never acknowledged.
	TornTail bool
	// MadeUncertain lists the comments Open found in Delivering and moved to Uncertain.
	MadeUncertain []int64
}

// OpenJournal reads the journal at path, creating it if absent. A line that is not JSON
// anywhere but at the very end is an error: the journal is not guessed at. A mandate found
// in Delivering is moved to Uncertain, durably, before OpenJournal returns.
func OpenJournal(path string, now func() time.Time) (*Journal, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	j := &Journal{f: f, last: map[int64]Entry{}, seen: map[string]bool{}, coord: map[string]int64{}}
	data, err := io.ReadAll(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	good := int64(0) // bytes of whole, valid lines
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := sc.Bytes()
		end := good + int64(len(raw)) + 1
		whole := end <= int64(len(data)) // the line has its newline
		var e Entry
		if err := json.Unmarshal(raw, &e); err != nil || !whole {
			if end >= int64(len(data)) { // the last line
				j.TornTail = true
				break
			}
			f.Close()
			return nil, fmt.Errorf("journal %s: line %d is not an entry: %v", path, line, err)
		}
		good = end
		j.apply(e)
	}
	if err := sc.Err(); err != nil {
		f.Close()
		return nil, err
	}
	if j.TornTail {
		if err := f.Truncate(good); err != nil {
			f.Close()
			return nil, err
		}
	}
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		f.Close()
		return nil, err
	}
	var stuck []int64
	for id, e := range j.last {
		if e.State == Delivering {
			stuck = append(stuck, id)
		}
	}
	sort.Slice(stuck, func(a, b int) bool { return stuck[a] < stuck[b] })
	for _, id := range stuck {
		if _, err := j.Append(Entry{At: now(), CommentID: id, State: Uncertain,
			Note: "the bridge stopped while delivering: whether the target got the mandate is unknown"}); err != nil {
			f.Close()
			return nil, err
		}
	}
	j.MadeUncertain = stuck
	return j, nil
}

func (j *Journal) apply(e Entry) {
	if e.Seq > j.seq {
		j.seq = e.Seq
	}
	// The body and author are written once, with the first entry; later entries carry them on.
	if prev, ok := j.last[e.CommentID]; ok {
		if e.Body == "" {
			e.Body = prev.Body
		}
		if e.Author == "" {
			e.Author = prev.Author
		}
		if e.UpdatedAt == "" {
			e.UpdatedAt = prev.UpdatedAt
		}
		if e.CoordID == "" {
			e.CoordID, e.From = prev.CoordID, prev.From
		}
	}
	j.last[e.CommentID] = e
	if e.Delivery != "" {
		j.seen[e.Delivery] = true
	}
	if e.CoordID != "" && e.State != Refused {
		if _, ok := j.coord[e.CoordID]; !ok {
			j.coord[e.CoordID] = e.CommentID
		}
	}
}

// Append writes e as the next entry and syncs it to disk. It returns the entry as stored.
func (j *Journal) Append(e Entry) (Entry, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.f == nil {
		return Entry{}, errors.New("journal is closed")
	}
	e.Seq = j.seq + 1
	line, err := json.Marshal(e)
	if err != nil {
		return Entry{}, err
	}
	if _, err := j.f.Write(append(line, '\n')); err != nil {
		return Entry{}, err
	}
	if err := j.f.Sync(); err != nil {
		return Entry{}, err
	}
	j.apply(e)
	return e, nil
}

// Get returns the current entry of a comment.
func (j *Journal) Get(commentID int64) (Entry, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	e, ok := j.last[commentID]
	return e, ok
}

// SeenDelivery says whether a GitHub delivery id is already in the journal.
func (j *Journal) SeenDelivery(id string) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.seen[id]
}

// In returns the comments currently in state s, oldest entry first.
func (j *Journal) In(s State) []Entry {
	j.mu.Lock()
	defer j.mu.Unlock()
	var out []Entry
	for _, e := range j.last {
		if e.State == s {
			out = append(out, e)
		}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].CommentID < out[b].CommentID })
	return out
}

// Close closes the file.
func (j *Journal) Close() error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.f == nil {
		return nil
	}
	err := j.f.Close()
	j.f = nil
	return err
}

// CoordOwner returns the comment that first carried a coord id, if any did.
func (j *Journal) CoordOwner(coordID string) (int64, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	id, ok := j.coord[coordID]
	return id, ok
}
