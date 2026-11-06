package server

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
)

// HoldEnv, a test hook: while the file it names exists, a fake session holds each delivery after
// its effect, ignoring the deadline, as a session that took the mandate but never answered would.
const HoldEnv = "IXMUX_FAKE_HOLD_AFTER_EFFECT"

// FakeSession is the adapter of this slice: a "session" that is a JSONL file. Deliver appends one
// line {mandate_id, incarnation, text}; Status reads the file back. Its incarnation names its
// absolute path and the identity of the file there (volume and file index, or device and inode):
// the same across restarts of ixmux, another one once another file is put at the path.
type FakeSession struct {
	Name, Path string
	hold       string
	mu         sync.Mutex
}

type fakeLine struct {
	MandateID   string `json:"mandate_id"`
	Incarnation string `json:"incarnation"`
	Text        string `json:"text"`
}

// NewFakeSession declares the fake session name at path, creating its file if missing: the
// session has an incarnation from the start.
func NewFakeSession(name, path string) (*FakeSession, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	fh, err := os.OpenFile(abs, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return nil, err
	}
	fh.Close()
	return &FakeSession{Name: name, Path: abs, hold: os.Getenv(HoldEnv)}, nil
}

// incOf is the incarnation of the file fh holds open.
func (f *FakeSession) incOf(fh *os.File) (string, error) {
	id, err := fileIdentity(fh)
	if err != nil {
		return "", err
	}
	h := sha256.Sum256([]byte("ixmux-fake-session\n" + f.Path + "\n" + id))
	return hex.EncodeToString(h[:6]), nil
}

// Incarnation is the incarnation of the file now at Path.
func (f *FakeSession) Incarnation() (string, error) {
	fh, err := os.Open(f.Path)
	if err != nil {
		return "", err
	}
	defer fh.Close()
	return f.incOf(fh)
}

// Deliver appends the mandate. The incarnation is compared on the file it opened, and the line
// written through that same handle, under the session's lock: a file put at the path after the
// engine's check is refused, untouched (the Adapter contract).
func (f *FakeSession) Deliver(ctx context.Context, m deliver.Mandate) (deliver.Ack, error) {
	b, err := json.Marshal(fakeLine{m.MandateID, m.Incarnation, m.Text})
	if err != nil {
		return deliver.Ack{}, err
	}
	f.mu.Lock()
	at, err := f.appendIf(m.Incarnation, append(b, '\n'))
	f.mu.Unlock()
	if err != nil {
		return deliver.Ack{}, err
	}
	if f.hold != "" {
		for {
			if _, err := os.Stat(f.hold); err != nil {
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
	return deliver.Ack{Receipt: fmt.Sprintf("%s@%d", f.Name, at)}, nil
}

// appendIf opens the session, refuses unless the file it opened is incarnation inc, then appends
// b through the same handle and syncs it. Every refusal comes before the write.
// @ai:invariant a fake session writes only into the file whose identity is the mandate's incarnation [T:test conf:0.85 src:go/ixmux/server/fake_test.go:TestFakeSessionChecksTheFileItWrites]
func (f *FakeSession) appendIf(inc string, b []byte) (int64, error) {
	fh, err := os.OpenFile(f.Path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return 0, fmt.Errorf("%w: session %s: %v", deliver.ErrRefused, f.Name, err)
	}
	defer fh.Close()
	cur, err := f.incOf(fh)
	if err != nil {
		return 0, fmt.Errorf("%w: session %s: %v", deliver.ErrRefused, f.Name, err)
	}
	if cur != inc {
		return 0, fmt.Errorf("%w: session %s is incarnation %s, not %s", deliver.ErrRefused, f.Name, cur, inc)
	}
	st, err := fh.Stat()
	if err != nil {
		return 0, fmt.Errorf("%w: session %s: %v", deliver.ErrRefused, f.Name, err)
	}
	if _, err := fh.Write(b); err != nil {
		return 0, err
	}
	return st.Size(), fh.Sync()
}
// Status answers for the session's own incarnation only (the Reconciler contract): Unknown for
// any other.
func (f *FakeSession) Status(ctx context.Context, mandateID, incarnation string) (deliver.Outcome, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	fh, err := os.Open(f.Path)
	if errors.Is(err, os.ErrNotExist) {
		return deliver.Unknown, nil // the file of that incarnation is gone
	}
	if err != nil {
		return deliver.Unknown, err
	}
	defer fh.Close()
	if cur, err := f.incOf(fh); err != nil || cur != incarnation {
		return deliver.Unknown, err
	}
	sc := bufio.NewScanner(fh)
	sc.Buffer(make([]byte, 64<<10), 2*deliver.MaxText+1024)
	for sc.Scan() {
		var l fakeLine
		if json.Unmarshal(sc.Bytes(), &l) == nil && l.MandateID == mandateID && l.Incarnation == incarnation {
			return deliver.Received, nil
		}
	}
	if err := sc.Err(); err != nil {
		return deliver.Unknown, err
	}
	return deliver.NotReceived, nil
}

// Target registers f with an engine.
func (f *FakeSession) Target() deliver.Target {
	return deliver.Target{Adapter: f, Incarnation: f.Incarnation}
}
