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
// absolute path, so it is the same across restarts of ixmux.
type FakeSession struct {
	Name, Path, Inc string
	hold            string
	mu              sync.Mutex
}

type fakeLine struct {
	MandateID   string `json:"mandate_id"`
	Incarnation string `json:"incarnation"`
	Text        string `json:"text"`
}

// NewFakeSession declares the fake session name at path.
func NewFakeSession(name, path string) (*FakeSession, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	h := sha256.Sum256([]byte("ixmux-fake-session\n" + abs))
	return &FakeSession{Name: name, Path: abs, Inc: hex.EncodeToString(h[:6]), hold: os.Getenv(HoldEnv)}, nil
}

// Deliver appends the mandate. The incarnation is the session's own, compared with the mandate's
// in the same call that takes effect, as the Adapter contract asks.
func (f *FakeSession) Deliver(ctx context.Context, m deliver.Mandate) (deliver.Ack, error) {
	if m.Incarnation != f.Inc {
		return deliver.Ack{}, fmt.Errorf("%w: session %s is incarnation %s, not %s", deliver.ErrRefused, f.Name, f.Inc, m.Incarnation)
	}
	b, err := json.Marshal(fakeLine{m.MandateID, m.Incarnation, m.Text})
	if err != nil {
		return deliver.Ack{}, err
	}
	f.mu.Lock()
	at, err := appendLine(f.Path, append(b, '\n'))
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

func appendLine(path string, b []byte) (int64, error) {
	fh, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return 0, err
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return 0, err
	}
	if _, err := fh.Write(b); err != nil {
		return 0, err
	}
	return st.Size(), fh.Sync()
}

// Status answers for the session's own incarnation only (the Reconciler contract): Unknown for
// any other.
func (f *FakeSession) Status(ctx context.Context, mandateID, incarnation string) (deliver.Outcome, error) {
	if incarnation != f.Inc {
		return deliver.Unknown, nil
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	fh, err := os.Open(f.Path)
	if errors.Is(err, os.ErrNotExist) {
		return deliver.NotReceived, nil
	}
	if err != nil {
		return deliver.Unknown, err
	}
	defer fh.Close()
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
	return deliver.Target{Adapter: f, Incarnation: func() (string, error) { return f.Inc, nil }}
}
