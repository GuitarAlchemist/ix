package deliver

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// S2G-1 (Codex round 5): the same mandate id sent at once to several targets is delivered once.
// The global id check reserved the id only when accepted was written, so concurrent deliveries
// to different targets all passed it and every adapter was called.
func TestConcurrentSameIDToTargetsIsDeliveredOnce(t *testing.T) {
	const n = 8
	e, _ := openEngine(t, filepath.Join(t.TempDir(), "cmd.jsonl"), Options{})
	tgs := make([]*fakeTarget, n)
	for i := range tgs {
		tgs[i] = &fakeTarget{inc: "i"}
		e.Register(fmt.Sprint("t", i), tgs[i].target())
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	codes := make([]string, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := e.Deliver(context.Background(), Mandate{MandateID: "m", Target: fmt.Sprint("t", i), Incarnation: "i", Text: "go"})
			codes[i] = refusalCode(err)
			if err != nil && codes[i] == "" {
				codes[i] = err.Error()
			}
		}()
	}
	close(start)
	wg.Wait()
	calls, ok := int64(0), 0
	for i, tg := range tgs {
		calls += tg.calls.Load()
		switch codes[i] {
		case "":
			ok++
		case MandateIDReuse, InFlight:
		default:
			t.Errorf("t%d: %s", i, codes[i])
		}
	}
	if calls != 1 || ok != 1 {
		t.Errorf("%d adapter calls and %d successes for one mandate id sent to %d targets (answers %v); want 1 and 1", calls, ok, n, codes)
	}
}

// S2G-3 (P2): in a v1 log, an accepted or delivering record without its fingerprint is corrupt:
// rebuilt without it, the same mandate would later be refused mandate-id-reuse.
func TestV1RecordWithoutFingerprintIsCorrupt(t *testing.T) {
	for _, state := range []State{Accepted, Delivering} {
		t.Run(string(state), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "cmd.jsonl")
			log := logHeader + `{"at":"2026-11-05T00:00:00Z","target":"a","mandate_id":"m1","incarnation":"i","state":"` + string(state) + `"}` + "\n"
			if err := os.WriteFile(path, []byte(log), 0o600); err != nil {
				t.Fatal(err)
			}
			e, rec, err := Open(path, Options{})
			if err == nil {
				e.Close()
			}
			if refusalCode(err) != LogCorrupt {
				t.Errorf("%s without fp: recovery %+v, %v; want %s", state, rec, err, LogCorrupt)
			}
			if b, _ := os.ReadFile(path); string(b) != log {
				t.Errorf("the refused log was changed:\n%s", b)
			}
		})
	}
}

// S2G-3 (P2): a MaxLogBytes below the header is refused at Open, before anything is written.
func TestMaxLogBytesBelowTheHeaderIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cmd.jsonl")
	e, _, err := Open(path, Options{MaxLogBytes: 1})
	if err == nil {
		e.Close()
	}
	if refusalCode(err) != LogTooLarge {
		t.Errorf("MaxLogBytes 1: %v, want %s", err, LogTooLarge)
	}
	if _, serr := os.Stat(path); !errors.Is(serr, os.ErrNotExist) {
		t.Errorf("the refused Open wrote %s (%v)", path, serr)
	}
}
