package deliver

import (
	"context"
	"fmt"
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
