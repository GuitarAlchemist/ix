package main

import (
	"context"
	"sync"
)

// genLedger is the life of the plane's generations (ADR 0014, 2.1): it numbers them at
// admission, records each one's outcome as it ends (in any order), keeps the settled frontier
// (every generation up to it has ended), and answers a generation's result, now or once it
// ends. The plane admits and ends; Status and the waits read.
//
// Lock order: the ledger's lock is taken before any service's (admit holds it while it reads
// the services named by a document); no path takes a service's lock and then the ledger's.
type genLedger struct {
	mu       sync.Mutex
	gen      uint64
	observed uint64          // the highest generation that converged
	settled  uint64          // every generation <= settled has ended, converged or diverged
	ended    map[uint64]bool // generations above settled that have ended
	diverged *Diverged       // the diverged apply of highest generation, if any
	failed   []Diverged      // every diverged apply since boot, by generation
	changed  chan struct{}   // closed, then replaced, when a generation ends
}

func newGenLedger() *genLedger {
	return &genLedger{ended: map[uint64]bool{}, changed: make(chan struct{})}
}

// admit numbers the next generation and runs decide with it under the ledger's lock: what
// decide reads (the services it locks) and the number it gets are one instant.
func (l *genLedger) admit(decide func(g uint64)) uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.gen++
	decide(l.gen)
	return l.gen
}

// current is the last generation admitted.
func (l *genLedger) current() uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.gen
}

// end records g's outcome and wakes the waits.
func (l *genLedger) end(res applyResult) {
	g := res.Generation
	l.mu.Lock()
	defer l.mu.Unlock()
	if res.Converged {
		if g > l.observed {
			l.observed = g
		}
	} else {
		d := Diverged{Generation: g, Service: res.Service, Reason: res.Reason}
		l.failed = append(l.failed, d)
		if l.diverged == nil || g > l.diverged.Generation {
			l.diverged = &d
		}
	}
	// Generations end in any order; settled only moves past an unbroken run of ended ones,
	// so a later generation never counts an earlier, still pending one as done.
	l.ended[g] = true
	for l.ended[l.settled+1] {
		delete(l.ended, l.settled+1)
		l.settled++
	}
	close(l.changed)
	l.changed = make(chan struct{})
}

// ledgerStatus is what Status reports of the generations, read at one instant.
type ledgerStatus struct {
	gen, observed, settled uint64
	diverged               *Diverged
	failed                 []Diverged
}

func (l *genLedger) status() ledgerStatus {
	l.mu.Lock()
	defer l.mu.Unlock()
	st := ledgerStatus{gen: l.gen, observed: l.observed, settled: l.settled, failed: append([]Diverged{}, l.failed...)}
	if l.diverged != nil {
		d := *l.diverged
		st.diverged = &d
	}
	return st
}

// result answers g's outcome if it has ended (done), and refuses a generation not yet admitted.
func (l *genLedger) result(g uint64) (WaitReply, bool, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.resultLocked(g)
}

func (l *genLedger) resultLocked(g uint64) (WaitReply, bool, error) {
	if g > l.gen {
		return WaitReply{}, false, badRequest("unknown generation %d", g)
	}
	for _, d := range l.failed {
		if d.Generation == g {
			return WaitReply{Outcome: waitDiverged, Service: d.Service, Reason: d.Reason}, true, nil
		}
	}
	if g <= l.settled || l.ended[g] {
		return WaitReply{Outcome: waitConverged}, true, nil
	}
	return WaitReply{}, false, nil
}

// wait blocks until g has ended, and answers its outcome; an error of ctx ends it first.
//
// @ai:invariant a wait on an admitted generation answers once its end is recorded, with no ctl event read [P:assumed conf:0.8] (pinned by TestALedgerWaitWakesOnTheEndOfItsGeneration)
func (l *genLedger) wait(ctx context.Context, g uint64) (WaitReply, error) {
	for {
		l.mu.Lock()
		reply, done, err := l.resultLocked(g)
		changed := l.changed
		l.mu.Unlock()
		if done || err != nil {
			return reply, err
		}
		select {
		case <-changed:
		case <-ctx.Done():
			return WaitReply{}, ctx.Err()
		}
	}
}
