package main

import (
	"context"
	"testing"
	"time"
)

// Generations end in any order; the settled frontier moves only past an unbroken run of ended
// ones, and each one's result is its own.
func TestTheLedgerSettlesOnlyAnUnbrokenRun(t *testing.T) {
	l := newGenLedger()
	for i := 0; i < 3; i++ {
		l.admit(func(uint64) {})
	}
	l.end(applyResult{Generation: 2, Converged: true})
	l.end(applyResult{Generation: 3, Service: "x", Reason: "exit 1"})
	if st := l.status(); st.settled != 0 || st.observed != 2 || st.diverged == nil || st.diverged.Generation != 3 {
		t.Fatalf("after 2 and 3 ended: %+v, want settled 0, observed 2, diverged 3", st)
	}
	if _, done, _ := l.result(1); done {
		t.Fatal("generation 1 answered before it ended")
	}
	if r, done, _ := l.result(2); !done || r.Outcome != waitConverged {
		t.Fatalf("generation 2: %+v %v, want converged", r, done)
	}
	if r, done, _ := l.result(3); !done || r.Outcome != waitDiverged || r.Service != "x" {
		t.Fatalf("generation 3: %+v %v, want diverged on x", r, done)
	}
	l.end(applyResult{Generation: 1, Converged: true})
	if st := l.status(); st.settled != 3 {
		t.Fatalf("after 1 ended: settled %d, want 3", st.settled)
	}
	if _, _, err := l.result(4); err == nil {
		t.Fatal("a generation not yet admitted was not refused")
	}
}

// A wait on an admitted generation answers once its end is recorded: the ledger wakes it, not
// an event on ctl.
func TestALedgerWaitWakesOnTheEndOfItsGeneration(t *testing.T) {
	l := newGenLedger()
	g := l.admit(func(uint64) {})
	l.admit(func(uint64) {})
	got := make(chan WaitReply, 1)
	go func() {
		r, err := l.wait(context.Background(), g)
		if err != nil {
			t.Error(err)
		}
		got <- r
	}()
	l.end(applyResult{Generation: g + 1, Converged: true}) // another generation's end: still waiting
	select {
	case r := <-got:
		t.Fatalf("the wait answered %+v on another generation's end", r)
	case <-time.After(50 * time.Millisecond):
	}
	l.end(applyResult{Generation: g, Service: "x", Reason: "timeout"})
	select {
	case r := <-got:
		if r.Outcome != waitDiverged || r.Reason != "timeout" {
			t.Fatalf("the wait answered %+v, want diverged (timeout)", r)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the wait never woke on its generation's end")
	}
}
