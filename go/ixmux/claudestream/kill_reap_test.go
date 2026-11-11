package claudestream

import (
	"testing"
	"time"
)

// A killed child that died is not lost because ixmux was slow to note its exit: the wait after a
// kill is for the child's death, not for the line and the fsync that record it. On a loaded
// Linux runner TestStopWithBlockedStdin failed kill-failed that way (ix#390's CI, 2026-11-11).
func TestAKilledChildIsNotLostWhileItsExitIsStillBeingNoted(t *testing.T) {
	const stopWait = 200 * time.Millisecond
	testHookBeforeExitNote = func() { time.Sleep(3 * stopWait) }
	a, _, _ := setupCfg(t, 100*time.Millisecond, func(c *Config) {
		c.TurnMax = time.Minute
		c.StopWait = stopWait
	}, "FAKE_NOREAD=1")
	c := a.live()
	defer func() { // let router end before the hook is cleared for the next test
		<-c.exited
		testHookBeforeExitNote = nil
	}()
	defer c.cmd.Process.Kill()
	if err := a.Stop(); err != nil {
		t.Errorf("Stop: %v, want nil: the child was killed and died", err)
	}
	if a.Lost() {
		t.Error("the incarnation is lost, though its child died within StopWait of the kill")
	}
	if kinds := noteKinds(t, a.StreamPath()); kinds[ReasonKillFailed] != 0 {
		t.Errorf("notes %v, want no %s", kinds, ReasonKillFailed)
	}
}

// The wait for the exit note after a reaped child is bounded too: a journal write that does not
// return (a stuck disk) must not hold kill, and with it Stop and the watcher's reservation.
func TestAKillDoesNotWaitForeverOnAnExitNoteThatNeverComes(t *testing.T) {
	const stopWait = 200 * time.Millisecond
	release := make(chan struct{})
	testHookBeforeExitNote = func() { <-release }
	a, _, _ := setupCfg(t, 100*time.Millisecond, func(c *Config) {
		c.TurnMax = time.Minute
		c.StopWait = stopWait
	}, "FAKE_NOREAD=1")
	c := a.live()
	defer func() { // let router end before the hook is cleared for the next test
		close(release)
		<-c.exited
		testHookBeforeExitNote = nil
	}()
	defer c.cmd.Process.Kill()
	stopped := make(chan error, 1)
	go func() { stopped <- a.Stop() }()
	select {
	case err := <-stopped:
		if err != nil {
			t.Errorf("Stop: %v, want nil: the child was killed and died", err)
		}
	case <-time.After(10 * stopWait):
		t.Fatal("Stop waits on an exit note that never comes")
	}
	if a.Lost() {
		t.Error("the incarnation is lost, though its child died")
	}
	if kinds := noteKinds(t, a.StreamPath()); kinds[ReasonKillFailed] != 0 {
		t.Errorf("notes %v, want no %s", kinds, ReasonKillFailed)
	}
}

// With nothing slow, the exit of a killed child is in the journal when Stop returns.
func TestAKilledChildsExitIsNotedWhenStopReturns(t *testing.T) {
	a, _, _ := setupCfg(t, 100*time.Millisecond, func(c *Config) {
		c.TurnMax = time.Minute
		c.StopWait = 200 * time.Millisecond
	}, "FAKE_NOREAD=1")
	defer a.live().cmd.Process.Kill()
	if err := a.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if kinds := noteKinds(t, a.StreamPath()); kinds["exit"] != 1 || kinds["stop"] != 2 {
		t.Errorf("notes %v, want one exit and the two stop steps", kinds)
	}
}
