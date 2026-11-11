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
	t.Cleanup(func() { testHookBeforeExitNote = nil })
	a, _, _ := setupCfg(t, 100*time.Millisecond, func(c *Config) {
		c.TurnMax = time.Minute
		c.StopWait = stopWait
	}, "FAKE_NOREAD=1")
	c := a.live()
	defer c.cmd.Process.Kill()
	if err := a.Stop(); err != nil {
		t.Errorf("Stop: %v, want nil: the child was killed and died", err)
	}
	if a.Lost() {
		t.Error("the incarnation is lost, though its child died within StopWait of the kill")
	}
	kinds := noteKinds(t, a.StreamPath())
	if kinds[ReasonKillFailed] != 0 || kinds["exit"] != 1 {
		t.Errorf("notes %v, want one exit and no %s", kinds, ReasonKillFailed)
	}
}
