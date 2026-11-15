package main

import "time"

// finishable is what finishTree needs of a process tree: the OS adapter is *procTree (a job
// object on Windows, a process group elsewhere), through liveTree; a test hands a fake.
type finishable interface {
	kill() error
	waitGone(limit time.Duration) bool
	drain(limit time.Duration)
	exceeded() <-chan string
	usage() (peakBytes, cpuMs, peakProcs int64)
	close()
}

// liveTree is the OS adapter: *procTree, with waitGone through the treeGone seam.
type liveTree struct{ *procTree }

func (t liveTree) waitGone(limit time.Duration) bool { return treeGone(t.procTree, limit) }

// treeEnd is how a run of a tree ended, once: the reason (the limit the plane killed it for,
// or the one its monitor or its figures say it went past), what it cost, and whether every
// process was seen gone within the bound.
type treeEnd struct {
	reason           string
	peak, cpu, procs int64
	seenGone         bool
}

// finishTree ends a tree whose leader has exited, or was given up on (killAndWait's code -1;
// leader is how): whatever it left behind is
// killed and waited for within treeGoneLimit; the monitor drained within monitorLimit; then,
// unless the plane ended the run (requested) or already named a reason, the limit the monitor
// saw it go past, if any; the usage, read before close since the job is gone after, and raised
// to the leader's own figures; the processor-time and process limits read from those very
// figures (a run that went past one and ended before a poll could kill it is past it all the
// same); then the tree is closed.
//
// Whether the tree was seen gone is known after waitGone but returned only at the end: a caller
// counting the trees not seen gone counts one later than before this module, by the drain and
// the close (on Windows close may wait for the job once more and drain again: a few bounds).
// The one reader of that count (runHost) reads it after the plane closed, once every reap
// returned.
//
// @ai:invariant usage is read after the drain and before close; a requested end names no limit of its own (it keeps the reason it was given) [P:assumed conf:0.8] (pinned by TestFinishTreeReadsInOrder, TestARequestedEndNamesNoLimit)
func finishTree(t finishable, leader exitStat, limits *Limits, reason string, requested bool) treeEnd {
	t.kill()
	gone := t.waitGone(treeGoneLimit)
	t.drain(monitorLimit)
	if reason == "" && !requested {
		select {
		case reason = <-t.exceeded():
		default:
		}
	}
	peak, cpu, procs := t.usage()
	if leader.rss > peak {
		peak = leader.rss
	}
	if leader.cpu > cpu {
		cpu = leader.cpu
	}
	if reason == "" && !requested && limits != nil {
		switch {
		case limits.CpuMs > 0 && cpu > limits.CpuMs:
			reason = reasonCPU
		case limits.Processes > 0 && procs > limits.Processes:
			reason = reasonProcesses
		}
	}
	t.close()
	return treeEnd{reason: reason, peak: peak, cpu: cpu, procs: procs, seenGone: gone}
}

// killAndWait kills a run's tree and waits for its leader's exit within treeGoneLimit; past
// it the exit is unknown (code -1).
func killAndWait(t finishable, exited <-chan exitStat) exitStat {
	t.kill()
	select {
	case st := <-exited:
		return st
	case <-time.After(treeGoneLimit):
		return exitStat{code: -1}
	}
}
