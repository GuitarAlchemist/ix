package main

// instanceOptions is everything an instance starts with, one value whichever manager runs it:
// `ixctl up` (the console) and the Windows service (the SCM) fill it from their own flags and
// differ only in what they leave zero.
type instanceOptions struct {
	spec      *Spec
	name      string // the spec's name when empty
	addr      string
	stateDir  string
	bookmarks string       // "disk" or "memory"
	budget    *probeBudget // nil: the plane's default probe budget
	// deltaJournal and claudeBin configure the delta@1 and claude-sessions@1 probe kinds;
	// empty leaves the kind refused at create.
	deltaJournal string
	claudeBin    string
	workshop     workshopSettings // zero: Workshop disabled
}

// errWorkshopStart wraps a Workshop that did not start, so a manager can tell it from the
// instance's own refusals (errInstanceRunning) and failures.
type errWorkshopStart struct{ err error }

func (e errWorkshopStart) Error() string { return e.err.Error() }
func (e errWorkshopStart) Unwrap() error { return e.err }

// bootInstance runs the start of an instance in its one order: the name's lock, the listener,
// the plane, the state file (startInstance), then the Workshop, then the probe kinds' settings,
// all before the manager announces readiness. A client that reads the state file (it holds the
// token) during the boot may still find a probe kind unset, as before this module: the settings
// follow the state file. On a failure what was started is closed. The caller announces
// readiness its own way, then starts the plane.
//
// @ai:invariant up (cmdUp) and the service host (runHost) start an instance through bootInstance only [P:assumed conf:0.7] (by reading; TestBootInstanceAppliesEveryOption pins the options, TestAHostWhoseWorkshopFailsSaysSo the host's use)
func bootInstance(o instanceOptions) (*instance, error) {
	in, err := startInstance(o.spec, o.name, o.addr, o.stateDir, o.bookmarks, o.budget)
	if err != nil {
		return nil, err
	}
	in.workshop, err = startWorkshop(o.workshop, o.stateDir, in.ready.Name)
	if err != nil {
		in.close()
		return nil, errWorkshopStart{err}
	}
	in.plane.SetDeltaJournal(o.deltaJournal)
	in.plane.SetClaudeBin(o.claudeBin)
	return in, nil
}
