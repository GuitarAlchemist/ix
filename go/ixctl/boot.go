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
// all before any client knows the token, so no node can exist yet. On a failure what was
// started is closed. The caller announces readiness its own way, then starts the plane.
//
// @ai:invariant up and the service start an instance through bootInstance only, in this order [P:assumed conf:0.8] (pinned by TestUpAndTheServiceBootTheSameInstance)
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
