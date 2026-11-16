package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

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

// bootInstance starts an instance in its one order: the name's lock in the state directory,
// the listener, the plane (bookmarks, probe budget, need store), the HTTP server, the state
// file, then the Workshop, then the probe kinds' settings, all before the manager announces
// readiness. A client that reads the state file (it holds the token) during the boot may still
// find a probe kind unset: the settings follow the state file. On a failure what was started
// is closed. The caller announces readiness its own way, then starts the plane (supervisors,
// peers, autostart).
//
// @ai:invariant every instance, in production (cmdUp, runHost) and in tests, starts through bootInstance: there is no other way to start one [T:compiler conf:0.9] (TestBootInstanceAppliesEveryOption pins the options, TestAHostWhoseWorkshopFailsSaysSo the host's use)
func bootInstance(o instanceOptions) (*instance, error) {
	spec, name, addr, stateDir, bookmarks, budget := o.spec, o.name, o.addr, o.stateDir, o.bookmarks, o.budget
	if name == "" {
		name = spec.Name
	}
	if !nameRE.MatchString(name) {
		return nil, fmt.Errorf("instance name %q must match %s", name, nameRE)
	}
	if bookmarks != "disk" && bookmarks != "memory" {
		return nil, fmt.Errorf("bookmark store %q: want disk or memory", bookmarks)
	}
	if budget != nil {
		if err := budget.validate(); err != nil {
			return nil, fmt.Errorf("probe budget: %w", err)
		}
	}
	if err := checkLoopback(addr); err != nil {
		return nil, err
	}
	if stateDir == "" {
		stateDir = defaultStateDir()
	}
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		return nil, err
	}
	statePath := filepath.Join(stateDir, name+".json")
	// One instance per name per state dir, held as an OS lock for the
	// instance's life. The state-file check below is advisory: two startups
	// could both pass it before either wrote the file, then append their boot
	// headers and records to one journal, interleaved. The lock dies with the
	// process, so a crash leaves nothing to clean up.
	lock, held, err := lockFile(filepath.Join(stateDir, name+".lock"))
	if err != nil {
		return nil, err
	}
	if !held {
		return nil, fmt.Errorf("%w: instance %q (its lock in %s is held)", errInstanceRunning, name, stateDir)
	}
	fail := func(err error) (*instance, error) {
		lock.Close()
		return nil, err
	}
	if st, err := readStateFile(statePath); err == nil && instanceAlive(st) {
		return fail(fmt.Errorf("%w: instance %q at %s (pid %d)", errInstanceRunning, name, st.Addr, st.Pid))
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fail(err)
	}
	tok := make([]byte, 32)
	if _, err := rand.Read(tok); err != nil {
		ln.Close()
		return fail(err)
	}
	in := &instance{addr: ln.Addr().String(), token: hex.EncodeToString(tok), stateFile: statePath, lock: lock}
	bookmarksPath := ""
	if bookmarks == "disk" {
		bookmarksPath = filepath.Join(stateDir, name+".bookmarks.json")
	}
	in.plane, err = NewPlane(spec, name, filepath.Join(stateDir, name+".journal.jsonl"), bookmarksPath)
	if err != nil {
		ln.Close()
		return fail(err)
	}
	if budget != nil {
		in.plane.SetProbeBudget(budget.maxRate, budget.treeRate) // validated above
	}
	in.plane.SetNeedStore(filepath.Join(stateDir, name+".needs.json")) // under the instance lock: its only writer
	in.web = newServer(in.plane, name, in.token, ln.Addr().(*net.TCPAddr))
	in.srv = &http.Server{Handler: in.web, ReadHeaderTimeout: 5 * time.Second}
	go in.srv.Serve(ln)
	b, _ := json.Marshal(stateFile{Name: name, Addr: in.addr, Pid: os.Getpid(), Token: in.token})
	if err := writeFileAtomic(statePath, b); err != nil {
		in.srv.Close()
		in.plane.Shutdown()
		return fail(err)
	}
	in.ready = readyLine{Event: "ready", Name: name, Addr: in.addr, Pid: os.Getpid(), StateFile: statePath}
	in.workshop, err = startWorkshop(o.workshop, o.stateDir, in.ready.Name)
	if err != nil {
		in.close()
		return nil, errWorkshopStart{err}
	}
	in.plane.SetDeltaJournal(o.deltaJournal)
	in.plane.SetClaudeBin(o.claudeBin)
	return in, nil
}

// errInstanceRunning: another process holds the name, or answers at its state file.
var errInstanceRunning = errors.New("already starting or running")
