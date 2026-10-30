package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// A host runs one instance for a service manager: the Windows SCM (service_windows.go) or a
// console. It reports the SCM's states, and answers its controls, without knowing which.

type hostState string

const (
	hostStartPending hostState = "start-pending"
	hostRunning      hostState = "running"
	hostStopPending  hostState = "stop-pending"
	hostStopped      hostState = "stopped"
)

// hostStatus is one report to the manager. Accepts says stop and shutdown are taken;
// Checkpoint grows during a pending state, and WaitHint is the time to the next report or
// the end; ExitCode is the service-specific code, once stopped.
type hostStatus struct {
	State      hostState
	Accepts    bool
	Checkpoint uint32
	WaitHint   time.Duration
	ExitCode   uint32
}

const (
	controlStop        = "stop"
	controlShutdown    = "shutdown"
	controlInterrogate = "interrogate"
)

// hostControl is a request from the manager; Source names who asked (scm, signal).
type hostControl struct {
	Cmd    string
	Source string
}

// Service-specific exit codes.
const (
	exitOK             uint32 = 0
	exitStartFailed    uint32 = 1
	exitAlreadyRunning uint32 = 2
	// exitStopUnobserved: the trees were not seen gone within the bound. The process exits
	// anyway, and the jobs close with it, which kills what is left of them.
	exitStopUnobserved uint32 = 3
)

const (
	defaultStopTimeout = 30 * time.Second
	hostStartHint      = 10 * time.Second
	hostTick           = time.Second
	serviceLogMax      = 4 << 20
)

type hostOptions struct {
	spec        *Spec
	name        string
	addr        string
	stateDir    string
	stopTimeout time.Duration
	log         io.Writer
	// close ends the instance (a test seam; (*instance).close when nil).
	close func(*instance)
}

// hostLog writes one JSON object per line, with its time. It never carries the token.
type hostLog struct {
	mu sync.Mutex
	w  io.Writer
}

func (l *hostLog) event(kind string, fields map[string]any) {
	m := map[string]any{"t": time.Now().UnixMilli(), "event": kind}
	for k, v := range fields {
		m[k] = v
	}
	b, _ := json.Marshal(m)
	l.mu.Lock()
	l.w.Write(append(b, '\n'))
	l.mu.Unlock()
}

// runHost starts the instance, reports running, and on a stop (from the manager, or the
// plane's own /v1/shutdown) ends it within the bound. It returns the exit code, after
// reporting stopped.
//
// A stop has three moments: requested (the control reached the host), received (the plane
// admits nothing more), observed (every tree is gone). The first two go on ctl, before any
// service stops; the third goes to the log, since the journal is closed by then.
func runHost(o hostOptions, controls <-chan hostControl, report func(hostStatus)) uint32 {
	log := &hostLog{w: o.log}
	if o.stopTimeout <= 0 {
		o.stopTimeout = defaultStopTimeout
	}
	closeIn := o.close
	if closeIn == nil {
		closeIn = (*instance).close
	}
	cur := hostStatus{State: hostStartPending, Checkpoint: 1, WaitHint: hostStartHint}
	set := func(st hostStatus) { cur = st; report(st) }
	set(cur)
	stopped := func(code uint32) uint32 {
		set(hostStatus{State: hostStopped, ExitCode: code})
		return code
	}

	in, err := startInstance(o.spec, o.name, o.addr, o.stateDir, "disk", nil)
	if err != nil {
		if errors.Is(err, errInstanceRunning) {
			log.event("host.refused", map[string]any{"error": err.Error()})
			return stopped(exitAlreadyRunning)
		}
		log.event("host.failed", map[string]any{"error": err.Error()})
		return stopped(exitStartFailed)
	}
	in.plane.Start()
	log.event("host.ready", map[string]any{"name": in.ready.Name, "addr": in.addr, "pid": in.ready.Pid,
		"epoch": in.plane.Epoch(), "state_file": in.stateFile, "version": version})
	set(hostStatus{State: hostRunning, Accepts: true})

	source := ""
	for source == "" {
		select {
		case c := <-controls:
			switch c.Cmd {
			case controlStop, controlShutdown:
				source = c.Source
			default:
				report(cur)
			}
		case <-in.plane.Quit():
			source = "api"
		}
	}

	t0 := time.Now()
	deadline := t0.Add(o.stopTimeout)
	checkpoint := uint32(1)
	set(hostStatus{State: hostStopPending, Checkpoint: checkpoint, WaitHint: o.stopTimeout})
	in.plane.Publish("ctl", event("host.stop", map[string]any{"phase": "requested", "source": source}))
	log.event("host.stop", map[string]any{"phase": "requested", "source": source})
	in.plane.RequestShutdown()
	in.plane.Publish("ctl", event("host.stop", map[string]any{"phase": "received", "source": source}))
	log.event("host.stop", map[string]any{"phase": "received", "source": source})

	closed := make(chan struct{})
	go func() { closeIn(in); close(closed) }()
	tick := time.NewTicker(hostTick)
	defer tick.Stop()
	bound := time.NewTimer(o.stopTimeout)
	defer bound.Stop()
	unobserved := func(fields map[string]any) uint32 {
		fields["phase"], fields["source"], fields["ms"] = "unobserved", source, time.Since(t0).Milliseconds()
		log.event("host.stop", fields)
		return stopped(exitStopUnobserved)
	}
	for {
		select {
		case <-closed:
			// A returned close is not proof enough: a tree whose job still held a process at
			// the supervisor's bound was killed, not seen gone.
			if n := in.plane.TreesNotSeenGone(); n > 0 {
				return unobserved(map[string]any{"trees_not_seen_gone": n})
			}
			if time.Now().After(deadline) {
				return unobserved(map[string]any{"bound_ms": o.stopTimeout.Milliseconds()})
			}
			log.event("host.stop", map[string]any{"phase": "observed", "source": source, "ms": time.Since(t0).Milliseconds()})
			return stopped(exitOK)
		case <-bound.C:
			return unobserved(map[string]any{"bound_ms": o.stopTimeout.Milliseconds()})
		case <-tick.C:
			left := time.Until(deadline)
			checkpoint++
			set(hostStatus{State: hostStopPending, Checkpoint: checkpoint, WaitHint: left})
		case c := <-controls:
			if c.Cmd == controlInterrogate {
				report(cur)
			}
		}
	}
}

// consoleHost runs a host whose manager is the console: an interrupt or a SIGTERM is a stop.
func consoleHost(o hostOptions) uint32 {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sig)
	controls := make(chan hostControl, 1)
	go func() {
		for range sig {
			select {
			case controls <- hostControl{Cmd: controlStop, Source: "signal"}:
			default:
			}
		}
	}()
	return runHost(o, controls, func(hostStatus) {})
}

// rotatingLog is a file that, before a write would take it past max bytes, moves to
// path.1 (replacing the one before) and starts again: two files at most.
type rotatingLog struct {
	mu   sync.Mutex
	path string
	max  int64
	f    *os.File
	size int64
}

func openRotatingLog(path string, max int64) (*rotatingLog, error) {
	l := &rotatingLog{path: path, max: max}
	if err := l.open(); err != nil {
		return nil, err
	}
	return l, nil
}

func (l *rotatingLog) open() error {
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}
	l.f, l.size = f, st.Size()
	return nil
}

func (l *rotatingLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.size > 0 && l.size+int64(len(p)) > l.max {
		l.f.Close()
		os.Remove(l.path + ".1")
		if err := os.Rename(l.path, l.path+".1"); err != nil {
			return 0, err
		}
		if err := l.open(); err != nil {
			return 0, err
		}
	}
	n, err := l.f.Write(p)
	l.size += int64(n)
	return n, err
}

func (l *rotatingLog) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.f.Close()
}

const serviceUsage = `usage:
  ixctl service run <spec.json> --state DIR [--name NAME] [--addr 127.0.0.1:0] [--stop-timeout 30s] [--service-name ixctl]
  ixctl service plan --spec FILE --client-user DOMAIN\USER [--service-name ixctl] [--exe PATH] [--dir DIR]`

func cmdService(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New(serviceUsage)
	}
	switch args[0] {
	case "run":
		return cmdServiceRun(args[1:])
	case "plan":
		return cmdServicePlan(args[1:], stdout)
	}
	return errors.New(serviceUsage)
}

func cmdServiceRun(args []string) error {
	fs := flag.NewFlagSet("service run", flag.ContinueOnError)
	var o hostOptions
	var serviceName string
	fs.StringVar(&o.addr, "addr", "127.0.0.1:0", "loopback address to listen on")
	fs.StringVar(&o.name, "name", "", "instance name (default: the spec's name)")
	fs.StringVar(&o.stateDir, "state", "", "state directory (required)")
	fs.DurationVar(&o.stopTimeout, "stop-timeout", defaultStopTimeout, "bound on a stop, past which the host exits unobserved")
	fs.StringVar(&serviceName, "service-name", "ixctl", "the name the SCM knows the service by")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 || o.stateDir == "" {
		return errors.New(serviceUsage)
	}
	logPath := ""
	// setup runs once the manager is connected: under the SCM, a spec or a log that fails
	// here is a start failure the SCM sees, not a process gone before it was told anything.
	setup := func() (hostOptions, func(), error) {
		spec, err := LoadSpec(pos[0])
		if err != nil {
			return o, nil, err
		}
		o.spec = spec
		name := o.name
		if name == "" {
			name = spec.Name
		}
		if !nameRE.MatchString(name) {
			return o, nil, fmt.Errorf("instance name %q must match %s", name, nameRE)
		}
		if err := os.MkdirAll(o.stateDir, 0o700); err != nil {
			return o, nil, err
		}
		logPath = filepath.Join(o.stateDir, name+".service.log")
		l, err := openRotatingLog(logPath, serviceLogMax)
		if err != nil {
			return o, nil, err
		}
		o.log = l
		return o, func() { l.Close() }, nil
	}
	code, err := hostService(serviceName, setup)
	if err != nil {
		return err
	}
	if code != exitOK {
		return exitCodeError{code: code, log: logPath}
	}
	return nil
}

// exitCodeError carries a host's exit code to the process.
type exitCodeError struct {
	code uint32
	log  string
}

func (e exitCodeError) Error() string {
	return fmt.Sprintf("service host exit code %d (see %s)", e.code, e.log)
}

type planOptions struct {
	service, exe, dir, spec, clientUser string
	// name is the instance name (the spec's), which names the state and log files.
	name string
}

func cmdServicePlan(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("service plan", flag.ContinueOnError)
	var o planOptions
	fs.StringVar(&o.service, "service-name", "ixctl", "service name")
	fs.StringVar(&o.exe, "exe", `C:\Program Files\ixctl\ixctl.exe`, "where the binary is installed")
	fs.StringVar(&o.dir, "dir", `C:\ProgramData\ixctl`, "state directory of the service")
	fs.StringVar(&o.spec, "spec", "", "spec file to copy into the state directory (required)")
	fs.StringVar(&o.clientUser, "client-user", "", `the one user who may read the state file, DOMAIN\USER (required)`)
	if _, err := parseArgs(fs, args); err != nil {
		return err
	}
	if o.spec == "" || o.clientUser == "" {
		return errors.New(serviceUsage)
	}
	spec, err := LoadSpec(o.spec)
	if err != nil {
		return err
	}
	o.name = spec.Name
	fmt.Fprint(stdout, installPlan(o))
	return nil
}

// installPlan prints the commands an administrator runs, in order; ixctl runs none of them.
// The account is the service's virtual account (no password, its own SID); the start is
// manual and nothing restarts the service on its own, since a restarted plane starts from
// the spec, not from what the dead one was running.
func installPlan(o planOptions) string {
	spec := o.spec
	if spec == "" {
		spec = "spec.json"
	}
	svcSpec := o.dir + `\spec.json`
	acct := `NT SERVICE\` + o.service
	bin := fmt.Sprintf(`"%s" service run "%s" --state "%s" --addr 127.0.0.1:0 --service-name %s`,
		o.exe, svcSpec, o.dir, o.service)
	var b strings.Builder
	w := func(format string, a ...any) { fmt.Fprintf(&b, format+"\n", a...) }
	w("# ixctl as a Windows service: run these in an elevated PowerShell 7.3 or later (the nested quotes")
	w("# of binPath need its native argument passing), in order. ixctl runs none of them.")
	w("# account: %s (virtual: no password stored, its own SID; on the network it is the computer)", acct)
	w("# listens on loopback only; state, journal and log in %s; log bounded to 2 x %d MiB", o.dir, serviceLogMax>>20)
	w("")
	w("# install")
	w(`New-Item -ItemType Directory -Force "%s"`, o.exe[:max(strings.LastIndex(o.exe, `\`), 0)])
	w(`Copy-Item ixctl.exe "%s"`, o.exe)
	w(`New-Item -ItemType Directory -Force "%s"`, o.dir)
	w(`Copy-Item "%s" "%s"`, spec, svcSpec)
	w(`sc.exe create %s binPath= '%s' obj= "%s" start= demand DisplayName= "IX control plane"`, o.service, bin, acct)
	w(`sc.exe sidtype %s unrestricted`, o.service)
	w(`icacls "%s" /inheritance:r /grant:r "SYSTEM:(OI)(CI)F" "BUILTIN\Administrators:(OI)(CI)F" "%s:(OI)(CI)M" "%s:(OI)(CI)RX"`,
		o.dir, acct, o.clientUser)
	w("")
	w("# start")
	w(`sc.exe start %s`, o.service)
	w("")
	w("# status")
	w(`sc.exe query %s`, o.service)
	w(`Get-Content "%s\%s.json"   # addr and token: the service, SYSTEM, Administrators and %s read it`, o.dir, o.name, o.clientUser)
	w(`Get-Content -Tail 20 "%s\%s.service.log"`, o.dir, o.name)
	// sc.exe stop only asks; the stop itself takes up to the host's --stop-timeout (30 s).
	waitStopped := fmt.Sprintf(`(Get-Service %s).WaitForStatus('Stopped', [TimeSpan]::FromSeconds(60))`, o.service)
	w("")
	w("# stop")
	w(`sc.exe stop %s`, o.service)
	w("%s", waitStopped)
	w("")
	w("# uninstall (the way back: stop first, and wait for it)")
	w(`sc.exe stop %s`, o.service)
	w("%s", waitStopped)
	w(`sc.exe delete %s`, o.service)
	w(`Remove-Item -Recurse "%s"   # after reading what it holds: journal and log`, o.dir)
	w(`Remove-Item "%s"`, o.exe)
	return b.String()
}
