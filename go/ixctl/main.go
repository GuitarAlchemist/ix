// Command ixctl is IX's control plane: it supervises declared services,
// reconciles desired state (plan/apply) and federates with peer instances.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"
)

var initTime = time.Now()

const usage = `usage:
  ixctl up <spec.json> [--addr 127.0.0.1:0] [--name NAME] [--state DIR]
  ixctl bench <spec.json>
  ixctl version`

func main() { os.Exit(cli(os.Args[1:], os.Stdout, os.Stderr)) }

func cli(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	var err error
	switch args[0] {
	case "up":
		err = cmdUp(args[1:], stdout)
	case "bench":
		err = cmdBench(args[1:], stdout)
	case "version", "--version":
		fmt.Fprintln(stdout, "ixctl", version)
	case "help", "-h", "--help":
		fmt.Fprintln(stdout, usage)
	default:
		fmt.Fprintln(stderr, usage)
		return 2
	}
	if err != nil {
		fmt.Fprintln(stderr, "ixctl:", err)
		return 1
	}
	return 0
}

// parseArgs parses flags placed before or after the positional arguments.
func parseArgs(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		if fs.NArg() == 0 {
			return pos, nil
		}
		pos = append(pos, fs.Arg(0))
		args = fs.Args()[1:]
	}
}

type upOptions struct {
	spec  string
	addr  string
	name  string
	state string
}

func cmdUp(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	var o upOptions
	fs.StringVar(&o.addr, "addr", "127.0.0.1:0", "loopback address to listen on")
	fs.StringVar(&o.name, "name", "", "instance name (default: the spec's name)")
	fs.StringVar(&o.state, "state", "", "state directory")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return errors.New("up needs exactly one spec file")
	}
	o.spec = pos[0]
	spec, err := LoadSpec(o.spec)
	if err != nil {
		return err
	}
	in, err := startInstance(spec, o.name, o.addr, o.state)
	if err != nil {
		return err
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sig)
	in.ready.MsToReady = msSinceProcessStart()
	line, _ := json.Marshal(in.ready)
	fmt.Fprintln(stdout, string(line))
	in.plane.Start()
	select {
	case <-sig:
	case <-in.plane.Quit():
	}
	in.close()
	return nil
}

// instance is a plane served on a loopback listener, with its state file.
type instance struct {
	plane     *Plane
	srv       *http.Server
	web       *server
	addr      string
	token     string
	stateFile string
	lock      *os.File // the per-name lock, held until close
	ready     readyLine
}

type readyLine struct {
	Event     string  `json:"event"`
	Name      string  `json:"name"`
	Addr      string  `json:"addr"`
	Pid       int     `json:"pid"`
	StateFile string  `json:"state_file"`
	MsToReady float64 `json:"ms_to_ready"`
}

// startInstance listens, serves and writes the state file. The caller starts
// the plane (supervisors, peers, autostart) once it has announced readiness.
func startInstance(spec *Spec, name, addr, stateDir string) (*instance, error) {
	if name == "" {
		name = spec.Name
	}
	if !nameRE.MatchString(name) {
		return nil, fmt.Errorf("instance name %q must match %s", name, nameRE)
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
		return nil, fmt.Errorf("instance %q is already starting or running (its lock in %s is held)", name, stateDir)
	}
	fail := func(err error) (*instance, error) {
		lock.Close()
		return nil, err
	}
	if st, err := readStateFile(statePath); err == nil && instanceAlive(st) {
		return fail(fmt.Errorf("instance %q already runs at %s (pid %d)", name, st.Addr, st.Pid))
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
	in.plane, err = NewPlane(spec, name, filepath.Join(stateDir, name+".journal.jsonl"))
	if err != nil {
		ln.Close()
		return fail(err)
	}
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
	return in, nil
}

// close stops every service tree, then the transports, then removes the
// state file and lets go of the name.
func (in *instance) close() {
	in.plane.Shutdown()
	in.web.cancel()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	in.srv.Shutdown(ctx)
	cancel()
	in.srv.Close()
	os.Remove(in.stateFile)
	if in.lock != nil {
		in.lock.Close()
	}
}

// instanceAlive asks a state file's address whether that instance still runs.
func instanceAlive(st stateFile) bool {
	if checkLoopback(st.Addr) != nil {
		return false
	}
	c := &http.Client{Transport: loopbackHTTP.Transport, CheckRedirect: noRedirect, Timeout: 300 * time.Millisecond}
	resp, err := c.Get("http://" + st.Addr + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var h struct {
		OK   bool   `json:"ok"`
		Name string `json:"name"`
	}
	return json.NewDecoder(resp.Body).Decode(&h) == nil && h.OK && h.Name == st.Name
}

func writeFileAtomic(path string, b []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func defaultStateDir() string {
	if runtime.GOOS == "windows" {
		if d := os.Getenv("LOCALAPPDATA"); d != "" {
			return filepath.Join(d, "ixctl")
		}
	}
	if d := os.Getenv("XDG_STATE_HOME"); d != "" && filepath.IsAbs(d) {
		return filepath.Join(d, "ixctl")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "state", "ixctl")
}
