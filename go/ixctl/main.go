// Command ixctl is IX's control plane: it supervises declared services,
// reconciles desired state (plan/apply) and federates with peer instances.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
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
  ixctl up <spec.json> [--addr 127.0.0.1:0] [--name NAME] [--state DIR] [--bookmarks disk|memory] [--probe-rate N] [--probe-tree-rate N] [--delta-journal FILE]
  ixctl bench <spec.json>
  ixctl service run|plan    # host under the Windows SCM (or the console); print the install plan
  ixctl probes              # the probe kinds this binary carries, with the system's limits (JSON, offline)
  ixctl resume check <inventory.json> [--json]   # after a reboot: what came back, in start order (observes only; exit 1 when not ready)
  ixctl need submit|status|claim|resolve|refuse|reserve|settle|wait --name NAME [--state DIR]   # the op's JSON on stdin
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
	case "service":
		err = cmdService(args[1:], stdout)
	case "version", "--version":
		fmt.Fprintln(stdout, "ixctl", version)
	case "probes":
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		err = enc.Encode(kindsReply(defaultProbeBudget))
	case "resume":
		var ready bool
		if ready, err = cmdResume(args[1:], stdout, realResumeProbes()); err == nil && !ready {
			return 1
		}
	case "need":
		err = cmdNeed(args[1:], os.Stdin, stdout)
	case "help", "-h", "--help":
		fmt.Fprintln(stdout, usage)
	default:
		fmt.Fprintln(stderr, usage)
		return 2
	}
	if err != nil {
		fmt.Fprintln(stderr, "ixctl:", err)
		var ec exitCodeError
		if errors.As(err, &ec) {
			return int(ec.code)
		}
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
	// bookmarks is the bookmark store: "disk" (a file beside the journal, kept across
	// restarts) or "memory" (this boot only).
	bookmarks string
	// probeRate and probeTreeRate are the probe budget (D10): records/s a subtree may declare,
	// and the whole tree.
	probeRate     int64
	probeTreeRate int64
	// deltaJournal is the JSONL journal the `delta@1` probe kind tails; empty leaves the kind
	// refused at create.
	deltaJournal string
	// claudeBin is the Claude Code CLI the `claude-sessions@1` probe kind runs; empty leaves the
	// kind refused at create.
	claudeBin string
}

func cmdUp(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	var o upOptions
	fs.StringVar(&o.addr, "addr", "127.0.0.1:0", "loopback address to listen on")
	fs.StringVar(&o.name, "name", "", "instance name (default: the spec's name)")
	fs.StringVar(&o.state, "state", "", "state directory")
	fs.StringVar(&o.bookmarks, "bookmarks", "disk", "bookmark store: disk (kept across restarts) or memory")
	fs.Int64Var(&o.probeRate, "probe-rate", defaultProbeBudget.maxRate, "records/s a probe subtree may declare (max_rate_per_s)")
	fs.Int64Var(&o.probeTreeRate, "probe-tree-rate", defaultProbeBudget.treeRate, "records/s the whole probe tree may declare (tree_rate_per_s)")
	fs.StringVar(&o.deltaJournal, "delta-journal", "", "JSONL journal the delta@1 probe kind tails (Delta's brain writes it); unset refuses the kind")
	fs.StringVar(&o.claudeBin, "claude-bin", "", "Claude Code CLI the claude-sessions@1 probe kind runs as `agents --json` (the plane must run in that user's session); unset refuses the kind")
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
	in, err := bootInstance(instanceOptions{spec: spec, name: o.name, addr: o.addr, stateDir: o.state, bookmarks: o.bookmarks,
		budget:       &probeBudget{maxRate: o.probeRate, treeRate: o.probeTreeRate},
		deltaJournal: o.deltaJournal, claudeBin: o.claudeBin})
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
	workshop  *workshopHost
}

type readyLine struct {
	Event     string  `json:"event"`
	Name      string  `json:"name"`
	Addr      string  `json:"addr"`
	Pid       int     `json:"pid"`
	StateFile string  `json:"state_file"`
	MsToReady float64 `json:"ms_to_ready"`
}

// close stops every service tree, then the transports, then removes the
// state file and lets go of the name.
func (in *instance) close() {
	in.workshop.close()
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

// writeFileAtomic writes through `path.tmp` and renames (replaceFile). The content is the
// state file, token included: a failed rename leaves the copy (mode 0600) only when it cannot
// see the state file any more, the copy then being the only one.
func writeFileAtomic(path string, b []byte) error {
	return replaceFile(path, b)
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
