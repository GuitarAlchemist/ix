// Command ixmux is the durable ixmux process.
//
//	ixmux serve --addr 127.0.0.1:0 --state <dir> [--journal name=path ...] [--target name=path ...]
//	            [--claude-target name ...] [--claude-bin path] [--claude-env NAME ...] [--dedup-window 720h]
//	ixmux mcp --state <dir>        a local MCP server on stdio over that ixmux (MCP.md)
//	ixmux version
//
// Exit codes: 0 stopped cleanly; 1 serving failed; 2 bad command line (a --claude-target named
// twice included); 3 start refused (address not loopback, state held by another ixmux, token
// file); 4 stopped with adapter calls still running; 5 stopped, but its files not cleaned up.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/deliver"
	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

// recoveryLine says what a start found in the command log and did with it, for the operator.
func recoveryLine(r deliver.Recovery) string {
	b := &strings.Builder{}
	fmt.Fprintf(b, "ixmux: command log: %d records replayed", r.Records)
	if r.Torn != nil {
		fmt.Fprintf(b, ", a torn last record of %d bytes cut", r.Torn.Bytes)
	}
	fmt.Fprintf(b, ", %d made uncertain, %d made refused", r.MadeUncertain, r.MadeRefused)
	if r.InMemoryOnly > 0 {
		fmt.Fprintf(b, " (%d of them in memory only: the log is full)", r.InMemoryOnly)
	}
	if r.Compacted != nil {
		fmt.Fprintf(b, ", compacted from %d to %d bytes", r.Compacted.From, r.Compacted.To)
		if r.Compacted.Forgotten > 0 {
			fmt.Fprintf(b, " (%d finished mandates past the deduplication window forgotten)", r.Compacted.Forgotten)
		}
	}
	if r.CompactFailed != "" {
		fmt.Fprintf(b, ", compaction not done (the log is kept as read): %s", r.CompactFailed)
	}
	fmt.Fprintf(b, "; a finished mandate is remembered at least %s", r.DedupWindow)
	return b.String()
}

// version is set at build time: go build -ldflags "-X main.version=$(git describe --always --dirty)".
var version = "dev"

type named []server.Named

func (n *named) String() string { return fmt.Sprint(*n) }
func (n *named) Set(v string) error {
	name, path, ok := strings.Cut(v, "=")
	if !ok || name == "" || path == "" {
		return fmt.Errorf("want name=path, got %q", v)
	}
	*n = append(*n, server.Named{Name: name, Path: path})
	return nil
}

type names []string

func (n *names) String() string { return fmt.Sprint(*n) }
func (n *names) Set(v string) error {
	if v == "" || strings.ContainsAny(v, `/\:=`) {
		return fmt.Errorf("want a plain name, got %q", v)
	}
	*n = append(*n, v)
	return nil
}

func main() { os.Exit(run(os.Args[1:], os.Stderr)) }

func run(args []string, stderr io.Writer) int {
	if len(args) > 0 && args[0] == "version" {
		fmt.Fprintln(stderr, version)
		return server.ExitOK
	}
	if len(args) > 0 && args[0] == "mcp" {
		return runMCP(args[1:], os.Stdin, os.Stdout, stderr)
	}
	if len(args) == 0 || args[0] != "serve" {
		fmt.Fprintln(stderr, "usage: ixmux serve --addr 127.0.0.1:0 --state <dir> [--journal name=path ...] [--target name=path ...]")
		return server.ExitUsage
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	cfg := server.Config{Version: version}
	var journals, targets named
	var claude, claudeEnv names
	fs.StringVar(&cfg.Addr, "addr", "127.0.0.1:0", "loopback IP address and port")
	fs.StringVar(&cfg.State, "state", "", "state directory: command log, ixmux.json, token")
	fs.Var(&journals, "journal", "a journal to serve and follow, name=path (repeatable)")
	fs.Var(&targets, "target", "a fake session (JSONL file), name=path (repeatable)")
	fs.Var(&claude, "claude-target", "a real Claude session: one claude -p child in stream-json, no tools (repeatable)")
	fs.StringVar(&cfg.ClaudeBin, "claude-bin", "", "the claude executable (default: claude on PATH)")
	fs.Var(&claudeEnv, "claude-env", "a variable of ixmux's environment the claude children also get, by name (repeatable)")
	fs.DurationVar(&cfg.Timeout, "timeout", 0, "deadline of one adapter call (default 30s)")
	fs.DurationVar(&cfg.DedupWindow, "dedup-window", 0, "a finished mandate is remembered, and deduplicated, at least this long (default 720h)")
	fs.IntVar(&cfg.MaxConns, "max-conns", server.MaxConns, "simultaneous stream connections")
	fs.IntVar(&cfg.MaxHTTP, "max-http-conns", server.MaxHTTPConns, "simultaneous TCP connections, stream connections included")
	fs.DurationVar(&cfg.ReadLimit, "read-timeout", server.ReadTimeout, "to read one request's headers and body")
	fs.IntVar(&cfg.Queue, "queue", server.QueueSize, "messages queued per stream connection")
	fs.DurationVar(&cfg.Poll, "poll", server.PollInterval, "between two reads of a followed journal")
	if err := fs.Parse(args[1:]); err != nil {
		return server.ExitUsage
	}
	if cfg.DedupWindow < 0 {
		fmt.Fprintf(stderr, "ixmux: --dedup-window %s is negative\n", cfg.DedupWindow)
		return server.ExitUsage
	}
	for i, name := range claude {
		if slices.Contains(claude[:i], name) {
			fmt.Fprintf(stderr, "ixmux: --claude-target %q is given twice; each claude target needs its own name\n", name)
			return server.ExitUsage
		}
	}
	cfg.Journals, cfg.Targets, cfg.Claude, cfg.ClaudeEnv = journals, targets, claude, claudeEnv
	s, err := server.Start(cfg)
	if err != nil {
		fmt.Fprintln(stderr, "ixmux:", err)
		if se := (*server.StartError)(nil); errors.As(err, &se) {
			return server.ExitRefused
		}
		return server.ExitError
	}
	fmt.Fprintf(stderr, "ixmux %s listening on %s (pid %d)\n", version, s.Addr(), os.Getpid())
	fmt.Fprintln(stderr, recoveryLine(s.Recovery()))
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt, syscall.SIGTERM) // on Windows, Ctrl+C and Ctrl+Break are os.Interrupt
	go func() {
		<-sig
		s.RequestStop()
	}()
	start := time.Now()
	code := s.Wait()
	fmt.Fprintf(stderr, "ixmux stopped in %s, exit code %d\n", time.Since(start).Round(time.Millisecond), code)
	return code
}
