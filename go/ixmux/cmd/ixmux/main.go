// Command ixmux is the durable ixmux process.
//
//	ixmux serve --addr 127.0.0.1:0 --state <dir> [--journal name=path ...] [--target name=path ...]
//	            [--claude-target name ...] [--claude-bin path]
//	ixmux version
//
// Exit codes: 0 stopped cleanly; 1 serving failed; 2 bad command line; 3 start refused (address
// not loopback, state held by another ixmux, token file); 4 stopped with adapter calls still
// running.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/GuitarAlchemist/ix/go/ixmux/server"
)

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
	if len(args) == 0 || args[0] != "serve" {
		fmt.Fprintln(stderr, "usage: ixmux serve --addr 127.0.0.1:0 --state <dir> [--journal name=path ...] [--target name=path ...]")
		return server.ExitUsage
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	cfg := server.Config{Version: version}
	var journals, targets named
	var claude names
	fs.StringVar(&cfg.Addr, "addr", "127.0.0.1:0", "loopback IP address and port")
	fs.StringVar(&cfg.State, "state", "", "state directory: command log, ixmux.json, token")
	fs.Var(&journals, "journal", "a journal to serve and follow, name=path (repeatable)")
	fs.Var(&targets, "target", "a fake session (JSONL file), name=path (repeatable)")
	fs.Var(&claude, "claude-target", "a real Claude session: one claude -p child in stream-json, no tools (repeatable)")
	fs.StringVar(&cfg.ClaudeBin, "claude-bin", "", "the claude executable (default: claude on PATH)")
	fs.DurationVar(&cfg.Timeout, "timeout", 0, "deadline of one adapter call (default 30s)")
	fs.IntVar(&cfg.MaxConns, "max-conns", server.MaxConns, "simultaneous stream connections")
	fs.IntVar(&cfg.MaxHTTP, "max-http-conns", server.MaxHTTPConns, "simultaneous TCP connections, stream connections included")
	fs.DurationVar(&cfg.ReadLimit, "read-timeout", server.ReadTimeout, "to read one request's headers and body")
	fs.IntVar(&cfg.Queue, "queue", server.QueueSize, "messages queued per stream connection")
	fs.DurationVar(&cfg.Poll, "poll", server.PollInterval, "between two reads of a followed journal")
	if err := fs.Parse(args[1:]); err != nil {
		return server.ExitUsage
	}
	cfg.Journals, cfg.Targets, cfg.Claude = journals, targets, claude
	s, err := server.Start(cfg)
	if err != nil {
		fmt.Fprintln(stderr, "ixmux:", err)
		if se := (*server.StartError)(nil); errors.As(err, &se) {
			return server.ExitRefused
		}
		return server.ExitError
	}
	fmt.Fprintf(stderr, "ixmux %s listening on %s (pid %d)\n", version, s.Addr(), os.Getpid())
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
