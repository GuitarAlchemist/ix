package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

// run is main without the process: 0 on a report, 2 on anything else. It never writes a
// partial report: an input it cannot read is an error, not an empty map.
func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("prmap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	in := fs.String("in", "", "file of `gh pr list --json` output; - for stdin; empty runs gh")
	repo := fs.String("repo", "", "owner/name, passed to gh when -in is empty (default: gh's current repository)")
	claims := fs.String("claims", "", "claims ledger (JSON lines) to name each branch's holder")
	base := fs.String("default-branch", "main", "the branch a pull request normally targets")
	asJSON := fs.Bool("json", false, "print the report as JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	var prs io.Reader
	switch *in {
	case "":
		ghArgs := []string{"pr", "list", "--state", "open", "--limit", "1000", "--json", GHFields}
		if *repo != "" {
			ghArgs = append(ghArgs, "-R", *repo)
		}
		cmd := exec.Command("gh", ghArgs...)
		cmd.Stderr = stderr
		out, err := cmd.Output()
		if err != nil {
			fmt.Fprintf(stderr, "prmap: gh pr list: %v\n", err)
			return 2
		}
		prs = bytes.NewReader(out)
	case "-":
		prs = stdin
	default:
		f, err := os.Open(*in)
		if err != nil {
			fmt.Fprintf(stderr, "prmap: %v\n", err)
			return 2
		}
		defer f.Close()
		prs = f
	}

	var ledger io.Reader
	if *claims != "" {
		f, err := os.Open(*claims)
		if err != nil {
			fmt.Fprintf(stderr, "prmap: %v\n", err)
			return 2
		}
		defer f.Close()
		ledger = f
	}

	r, err := Build(prs, ledger, *base)
	if err != nil {
		fmt.Fprintf(stderr, "prmap: %v\n", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			fmt.Fprintf(stderr, "prmap: %v\n", err)
			return 2
		}
		return 0
	}
	Text(stdout, r)
	return 0
}
