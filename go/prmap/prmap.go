// Package main is prmap: a read-only map of a repository's open pull requests, for several
// agents working at once. It says who holds each branch, which files two pull requests both
// change, which pull request is built on another, and which checks are red or still running.
//
// It reads the JSON `gh pr list` prints and, optionally, the claims ledger; it writes nothing
// but its report. The same input gives the same report, byte for byte.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

// GHFields is the --json list prmap needs from `gh pr list`.
const GHFields = "number,title,headRefName,baseRefName,isDraft,mergeable,changedFiles,files,statusCheckRollup"

// ghPR is one pull request as `gh pr list --json` prints it.
type ghPR struct {
	Number       int    `json:"number"`
	Title        string `json:"title"`
	Head         string `json:"headRefName"`
	Base         string `json:"baseRefName"`
	Draft        bool   `json:"isDraft"`
	Mergeable    string `json:"mergeable"`
	ChangedFiles int    `json:"changedFiles"`
	Files        []struct {
		Path string `json:"path"`
	} `json:"files"`
	Checks []struct {
		Name       string `json:"name"`    // a check run
		Context    string `json:"context"` // a commit status
		Status     string `json:"status"`
		Conclusion string `json:"conclusion"`
		State      string `json:"state"`
	} `json:"statusCheckRollup"`
}

// Owner is the last line of the claims ledger that names a pull request's branch.
type Owner struct {
	Session string `json:"session"`
	Status  string `json:"status"`
	At      string `json:"at"`
}

// PR is one open pull request in the report.
type PR struct {
	Number    int    `json:"number"`
	Title     string `json:"title"`
	Head      string `json:"head"`
	Base      string `json:"base"`
	Draft     bool   `json:"draft"`
	Mergeable string `json:"mergeable"`
	// Owner is nil when no claim names the branch: unknown, not free.
	Owner   *Owner   `json:"owner"`
	Red     []string `json:"red"`
	Running []string `json:"running"`
	// FilesKnown is how many changed paths the input lists; FilesChanged is GitHub's count.
	// When they differ the overlaps of this pull request are incomplete.
	FilesKnown   int `json:"files_known"`
	FilesChanged int `json:"files_changed"`
}

// Overlap is a path that two or more open pull requests change.
type Overlap struct {
	Path string `json:"path"`
	PRs  []int  `json:"prs"`
}

// Dependency is a pull request whose base is not the default branch. On is the open pull
// request that owns that base branch, 0 when none does.
type Dependency struct {
	PR   int    `json:"pr"`
	Base string `json:"base"`
	On   int    `json:"on"`
}

// Report is the whole map. Warnings name what the report could not know.
type Report struct {
	PRs          []PR         `json:"prs"`
	Overlaps     []Overlap    `json:"overlaps"`
	Dependencies []Dependency `json:"dependencies"`
	Warnings     []string     `json:"warnings"`
}

// Build makes the report from `gh pr list` JSON and, when claims is not nil, the claims
// ledger (JSON lines). defaultBranch is the branch a pull request normally targets.
func Build(prs io.Reader, claims io.Reader, defaultBranch string) (Report, error) {
	var in []ghPR
	dec := json.NewDecoder(prs)
	if err := dec.Decode(&in); err != nil {
		return Report{}, fmt.Errorf("pull requests: %w", err)
	}
	r := Report{PRs: []PR{}, Overlaps: []Overlap{}, Dependencies: []Dependency{}, Warnings: []string{}}

	owners := map[string]Owner{}
	if claims == nil {
		r.Warnings = append(r.Warnings, "no claims ledger given: every owner is unknown")
	} else {
		var bad int
		var err error
		owners, bad, err = readClaims(claims)
		if err != nil {
			return Report{}, fmt.Errorf("claims: %w", err)
		}
		if bad > 0 {
			r.Warnings = append(r.Warnings, fmt.Sprintf("claims ledger: %d line(s) are not JSON and were not read", bad))
		}
	}

	sort.Slice(in, func(i, j int) bool { return in[i].Number < in[j].Number })
	byHead := map[string]int{}
	byPath := map[string][]int{}
	for _, p := range in {
		byHead[p.Head] = p.Number
		out := PR{Number: p.Number, Title: p.Title, Head: p.Head, Base: p.Base, Draft: p.Draft,
			Mergeable: p.Mergeable, Red: []string{}, Running: []string{},
			FilesKnown: len(p.Files), FilesChanged: p.ChangedFiles}
		if o, ok := owners[p.Head]; ok {
			out.Owner = &o
		}
		red, running := map[string]bool{}, map[string]bool{}
		for _, c := range p.Checks {
			name := c.Name
			if name == "" {
				name = c.Context
			}
			switch {
			case c.Conclusion == "FAILURE" || c.Conclusion == "TIMED_OUT" || c.Conclusion == "STARTUP_FAILURE" ||
				c.State == "FAILURE" || c.State == "ERROR":
				red[name] = true
			case c.State == "PENDING" || (c.State == "" && c.Status != "COMPLETED"):
				running[name] = true
			}
		}
		out.Red, out.Running = keys(red), keys(running)
		if out.FilesKnown != out.FilesChanged {
			r.Warnings = append(r.Warnings, fmt.Sprintf("#%d: %d of %d changed files listed: its overlaps are incomplete",
				p.Number, out.FilesKnown, out.FilesChanged))
		}
		seen := map[string]bool{}
		for _, f := range p.Files {
			if !seen[f.Path] {
				seen[f.Path] = true
				byPath[f.Path] = append(byPath[f.Path], p.Number)
			}
		}
		r.PRs = append(r.PRs, out)
	}
	for path, ns := range byPath {
		if len(ns) > 1 {
			r.Overlaps = append(r.Overlaps, Overlap{Path: path, PRs: ns})
		}
	}
	sort.Slice(r.Overlaps, func(i, j int) bool { return r.Overlaps[i].Path < r.Overlaps[j].Path })
	for _, p := range r.PRs {
		if p.Base != defaultBranch {
			r.Dependencies = append(r.Dependencies, Dependency{PR: p.Number, Base: p.Base, On: byHead[p.Base]})
		}
	}
	return r, nil
}

// readClaims keeps, for each branch, the last ledger line that names it. The ledger is
// append-only, so the last line is the current claim. It counts the lines that are not JSON.
func readClaims(claims io.Reader) (map[string]Owner, int, error) {
	owners := map[string]Owner{}
	bad := 0
	sc := bufio.NewScanner(claims)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var c struct {
			Branch  string `json:"branch"`
			Session string `json:"session"`
			Status  string `json:"status"`
			TS      string `json:"ts"`
		}
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			bad++
			continue
		}
		if c.Branch != "" {
			owners[c.Branch] = Owner{Session: c.Session, Status: c.Status, At: c.TS}
		}
	}
	return owners, bad, sc.Err()
}

func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Text writes the report for a person to read.
func Text(w io.Writer, r Report) {
	fmt.Fprintf(w, "%d open pull requests\n\n", len(r.PRs))
	for _, p := range r.PRs {
		state := "ready"
		if p.Draft {
			state = "draft"
		}
		owner := "owner unknown"
		if p.Owner != nil {
			owner = fmt.Sprintf("held by %s (%s, %s)", short(p.Owner.Session), p.Owner.Status, p.Owner.At)
		}
		fmt.Fprintf(w, "#%d %s -> %s  %s  %s  %s\n", p.Number, p.Head, p.Base, state, p.Mergeable, owner)
		if len(p.Red) > 0 {
			fmt.Fprintf(w, "    red: %s\n", strings.Join(p.Red, ", "))
		}
		if len(p.Running) > 0 {
			fmt.Fprintf(w, "    running: %s\n", strings.Join(p.Running, ", "))
		}
	}
	fmt.Fprintf(w, "\nFiles changed by more than one pull request: %d\n", len(r.Overlaps))
	for _, o := range r.Overlaps {
		fmt.Fprintf(w, "    %s: %s\n", o.Path, numbers(o.PRs))
	}
	fmt.Fprintf(w, "\nPull requests built on another branch: %d\n", len(r.Dependencies))
	for _, d := range r.Dependencies {
		if d.On != 0 {
			fmt.Fprintf(w, "    #%d is built on #%d (%s)\n", d.PR, d.On, d.Base)
		} else {
			fmt.Fprintf(w, "    #%d is built on %s, which no open pull request owns\n", d.PR, d.Base)
		}
	}
	if len(r.Warnings) > 0 {
		fmt.Fprintf(w, "\nNot known:\n")
		for _, s := range r.Warnings {
			fmt.Fprintf(w, "    %s\n", s)
		}
	}
}

func short(session string) string {
	if session == "" {
		return "an unnamed session"
	}
	if len(session) > 8 {
		return session[:8]
	}
	return session
}

func numbers(ns []int) string {
	s := make([]string, len(ns))
	for i, n := range ns {
		s[i] = fmt.Sprintf("#%d", n)
	}
	return strings.Join(s, ", ")
}
