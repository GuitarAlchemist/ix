// Command ghbridge watches one GitHub issue (the room) by reading its comments through the
// API, and records what it sees in a journal. It opens no port and needs no webhook.
//
// It observes only: it delivers nothing and posts nothing. `ghbridge watch` is the first
// half of the pilot, the part that can be proven without a target.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/GuitarAlchemist/ix/go/ghbridge"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr, ghComments))
}

// fetch returns every comment of the room.
type fetch func(repo string, room int) ([]ghbridge.Comment, error)

func run(args []string, stdout, stderr io.Writer, get fetch) int {
	if len(args) == 0 || args[0] != "watch" {
		fmt.Fprintln(stderr, "usage: ghbridge watch -repo owner/name -room N -allow-id ID[,ID] -journal FILE [-since RFC3339] [-interval 30s] [-once]")
		return 2
	}
	fs := flag.NewFlagSet("watch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "owner/name of the repository")
	room := fs.Int("room", 0, "issue or pull request number of the room")
	allow := fs.String("allow-id", "", "numeric GitHub account ids allowed to give mandates, comma separated")
	journal := fs.String("journal", "", "journal file; keep it outside a public repository")
	since := fs.String("since", "", "RFC 3339 time the bridge started: older comments it never recorded are left out")
	interval := fs.Duration("interval", 30*time.Second, "time between two reads")
	once := fs.Bool("once", false, "read once and stop")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	var ids []int64
	for _, s := range strings.Split(*allow, ",") {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		id, err := strconv.ParseInt(s, 10, 64)
		if err != nil || id <= 0 {
			fmt.Fprintf(stderr, "ghbridge: -allow-id %q is not an account id\n", s)
			return 2
		}
		ids = append(ids, id)
	}
	if *repo == "" || *room <= 0 || *journal == "" || len(ids) == 0 {
		fmt.Fprintln(stderr, "ghbridge: -repo, -room, -allow-id and -journal are required")
		return 2
	}
	if *since != "" {
		if _, err := time.Parse(time.RFC3339, *since); err != nil {
			fmt.Fprintf(stderr, "ghbridge: -since: %v\n", err)
			return 2
		}
	}
	if *interval < 10*time.Second && !*once {
		fmt.Fprintln(stderr, "ghbridge: -interval under 10s would spend the API quota for nothing")
		return 2
	}

	j, err := ghbridge.OpenJournal(*journal, time.Now)
	if err != nil {
		fmt.Fprintf(stderr, "ghbridge: %v\n", err)
		return 2
	}
	defer j.Close()
	if j.TornTail {
		fmt.Fprintln(stdout, "journal: the last line was cut by a crash and was left out")
	}
	for _, id := range j.MadeUncertain {
		fmt.Fprintf(stdout, "journal: comment %d was being delivered when the bridge stopped: now uncertain\n", id)
	}
	b := &ghbridge.Bridge{J: j, Cfg: ghbridge.Config{Repo: *repo, Room: *room, Allow: ids, Since: *since, Now: time.Now}}

	for {
		comments, err := get(*repo, *room)
		if err != nil {
			// A read that failed is not an empty room: nothing is withdrawn on it.
			fmt.Fprintf(stderr, "ghbridge: read failed, journal untouched: %v\n", err)
			if *once {
				return 2
			}
		} else {
			lost, err := b.CatchUp(comments)
			if err != nil {
				fmt.Fprintf(stderr, "ghbridge: %v\n", err)
				return 2
			}
			fmt.Fprintf(stdout, "%s read %d comments:", time.Now().UTC().Format(time.RFC3339), len(comments))
			for _, s := range []ghbridge.State{ghbridge.Received, ghbridge.Delivered, ghbridge.Result,
				ghbridge.Uncertain, ghbridge.Refused, ghbridge.Withdrawn} {
				fmt.Fprintf(stdout, " %s=%d", s, len(j.In(s)))
			}
			fmt.Fprintln(stdout)
			for _, l := range lost {
				fmt.Fprintf(stdout, "  not reconstructed: %s\n", l)
			}
			if *once {
				return 0
			}
		}
		time.Sleep(*interval)
	}
}

// ghComments reads the room's comments with the gh CLI, every page.
func ghComments(repo string, room int) ([]ghbridge.Comment, error) {
	cmd := exec.Command("gh", "api", "--paginate", fmt.Sprintf("repos/%s/issues/%d/comments?per_page=100", repo, room),
		"--jq", ".[] | {id, body, created_at, updated_at, user: {login: .user.login, id: .user.id}}")
	var errb strings.Builder
	cmd.Stderr = &errb
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("gh api: %v: %s", err, strings.TrimSpace(errb.String()))
	}
	var all []ghbridge.Comment
	dec := json.NewDecoder(strings.NewReader(string(out)))
	for {
		var c ghbridge.Comment
		if err := dec.Decode(&c); errors.Is(err, io.EOF) {
			return all, nil
		} else if err != nil {
			return nil, fmt.Errorf("gh api output: %w", err)
		}
		all = append(all, c)
	}
}
