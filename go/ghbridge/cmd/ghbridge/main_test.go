package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/GuitarAlchemist/ix/go/ghbridge"
)

func c(id, author int64, body, at string) ghbridge.Comment {
	x := ghbridge.Comment{ID: id, CreatedAt: at, UpdatedAt: at}
	x.Body = fmt.Sprintf("```coord\nv: 1\nid: gaia-%04d\nfrom: gaia\nto: claude-pilot\nreply_to: none\ntype: request\nat: 2026-11-11T19:00:00Z\nstate: open\nnext: claude-pilot posts one ack\neta: unknown\n```\n\n%s", id, body)
	x.User.ID = author
	return x
}

func watch(t *testing.T, journal string, get fetch, extra ...string) (int, string, string) {
	t.Helper()
	var out, errOut bytes.Buffer
	args := append([]string{"watch", "-repo", "o/r", "-room", "388", "-allow-id", "1001", "-target", "claude-pilot", "-journal", journal, "-once"}, extra...)
	code := run(args, &out, &errOut, get)
	return code, out.String(), errOut.String()
}

func TestOneReadRecordsTheAllowedAuthorsCommentAndRefusesTheOthers(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "j.jsonl")
	get := func(string, int) ([]ghbridge.Comment, error) {
		return []ghbridge.Comment{
			c(1, 1001, "old", "2026-10-17T10:00:00Z"),
			c(2, 1001, "a mandate", "2026-11-11T20:00:00Z"),
			c(3, 2002, "not mine to give", "2026-11-11T20:01:00Z"),
		}, nil
	}
	code, out, errOut := watch(t, journal, get, "-since", "2026-11-11T19:00:00Z")
	if code != 0 {
		t.Fatalf("exit %d: %s", code, errOut)
	}
	if !strings.Contains(out, "read 3 comments: received=1 delivered=0 result=0 uncertain=0 refused=1 withdrawn=0") {
		t.Errorf("summary: %s", out)
	}
	// A second read of the same room writes nothing more.
	before, _ := os.ReadFile(journal)
	watch(t, journal, get, "-since", "2026-11-11T19:00:00Z")
	if after, _ := os.ReadFile(journal); !bytes.Equal(before, after) {
		t.Error("a second read of the same comments wrote to the journal")
	}
}

func TestAReadThatFailsWithdrawsNothing(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "j.jsonl")
	ok := func(string, int) ([]ghbridge.Comment, error) {
		return []ghbridge.Comment{c(2, 1001, "a mandate", "2026-11-11T20:00:00Z")}, nil
	}
	watch(t, journal, ok)
	before, _ := os.ReadFile(journal)
	down := func(string, int) ([]ghbridge.Comment, error) { return nil, errors.New("network down") }
	code, _, errOut := watch(t, journal, down)
	if code != 2 || !strings.Contains(errOut, "journal untouched") {
		t.Errorf("exit %d, stderr %q", code, errOut)
	}
	if after, _ := os.ReadFile(journal); !bytes.Equal(before, after) {
		t.Error("a failed read changed the journal: an outage was taken for an empty room")
	}
}

func TestMissingOrBadFlagsAreRefusedBeforeAnythingIsRead(t *testing.T) {
	never := func(string, int) ([]ghbridge.Comment, error) {
		t.Error("the room was read")
		return nil, nil
	}
	j := filepath.Join(t.TempDir(), "j.jsonl")
	for name, args := range map[string][]string{
		"no subcommand":  {},
		"no allow list":  {"watch", "-repo", "o/r", "-room", "1", "-journal", j, "-once"},
		"login as id":    {"watch", "-repo", "o/r", "-room", "1", "-journal", j, "-allow-id", "stef", "-once"},
		"no journal":     {"watch", "-repo", "o/r", "-room", "1", "-allow-id", "1", "-once"},
		"bad since":      {"watch", "-repo", "o/r", "-room", "1", "-journal", j, "-allow-id", "1", "-since", "yesterday", "-once"},
		"tight interval": {"watch", "-repo", "o/r", "-room", "1", "-journal", j, "-allow-id", "1", "-interval", "1s"},
	} {
		var out, errOut bytes.Buffer
		if code := run(args, &out, &errOut, never); code != 2 {
			t.Errorf("%s: exit %d, want 2", name, code)
		}
	}
}
