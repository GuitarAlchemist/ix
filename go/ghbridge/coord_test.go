package ghbridge

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func block(lines ...string) string {
	return "```coord\n" + strings.Join(lines, "\n") + "\n```\n"
}

var (
	// The three valid fixtures of the contract (#396), by type.
	validRequest = block("v: 1", "id: claude-indigo-0001", "from: claude/indigo", "to: gaia", "reply_to: none",
		"type: request", "at: 2026-10-17T20:10:00Z", "state: open",
		"next: gaia posts one result with reply_to claude-indigo-0001", "eta: unknown") +
		"\nGaia, please confirm that your trigger detected this comment."
	validAck = block("v: 1", "id: gaia-0001", "from: gaia", "to: claude/indigo", "reply_to: claude-indigo-0001",
		"type: ack", "at: 2026-10-17T20:14:00Z", "state: received", "next: gaia posts one result",
		"eta: measured 4 minutes (the previous run)")
	validResult = block("v: 1", "id: gaia-0002", "from: gaia", "to: claude/indigo", "reply_to: claude-indigo-0001",
		"type: result", "at: 2026-10-17T20:30:00Z", "state: done", "next: none", "eta: unknown")
)

func TestTheContractsValidMessagesParseAndItsInvalidOnesDoNot(t *testing.T) {
	for name, body := range map[string]string{"request": validRequest, "ack": validAck, "result": validResult} {
		c, found, err := ParseCoord(body)
		if !found || err != nil || c.Type != name {
			t.Errorf("%s: found=%v err=%v type=%q", name, found, err, c.Type)
		}
	}
	invalid := map[string]string{
		"missing id": block("v: 1", "from: gaia", "to: claude/indigo", "reply_to: none", "type: request",
			"at: 2026-10-17T20:10:00Z", "state: open", "next: none", "eta: unknown"),
		"ack with state done": strings.Replace(validAck, "state: received", "state: done", 1),
		"result without reply_to": block("v: 1", "id: gaia-0002", "from: gaia", "to: claude/indigo",
			"type: result", "at: 2026-10-17T20:30:00Z", "state: done", "next: none", "eta: unknown"),
		"unknown type":            strings.Replace(validRequest, "type: request", "type: notify", 1),
		"type status":             strings.Replace(validRequest, "type: request", "type: status", 1),
		"request that replies":    strings.Replace(validRequest, "reply_to: none", "reply_to: gaia-0001", 1),
		"extra field":             strings.Replace(validRequest, "v: 1\n", "v: 1\npriority: high\n", 1),
		"field twice":             strings.Replace(validRequest, "v: 1\n", "v: 1\nv: 1\n", 1),
		"a mention in to":         strings.Replace(validRequest, "to: gaia", "to: @gaia", 1),
		"local time":              strings.Replace(validRequest, "20:10:00Z", "20:10:00+01:00", 1),
		"a guessed eta":           strings.Replace(validRequest, "eta: unknown", "eta: about an hour", 1),
		"block never closed":      "```coord\nv: 1\nid: gaia-0001\n",
		"version 2":               strings.Replace(validRequest, "v: 1", "v: 2", 1),
		"id without four digits":  strings.Replace(validRequest, "id: claude-indigo-0001", "id: claude-indigo-1", 1),
		"a line that is not k: v": strings.Replace(validRequest, "v: 1\n", "v: 1\nhello\n", 1),
	}
	for name, body := range invalid {
		if _, found, err := ParseCoord(body); !found || err == nil {
			t.Errorf("%s: found=%v err=%v, want a block that is refused", name, found, err)
		}
	}
	for name, body := range map[string]string{
		"plain talk":             "Looks good to me.",
		"a block after text":     "See below.\n\n" + validRequest,
		"another fenced block":   "```json\n{}\n```",
		"coord with a suffix":    "```coordination\nv: 1\n```",
		"a risk report":          "<!-- agent-blackbox-risk-report -->\n# Agent Blackbox Risk Report",
		"empty":                  "",
		"indented four (a code)": "    ```coord\n    v: 1\n    ```",
	} {
		if _, found, _ := ParseCoord(body); found {
			t.Errorf("%s: taken for a coord message", name)
		}
	}
}

func mk(id int64, uid int64, body string) Comment {
	c := Comment{ID: id, Body: body, CreatedAt: "2026-11-11T19:00:00Z", UpdatedAt: "2026-11-11T19:00:00Z"}
	c.User.ID, c.User.Login = uid, "someone"
	return c
}

// In the room every agent posts under the operator's account, so the author says nothing:
// only the type and the recipient make a mandate.
func TestOnlyAValidRequestToTheTargetIsEverDelivered(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	toPilot := func(s string) string { return strings.Replace(s, "to: claude/indigo", "to: claude-pilot", 1) }
	list := []Comment{
		mk(1, 1001, "ordinary talk, from the allowed account"),
		mk(2, 1001, toPilot(validAck)),                                                // an ack addressed to the target
		mk(3, 1001, toPilot(validResult)),                                             // a result addressed to the target
		mk(4, 1001, validRequest),                                                     // a valid request, to gaia: not ours
		mk(5, 1001, strings.Replace(req(5, "x"), "type: request", "type: status", 1)), // invalid, allowed author
		mk(6, 2002, strings.Replace(req(6, "x"), "type: request", "type: status", 1)), // invalid, stranger
		mk(7, 2002, req(7, "from a stranger")),
		mk(8, 1001, req(8, "the one mandate")),
		mk(9, 1001, req(8, "the same id again, other text")), // id gaia-0008 reused
	}
	if _, err := b.CatchUp(list); err != nil {
		t.Fatal(err)
	}
	want := map[int64]State{1: "", 2: "", 3: "", 4: "", 5: Refused, 6: "", 7: Refused, 8: Received, 9: Refused}
	for id, s := range want {
		if got := state(t, b, id); got != s {
			t.Errorf("comment %d is %q, want %q", id, got, s)
		}
	}
	if e, _ := b.J.Get(9); !strings.Contains(e.Note, "gaia-0008 is already the id of comment 8") {
		t.Errorf("comment 9's refusal: %q", e.Note)
	}
	if e, _ := b.J.Get(8); e.CoordID != "gaia-0008" || e.From != "gaia" {
		t.Errorf("comment 8 lost its coord id or sender: %+v", e)
	}
	var sent []int64
	for {
		_, ok, err := b.DeliverNext(context.Background(), func(_ context.Context, m Entry) (string, error) {
			sent = append(sent, m.CommentID)
			return "ack", nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
	}
	if len(sent) != 1 || sent[0] != 8 {
		t.Errorf("delivered %v, want only [8]", sent)
	}
}

// The bridge's own ack, read back from the room, is not a mandate either.
func TestTheAckTheBridgePostsIsAValidAckAndNeverAMandate(t *testing.T) {
	r, _, _ := ParseCoord(req(8, "the one mandate"))
	ack := Ack("claude-pilot-0001", "claude-pilot", r, "2026-11-11T20:00:05Z")
	c, found, err := ParseCoord(ack)
	if !found || err != nil || c.Type != "ack" || c.ReplyTo != "gaia-0008" || c.To != "gaia" {
		t.Fatalf("the ack does not parse as an ack to the sender: %+v found=%v err=%v", c, found, err)
	}
	for _, bad := range []string{"\\", "/", "@", "C:", "pid"} {
		if strings.Contains(ack, bad) {
			t.Errorf("the public ack holds %q", bad)
		}
	}
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	// Even addressed to the target itself, under the allowed account.
	self := strings.Replace(ack, "to: gaia", "to: claude-pilot", 1)
	if _, err := b.CatchUp([]Comment{mk(20, 1001, ack), mk(21, 1001, self)}); err != nil {
		t.Fatal(err)
	}
	if len(b.J.In(Received)) != 0 {
		t.Error("an ack became a mandate")
	}
}
