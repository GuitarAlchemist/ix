package ghbridge

import (
	"fmt"
	"regexp"
	"strings"
)

// Coord is the `coord` block of a room message, as the PR coordination room contract v0.1
// defines it (docs/contracts/2026-10-17-pr-coordination-room.contract.md, in #388/#396).
type Coord struct {
	V, ID, From, To, ReplyTo, Type, At, State, Next, ETA string
}

var (
	coordID    = regexp.MustCompile(`^[a-z0-9-]+-[0-9]{4}$`)
	coordAgent = regexp.MustCompile(`^[a-z0-9-]+(/[a-z0-9-]+)?$`)
	coordAt    = regexp.MustCompile(`^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$`)
	coordETA   = regexp.MustCompile(`^(unknown|measured [^()]+ \([^()]+\))$`)
)

// ParseCoord reads the coord block that opens a comment. found is false when the comment
// does not open with one: it is conversation, not a message. err is set when the block is
// there and is not a valid message of the contract.
func ParseCoord(body string) (c Coord, found bool, err error) {
	lines := strings.Split(strings.ReplaceAll(body, "\r\n", "\n"), "\n")
	i := 0
	for i < len(lines) && strings.TrimSpace(lines[i]) == "" {
		i++
	}
	if i == len(lines) || strings.TrimRight(lines[i], " \t") != "```coord" {
		return Coord{}, false, nil
	}
	fields := map[string]string{}
	closed := false
	for i++; i < len(lines); i++ {
		if strings.TrimRight(lines[i], " \t") == "```" {
			closed = true
			break
		}
		k, v, ok := strings.Cut(lines[i], ": ")
		if !ok {
			return Coord{}, true, fmt.Errorf("line %q is not `key: value`", lines[i])
		}
		if _, dup := fields[k]; dup {
			return Coord{}, true, fmt.Errorf("field %q is given twice", k)
		}
		fields[k] = v
	}
	if !closed {
		return Coord{}, true, fmt.Errorf("the block is not closed")
	}
	dst := map[string]*string{"v": &c.V, "id": &c.ID, "from": &c.From, "to": &c.To, "reply_to": &c.ReplyTo,
		"type": &c.Type, "at": &c.At, "state": &c.State, "next": &c.Next, "eta": &c.ETA}
	for k, v := range fields {
		p, ok := dst[k]
		if !ok {
			return Coord{}, true, fmt.Errorf("unknown field %q", k)
		}
		*p = v
	}
	for k := range dst {
		if _, ok := fields[k]; !ok {
			return Coord{}, true, fmt.Errorf("field %q is missing", k)
		}
	}
	switch {
	case c.V != "1":
		err = fmt.Errorf("v is %q, not 1", c.V)
	case !coordID.MatchString(c.ID):
		err = fmt.Errorf("id %q is not <sender>-<4 digits>", c.ID)
	case !coordAgent.MatchString(c.From) || !coordAgent.MatchString(c.To):
		err = fmt.Errorf("from or to is not an agent name")
	case c.ReplyTo != "none" && !coordID.MatchString(c.ReplyTo):
		err = fmt.Errorf("reply_to %q is neither none nor an id", c.ReplyTo)
	case !coordAt.MatchString(c.At):
		err = fmt.Errorf("at %q is not UTC ISO 8601", c.At)
	case c.Next == "":
		err = fmt.Errorf("next is empty")
	case !coordETA.MatchString(c.ETA):
		err = fmt.Errorf("eta %q is neither unknown nor measured", c.ETA)
	}
	if err != nil {
		return Coord{}, true, err
	}
	switch c.Type {
	case "request":
		if c.State != "open" || c.ReplyTo != "none" {
			err = fmt.Errorf("a request has state open and reply_to none")
		}
	case "ack":
		if c.State != "received" || c.ReplyTo == "none" {
			err = fmt.Errorf("an ack has state received and replies to a request")
		}
	case "result":
		if (c.State != "done" && c.State != "failed" && c.State != "cancelled") || c.ReplyTo == "none" {
			err = fmt.Errorf("a result has state done, failed or cancelled and replies to a request")
		}
	default:
		err = fmt.Errorf("type %q is not request, ack or result", c.Type)
	}
	if err != nil {
		return Coord{}, true, err
	}
	return c, true, nil
}

// Ack is the text of the one ack a recipient posts for a request: a coord block and nothing
// else. It names no path, no process and no machine.
func Ack(id, from string, req Coord, at string) string {
	return strings.Join([]string{"```coord", "v: 1", "id: " + id, "from: " + from, "to: " + req.From,
		"reply_to: " + req.ID, "type: ack", "at: " + at, "state: received",
		"next: " + from + " posts one result with reply_to " + req.ID, "eta: unknown", "```"}, "\n")
}
