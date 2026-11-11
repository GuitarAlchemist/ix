package ghbridge

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Config says which comments are mandates.
type Config struct {
	Secret []byte   // the webhook's secret
	Repo   string   // "owner/name": deliveries from another repository are ignored
	Room   int      // the issue number of the dedicated room
	Allow  []string // logins whose comments are mandates
	Now    func() time.Time
}

func (c Config) allowed(login string) bool {
	for _, a := range c.Allow {
		if strings.EqualFold(a, login) {
			return true
		}
	}
	return false
}

// VerifySignature checks GitHub's X-Hub-Signature-256 header against the body.
func VerifySignature(secret, body []byte, header string) error {
	if len(secret) == 0 {
		return errors.New("no webhook secret configured")
	}
	hexsum, ok := strings.CutPrefix(header, "sha256=")
	if !ok {
		return errors.New("no sha256 signature")
	}
	got, err := hex.DecodeString(hexsum)
	if err != nil {
		return errors.New("signature is not hex")
	}
	mac := hmac.New(sha256.New, secret)
	mac.Write(body)
	if !hmac.Equal(got, mac.Sum(nil)) {
		return errors.New("signature does not match")
	}
	return nil
}

// Comment is an issue comment, from a webhook payload or from the API.
type Comment struct {
	ID        int64  `json:"id"`
	Body      string `json:"body"`
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
}

type payload struct {
	Action  string  `json:"action"`
	Comment Comment `json:"comment"`
	Issue   struct {
		Number int `json:"number"`
	} `json:"issue"`
	Repository struct {
		FullName string `json:"full_name"`
	} `json:"repository"`
}

// Outcome is what the bridge did with one delivery or one caught-up comment.
type Outcome string

const (
	Recorded  Outcome = "recorded"  // a new mandate, now Received
	Duplicate Outcome = "duplicate" // already in the journal: nothing written
	Ignored   Outcome = "ignored"   // not for this bridge: nothing written
	RefusedBy Outcome = "refused"   // author not allowed: recorded as Refused, without the body
	Changed   Outcome = "changed"   // an edit or a deletion of a recorded comment, recorded
)

// Bridge takes deliveries into a journal.
type Bridge struct {
	Cfg Config
	J   *Journal
}

const maxBody = 1 << 20

// ServeHTTP answers a GitHub webhook delivery. It answers 2xx only once the event is on
// disk, so a delivery GitHub believes received is one the journal holds.
func (b *Bridge) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil || len(body) > maxBody {
		http.Error(w, "body unreadable or too large", http.StatusBadRequest)
		return
	}
	// An unsigned or badly signed delivery writes nothing: the journal is not a place
	// anyone on the network can fill.
	if err := VerifySignature(b.Cfg.Secret, body, r.Header.Get("X-Hub-Signature-256")); err != nil {
		http.Error(w, "signature refused", http.StatusUnauthorized)
		return
	}
	out, err := b.Take(r.Header.Get("X-GitHub-Event"), r.Header.Get("X-GitHub-Delivery"), body)
	if err != nil {
		// Not recorded: GitHub sees a failure and the delivery can be redelivered.
		http.Error(w, "not recorded", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusAccepted)
	fmt.Fprintln(w, out)
}

// Take records one verified delivery. The signature is the caller's to check.
func (b *Bridge) Take(event, delivery string, body []byte) (Outcome, error) {
	if event != "issue_comment" {
		return Ignored, nil
	}
	var p payload
	if err := json.Unmarshal(body, &p); err != nil {
		return "", fmt.Errorf("payload: %w", err)
	}
	if p.Repository.FullName != b.Cfg.Repo || p.Issue.Number != b.Cfg.Room {
		return Ignored, nil
	}
	if delivery != "" && b.J.SeenDelivery(delivery) {
		return Duplicate, nil
	}
	return b.take(p.Action, p.Comment, delivery, "webhook")
}

func (b *Bridge) take(action string, c Comment, delivery, source string) (Outcome, error) {
	now := b.Cfg.Now()
	prev, known := b.J.Get(c.ID)
	switch action {
	case "created":
		if known {
			return Duplicate, nil
		}
		if !b.Cfg.allowed(c.User.Login) {
			_, err := b.J.Append(Entry{At: now, CommentID: c.ID, State: Refused, Delivery: delivery,
				Source: source, Author: c.User.Login, Note: "author is not on the allow list"})
			return RefusedBy, err
		}
		e := Entry{At: now, CommentID: c.ID, State: Received, Delivery: delivery, Source: source,
			Author: c.User.Login, Body: c.Body, UpdatedAt: c.UpdatedAt}
		if source == "catch-up" && c.CreatedAt != c.UpdatedAt {
			e.Note = "edited before the bridge saw it: the text first posted cannot be reconstructed"
		}
		_, err := b.J.Append(e)
		return Recorded, err
	case "edited", "deleted":
		if !known {
			if action == "edited" {
				// The creation was missed: take the comment as it is now, and say so.
				c2 := c
				out, err := b.take("created", c2, delivery, source)
				return out, err
			}
			return Ignored, nil // a deletion of a comment never seen: nothing to withdraw
		}
		if action == "edited" && prev.UpdatedAt == c.UpdatedAt {
			return Duplicate, nil
		}
		e := Entry{At: now, CommentID: c.ID, Delivery: delivery, Source: source, State: prev.State}
		switch prev.State {
		case Received:
			// Not handed over yet: the mandate as recorded no longer stands.
			e.State = Withdrawn
			e.Note = action + " before delivery: not delivered"
		case Refused, Withdrawn:
			e.Note = action
		default:
			// Already handed over, or uncertain: the state stays, the change is on record.
			e.Note = action + " after the hand-over began: the target has, or may have, the earlier text"
		}
		if action == "edited" {
			e.UpdatedAt = c.UpdatedAt
		}
		_, err := b.J.Append(e)
		return Changed, err
	}
	return Ignored, nil
}

// CatchUp reconciles the journal with the room's full list of comments, as the API gives it
// after an outage. It returns what it could not reconstruct. A comment created and deleted
// while the bridge was down leaves no trace in the list: CatchUp cannot know of it.
func (b *Bridge) CatchUp(all []Comment) (lost []string, err error) {
	present := map[int64]bool{}
	for _, c := range all {
		present[c.ID] = true
		prev, known := b.J.Get(c.ID)
		switch {
		case !known:
			if _, err := b.take("created", c, "", "catch-up"); err != nil {
				return lost, err
			}
			if e, _ := b.J.Get(c.ID); e.State == Received && e.Note != "" {
				lost = append(lost, fmt.Sprintf("comment %d: %s", c.ID, e.Note))
			}
		case prev.UpdatedAt != "" && prev.UpdatedAt != c.UpdatedAt:
			if _, err := b.take("edited", c, "", "catch-up"); err != nil {
				return lost, err
			}
		}
	}
	for _, s := range []State{Received, Delivered, Result, Uncertain} {
		for _, e := range b.J.In(s) {
			if !present[e.CommentID] {
				if _, err := b.take("deleted", Comment{ID: e.CommentID}, "", "catch-up"); err != nil {
					return lost, err
				}
			}
		}
	}
	return lost, nil
}

// ErrNotSent wraps a delivery error that proves the target did not get the mandate (it was
// refused before anything was sent). Any other error leaves the mandate Uncertain.
var ErrNotSent = errors.New("not sent")

// Deliver hands one mandate to a target and returns the target's acknowledgement.
type Deliver func(ctx context.Context, e Entry) (ack string, err error)

// DeliverNext hands over the oldest Received mandate, if any. The intent is on disk before
// the hand-over begins, so a crash in between is found, at the next open, as Uncertain.
// A mandate that is Uncertain is never picked: DeliverNext delivers each mandate at most once.
func (b *Bridge) DeliverNext(ctx context.Context, deliver Deliver) (Entry, bool, error) {
	waiting := b.J.In(Received)
	if len(waiting) == 0 {
		return Entry{}, false, nil
	}
	m := waiting[0]
	if _, err := b.J.Append(Entry{At: b.Cfg.Now(), CommentID: m.CommentID, State: Delivering}); err != nil {
		return Entry{}, false, err
	}
	ack, derr := deliver(ctx, m)
	next := Entry{At: b.Cfg.Now(), CommentID: m.CommentID}
	switch {
	case derr == nil:
		next.State, next.Note = Delivered, ack
	case errors.Is(derr, ErrNotSent):
		next.State, next.Note = Received, "not sent: "+derr.Error()
	default:
		next.State, next.Note = Uncertain, "delivery outcome unknown: "+derr.Error()
	}
	e, err := b.J.Append(next)
	if err != nil {
		return Entry{}, false, err
	}
	return e, true, nil
}

// RecordResult records the result of a delivered mandate.
func (b *Bridge) RecordResult(commentID int64, result string) error {
	e, ok := b.J.Get(commentID)
	if !ok || e.State != Delivered {
		return fmt.Errorf("comment %d is %q, not delivered: no result recorded", commentID, e.State)
	}
	_, err := b.J.Append(Entry{At: b.Cfg.Now(), CommentID: commentID, State: Result, Note: result})
	return err
}
