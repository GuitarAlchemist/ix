// Package deliver hands mandates to session adapters at most once per (target, mandate id),
// through a durable append-only command log. A delivery whose outcome is unknown (a timeout, a
// crash between the call and its answer) is uncertain: neither a success nor replayed, until
// Reconcile asks the target.
package deliver

import (
	"context"
	"errors"
	"time"
)

// Named bounds.
const (
	MaxInFlight    = 16               // deliveries in flight at once, across all targets
	MaxText        = 64 << 10         // bytes of a mandate's text
	MaxLogBytes    = 64 << 20         // bytes of command log read back at Open
	DefaultTimeout = 30 * time.Second // one Deliver or Status call to an adapter
)

// State of one (target, mandate id).
type State string

const (
	Accepted   State = "accepted"
	Delivering State = "delivering"
	Delivered  State = "delivered"
	Refused    State = "refused"
	Uncertain  State = "uncertain"
)

// Refusal codes. A refusal means nothing was sent.
const (
	BadMandate              = "bad-mandate"
	TextTooLong             = "text-too-long"
	UnknownTarget           = "unknown-target"
	StaleIncarnation        = "stale-incarnation"
	UncertainNeedsReconcile = "uncertain-needs-reconcile"
	InFlight                = "in-flight"
	TargetBusy              = "target-busy"
	Busy                    = "busy"
	UnknownMandate          = "unknown-mandate"
	NotUncertain            = "not-uncertain"
	ReconcileUnsupported    = "reconcile-unsupported"
	LogTooLarge             = "log-too-large"
	LogCorrupt              = "log-corrupt"
	MandateIDReuse          = "mandate-id-reuse"
	JournalLocked           = "journal-locked"
	DeliveryStillRunning    = "delivery-still-running"
	LogBroken               = "log-broken"
	LogFormatUnsupported    = "log-format-unsupported"
	LockUnsupported         = "lock-unsupported"
	CallsInFlight           = "calls-in-flight"
	EngineClosed            = "engine-closed"
)

// Refusal is a typed refusal.
type Refusal struct {
	Code   string
	Detail string
}

func (r *Refusal) Error() string { return r.Code + ": " + r.Detail }

// Mandate is one command for one session. Its MandateID is unique in the whole command log:
// the engine keeps a fingerprint of (Target, Incarnation, Text), and the same id sent again with
// any of them different, even by a byte (a re-encoded JSON text, other whitespace or line
// endings), is refused mandate-id-reuse. Byte-for-byte equality is intended: the engine does not
// know what makes two texts mean the same.
type Mandate struct {
	MandateID   string
	Target      string // the effective target
	Incarnation string // the incarnation of the session it is meant for
	Text        string
}

// Ack is what an adapter returns once the target took the mandate.
type Ack struct {
	Receipt string `json:"receipt,omitempty"`
}

// Adapter delivers a mandate to its target. An error wrapping ErrRefused says the target
// certainly did not take it; any other error leaves the outcome unknown.
//
// Contract on incarnation: the engine checks m.Incarnation against Target.Incarnation and,
// under the same lock as its claim on the delivery, that the target was not registered again
// since; but the session can still restart between that check and the call. Deliver must
// therefore deliver only if the session's current incarnation equals m.Incarnation, checked
// atomically with the delivery itself (for instance by passing it to the session, which refuses
// another), and return an error wrapping ErrRefused otherwise.
type Adapter interface {
	Deliver(ctx context.Context, m Mandate) (Ack, error)
}

// ErrRefused marks an adapter error after which the target certainly did not take the mandate.
var ErrRefused = errors.New("refused by the target")

// Outcome is a target's answer to "did you receive this mandate?".
type Outcome string

const (
	Received    Outcome = "received"
	NotReceived Outcome = "not-received"
	Unknown     Outcome = "unknown"
)

// Reconciler is an optional face of an Adapter: it asks the target about a mandate.
type Reconciler interface {
	Status(ctx context.Context, mandateID string) (Outcome, error)
}

// Target is a registered session: its adapter and how to read its current incarnation.
type Target struct {
	Adapter     Adapter
	Incarnation func() (string, error)
}

// Result is the answer of Deliver or Reconcile.
type Result struct {
	Target    string `json:"target"`
	MandateID string `json:"mandate_id"`
	State     State  `json:"state"`
	Reason    string `json:"reason,omitempty"`
	Ack       *Ack   `json:"ack,omitempty"`
	Replayed  bool   `json:"replayed"`
}

// Entry is one delivery in Status.
type Entry struct {
	Target    string        `json:"target"`
	MandateID string        `json:"mandate_id"`
	State     State         `json:"state"`
	Reason    string        `json:"reason,omitempty"`
	Age       time.Duration `json:"age"`   // since it was first accepted
	Since     time.Time     `json:"since"` // time of the last transition
	Last      string        `json:"last"`  // the last transition, e.g. "delivering->uncertain"
}

// Options of an Engine; zero values take the named bounds.
type Options struct {
	Timeout     time.Duration
	MaxInFlight int
	MaxLogBytes int64
}

// Torn is the incomplete last record of the log, ignored and cut at Open.
type Torn struct {
	At    int64 `json:"at"`
	Bytes int64 `json:"bytes"`
}

// Recovery says what Open found in the log.
type Recovery struct {
	Records       int   // log records replayed
	Torn          *Torn // nil when the log ended on a complete record
	MadeUncertain int   // deliveries found delivering, now uncertain
	MadeRefused   int   // deliveries found accepted (adapter never called), now refused
}
