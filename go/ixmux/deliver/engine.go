package deliver

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

type key struct{ target, id string }

type record struct {
	target, id, inc string
	state, prev     State
	reason          string
	ack             *Ack
	created, since  time.Time
}

// Engine delivers mandates. Its lock is never held while an adapter or a reconciler runs, so a
// target that never answers holds only its own delivery, its target and one slot.
type Engine struct {
	log     *cmdLog
	timeout time.Duration
	slots   chan struct{} // one per adapter call not yet returned

	mu      sync.Mutex
	targets map[string]Target
	recs    map[key]*record
	claims  map[key]bool    // a Deliver or Reconcile of this key is running
	running map[string]bool // targets with an adapter call not yet returned
}

// Open replays the command log at path (created if missing) and returns an Engine appending to
// it. A delivery left delivering becomes uncertain; one left accepted (its adapter never called)
// becomes refused, so it may be sent again. Both transitions are written to the log.
func Open(path string, opt Options) (*Engine, Recovery, error) {
	if opt.Timeout <= 0 {
		opt.Timeout = DefaultTimeout
	}
	if opt.MaxInFlight <= 0 {
		opt.MaxInFlight = MaxInFlight
	}
	if opt.MaxLogBytes <= 0 {
		opt.MaxLogBytes = MaxLogBytes
	}
	l, recs, torn, err := openLog(path, opt.MaxLogBytes)
	if err != nil {
		return nil, Recovery{}, err
	}
	e := &Engine{log: l, timeout: opt.Timeout, slots: make(chan struct{}, opt.MaxInFlight),
		targets: map[string]Target{}, recs: map[key]*record{}, claims: map[key]bool{}, running: map[string]bool{}}
	for _, r := range recs {
		e.apply(r)
	}
	rec := Recovery{Records: len(recs), Torn: torn}
	for _, r := range e.sorted() {
		k := key{r.target, r.id}
		switch r.state {
		case Delivering:
			err = e.transition(k, r.inc, Uncertain, "restart: delivering without a final state", nil)
			rec.MadeUncertain++
		case Accepted:
			err = e.transition(k, r.inc, Refused, "restart: the adapter was never called", nil)
			rec.MadeRefused++
		}
		if err != nil {
			l.close()
			return nil, Recovery{}, err
		}
	}
	return e, rec, nil
}

// apply folds one log record into the in-memory state.
func (e *Engine) apply(lr logRecord) {
	k := key{lr.Target, lr.MandateID}
	r := e.recs[k]
	if r == nil {
		r = &record{target: lr.Target, id: lr.MandateID, created: lr.At}
		e.recs[k] = r
	}
	r.prev, r.state, r.reason, r.since, r.ack = r.state, lr.State, lr.Reason, lr.At, lr.Ack
	if lr.Incarnation != "" {
		r.inc = lr.Incarnation
	}
}

// transition writes one state change to the log, then applies it; nothing changes if the write
// fails.
func (e *Engine) transition(k key, inc string, to State, reason string, ack *Ack) error {
	lr := logRecord{At: time.Now().UTC(), Target: k.target, MandateID: k.id, Incarnation: inc, State: to, Reason: reason, Ack: ack}
	if err := e.log.append(lr); err != nil {
		return err
	}
	e.mu.Lock()
	e.apply(lr)
	e.mu.Unlock()
	return nil
}

// Register names a target.
func (e *Engine) Register(name string, t Target) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.targets[name] = t
}

// gate decides, under e.mu, whether k may be sent again: done is true when the answer is
// already known (a replay or a refusal).
func (e *Engine) gate(k key) (res Result, err error, done bool) {
	if e.claims[k] {
		return Result{}, &Refusal{InFlight, fmt.Sprintf("%s/%s is being delivered or reconciled", k.target, k.id)}, true
	}
	r := e.recs[k]
	if r == nil {
		return Result{}, nil, false
	}
	switch r.state {
	case Delivered:
		res = r.result()
		res.Replayed = true
		return res, nil, true
	case Uncertain, Delivering:
		return Result{}, &Refusal{UncertainNeedsReconcile, fmt.Sprintf("%s/%s may have been received; Reconcile it first", k.target, k.id)}, true
	}
	return Result{}, nil, false // accepted or refused: the target did not take it, a new attempt is allowed
}

func (r *record) result() Result {
	return Result{Target: r.target, MandateID: r.id, State: r.state, Reason: r.reason, Ack: r.ack}
}

type callResult struct {
	ack Ack
	err error
}

// Deliver hands m to its target at most once. A (target, mandate id) already delivered returns
// the first result marked replayed, without calling the adapter; an uncertain one is refused
// until Reconcile. The adapter runs under a deadline: past it, the delivery is uncertain.
func (e *Engine) Deliver(ctx context.Context, m Mandate) (Result, error) {
	switch {
	case m.MandateID == "" || m.Target == "":
		return Result{}, &Refusal{BadMandate, "a mandate needs a mandate id and a target"}
	case len(m.Text) > MaxText:
		return Result{}, &Refusal{TextTooLong, fmt.Sprintf("text of %d bytes, bound %d", len(m.Text), MaxText)}
	}
	k := key{m.Target, m.MandateID}
	e.mu.Lock()
	t, ok := e.targets[m.Target]
	if !ok {
		e.mu.Unlock()
		return Result{}, &Refusal{UnknownTarget, "no target " + m.Target}
	}
	if res, err, done := e.gate(k); done {
		e.mu.Unlock()
		return res, err
	}
	e.mu.Unlock()

	inc, err := t.Incarnation()
	if err != nil {
		return Result{}, fmt.Errorf("incarnation of %s: %w", m.Target, err)
	}
	if inc != m.Incarnation {
		return Result{}, &Refusal{StaleIncarnation, fmt.Sprintf("mandate is for incarnation %q, %s is now %q", m.Incarnation, m.Target, inc)}
	}

	e.mu.Lock()
	if res, err, done := e.gate(k); done {
		e.mu.Unlock()
		return res, err
	}
	if e.running[m.Target] {
		e.mu.Unlock()
		return Result{}, &Refusal{TargetBusy, m.Target + " has a delivery that has not returned"}
	}
	select {
	case e.slots <- struct{}{}:
	default:
		e.mu.Unlock()
		return Result{}, &Refusal{Busy, fmt.Sprintf("%d deliveries in flight", cap(e.slots))}
	}
	e.claims[k], e.running[m.Target] = true, true
	e.mu.Unlock()
	release := func() {
		e.mu.Lock()
		delete(e.running, m.Target)
		e.mu.Unlock()
		<-e.slots
	}
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transition(k, inc, s, "", nil); err != nil {
			e.unclaim(k)
			release()
			return Result{}, err
		}
	}

	cctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	done := make(chan callResult, 1)
	go func() {
		ack, err := t.Adapter.Deliver(cctx, m)
		release() // before the answer, so a caller that has it finds the target free
		done <- callResult{ack, err}
	}()
	var c callResult
	select {
	case c = <-done:
	case <-cctx.Done():
		select {
		case c = <-done:
		default:
			return e.finish(k, inc, Uncertain, "no answer within the delivery deadline: "+cctx.Err().Error(), nil)
		}
	}
	switch {
	case c.err == nil:
		return e.finish(k, inc, Delivered, "", &c.ack)
	case errors.Is(c.err, ErrRefused):
		return e.finish(k, inc, Refused, c.err.Error(), nil)
	}
	return e.finish(k, inc, Uncertain, "adapter error, outcome unknown: "+c.err.Error(), nil)
}

// finish records the final state of a delivery or reconcile and releases its claim. If the log
// cannot be written, the delivery is uncertain in memory, as it will be after a restart.
func (e *Engine) finish(k key, inc string, to State, reason string, ack *Ack) (Result, error) {
	err := e.transition(k, inc, to, reason, ack)
	e.mu.Lock()
	defer e.mu.Unlock()
	r := e.recs[k]
	if err != nil {
		r.prev, r.state, r.reason, r.since, r.ack = r.state, Uncertain, "log write failed: "+err.Error(), time.Now().UTC(), nil
	}
	delete(e.claims, k)
	return r.result(), err
}

func (e *Engine) unclaim(k key) {
	e.mu.Lock()
	delete(e.claims, k)
	e.mu.Unlock()
}

type statusResult struct {
	o   Outcome
	err error
}

// Reconcile asks the target of an uncertain delivery whether it received the mandate: received
// makes it delivered, not received makes it refused (a new attempt is then allowed); any other
// answer, an error or no answer within the deadline leaves it uncertain.
func (e *Engine) Reconcile(ctx context.Context, target, mandateID string) (Result, error) {
	k := key{target, mandateID}
	e.mu.Lock()
	r := e.recs[k]
	var refusal *Refusal
	var rc Reconciler
	switch {
	case r == nil:
		refusal = &Refusal{UnknownMandate, fmt.Sprintf("no delivery %s/%s", target, mandateID)}
	case e.claims[k]:
		refusal = &Refusal{InFlight, fmt.Sprintf("%s/%s is being delivered or reconciled", target, mandateID)}
	case r.state != Uncertain:
		refusal = &Refusal{NotUncertain, fmt.Sprintf("%s/%s is %s", target, mandateID, r.state)}
	default:
		t, ok := e.targets[target]
		if !ok {
			refusal = &Refusal{UnknownTarget, "no target " + target}
		} else if rc, ok = t.Adapter.(Reconciler); !ok {
			refusal = &Refusal{ReconcileUnsupported, "the adapter of " + target + " cannot answer for a mandate"}
		}
	}
	if refusal != nil {
		e.mu.Unlock()
		return Result{}, refusal
	}
	e.claims[k] = true
	inc := r.inc
	e.mu.Unlock()

	cctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	done := make(chan statusResult, 1)
	go func() {
		o, err := rc.Status(cctx, mandateID)
		done <- statusResult{o, err}
	}()
	why := "no answer within the deadline"
	select {
	case s := <-done:
		switch {
		case s.err != nil:
			why = "status error: " + s.err.Error()
		case s.o == Received:
			return e.finish(k, inc, Delivered, "reconciled: the target received it", nil)
		case s.o == NotReceived:
			return e.finish(k, inc, Refused, "reconciled: the target did not receive it", nil)
		default:
			why = "the target answered " + string(s.o)
		}
	case <-cctx.Done():
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.claims, k)
	res := e.recs[k].result()
	res.Reason = "still uncertain: " + why
	return res, nil
}

func (e *Engine) sorted() []*record {
	out := make([]*record, 0, len(e.recs))
	for _, r := range e.recs {
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].target != out[j].target {
			return out[i].target < out[j].target
		}
		return out[i].id < out[j].id
	})
	return out
}

// Status lists every delivery: its state, its age and its last transition.
func (e *Engine) Status() []Entry {
	e.mu.Lock()
	defer e.mu.Unlock()
	now := time.Now()
	out := []Entry{}
	for _, r := range e.sorted() {
		prev := string(r.prev)
		if prev == "" {
			prev = "none"
		}
		out = append(out, Entry{Target: r.target, MandateID: r.id, State: r.state, Reason: r.reason,
			Age: now.Sub(r.created), Since: r.since, Last: prev + "->" + string(r.state)})
	}
	return out
}

// Close closes the log.
func (e *Engine) Close() error { return e.log.close() }
