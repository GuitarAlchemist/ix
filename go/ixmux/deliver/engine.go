package deliver

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

type key struct{ target, id string }

// idHold is a mandate id reserved by the Deliver that claimed k, for the mandate fp.
type idHold struct {
	k  key
	fp string
}

type record struct {
	target, id, inc string
	fp              string // fingerprint of the mandate first accepted under this id
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
	targets map[string]*Target // one allocation per Register: a pointer names a registration
	recs    map[key]*record
	ids     map[string]*record // by mandate id alone: an id is unique in the whole log
	idClaim map[string]idHold  // ids reserved by a Deliver's claim, before accepted is written
	claims  map[key]bool       // a Deliver or Reconcile of this key is running
	running map[string]bool    // targets with an adapter call not yet returned
	calling map[key]bool       // keys whose adapter call has not returned, even past its deadline
	asking  map[key]bool       // keys whose Status call has not returned, even past its deadline
	closed  bool               // Close has begun releasing the log: no new adapter call
}

// call is one adapter call, shared by the caller and the goroutine that runs it.
type call struct {
	returned  bool          // the adapter has answered (under e.mu)
	abandoned bool          // the caller gave up at its deadline (under e.mu)
	res       callResult    // the answer, once returned
	answered  chan struct{} // closed once the answer is in res, for a caller still waiting
	finished  chan struct{} // closed once an abandoning caller has recorded uncertain
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
	lock, err := lockFile(path + ".lock")
	if errors.Is(err, errLocked) {
		return nil, Recovery{}, &Refusal{JournalLocked, path + " is held by another engine"}
	}
	if errors.Is(err, errLockUnsupported) {
		return nil, Recovery{}, &Refusal{LockUnsupported, "no exclusive file lock on this system: an engine cannot be sure to be alone on " + path}
	}
	if err != nil {
		return nil, Recovery{}, err
	}
	l, recs, torn, err := openLog(path, opt.MaxLogBytes)
	if err != nil {
		lock.Close()
		return nil, Recovery{}, err
	}
	l.lock = lock
	e := &Engine{log: l, timeout: opt.Timeout, slots: make(chan struct{}, opt.MaxInFlight),
		targets: map[string]*Target{}, recs: map[key]*record{}, ids: map[string]*record{}, idClaim: map[string]idHold{}, claims: map[key]bool{}, running: map[string]bool{},
		calling: map[key]bool{}, asking: map[key]bool{}}
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
		if e.ids[lr.MandateID] == nil {
			e.ids[lr.MandateID] = r
		}
	}
	r.prev, r.state, r.reason, r.since, r.ack = r.state, lr.State, lr.Reason, lr.At, lr.Ack
	if lr.Incarnation != "" {
		r.inc = lr.Incarnation
	}
	if r.fp == "" {
		r.fp = lr.Fingerprint
	}
}

// fingerprint names what a mandate asks: its target, incarnation and text, each prefixed by its
// length, so no choice of bytes (NUL included) makes two different mandates collide.
func fingerprint(m Mandate) string {
	h := sha256.New()
	for _, f := range []string{m.Target, m.Incarnation, m.Text} {
		fmt.Fprintf(h, "%d:", len(f))
		h.Write([]byte(f))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// transition writes one state change to the log, then applies it; nothing changes if the write
// fails.
func (e *Engine) transition(k key, inc string, to State, reason string, ack *Ack) error {
	return e.transitionFP(k, inc, to, reason, ack, "")
}

func (e *Engine) transitionFP(k key, inc string, to State, reason string, ack *Ack, fp string) error {
	lr := logRecord{At: time.Now().UTC(), Target: k.target, MandateID: k.id, Incarnation: inc, State: to, Reason: reason, Ack: ack, Fingerprint: fp}
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
	e.targets[name] = &t
}

// gate decides, under e.mu, whether k may be sent again: done is true when the answer is
// already known (a replay or a refusal).
func (e *Engine) gate(k key, fp string) (res Result, err error, done bool) {
	if first := e.ids[k.id]; first != nil && first.fp != fp { // v1 logs carry fp on every accepted record
		return Result{}, &Refusal{MandateIDReuse, fmt.Sprintf("mandate id %s was first accepted for %s with another incarnation or text", k.id, first.target)}, true
	}
	if h, ok := e.idClaim[k.id]; ok && h.fp != fp {
		return Result{}, &Refusal{MandateIDReuse, fmt.Sprintf("mandate id %s is being delivered to %s with another incarnation or text", k.id, h.k.target)}, true
	}
	r := e.recs[k]
	if e.claims[k] {
		return Result{}, &Refusal{InFlight, fmt.Sprintf("%s/%s is being delivered or reconciled", k.target, k.id)}, true
	}
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
	k, fp := key{m.Target, m.MandateID}, fingerprint(m)
	e.mu.Lock()
	t, ok := e.targets[m.Target]
	if !ok {
		e.mu.Unlock()
		return Result{}, &Refusal{UnknownTarget, "no target " + m.Target}
	}
	if res, err, done := e.gate(k, fp); done {
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
	if testHookAfterIncarnation != nil {
		testHookAfterIncarnation()
	}

	e.mu.Lock()
	if res, err, done := e.gate(k, fp); done {
		e.mu.Unlock()
		return res, err
	}
	if e.targets[m.Target] != t {
		// Re-registered since the check: the incarnation checked is not the target's any more.
		e.mu.Unlock()
		return Result{}, &Refusal{StaleIncarnation, m.Target + " was registered again after its incarnation was checked"}
	}
	if e.closed {
		e.mu.Unlock()
		return Result{}, &Refusal{EngineClosed, "the engine is closed"}
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
	// @ai:invariant a mandate id is reserved under the same lock as the claim, before any write, so concurrent deliveries of one id to two targets cannot both pass the gate [T:test conf:0.85 src:go/ixmux/deliver/s2g_test.go:TestConcurrentSameIDToTargetsIsDeliveredOnce]
	e.claims[k], e.running[m.Target], e.calling[k] = true, true, true
	e.idClaim[k.id] = idHold{k, fp}
	e.mu.Unlock()
	release := func() {
		e.mu.Lock()
		delete(e.running, m.Target)
		delete(e.calling, k)
		e.mu.Unlock()
		<-e.slots
	}
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transitionFP(k, inc, s, "", nil, fp); err != nil {
			e.unclaim(k)
			release()
			return Result{}, err
		}
	}

	// @ai:invariant the adapter is called only once delivering is synced, so a crash during the call restarts as uncertain [T:test conf:0.85 src:go/ixmux/deliver/kill_test.go:TestKillAfterEffectThenReconcileReceived]
	cctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	cl := &call{answered: make(chan struct{}), finished: make(chan struct{})}
	go func() {
		ack, err := t.Adapter.Deliver(cctx, m)
		e.mu.Lock()
		cl.returned, cl.res = true, callResult{ack, err}
		abandoned := cl.abandoned
		e.mu.Unlock()
		if abandoned {
			// The caller recorded uncertain at its deadline; this answer is an observation.
			// Reconcile stays refused (calling) until it is written.
			<-cl.finished
			e.observeLate(k, inc, cl.res)
		}
		release()
		close(cl.answered)
	}()
	select {
	case <-cl.answered:
	case <-cctx.Done():
		e.mu.Lock()
		if !cl.returned {
			cl.abandoned = true
			e.mu.Unlock()
			res, err := e.finish(k, inc, Uncertain, "no answer within the delivery deadline: "+cctx.Err().Error(), nil)
			close(cl.finished)
			return res, err
		}
		e.mu.Unlock()
		<-cl.answered
	}
	to, reason, ack := outcome(cl.res)
	return e.finish(k, inc, to, reason, ack)
}

// testHookAfterIncarnation, when set by a test, runs between the incarnation check and the claim.
var testHookAfterIncarnation func()

// outcome maps an adapter's answer to a state: an error wrapping ErrRefused is certain, any
// other error leaves the outcome unknown.
func outcome(c callResult) (State, string, *Ack) {
	switch {
	case c.err == nil:
		return Delivered, "", &c.ack
	case errors.Is(c.err, ErrRefused):
		return Refused, c.err.Error(), nil
	}
	return Uncertain, "adapter error, outcome unknown: " + c.err.Error(), nil
}

// observeLate records the answer of an adapter call that returned after its caller's deadline.
// A certain answer settles the uncertain delivery; an unknown one is logged and leaves it so.
func (e *Engine) observeLate(k key, inc string, c callResult) {
	to, reason, ack := outcome(c)
	if reason == "" {
		reason = "late answer of the adapter after the deadline"
	} else {
		reason = "late answer of the adapter after the deadline: " + reason
	}
	e.mu.Lock()
	r := e.recs[k]
	still := r != nil && r.state == Uncertain && !e.claims[k]
	e.mu.Unlock()
	if still {
		e.transition(k, inc, to, reason, ack) // a failed write leaves it uncertain, as a restart would
	}
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
	e.dropIDClaim(k)
	return r.result(), err
}

func (e *Engine) unclaim(k key) {
	e.mu.Lock()
	delete(e.claims, k)
	e.dropIDClaim(k)
	e.mu.Unlock()
}

// dropIDClaim ends k's reservation of its id (under e.mu): ids holds the id once accepted is
// written, and nothing does if that write failed.
func (e *Engine) dropIDClaim(k key) {
	if h, ok := e.idClaim[k.id]; ok && h.k == k {
		delete(e.idClaim, k.id)
	}
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
	var t *Target
	switch {
	case r == nil:
		refusal = &Refusal{UnknownMandate, fmt.Sprintf("no delivery %s/%s", target, mandateID)}
	case e.claims[k]:
		refusal = &Refusal{InFlight, fmt.Sprintf("%s/%s is being delivered or reconciled", target, mandateID)}
	case e.calling[k]:
		// The adapter call that made it uncertain may still take effect: asking the target now
		// could refuse a delivery that is about to happen.
		refusal = &Refusal{DeliveryStillRunning, fmt.Sprintf("the adapter call of %s/%s has not returned", target, mandateID)}
	case e.asking[k]:
		refusal = &Refusal{InFlight, fmt.Sprintf("the Status call of %s/%s has not returned", target, mandateID)}
	case r.state != Uncertain:
		refusal = &Refusal{NotUncertain, fmt.Sprintf("%s/%s is %s", target, mandateID, r.state)}
	default:
		var ok bool
		if t, ok = e.targets[target]; !ok {
			refusal = &Refusal{UnknownTarget, "no target " + target}
		} else if rc, ok = t.Adapter.(Reconciler); !ok {
			refusal = &Refusal{ReconcileUnsupported, "the adapter of " + target + " cannot answer for a mandate"}
		}
	}
	if refusal == nil {
		select {
		case e.slots <- struct{}{}: // released when Status returns, even past its deadline
		default:
			refusal = &Refusal{Busy, fmt.Sprintf("%d calls in flight", cap(e.slots))}
		}
	}
	if refusal != nil {
		e.mu.Unlock()
		return Result{}, refusal
	}
	e.claims[k], e.asking[k] = true, true
	inc := r.inc
	e.mu.Unlock()

	// Only the incarnation the mandate was for can say whether it received it: checked under the
	// claim before Status, and again before its answer is written.
	if err := e.sameIncarnation(target, t, inc); err != nil {
		e.mu.Lock()
		delete(e.asking, k)
		delete(e.claims, k)
		e.mu.Unlock()
		<-e.slots
		return Result{}, err
	}

	cctx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()
	done := make(chan statusResult, 1)
	go func() {
		o, err := rc.Status(cctx, mandateID, inc)
		e.mu.Lock()
		delete(e.asking, k)
		e.mu.Unlock()
		<-e.slots
		done <- statusResult{o, err}
	}()
	why := "no answer within the deadline"
	select {
	case s := <-done:
		switch {
		case s.err != nil:
			why = "status error: " + s.err.Error()
		case s.o == Received || s.o == NotReceived:
			if err := e.sameIncarnation(target, t, inc); err != nil {
				e.unclaim(k)
				return Result{}, err
			}
			if s.o == NotReceived {
				return e.finish(k, inc, Refused, "reconciled: the target did not receive it", nil)
			}
			return e.finish(k, inc, Delivered, "reconciled: the target received it", nil)
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

// sameIncarnation refuses stale-incarnation unless target is still registered as t and its
// current incarnation is inc.
// @ai:invariant Reconcile writes an outcome only if the mandate's incarnation answered it, checked before and after Status [T:test conf:0.85 src:go/ixmux/deliver/s2f_test.go:TestReconcileAsksOnlyTheMandatesIncarnation]
func (e *Engine) sameIncarnation(target string, t *Target, inc string) error {
	cur, err := t.Incarnation()
	if err != nil {
		return fmt.Errorf("incarnation of %s: %w", target, err)
	}
	if cur != inc {
		return &Refusal{StaleIncarnation, fmt.Sprintf("the delivery was for incarnation %q, %s is now %q", inc, target, cur)}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.targets[target] != t {
		return &Refusal{StaleIncarnation, target + " was registered again"}
	}
	return nil
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

// Status lists every delivery: its state, its age, its last transition, and whether its adapter
// call is still running.
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
			Age: now.Sub(r.created), Since: r.since, Last: prev + "->" + string(r.state), StillRunning: e.calling[key{r.target, r.id}]})
	}
	return out
}

// Close waits, up to the delivery deadline, for the adapter calls that have not returned (a late
// answer is still recorded), then closes the log and releases its lock. If calls still run, it
// refuses calls-in-flight and keeps both: a call can still take effect, and another engine opened
// on the log would not know it (it could reconcile not-received and send again). Close may be
// called again later.
// @ai:invariant the log's lock is released only once no adapter call of this engine can still take effect [T:test conf:0.85 src:go/ixmux/deliver/s2f_test.go:TestCloseKeepsTheGuardOfARunningCall]
func (e *Engine) Close() error {
	deadline := time.Now().Add(e.timeout)
	for {
		e.mu.Lock()
		n := len(e.calling)
		if n == 0 {
			e.closed = true // under the same lock as a new claim: no call starts after this check
			e.mu.Unlock()
			return e.log.close()
		}
		e.mu.Unlock()
		if time.Now().After(deadline) {
			return &Refusal{CallsInFlight, fmt.Sprintf("%d adapter calls have not returned; the engine keeps the log and its lock", n)}
		}
		time.Sleep(5 * time.Millisecond)
	}
}
