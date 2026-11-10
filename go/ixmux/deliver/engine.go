package deliver

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
	"time"
	"unicode/utf8"
)

// timeNow is the clock of transitions and of the deduplication window; a test replaces it.
var timeNow = time.Now

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
	reason, cause   string
	ack             *Ack
	created, since  time.Time
}

// Engine delivers mandates. Its lock is never held while an adapter or a reconciler runs, so a
// target that never answers holds only its own delivery, its target and one slot.
type Engine struct {
	log     *cmdLog
	notify  func(Transition)
	timeout time.Duration
	slots   chan struct{} // one per adapter call not yet returned
	path    string        // of the log, for a compaction
	window  time.Duration // DedupWindow
	// small and big bound a record without and with an Ack: they size the room a mandate holds.
	small, big int64
	// nextCompact: no compaction is tried for a new mandate before it, once one failed or freed
	// too little (under log.mu).
	nextCompact time.Time

	mu      sync.Mutex
	targets map[string]*Target // one allocation per Register: a pointer names a registration
	recs    map[key]*record
	ids     map[string]*record // by mandate id alone: an id is unique in the whole log
	idClaim map[string]idHold  // ids reserved by a Deliver's claim, before accepted is written
	claims  map[key]bool       // a Deliver or Reconcile of this key is running
	running map[string]bool    // targets with an adapter call not yet returned
	calling map[key]bool       // keys whose adapter call has not returned, even past its deadline
	asking  map[key]bool       // keys whose Status call has not returned, even past its deadline
	closing bool               // Close has begun: no new Deliver or Reconcile
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
	if opt.MaxLogBytes < int64(len(logHeader)) { // refused before anything is written
		return nil, Recovery{}, &Refusal{LogTooLarge, fmt.Sprintf("MaxLogBytes %d cannot hold the %d-byte log header", opt.MaxLogBytes, len(logHeader))}
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
	// A compacted copy left by a crash during a compaction is not the log: removed under the lock.
	os.Remove(path + ".compact")
	l, recs, torn, err := openLog(path, opt.MaxLogBytes)
	if err != nil {
		lock.Close()
		return nil, Recovery{}, err
	}
	l.lock = lock
	if opt.DedupWindow <= 0 {
		opt.DedupWindow = DefaultDedupWindow
	}
	big := max(min(int64(MaxRecordBytes), opt.MaxLogBytes/8), SmallRecordBytes)
	e := &Engine{log: l, notify: opt.Notify, timeout: opt.Timeout, slots: make(chan struct{}, opt.MaxInFlight),
		path: path, window: opt.DedupWindow, small: SmallRecordBytes, big: big,
		targets: map[string]*Target{}, recs: map[key]*record{}, ids: map[string]*record{}, idClaim: map[string]idHold{}, claims: map[key]bool{}, running: map[string]bool{},
		calling: map[key]bool{}, asking: map[key]bool{}}
	for _, r := range recs {
		e.apply(r)
	}
	rec := Recovery{Records: len(recs), Torn: torn, DedupWindow: e.window}
	l.mu.Lock()
	for k := range e.recs {
		e.rehold(k)
	}
	// Past three quarters of its bound, or with no room for a new mandate, the log is compacted
	// before the restart's own transitions are written: one record per mandate, in its current
	// state, finished mandates past the deduplication window left out, and only if that makes the
	// log smaller.
	// @ai:invariant a compaction keeps every mandate's state, fingerprint, incarnation, Ack and reason, so a restart neither forgets an id nor sends a delivered mandate again [T:test conf:0.85 src:go/ixmux/deliver/logbound_test.go:TestARestartCompactsAFullLogAndForgetsNothing]
	if from := l.size; from > opt.MaxLogBytes/4*3 || from+l.reserved+2*e.small+e.big > opt.MaxLogBytes {
		forgotten, fatal, err := e.compactLocked()
		switch {
		case fatal:
			l.mu.Unlock()
			l.close()
			return nil, rec, err
		case errors.Is(err, errNoGain):
		case err != nil:
			rec.CompactFailed = err.Error()
		default:
			rec.Compacted = &Compaction{From: from, To: l.size, Forgotten: forgotten}
		}
	}
	l.mu.Unlock()
	for _, r := range e.sorted() {
		k := key{r.target, r.id}
		var lr logRecord
		switch r.state {
		case Delivering:
			lr = logRecord{Target: r.target, MandateID: r.id, Incarnation: r.inc, State: Uncertain, Reason: "restart: delivering without a final state", Cause: CauseRestart}
			rec.MadeUncertain++
		case Accepted:
			lr = logRecord{Target: r.target, MandateID: r.id, Incarnation: r.inc, State: Refused, Reason: "restart: the adapter was never called"}
			rec.MadeRefused++
		default:
			continue
		}
		err = e.transition(k, lr.Incarnation, lr.State, lr.Reason, lr.Cause, nil)
		if errors.Is(err, errLogFull) {
			// The log is full: the transition is applied in memory only. The log already makes it:
			// every Open finds this delivery delivering (or accepted) and does the same again.
			// @ai:invariant a restart whose own transitions do not fit in the log still starts, with them in memory [T:test conf:0.85 src:go/ixmux/deliver/logbound_review_test.go:TestARestartWhoseTransitionsDoNotFitStillStarts]
			lr.At = timeNow().UTC()
			e.mu.Lock()
			e.apply(lr)
			e.mu.Unlock()
			l.mu.Lock()
			e.rehold(k)
			l.mu.Unlock()
			rec.InMemoryOnly++
			err = nil
		}
		if err != nil {
			l.close()
			return nil, rec, err
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
	r.prev, r.state, r.reason, r.cause, r.since, r.ack = r.state, lr.State, lr.Reason, lr.Cause, lr.At, lr.Ack
	if lr.Prev != "" { // a compacted record
		r.prev = lr.Prev
	}
	if lr.Created != nil {
		r.created = *lr.Created
	}
	if lr.Incarnation != "" {
		r.inc = lr.Incarnation
	}
	if r.fp == "" {
		r.fp = lr.Fingerprint
	}
}

// snapshot is every mandate as one log record in its current state, for a compaction, but for
// the finished mandates (delivered or refused, not claimed) whose last transition is older than
// the deduplication window: those it returns as gone. Under e.log.mu.
// @ai:invariant a compaction forgets only finished mandates past the window, never an uncertain, accepted or delivering one [T:test conf:0.85 src:go/ixmux/deliver/r2_test.go:TestOldFinishedMandatesAreForgottenAtRestart]
func (e *Engine) snapshot() (out []logRecord, gone []key) {
	e.mu.Lock()
	defer e.mu.Unlock()
	cutoff := timeNow().Add(-e.window)
	out = make([]logRecord, 0, len(e.recs))
	for _, r := range e.sorted() {
		k := key{r.target, r.id}
		if (r.state == Delivered || r.state == Refused) && r.since.Before(cutoff) && !e.claims[k] && e.log.held[k] == 0 {
			gone = append(gone, k)
			continue
		}
		created := r.created
		out = append(out, logRecord{At: r.since, Target: r.target, MandateID: r.id, Incarnation: r.inc, State: r.state,
			Reason: r.reason, Cause: r.cause, Ack: r.ack, Fingerprint: r.fp, Prev: r.prev, Created: &created})
	}
	return out, gone
}

// compactLocked rewrites the log from a snapshot, under e.log.mu, then forgets in memory the
// mandates the snapshot left out: the same id is a new mandate from then on.
func (e *Engine) compactLocked() (forgotten int, fatal bool, err error) {
	recs, gone := e.snapshot()
	if fatal, err := e.log.rewriteLocked(e.path, recs); err != nil {
		return 0, fatal, err
	}
	e.mu.Lock()
	for _, k := range gone {
		r := e.recs[k]
		delete(e.recs, k)
		if e.ids[k.id] == r {
			delete(e.ids, k.id)
		}
	}
	e.mu.Unlock()
	return len(gone), false, nil
}

// need is the room k must hold, once in state s, for the records it may still write (under
// e.mu). Accepted under its claim, it may write delivering, an uncertain final record and a late
// answer with an Ack; delivering, a final record with an Ack, or an uncertain one and its late
// answer; uncertain with its call still running, a late answer; uncertain otherwise, the record of
// a reconcile (no Ack). Found accepted or delivering at an Open, it writes its restart transition,
// then a reconcile's once uncertain. callDone: the call has answered, no late answer comes.
func (e *Engine) need(k key, s State, callDone bool) int64 {
	calling := e.calling[k] && !callDone
	switch s {
	case Accepted:
		if e.claims[k] {
			return 2*e.small + e.big
		}
		return e.small
	case Delivering:
		if calling {
			return e.small + e.big
		}
		return 2 * e.small
	case Uncertain:
		if calling {
			return e.big
		}
		return e.small
	}
	return 0
}

// rehold sets the room k holds from its state in memory (under e.log.mu).
func (e *Engine) rehold(k key) {
	e.mu.Lock()
	n := int64(0)
	if r := e.recs[k]; r != nil {
		n = e.need(k, r.state, false)
	}
	e.mu.Unlock()
	e.log.setHeld(k, n)
}

// fit marshals lr as one line within limit bytes, cutting its Ack's text (marked Truncated), then
// its reason, as needed. A record with nothing left to cut is returned whole: the append decides.
func fit(lr *logRecord, limit int64) ([]byte, error) {
	for {
		b, err := json.Marshal(lr)
		if err != nil {
			return nil, err
		}
		b = append(b, '\n')
		over := int64(len(b)) - limit
		switch {
		case over <= 0:
			return b, nil
		case lr.Ack != nil && lr.Ack.Text != "":
			a := *lr.Ack
			a.Text, a.Truncated = cut(a.Text, over), true
			lr.Ack = &a
		case lr.Reason != "":
			lr.Reason = cut(lr.Reason, over)
		default:
			return b, nil
		}
	}
}

// cut drops at least over bytes from the end of s, on a rune boundary.
func cut(s string, over int64) string {
	n := max(len(s)-int(over), 0)
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
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
func (e *Engine) transition(k key, inc string, to State, reason, cause string, ack *Ack) error {
	return e.write(logRecord{Target: k.target, MandateID: k.id, Incarnation: inc, State: to, Reason: reason, Cause: cause, Ack: ack}, false)
}

func (e *Engine) transitionFP(k key, inc string, to State, reason string, ack *Ack, fp string) error {
	return e.write(logRecord{Target: k.target, MandateID: k.id, Incarnation: inc, State: to, Reason: reason, Ack: ack, Fingerprint: fp}, false)
}

// write appends lr and applies it, both under e.log.mu: a compaction's snapshot always holds every
// record written. A new mandate the log has no room for first tries a compaction (at most once a
// minute once one did not help), which forgets the finished mandates past the window.
func (e *Engine) write(lr logRecord, callDone bool) error {
	lr.At = timeNow().UTC()
	limit := e.small
	if lr.Ack != nil {
		limit = e.big
	}
	b, err := fit(&lr, limit)
	if err != nil {
		return err
	}
	k, accept := key{lr.Target, lr.MandateID}, lr.State == Accepted
	l := e.log
	l.mu.Lock()
	e.mu.Lock()
	next := e.need(k, lr.State, callDone)
	e.mu.Unlock()
	err = l.appendLocked(b, k, next, accept)
	if rf := (*Refusal)(nil); accept && errors.As(err, &rf) && rf.Code == LogFull && !timeNow().Before(e.nextCompact) {
		if _, _, cerr := e.compactLocked(); cerr == nil {
			err = l.appendLocked(b, k, next, accept)
		}
		if err != nil {
			e.nextCompact = timeNow().Add(time.Minute)
		}
	}
	if err == nil {
		e.mu.Lock()
		e.apply(lr)
		e.mu.Unlock()
	}
	l.mu.Unlock()
	if err != nil {
		return err
	}
	if e.notify != nil {
		e.notify(Transition{At: lr.At, Target: lr.Target, MandateID: lr.MandateID, State: lr.State, Reason: lr.Reason})
	}
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
	return Result{Target: r.target, MandateID: r.id, State: r.state, Reason: r.reason, Cause: r.cause, Ack: r.ack}
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
	// Every record of the mandate must fit in a small record, with room left for a reason.
	if probe, _ := json.Marshal(logRecord{At: time.Now().UTC(), Target: m.Target, MandateID: m.MandateID, Incarnation: m.Incarnation,
		State: Uncertain, Cause: CauseLogWrite, Fingerprint: fp}); int64(len(probe)) > e.small/2 {
		return Result{}, &Refusal{BadMandate, fmt.Sprintf("its mandate id, target and incarnation take %d bytes of a log record, bound %d", len(probe), e.small/2)}
	}
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
		// No live incarnation (a child ended, or lost): the mandate's is certainly not the
		// current one, and nothing was written. A typed refusal, not an internal error.
		// @ai:invariant a target without a live incarnation refuses a mandate stale-incarnation [T:test conf:0.85 src:go/ixmux/claudestream/s6d_test.go:TestLostTargetIsSaidAndRefusedTyped]
		return Result{}, &Refusal{StaleIncarnation, fmt.Sprintf("%s has no live incarnation: %v", m.Target, err)}
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
	if e.closing {
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
		e.log.mu.Lock()
		e.mu.Lock()
		delete(e.running, m.Target)
		delete(e.calling, k)
		e.mu.Unlock()
		e.rehold(k) // no late answer can come now
		e.log.mu.Unlock()
		<-e.slots
	}
	for _, s := range []State{Accepted, Delivering} {
		if err := e.transitionFP(k, inc, s, "", nil, fp); err != nil {
			e.unclaim(k)
			release()
			if errors.Is(err, errLogFull) { // the adapter was not called: nothing was sent
				err = &Refusal{LogFull, err.Error()}
			}
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
			// Reconcile is refused and Close waits (calling) until it is written.
			<-cl.finished
			e.observeLate(k, inc, cl.res)
			release()
		}
		close(cl.answered)
	}()
	select {
	case <-cl.answered:
	case <-cctx.Done():
		e.mu.Lock()
		if !cl.returned {
			cl.abandoned = true
			e.mu.Unlock()
			res, err := e.finish(k, inc, Uncertain, "no answer within the delivery deadline: "+cctx.Err().Error(), CauseDeadline, nil)
			close(cl.finished)
			return res, err
		}
		e.mu.Unlock()
		<-cl.answered
	}
	to, reason, cause, ack := outcome(cl.res)
	if testHookBeforeFinish != nil {
		testHookBeforeFinish()
	}
	// @ai:invariant a call leaves calling only once its final state is written, so Close cannot release the log between an adapter's success and its record [T:test conf:0.85 src:go/ixmux/deliver/s3b_test.go:TestCloseWaitsForTheFinalWrite]
	res, err := e.finish(k, inc, to, reason, cause, ack)
	release()
	return res, err
}

// Test hooks, set by a test only. testHookAfterIncarnation runs between the incarnation check and
// the claim; testHookBeforeFinish between the adapter's answer and the write of the final state;
// testHookCloseWaits once, when Close finds work it must wait for.
var testHookAfterIncarnation, testHookBeforeFinish, testHookCloseWaits func()

// testHookMandateRead runs in Mandate between its read of the entry and its read of the Ack (a
// test starts a transition there).
var testHookMandateRead func()

// outcome maps an adapter's answer to a state: an error wrapping ErrRefused is certain, any
// other error leaves the outcome unknown.
func outcome(c callResult) (to State, reason, cause string, ack *Ack) {
	switch {
	case c.err == nil:
		return Delivered, "", "", &c.ack
	case errors.Is(c.err, ErrRefused):
		return Refused, c.err.Error(), "", nil
	}
	return Uncertain, "adapter error, outcome unknown: " + c.err.Error(), CauseAdapterError, nil
}

// observeLate records the answer of an adapter call that returned after its caller's deadline.
// A certain answer settles the uncertain delivery; an unknown one is logged and leaves it so.
func (e *Engine) observeLate(k key, inc string, c callResult) {
	to, reason, cause, ack := outcome(c)
	if reason == "" {
		reason = "late answer of the adapter after the deadline"
	} else {
		reason = "late answer of the adapter after the deadline: " + reason
	}
	e.mu.Lock()
	r := e.recs[k]
	still := r != nil && r.state == Uncertain && !e.claims[k]
	e.mu.Unlock()
	if still { // a failed write leaves it uncertain, as a restart would
		e.write(logRecord{Target: k.target, MandateID: k.id, Incarnation: inc, State: to, Reason: reason, Cause: cause, Ack: ack}, true)
	}
}

// finish records the final state of a delivery or reconcile and releases its claim. If the log
// cannot be written, the delivery is uncertain in memory, as it will be after a restart. Its error
// is then never a refusal: the adapter (or Status) was called, and a refusal says nothing was sent.
// @ai:invariant a delivery whose adapter was called never answers a refusal, even when another delivery broke the log [T:test conf:0.85 src:go/ixmux/deliver/r2_test.go:TestALogBrokenUnderAnotherDeliveryIsNoRefusalAfterTheCall]
func (e *Engine) finish(k key, inc string, to State, reason, cause string, ack *Ack) (Result, error) {
	err := e.transition(k, inc, to, reason, cause, ack)
	if rf := (*Refusal)(nil); errors.As(err, &rf) {
		err = fmt.Errorf("the final state was not written: %s", rf.Error())
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	r := e.recs[k]
	if err != nil {
		r.prev, r.state, r.reason, r.cause, r.since, r.ack = r.state, Uncertain, "log write failed: "+err.Error(), CauseLogWrite, timeNow().UTC(), nil
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
	case e.closing:
		refusal = &Refusal{EngineClosed, "the engine is closed"}
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
				return e.finish(k, inc, Refused, "reconciled: the target did not receive it", "", nil)
			}
			return e.finish(k, inc, Delivered, "reconciled: the target received it", "", nil)
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
		out = append(out, e.entry(r, now))
	}
	return out
}

// Mandate is the delivery of a mandate id (unique in the log) as in Status, with its Ack once
// delivered: the Ack is in the command log, so it is read back after a restart too.
// @ai:invariant the Ack of a delivered mandate is read back by its id after the engine reopens its log [T:test conf:0.85 src:go/ixmux/server/s6_test.go:TestMandateResultIsReadBackAfterARestart]
// The entry and its Ack are read under one lock: a transition is seen whole or not at all.
// @ai:invariant Mandate reads the state and the Ack of a delivery in one snapshot [T:test conf:0.85 src:go/ixmux/deliver/s6b_test.go:TestMandateReadsOneSnapshot]
func (e *Engine) Mandate(id string) (Entry, *Ack, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	r := e.ids[id] // an id is unique in the whole log: its first record is its only one
	if r == nil {
		return Entry{}, nil, false
	}
	en := e.entry(r, time.Now())
	if testHookMandateRead != nil {
		testHookMandateRead()
	}
	var ack *Ack
	if r.ack != nil {
		a := *r.ack
		ack = &a
	}
	return en, ack, true
}

// entry is r as Status lists it (under e.mu).
func (e *Engine) entry(r *record, now time.Time) Entry {
	prev := string(r.prev)
	if prev == "" {
		prev = "none"
	}
	return Entry{Target: r.target, MandateID: r.id, State: r.state, Reason: r.reason, Cause: r.cause,
		Age: now.Sub(r.created), Since: r.since, Last: prev + "->" + string(r.state), StillRunning: e.calling[key{r.target, r.id}]}
}

// Close refuses every new Deliver and Reconcile (engine-closed), then waits, up to the delivery
// deadline, for the deliveries and reconciles under way and the adapter and Status calls that
// have not returned, each until its final state is written (a late answer is still recorded).
// It then closes the log and releases its lock. If work is still under way, it refuses
// calls-in-flight and keeps both: a call can still take effect, and another engine opened on the
// log would not know it (it could reconcile not-received and send again). Close may be called
// again later. It returns log-broken, after closing, if a transition could not be written.
// @ai:invariant the log's lock is released only once no adapter call of this engine can still take effect [T:test conf:0.85 src:go/ixmux/deliver/s2f_test.go:TestCloseKeepsTheGuardOfARunningCall]
// @ai:invariant Close returns nil only once no transition of this engine is still being written [T:test conf:0.85 src:go/ixmux/deliver/s3b_test.go:TestCloseWaitsForAReconcile]
// A transition the full log could not hold (errLogFull) is not written and does not break the log:
// Close still returns nil; its delivery is uncertain in memory and is found uncertain at the next Open.
// @ai:invariant a final record refused for lack of room leaves Close nil and its delivery uncertain after a restart [T:test conf:0.85 src:go/ixmux/deliver/logbound_review_test.go:TestAFinalRecordThatDoesNotFitIsNoRefusal]
func (e *Engine) Close() error {
	deadline := time.Now().Add(e.timeout)
	e.mu.Lock()
	e.closing = true // under the same lock as a new claim: nothing is claimed after this
	e.mu.Unlock()
	for waited := false; ; waited = true {
		e.mu.Lock()
		n := len(e.claims) + len(e.calling) + len(e.asking)
		e.mu.Unlock()
		if n == 0 {
			break
		}
		if !waited && testHookCloseWaits != nil {
			testHookCloseWaits()
		}
		if time.Now().After(deadline) {
			return &Refusal{CallsInFlight, fmt.Sprintf("%d deliveries, reconciles or calls under way; the engine keeps the log and its lock", n)}
		}
		time.Sleep(5 * time.Millisecond)
	}
	e.log.mu.Lock()
	broken := e.log.broken
	e.log.mu.Unlock()
	if err := e.log.close(); err != nil {
		return err
	}
	if broken != nil {
		return &Refusal{LogBroken, "a transition could not be written (" + broken.Error() + ")"}
	}
	return nil
}
