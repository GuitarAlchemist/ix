package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// A need is one question a waiting task asks (issue #486): information, an authorization or
// capacity, with an absolute deadline that never moves. One resolver claims it under a lease
// and a fencing token; only that claim may answer, before the deadline, and only the asker
// (requester and incarnation) reads the answer. A need grants nothing: an `authorization`
// answer is a record for the asker's own authority checks (#420), never a permission, and the
// asker may not resolve its own. Waiting is not the need: a wait that times out or is cancelled
// leaves it as it was, and an expiry says nothing about the task or about effects.
//
// The needs live in `<state>/<name>.needs.json`, apart from the hub's ring: every transition
// rewrites the file whole (a temporary, fsync, rename) and is adopted only once written. A write
// whose outcome is uncertain, or a file that cannot be read, makes every need op answer
// "unknown" until the file reads back; nothing is replayed from memory.
const (
	maxNeeds       = 256
	maxNeedBody    = 4096 // bytes of the body, and of an answer
	maxNeedReason  = 256  // characters of a refusal
	maxNeedHorizon = 7 * 24 * time.Hour
	maxNeedsFile   = maxNeeds * (3*maxNeedBody + 2048) // a full store's bound, checked before decoding
	maxNeedLease   = time.Hour
	needPoll       = 20 * time.Millisecond // a wait re-reads the clock this often, to see an expiry

	needsSchema = "ixctl-needs/0"
)

var (
	needKinds = map[string]bool{"information": true, "authorization": true, "capacity": true}
	partyRE   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$`)
)

type needSubmit struct {
	ID          string          `json:"id"`
	Kind        string          `json:"kind"`
	Requester   string          `json:"requester"`
	Incarnation string          `json:"incarnation"`
	Body        json.RawMessage `json:"body"`
	DeadlineMS  int64           `json:"deadline_ms"`
}

type needClaim struct {
	Resolver     string `json:"resolver"`
	Fence        uint64 `json:"fence"`
	LeaseUntilMS int64  `json:"lease_until_ms"`
}

type needResolution struct {
	Resolver string `json:"resolver"`
	Fence    uint64 `json:"fence"`
	AtMS     int64  `json:"at_ms"`
}

// need is one record of the file.
type need struct {
	ID          string          `json:"id"`
	Kind        string          `json:"kind"`
	Requester   string          `json:"requester"`
	Incarnation string          `json:"incarnation"`
	Body        json.RawMessage `json:"body"`
	Digest      string          `json:"digest"`
	CreatedMS   int64           `json:"created_ms"`
	DeadlineMS  int64           `json:"deadline_ms"`
	Rev         uint64          `json:"rev"`
	State       string          `json:"state"` // open, claimed, resolved, refused, expired
	Claim       *needClaim      `json:"claim,omitempty"`
	Resolution  *needResolution `json:"resolution,omitempty"`
	Answer      json.RawMessage `json:"answer,omitempty"`
	Reason      string          `json:"reason,omitempty"`
}

func (n *need) terminal() bool {
	return n.State == "resolved" || n.State == "refused" || n.State == "expired"
}

type needFile struct {
	Schema string  `json:"schema"`
	Fence  uint64  `json:"fence"` // the last fencing token handed out, so a restart never reuses one
	Needs  []*need `json:"needs"`
}

// needReply is what every op answers: the need's state at its revision. The body (the question)
// goes to the asker and to the resolver holding the claim; the answer only to the asker.
type needReply struct {
	ID         string          `json:"id"`
	Rev        uint64          `json:"rev"`
	State      string          `json:"state"`
	Outcome    string          `json:"outcome,omitempty"` // wait: resolved, refused, expired or wait-timeout
	Duplicate  bool            `json:"duplicate,omitempty"`
	Kind       string          `json:"kind"`
	Digest     string          `json:"digest"`
	CreatedMS  int64           `json:"created_ms"`
	DeadlineMS int64           `json:"deadline_ms"`
	Claim      *needClaim      `json:"claim,omitempty"`
	Resolution *needResolution `json:"resolution,omitempty"`
	Body       json.RawMessage `json:"body,omitempty"`
	Answer     json.RawMessage `json:"answer,omitempty"`
	Reason     string          `json:"reason,omitempty"`
}

type needStore struct {
	path string
	now  func() time.Time

	mu      sync.Mutex
	needs   []*need // in submit order; a transition replaces the record, never edits it in place
	fence   uint64
	unknown string        // non-empty: the file's state is not known; ops reconcile first
	wake    chan struct{} // closed and replaced on every adopted transition

	// Tests: interrupt a write before the rename, at it, or after it (committed, result lost).
	failTemp, failRename, failAfterRename error
}

// newNeedStore opens the store at path; it never fails, an unreadable file makes it unknown.
func newNeedStore(path string, now func() time.Time) *needStore {
	s := &needStore{path: path, now: now, wake: make(chan struct{})}
	s.load()
	return s
}

func (s *needStore) load() {
	f, err := os.Open(s.path)
	if os.IsNotExist(err) {
		// No file and no temporary: an empty store. A temporary alone may be a write whose rename
		// removed the old file first (os.Rename promises no atomicity off Unix): unknown, not empty.
		if _, terr := os.Stat(s.path + ".tmp"); terr == nil {
			s.unknown = s.path + " is missing while " + s.path + ".tmp remains"
			return
		}
		s.needs, s.fence, s.unknown = nil, 0, ""
		return
	}
	if err != nil {
		s.unknown = err.Error()
		return
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, maxNeedsFile+1))
	if err != nil {
		s.unknown = err.Error()
		return
	}
	if len(b) > maxNeedsFile {
		s.unknown = fmt.Sprintf("%s is over %d bytes", s.path, maxNeedsFile)
		return
	}
	var nf needFile
	if err := decodeStrict(bytes.NewReader(b), &nf); err != nil {
		s.unknown = fmt.Sprintf("%s: %v", s.path, err)
		return
	}
	if err := checkNeeds(nf); err != nil {
		s.unknown = fmt.Sprintf("%s: %v", s.path, err)
		return
	}
	s.needs, s.fence, s.unknown = nf.Needs, nf.Fence, ""
}

// checkNeeds holds a loaded file to what the ops write: every record well formed, its digest
// its own, and its state, claim, resolution, answer and reason consistent. A file that breaks
// any of it is unknown, never used: an op must not meet a claimed need with no claim.
func checkNeeds(f needFile) error {
	if f.Schema != needsSchema {
		return fmt.Errorf("schema %q, want %q", f.Schema, needsSchema)
	}
	if len(f.Needs) > maxNeeds {
		return fmt.Errorf("%d needs, over %d", len(f.Needs), maxNeeds)
	}
	ids, fences := map[string]bool{}, map[uint64]bool{}
	for i, n := range f.Needs {
		if n == nil || !nameRE.MatchString(n.ID) || ids[n.ID] {
			return fmt.Errorf("needs[%d]: missing, badly named or repeated id", i)
		}
		ids[n.ID] = true
		if err := checkNeed(n, f.Fence, fences); err != nil {
			return fmt.Errorf("needs[%d] (%s): %v", i, n.ID, err)
		}
	}
	return nil
}

func checkNeed(n *need, fileFence uint64, fences map[uint64]bool) error {
	if !needKinds[n.Kind] || !partyRE.MatchString(n.Requester) || !partyRE.MatchString(n.Incarnation) {
		return fmt.Errorf("bad kind, requester or incarnation")
	}
	body, err := checkJSON("body", n.Body)
	if err != nil || n.Digest != needDigest(n.Kind, n.Requester, n.Incarnation, n.DeadlineMS, body) {
		return fmt.Errorf("body or digest does not match")
	}
	if n.Rev < 1 || n.CreatedMS <= 0 || n.DeadlineMS <= n.CreatedMS || n.DeadlineMS > n.CreatedMS+maxNeedHorizon.Milliseconds() {
		return fmt.Errorf("rev, created_ms or deadline_ms out of bounds")
	}
	if c := n.Claim; c != nil {
		if !partyRE.MatchString(c.Resolver) || c.Fence == 0 || c.Fence > fileFence || fences[c.Fence] || c.LeaseUntilMS <= n.CreatedMS {
			return fmt.Errorf("bad claim")
		}
		fences[c.Fence] = true
	}
	answered := n.Resolution != nil
	if answered && (n.Claim == nil || n.Resolution.Resolver != n.Claim.Resolver || n.Resolution.Fence != n.Claim.Fence) {
		return fmt.Errorf("resolution not bound to the claim")
	}
	// An answer is only ever written after the need's creation, before its deadline and inside
	// its lease: a resolution dated otherwise was not written by an op.
	if r := n.Resolution; answered && (r.AtMS < n.CreatedMS || r.AtMS >= n.DeadlineMS || r.AtMS >= n.Claim.LeaseUntilMS) {
		return fmt.Errorf("resolution at_ms outside the need's life or its lease")
	}
	var ok bool
	switch n.State {
	case "open":
		ok = n.Claim == nil && !answered && n.Answer == nil && n.Reason == ""
	case "claimed":
		ok = n.Claim != nil && !answered && n.Answer == nil && n.Reason == ""
	case "resolved":
		_, aerr := checkJSON("answer", n.Answer)
		ok = answered && aerr == nil && n.Reason == ""
	case "refused":
		ok = answered && n.Answer == nil && n.Reason != "" && len([]rune(n.Reason)) <= maxNeedReason
	case "expired":
		ok = !answered && n.Answer == nil && n.Reason == ""
	}
	if !ok {
		return fmt.Errorf("state %q inconsistent with its claim, resolution, answer or reason", n.State)
	}
	return nil
}

func needDigest(kind, requester, incarnation string, deadlineMS int64, body []byte) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s\n%s\n%s\n%d\n%s", kind, requester, incarnation, deadlineMS, body)))
	return hex.EncodeToString(sum[:])
}

// ready reconciles an unknown store from disk; it answers an error while the file stays unreadable.
func (s *needStore) ready() error {
	if s.unknown != "" {
		s.load()
	}
	if s.unknown != "" {
		return &opError{status: 503, msg: "need state unknown (" + s.unknown + "): the file must be reconciled by hand"}
	}
	return nil
}

func (s *needStore) find(id string) (int, *need) {
	for i, n := range s.needs {
		if n.ID == id {
			return i, n
		}
	}
	return -1, nil
}

// commit writes the store with next in place of record i (i == -1 appends) and adopts it only
// once the rename succeeded.
func (s *needStore) commit(i int, next *need, fence uint64) error {
	list := append([]*need(nil), s.needs...)
	if i < 0 {
		list = append(list, next)
	} else {
		list[i] = next
	}
	b, err := json.Marshal(needFile{Schema: needsSchema, Fence: fence, Needs: list})
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := s.writeTemp(tmp, append(b, '\n')); err != nil {
		return &opError{status: 500, msg: "need not persisted, nothing changed: " + err.Error()}
	}
	err = s.failRename
	if err == nil {
		err = os.Rename(tmp, s.path)
	}
	if err == nil {
		err = s.failAfterRename // committed on disk, the result lost before memory and reply
	}
	if err != nil {
		s.unknown = "write interrupted at or after rename: " + err.Error()
		return &opError{status: 503, msg: "need state unknown after an interrupted write; the next op reconciles from disk"}
	}
	s.needs, s.fence = list, fence
	close(s.wake)
	s.wake = make(chan struct{})
	return nil
}

func (s *needStore) writeTemp(tmp string, b []byte) error {
	if s.failTemp != nil {
		return s.failTemp
	}
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(b); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// expire moves a live need past its deadline to expired, persisted, before any op reads it.
func (s *needStore) expire(i int, n *need) (*need, error) {
	if n.terminal() || s.now().UnixMilli() < n.DeadlineMS {
		return n, nil
	}
	next := *n
	next.State, next.Rev = "expired", n.Rev+1
	if err := s.commit(i, &next, s.fence); err != nil {
		return nil, err
	}
	return &next, nil
}

// get finds a need, reconciled and expired as due.
func (s *needStore) get(id string) (int, *need, error) {
	if err := s.ready(); err != nil {
		return 0, nil, err
	}
	i, n := s.find(id)
	if n == nil {
		return 0, nil, &opError{status: 404, msg: fmt.Sprintf("need %q not found", id)}
	}
	n, err := s.expire(i, n)
	return i, n, err
}

// reply shows the need; the question (body) to its asker or its claim holder, the answer only
// to its asker. Both are declared names (see the README): correlation, not authentication.
func reply(n *need, asker, holder bool) needReply {
	r := needReply{ID: n.ID, Rev: n.Rev, State: n.State, Kind: n.Kind, Digest: n.Digest, CreatedMS: n.CreatedMS,
		DeadlineMS: n.DeadlineMS, Claim: n.Claim, Resolution: n.Resolution, Reason: n.Reason}
	if asker || holder {
		r.Body = n.Body
	}
	if asker {
		r.Answer = n.Answer
	}
	return r
}

func checkJSON(field string, raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 || len(raw) > maxNeedBody {
		return nil, badRequest("%s: 1 to %d bytes of JSON", field, maxNeedBody)
	}
	var b bytes.Buffer
	if err := json.Compact(&b, raw); err != nil {
		return nil, badRequest("%s: not JSON", field)
	}
	return b.Bytes(), nil
}

func (s *needStore) submit(r needSubmit) (needReply, error) {
	switch {
	case !nameRE.MatchString(r.ID):
		return needReply{}, badRequest("id must match %s", nameRE)
	case !needKinds[r.Kind]:
		return needReply{}, badRequest("kind must be information, authorization or capacity")
	case !partyRE.MatchString(r.Requester):
		return needReply{}, badRequest("requester must match %s", partyRE)
	case !partyRE.MatchString(r.Incarnation):
		return needReply{}, badRequest("incarnation must match %s", partyRE)
	}
	body, err := checkJSON("body", r.Body)
	if err != nil {
		return needReply{}, err
	}
	digest := needDigest(r.Kind, r.Requester, r.Incarnation, r.DeadlineMS, body)

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return needReply{}, err
	}
	// A resubmit finds its need first, whatever the clock says: its receipt, expired or not.
	if i, n := s.find(r.ID); n != nil {
		if n.Digest != digest {
			return needReply{}, &opError{status: 409, msg: fmt.Sprintf("need %q exists with another digest", r.ID)}
		}
		n, err := s.expire(i, n)
		if err != nil {
			return needReply{}, err
		}
		rep := reply(n, true, false)
		rep.Duplicate = true
		return rep, nil
	}
	now := s.now()
	if r.DeadlineMS <= now.UnixMilli() || r.DeadlineMS > now.Add(maxNeedHorizon).UnixMilli() {
		return needReply{}, badRequest("deadline_ms must be absolute, after now and within %s", maxNeedHorizon)
	}
	if len(s.needs) >= maxNeeds {
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("the store holds %d needs, its maximum", maxNeeds)}
	}
	n := &need{ID: r.ID, Kind: r.Kind, Requester: r.Requester, Incarnation: r.Incarnation, Body: body, Digest: digest,
		CreatedMS: now.UnixMilli(), DeadlineMS: r.DeadlineMS, Rev: 1, State: "open"}
	if err := s.commit(-1, n, s.fence); err != nil {
		return needReply{}, err
	}
	return reply(n, true, false), nil
}

// status shows the need; resolver, when it names the claim holder, gets the question too.
func (s *needStore) status(id, requester, incarnation, resolver string) (needReply, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, n, err := s.get(id)
	if err != nil {
		return needReply{}, err
	}
	holder := resolver != "" && n.Claim != nil && n.Claim.Resolver == resolver && !n.terminal()
	return reply(n, n.Requester == requester && n.Incarnation == incarnation, holder), nil
}

func (s *needStore) claim(id, resolver string, leaseMS int64) (needReply, error) {
	if !partyRE.MatchString(resolver) {
		return needReply{}, badRequest("resolver must match %s", partyRE)
	}
	if leaseMS <= 0 || leaseMS > maxNeedLease.Milliseconds() {
		return needReply{}, badRequest("lease_ms must be 1 to %d", maxNeedLease.Milliseconds())
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	i, n, err := s.get(id)
	if err != nil {
		return needReply{}, err
	}
	now := s.now().UnixMilli()
	switch {
	case n.terminal():
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("need %q is %s", id, n.State)}
	case n.Kind == "authorization" && resolver == n.Requester:
		return needReply{}, &opError{status: 403, msg: "the requester cannot resolve its own authorization need"}
	case n.State == "claimed" && now < n.Claim.LeaseUntilMS:
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("need %q is claimed until %d", id, n.Claim.LeaseUntilMS)}
	}
	fence := s.fence + 1
	next := *n
	next.State, next.Rev = "claimed", n.Rev+1
	next.Claim = &needClaim{Resolver: resolver, Fence: fence, LeaseUntilMS: now + leaseMS}
	if err := s.commit(i, &next, fence); err != nil {
		return needReply{}, err
	}
	return reply(&next, false, true), nil
}

// answerable checks that resolver holds the live claim fenced by fence.
func (s *needStore) answerable(id, resolver string, fence uint64) (int, *need, error) {
	i, n, err := s.get(id)
	if err != nil {
		return 0, nil, err
	}
	switch {
	case n.State == "expired":
		return 0, nil, &opError{status: 409, msg: fmt.Sprintf("need %q expired at %d: a late answer is refused", id, n.DeadlineMS)}
	case n.terminal():
		return 0, nil, &opError{status: 409, msg: fmt.Sprintf("need %q is %s", id, n.State)}
	case n.State != "claimed":
		return 0, nil, &opError{status: 409, msg: fmt.Sprintf("need %q is not claimed", id)}
	case fence != n.Claim.Fence:
		return 0, nil, &opError{status: 409, msg: fmt.Sprintf("stale fence %d for need %q", fence, id)}
	case resolver != n.Claim.Resolver:
		return 0, nil, &opError{status: 409, msg: fmt.Sprintf("resolver %q does not hold the claim on %q", resolver, id)}
	case s.now().UnixMilli() >= n.Claim.LeaseUntilMS:
		return 0, nil, &opError{status: 409, msg: fmt.Sprintf("the lease on %q was lost at %d", id, n.Claim.LeaseUntilMS)}
	}
	return i, n, nil
}

func (s *needStore) resolve(id, resolver string, fence uint64, answer json.RawMessage) (needReply, error) {
	answer, err := checkJSON("answer", answer)
	if err != nil {
		return needReply{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	i, n, err := s.answerable(id, resolver, fence)
	if err != nil {
		return needReply{}, err
	}
	next := *n
	next.State, next.Rev, next.Answer = "resolved", n.Rev+1, answer
	next.Resolution = &needResolution{Resolver: resolver, Fence: fence, AtMS: s.now().UnixMilli()}
	if err := s.commit(i, &next, s.fence); err != nil {
		return needReply{}, err
	}
	return reply(&next, false, false), nil
}

func (s *needStore) refuse(id, resolver string, fence uint64, reason string) (needReply, error) {
	if reason == "" || len([]rune(reason)) > maxNeedReason || strings.ContainsAny(reason, "\r\n") {
		return needReply{}, badRequest("reason must be one line of 1 to %d characters", maxNeedReason)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	i, n, err := s.answerable(id, resolver, fence)
	if err != nil {
		return needReply{}, err
	}
	next := *n
	next.State, next.Rev, next.Reason = "refused", n.Rev+1, reason
	next.Resolution = &needResolution{Resolver: resolver, Fence: fence, AtMS: s.now().UnixMilli()}
	if err := s.commit(i, &next, s.fence); err != nil {
		return needReply{}, err
	}
	return reply(&next, false, false), nil
}

// wait returns once the need is terminal (outcome = its state) or after timeout (outcome
// wait-timeout, the need unchanged). A cancelled wait returns ctx's error and changes nothing.
func (s *needStore) wait(ctx context.Context, id, requester, incarnation string, timeout time.Duration) (needReply, error) {
	if timeout <= 0 || timeout > maxNeedLease {
		return needReply{}, badRequest("timeout_ms must be 1 to %d", maxNeedLease.Milliseconds())
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	poll := time.NewTicker(needPoll)
	defer poll.Stop()
	for {
		s.mu.Lock()
		_, n, err := s.get(id)
		wake := s.wake
		s.mu.Unlock()
		if err != nil {
			return needReply{}, err
		}
		r := reply(n, n.Requester == requester && n.Incarnation == incarnation, false)
		if n.terminal() {
			r.Outcome = n.State
			return r, nil
		}
		select {
		case <-ctx.Done():
			return needReply{}, ctx.Err()
		case <-timer.C:
			r.Outcome = "wait-timeout"
			return r, nil
		case <-wake:
		case <-poll.C:
		}
	}
}

// needOp is the body of every need op, on every transport: the fields its op reads.
type needOp struct {
	ID          string          `json:"id"`
	Kind        string          `json:"kind,omitempty"`
	Requester   string          `json:"requester,omitempty"`
	Incarnation string          `json:"incarnation,omitempty"`
	Body        json.RawMessage `json:"body,omitempty"`
	DeadlineMS  int64           `json:"deadline_ms,omitempty"`
	Resolver    string          `json:"resolver,omitempty"`
	LeaseMS     int64           `json:"lease_ms,omitempty"`
	Fence       uint64          `json:"fence,omitempty"`
	Answer      json.RawMessage `json:"answer,omitempty"`
	Reason      string          `json:"reason,omitempty"`
	TimeoutMS   int64           `json:"timeout_ms,omitempty"`
}

var needOps = []string{"submit", "status", "claim", "resolve", "refuse", "wait"}

// SetNeedStore keeps the plane's needs in the file at path (`<state>/<name>.needs.json`); the
// instance lock makes this plane the file's only writer. A plane without it refuses need ops.
func (p *Plane) SetNeedStore(path string) { p.needs = newNeedStore(path, time.Now) }

// needRequest serves the need.* ops. They are not forwarded to peers: a need lives in the
// store of the plane that took it.
func (p *Plane) needRequest(ctx context.Context, req Request) (any, error) {
	if p.needs == nil {
		return nil, &opError{status: 503, msg: "this plane keeps no needs: it runs without a state dir"}
	}
	o := req.Need
	if o == nil {
		return nil, badRequest("%s needs its fields under `need`", req.Op)
	}
	switch req.Op {
	case "need.submit":
		return p.needs.submit(needSubmit{ID: o.ID, Kind: o.Kind, Requester: o.Requester, Incarnation: o.Incarnation,
			Body: o.Body, DeadlineMS: o.DeadlineMS})
	case "need.status":
		return p.needs.status(o.ID, o.Requester, o.Incarnation, o.Resolver)
	case "need.claim":
		return p.needs.claim(o.ID, o.Resolver, o.LeaseMS)
	case "need.resolve":
		return p.needs.resolve(o.ID, o.Resolver, o.Fence, o.Answer)
	case "need.refuse":
		return p.needs.refuse(o.ID, o.Resolver, o.Fence, o.Reason)
	}
	// need.wait holds a waiter slot like wait does, and ends with the plane.
	timeout, err := needTimeout(o.TimeoutMS)
	if err != nil {
		return nil, err
	}
	select {
	case <-p.quit:
		return nil, &opError{status: 503, msg: "shutting down"}
	case p.waiters <- struct{}{}:
		defer func() { <-p.waiters }()
	default:
		return nil, &opError{status: 429, msg: "too many waiters"}
	}
	wctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(p.stopping, cancel)
	defer func() {
		stop()
		cancel()
	}()
	r, err := p.needs.wait(wctx, o.ID, o.Requester, o.Incarnation, timeout)
	if err != nil && p.stopping.Err() != nil {
		return nil, &opError{status: 503, msg: "shutting down"}
	}
	return r, err
}

// needTimeout bounds a wait's timeout_ms before it becomes a duration: a huge value must be
// refused, not wrapped by the multiplication into a small or negative one.
func needTimeout(ms int64) (time.Duration, error) {
	if ms < 1 || ms > maxNeedLease.Milliseconds() {
		return 0, badRequest("timeout_ms must be 1 to %d", maxNeedLease.Milliseconds())
	}
	return time.Duration(ms) * time.Millisecond, nil
}

// cmdNeed is `ixctl need <op> [--state DIR] --name NAME`: the op's JSON body on stdin, sent to
// the running plane over its WebSocket, the reply printed as one JSON line. The need lives in
// the plane, not in this process: ending it (or a wait) leaves the need as it was.
func cmdNeed(args []string, stdin io.Reader, stdout io.Writer) error {
	if len(args) == 0 || !oneOf(needOps, args[0]) {
		return fmt.Errorf("need: want one of %v", needOps)
	}
	op := args[0]
	fs := flag.NewFlagSet("need "+op, flag.ContinueOnError)
	state := fs.String("state", "", "the plane's state dir")
	name := fs.String("name", "", "the plane's instance name")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	if *name == "" || fs.NArg() != 0 {
		return fmt.Errorf("need %s: --name NAME, and the op's JSON on stdin", op)
	}
	if *state == "" {
		*state = defaultStateDir()
	}
	var o needOp
	if err := decodeStrict(io.LimitReader(stdin, 1<<20), &o); err != nil {
		return fmt.Errorf("need %s: stdin: %v", op, err)
	}
	bound := 30 * time.Second
	if op == "wait" {
		timeout, err := needTimeout(o.TimeoutMS)
		if err != nil {
			return fmt.Errorf("need wait: %v", err)
		}
		bound += timeout
	}
	st, err := readStateFile(filepath.Join(*state, *name+".json"))
	if err != nil {
		return fmt.Errorf("need %s: %v", op, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), bound)
	defer cancel()
	cl, err := DialWS(ctx, st.Addr, st.Token)
	if err != nil {
		return fmt.Errorf("need %s: %v", op, err)
	}
	defer cl.Close()
	raw, err := cl.Do(ctx, Request{Op: "need." + op, Need: &o})
	if err != nil {
		return fmt.Errorf("need %s: %v", op, err)
	}
	_, err = fmt.Fprintln(stdout, string(raw))
	return err
}

func oneOf(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}
