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
	"reflect"
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
//
// A need may carry a token budget (see reserve and settle): accounting and admission only. The
// plane runs no responder and enforces no provider's cap, so it admits deterministic runners only.
const (
	maxNeeds       = 256
	maxNeedBody    = 4096 // bytes of the body, and of an answer
	maxNeedReason  = 256  // characters of a refusal
	maxNeedHorizon = 7 * 24 * time.Hour
	maxNeedLease   = time.Hour
	needPoll       = 20 * time.Millisecond // a wait re-reads the clock this often, to see an expiry

	maxNeedTokens       = 1 << 40 // a budget's total, a reservation, and each usage count
	maxNeedReservations = 64      // per need
	// a full store's bound, checked before decoding
	maxNeedsFile = maxNeeds * (3*maxNeedBody + 2048 + maxNeedReservations*1024)

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
	Budget      *needBudget     `json:"budget,omitempty"`
}

// needBudget is a need's token total and, for a child, the need of this store it is carved
// from. A root total is what its submitter declares: the plane does not authenticate it as
// anyone's authority, and a budget grants nothing.
type needBudget struct {
	TotalTokens int64  `json:"total_tokens"`
	Parent      string `json:"parent,omitempty"`
}

// needReservation is tokens set aside before a runner starts, then settled with what it used.
// Until a known usage settles it, the whole reservation stays committed: reserved (no usage
// reported yet) and unknown (reported as unknown) both count as spent.
type needReservation struct {
	ID       string     `json:"id"`
	Digest   string     `json:"digest"` // resolver, fence, runner and tokens: a retry with the same id must match
	Resolver string     `json:"resolver"`
	Fence    uint64     `json:"fence"`
	Runner   string     `json:"runner"`
	Tokens   int64      `json:"tokens"`
	AtMS     int64      `json:"at_ms"`
	State    string     `json:"state"` // reserved, unknown, settled
	Usage    *needUsage `json:"usage,omitempty"`
}

// needUsage is one run's own consumption, never a running total, in four disjoint counts: input
// excludes the cache's writes and reads. Unknown carries no count; a known usage settling an
// unknown one must say its effects were clarified.
type needUsage struct {
	Unit             string `json:"unit"`
	Source           string `json:"source"`
	Unknown          bool   `json:"unknown,omitempty"`
	EffectsClarified bool   `json:"effects_clarified,omitempty"`
	InputTokens      *int64 `json:"input_tokens,omitempty"`
	OutputTokens     *int64 `json:"output_tokens,omitempty"`
	CacheWriteTokens *int64 `json:"cache_write_tokens,omitempty"`
	CacheReadTokens  *int64 `json:"cache_read_tokens,omitempty"`
}

// needBudgetView is a budget as the ops show it. Enforcement is "accounting": the plane admits
// or refuses reservations, it stops no runner and enforces no provider cap.
type needBudgetView struct {
	TotalTokens     int64             `json:"total_tokens"`
	Parent          string            `json:"parent,omitempty"`
	CommittedTokens int64             `json:"committed_tokens"` // reserved + unknown + settled + children
	ReservedTokens  int64             `json:"reserved_tokens"`
	UnknownTokens   int64             `json:"unknown_tokens"`
	SettledTokens   int64             `json:"settled_tokens"`
	ChildrenTokens  int64             `json:"children_tokens"`
	RemainingTokens int64             `json:"remaining_tokens"`
	Overrun         bool              `json:"overrun,omitempty"`
	Enforcement     string            `json:"enforcement"`
	Reservations    []needReservation `json:"reservations,omitempty"`
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

	Budget       *needBudget       `json:"budget,omitempty"`
	Reservations []needReservation `json:"reservations,omitempty"`
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
	Budget     *needBudgetView `json:"budget,omitempty"`
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
	budgets, fences := map[string]*needBudget{}, map[uint64]bool{} // the needs seen so far, by id
	for i, n := range f.Needs {
		if n == nil || !nameRE.MatchString(n.ID) {
			return fmt.Errorf("needs[%d]: missing or badly named", i)
		}
		if _, seen := budgets[n.ID]; seen {
			return fmt.Errorf("needs[%d]: repeated id %q", i, n.ID)
		}
		if err := checkNeed(n, f.Fence, fences); err != nil {
			return fmt.Errorf("needs[%d] (%s): %v", i, n.ID, err)
		}
		// A child comes after its parent, which has a budget at least as large.
		if b := n.Budget; b != nil && b.Parent != "" {
			if pb := budgets[b.Parent]; pb == nil || b.TotalTokens > pb.TotalTokens {
				return fmt.Errorf("needs[%d] (%s): parent %q missing, after it, without a budget or smaller", i, n.ID, b.Parent)
			}
		}
		budgets[n.ID] = n.Budget
	}
	return nil
}

func checkNeed(n *need, fileFence uint64, fences map[uint64]bool) error {
	if !needKinds[n.Kind] || !partyRE.MatchString(n.Requester) || !partyRE.MatchString(n.Incarnation) {
		return fmt.Errorf("bad kind, requester or incarnation")
	}
	body, err := checkJSON("body", n.Body)
	if err != nil || n.Digest != needDigest(n.Kind, n.Requester, n.Incarnation, n.DeadlineMS, body, n.Budget) {
		return fmt.Errorf("body or digest does not match")
	}
	if err := checkReservations(n, fileFence); err != nil {
		return err
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

// needDigest covers what a submit declares. A budget is added only when there is one, so a need
// without a budget keeps the digest it had before budgets existed.
func needDigest(kind, requester, incarnation string, deadlineMS int64, body []byte, budget *needBudget) string {
	s := fmt.Sprintf("%s\n%s\n%s\n%d\n%s", kind, requester, incarnation, deadlineMS, body)
	if budget != nil {
		s += fmt.Sprintf("\nbudget %d %s", budget.TotalTokens, budget.Parent)
	}
	sum := sha256.Sum256([]byte(s))
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
func (s *needStore) reply(n *need, asker, holder bool) needReply {
	r := needReply{ID: n.ID, Rev: n.Rev, State: n.State, Kind: n.Kind, Digest: n.Digest, CreatedMS: n.CreatedMS,
		DeadlineMS: n.DeadlineMS, Claim: n.Claim, Resolution: n.Resolution, Reason: n.Reason}
	if asker || holder {
		r.Body = n.Body
	}
	if asker {
		r.Answer = n.Answer
	}
	if n.Budget != nil {
		v := s.spend(n)
		r.Budget = &v
	}
	return r
}

// spend sums a budgeted need's commitments: each reservation (its usage once a known usage
// settles it, all of it while reserved or unknown) and each child of this store, at the larger of
// the child's total and its own commitments, so a child is debited once and its overrun shows.
// A child always comes after its parent in the store (submit order), so this ends.
func (s *needStore) spend(n *need) needBudgetView {
	v := needBudgetView{TotalTokens: n.Budget.TotalTokens, Parent: n.Budget.Parent, Enforcement: "accounting",
		Reservations: n.Reservations}
	for _, r := range n.Reservations {
		switch r.State {
		case "settled":
			v.SettledTokens += r.Usage.total()
		case "unknown":
			v.UnknownTokens += r.Tokens
		default:
			v.ReservedTokens += r.Tokens
		}
	}
	for _, c := range s.needs {
		if c.Budget != nil && c.Budget.Parent == n.ID && c.ID != n.ID {
			cv := s.spend(c)
			v.ChildrenTokens += max(cv.TotalTokens, cv.CommittedTokens)
		}
	}
	v.CommittedTokens = v.ReservedTokens + v.UnknownTokens + v.SettledTokens + v.ChildrenTokens
	v.RemainingTokens = max(0, v.TotalTokens-v.CommittedTokens)
	v.Overrun = v.CommittedTokens > v.TotalTokens
	return v
}

func (u *needUsage) total() int64 {
	return *u.InputTokens + *u.OutputTokens + *u.CacheWriteTokens + *u.CacheReadTokens
}

// clone copies a usage with its own counts, so the store never shares a caller's memory.
func (u needUsage) clone() *needUsage {
	for _, p := range []**int64{&u.InputTokens, &u.OutputTokens, &u.CacheWriteTokens, &u.CacheReadTokens} {
		if *p != nil {
			v := **p
			*p = &v
		}
	}
	return &u
}

func checkBudget(b *needBudget) error {
	if b.TotalTokens < 1 || b.TotalTokens > maxNeedTokens {
		return badRequest("budget.total_tokens must be 1 to %d", int64(maxNeedTokens))
	}
	if b.Parent != "" && !nameRE.MatchString(b.Parent) {
		return badRequest("budget.parent must be a need id of this plane (%s)", nameRE)
	}
	return nil
}

// checkUsage refuses a usage before any effect: the unit, a source, and either no count at all
// (unknown) or the four counts, each 0 to maxNeedTokens, so their sum cannot overflow.
func checkUsage(u needUsage) error {
	if u.Unit != "tokens" {
		return badRequest("usage.unit must be tokens")
	}
	if !partyRE.MatchString(u.Source) {
		return badRequest("usage.source must match %s", partyRE)
	}
	counts := []*int64{u.InputTokens, u.OutputTokens, u.CacheWriteTokens, u.CacheReadTokens}
	if u.Unknown {
		for _, c := range counts {
			if c != nil {
				return badRequest("an unknown usage carries no count")
			}
		}
		if u.EffectsClarified {
			return badRequest("an unknown usage cannot have clarified effects")
		}
		return nil
	}
	for _, c := range counts {
		if c == nil || *c < 0 || *c > maxNeedTokens {
			return badRequest("usage needs input_tokens, output_tokens, cache_write_tokens and cache_read_tokens, each 0 to %d",
				int64(maxNeedTokens))
		}
	}
	return nil
}

func reservationDigest(resolver string, fence uint64, runner string, tokens int64) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s\n%d\n%s\n%d", resolver, fence, runner, tokens)))
	return hex.EncodeToString(sum[:])
}

// checkReservations holds a loaded need's reservations to what reserve and settle write.
func checkReservations(n *need, fileFence uint64) error {
	if n.Budget != nil && checkBudget(n.Budget) != nil {
		return fmt.Errorf("bad budget")
	}
	if len(n.Reservations) > 0 && n.Budget == nil || len(n.Reservations) > maxNeedReservations {
		return fmt.Errorf("reservations without a budget, or too many")
	}
	ids := map[string]bool{}
	for _, r := range n.Reservations {
		if !partyRE.MatchString(r.ID) || ids[r.ID] || !partyRE.MatchString(r.Resolver) || r.Runner != "deterministic" ||
			r.Tokens < 1 || r.Tokens > maxNeedTokens || r.Fence == 0 || r.Fence > fileFence ||
			r.AtMS < n.CreatedMS || r.AtMS >= n.DeadlineMS || r.Digest != reservationDigest(r.Resolver, r.Fence, r.Runner, r.Tokens) {
			return fmt.Errorf("bad reservation %q", r.ID)
		}
		ids[r.ID] = true
		var ok bool
		switch r.State {
		case "reserved":
			ok = r.Usage == nil
		case "unknown":
			ok = r.Usage != nil && r.Usage.Unknown && checkUsage(*r.Usage) == nil
		case "settled":
			ok = r.Usage != nil && !r.Usage.Unknown && checkUsage(*r.Usage) == nil
		}
		if !ok {
			return fmt.Errorf("reservation %q: state %q inconsistent with its usage", r.ID, r.State)
		}
	}
	return nil
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
	var budget *needBudget
	if r.Budget != nil {
		if err := checkBudget(r.Budget); err != nil {
			return needReply{}, err
		}
		b := *r.Budget
		budget = &b
	}
	digest := needDigest(r.Kind, r.Requester, r.Incarnation, r.DeadlineMS, body, budget)

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
		rep := s.reply(n, true, false)
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
	// A child's total is carved from its parent's remainder, now: the parent never gives more than
	// it has, and no later submit can raise it.
	if budget != nil && budget.Parent != "" {
		pi, p := s.find(budget.Parent)
		if p == nil {
			return needReply{}, &opError{status: 404, msg: fmt.Sprintf("parent need %q is not in this store: a parent outside it is unsupported", budget.Parent)}
		}
		p, err := s.expire(pi, p)
		if err != nil {
			return needReply{}, err
		}
		switch {
		case p.terminal():
			return needReply{}, &opError{status: 409, msg: fmt.Sprintf("parent need %q is %s", p.ID, p.State)}
		case p.Budget == nil:
			return needReply{}, &opError{status: 409, msg: fmt.Sprintf("parent need %q has no budget", p.ID)}
		}
		if v := s.spend(p); v.CommittedTokens+budget.TotalTokens > v.TotalTokens {
			return needReply{}, &opError{status: 409, msg: fmt.Sprintf("over budget: parent need %q has committed %d of %d tokens",
				p.ID, v.CommittedTokens, v.TotalTokens)}
		}
	}
	n := &need{ID: r.ID, Kind: r.Kind, Requester: r.Requester, Incarnation: r.Incarnation, Body: body, Digest: digest,
		CreatedMS: now.UnixMilli(), DeadlineMS: r.DeadlineMS, Rev: 1, State: "open", Budget: budget}
	if err := s.commit(-1, n, s.fence); err != nil {
		return needReply{}, err
	}
	return s.reply(n, true, false), nil
}

// status shows the need; resolver, when it names the claim holder, gets the question too.
func (s *needStore) status(id, requester, incarnation, resolver string) (needReply, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, n, err := s.get(id)
	if err != nil {
		return needReply{}, err
	}
	holder := resolver != "" && n.Claim != nil && n.Claim.Resolver == resolver && !n.terminal() &&
		s.now().UnixMilli() < n.Claim.LeaseUntilMS
	return s.reply(n, n.Requester == requester && n.Incarnation == incarnation, holder), nil
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
	return s.reply(&next, false, true), nil
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
	return s.reply(&next, false, false), nil
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
	return s.reply(&next, false, false), nil
}

// reserve sets tokens aside on a budgeted need before a runner starts: admitted only for the
// live claim's resolver and fence, and only if the need's commitments leave room. The plane
// accounts, it enforces no cap on a runner: a runner other than a deterministic one, which a
// hard total cap would have to bound, is unsupported rather than promised.
func (s *needStore) reserve(id, resolver string, fence uint64, rid, runner string, tokens int64) (needReply, error) {
	switch {
	case runner != "deterministic":
		return needReply{}, &opError{status: 501, msg: fmt.Sprintf("runner %q unsupported: ixctl only accounts for tokens and no adapter here enforces a hard total cap, so only a deterministic runner is admitted", runner)}
	case !partyRE.MatchString(rid):
		return needReply{}, badRequest("reservation_id must match %s", partyRE)
	case tokens < 1 || tokens > maxNeedTokens:
		return needReply{}, badRequest("tokens must be 1 to %d", int64(maxNeedTokens))
	}
	digest := reservationDigest(resolver, fence, runner, tokens)
	s.mu.Lock()
	defer s.mu.Unlock()
	_, n, err := s.get(id)
	if err != nil {
		return needReply{}, err
	}
	// A retry finds its reservation first, whatever the claim or the clock say now: its receipt.
	for _, r := range n.Reservations {
		if r.ID == rid {
			if r.Digest != digest {
				return needReply{}, &opError{status: 409, msg: fmt.Sprintf("reservation %q exists with another digest", rid)}
			}
			rep := s.reply(n, false, false)
			rep.Duplicate = true
			return rep, nil
		}
	}
	if n.Budget == nil {
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("need %q has no budget", id)}
	}
	i, n, err := s.answerable(id, resolver, fence)
	if err != nil {
		return needReply{}, err
	}
	if len(n.Reservations) >= maxNeedReservations {
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("need %q holds %d reservations, its maximum", id, maxNeedReservations)}
	}
	if v := s.spend(n); v.CommittedTokens+tokens > v.TotalTokens {
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("over budget: need %q has committed %d of %d tokens", id, v.CommittedTokens, v.TotalTokens)}
	}
	next := *n
	next.Rev = n.Rev + 1
	next.Reservations = append(append([]needReservation(nil), n.Reservations...), needReservation{ID: rid, Digest: digest,
		Resolver: resolver, Fence: fence, Runner: runner, Tokens: tokens, AtMS: s.now().UnixMilli(), State: "reserved"})
	if err := s.commit(i, &next, s.fence); err != nil {
		return needReply{}, err
	}
	return s.reply(&next, false, true), nil
}

// settle records what a reservation's run used, once: the same usage again is a duplicate,
// another is refused, so a running total re-sent cannot be counted twice. An unknown usage keeps
// the whole reservation committed; only a known usage whose effects were clarified replaces it.
// A settle needs no live claim or deadline: consumption is a fact, reported when it is known.
func (s *needStore) settle(id, resolver, rid string, u needUsage) (needReply, error) {
	if err := checkUsage(u); err != nil {
		return needReply{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	i, n, err := s.get(id)
	if err != nil {
		return needReply{}, err
	}
	k := -1
	for j, r := range n.Reservations {
		if r.ID == rid {
			k = j
		}
	}
	if k < 0 {
		return needReply{}, &opError{status: 404, msg: fmt.Sprintf("reservation %q not found on need %q", rid, id)}
	}
	r := n.Reservations[k]
	if resolver != r.Resolver {
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("resolver %q did not make reservation %q", resolver, rid)}
	}
	if r.Usage != nil && reflect.DeepEqual(*r.Usage, u) {
		rep := s.reply(n, false, false)
		rep.Duplicate = true
		return rep, nil
	}
	switch {
	case r.State == "settled":
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("reservation %q is already settled with another usage", rid)}
	case r.State == "unknown" && u.Unknown:
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("reservation %q is already unknown, from another source", rid)}
	case r.State == "unknown" && !u.EffectsClarified:
		return needReply{}, &opError{status: 409, msg: fmt.Sprintf("reservation %q was reported unknown: a known usage must come with effects_clarified", rid)}
	}
	next := *n
	next.Rev = n.Rev + 1
	next.Reservations = append([]needReservation(nil), n.Reservations...)
	r.Usage, r.State = u.clone(), "settled"
	if u.Unknown {
		r.State = "unknown"
	}
	next.Reservations[k] = r
	if err := s.commit(i, &next, s.fence); err != nil {
		return needReply{}, err
	}
	return s.reply(&next, false, false), nil
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
		var r needReply
		if err == nil { // the reply reads the store (a budget's children): under the lock
			r = s.reply(n, n.Requester == requester && n.Incarnation == incarnation, false)
		}
		s.mu.Unlock()
		if err != nil {
			return needReply{}, err
		}
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

	Budget        *needBudget `json:"budget,omitempty"`
	ReservationID string      `json:"reservation_id,omitempty"`
	Runner        string      `json:"runner,omitempty"`
	Tokens        int64       `json:"tokens,omitempty"`
	Usage         *needUsage  `json:"usage,omitempty"`
}

var needOps = []string{"submit", "status", "claim", "resolve", "refuse", "reserve", "settle", "wait"}

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
			Body: o.Body, DeadlineMS: o.DeadlineMS, Budget: o.Budget})
	case "need.status":
		return p.needs.status(o.ID, o.Requester, o.Incarnation, o.Resolver)
	case "need.claim":
		return p.needs.claim(o.ID, o.Resolver, o.LeaseMS)
	case "need.resolve":
		return p.needs.resolve(o.ID, o.Resolver, o.Fence, o.Answer)
	case "need.refuse":
		return p.needs.refuse(o.ID, o.Resolver, o.Fence, o.Reason)
	case "need.reserve":
		return p.needs.reserve(o.ID, o.Resolver, o.Fence, o.ReservationID, o.Runner, o.Tokens)
	case "need.settle":
		if o.Usage == nil {
			return nil, badRequest("need.settle needs a `usage`")
		}
		return p.needs.settle(o.ID, o.Resolver, o.ReservationID, *o.Usage)
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
