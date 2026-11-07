package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeClock is the store's injected clock: tests move time, nothing sleeps for a deadline.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *fakeClock) now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *fakeClock) add(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

func newNeedTest(t *testing.T) (*needStore, *fakeClock, string) {
	t.Helper()
	clk := &fakeClock{t: time.UnixMilli(1_800_000_000_000)}
	path := filepath.Join(t.TempDir(), "p.needs.json")
	return newNeedStore(path, clk.now), clk, path
}

func submitOf(clk *fakeClock, id string, in time.Duration) needSubmit {
	return needSubmit{ID: id, Kind: "information", Requester: "task-a", Incarnation: "inc-1",
		Body: json.RawMessage(`{"question":"which port?"}`), DeadlineMS: clk.now().Add(in).UnixMilli()}
}

func wantStatus(t *testing.T, err error, status int, frag string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want a %d refusal containing %q, got nil", status, frag)
	}
	if statusOf(err) != status || !strings.Contains(err.Error(), frag) {
		t.Fatalf("want %d containing %q, got %d %q", status, frag, statusOf(err), err)
	}
}

func TestNeedRestartKeepsDeadlineClaimAndFence(t *testing.T) {
	s, clk, path := newNeedTest(t)
	sub := submitOf(clk, "n1", time.Hour)
	if _, err := s.submit(sub); err != nil {
		t.Fatal(err)
	}
	c, err := s.claim("n1", "resolver-x", 60_000)
	if err != nil {
		t.Fatal(err)
	}

	s2 := newNeedStore(path, clk.now) // the plane restarted: same file, same clock
	r, err := s2.status("n1", "task-a", "inc-1", "")
	if err != nil {
		t.Fatal(err)
	}
	if r.DeadlineMS != sub.DeadlineMS || r.State != "claimed" || r.Claim == nil ||
		r.Claim.Fence != c.Claim.Fence || r.Claim.LeaseUntilMS != c.Claim.LeaseUntilMS || r.Rev != c.Rev {
		t.Fatalf("after restart: %+v (claim %+v), want deadline %d, claimed fence %d lease %d rev %d",
			r, r.Claim, sub.DeadlineMS, c.Claim.Fence, c.Claim.LeaseUntilMS, c.Rev)
	}
	// The lease is not extended by the restart: past it, another resolver claims with a new fence.
	clk.add(61 * time.Second)
	c2, err := s2.claim("n1", "resolver-y", 60_000)
	if err != nil {
		t.Fatal(err)
	}
	if c2.Claim.Fence <= c.Claim.Fence {
		t.Fatalf("fence after restart %d, want > %d", c2.Claim.Fence, c.Claim.Fence)
	}
	// The deadline is not extended either: past it, the need is expired, its rev moved once.
	clk.add(time.Hour)
	s3 := newNeedStore(path, clk.now)
	r, err = s3.status("n1", "task-a", "inc-1", "")
	if err != nil || r.State != "expired" || r.DeadlineMS != sub.DeadlineMS || r.Rev != c2.Rev+1 {
		t.Fatalf("past the deadline: %+v %v, want expired at the same deadline, rev %d", r, err, c2.Rev+1)
	}
}

func TestNeedSubmitDedupAndDigestConflict(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	sub := submitOf(clk, "n1", time.Hour)
	a, err := s.submit(sub)
	if err != nil || a.Duplicate || a.Rev != 1 || a.State != "open" {
		t.Fatalf("first submit: %+v %v", a, err)
	}
	b, err := s.submit(sub)
	if err != nil || !b.Duplicate || b.Rev != a.Rev || b.Digest != a.Digest {
		t.Fatalf("same id, same content: %+v %v, want the same need back as a duplicate", b, err)
	}
	other := sub
	other.Body = json.RawMessage(`{"question":"which host?"}`)
	_, err = s.submit(other)
	wantStatus(t, err, 409, "digest")
	if strings.Contains(err.Error(), "which") {
		t.Fatalf("a refusal must not echo the content: %q", err)
	}
	later := sub
	later.DeadlineMS += 1000
	_, err = s.submit(later)
	wantStatus(t, err, 409, "digest")
}

func TestNeedConcurrentClaimsOneWinner(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	if _, err := s.submit(submitOf(clk, "n1", time.Hour)); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wins := make(chan string, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := "resolver-" + string(rune('a'+i))
			if _, err := s.claim("n1", r, 60_000); err == nil {
				wins <- r
			} else if statusOf(err) != 409 {
				t.Errorf("losing claim: %v, want 409", err)
			}
		}(i)
	}
	wg.Wait()
	close(wins)
	if n := len(wins); n != 1 {
		t.Fatalf("%d claims won, want exactly 1", n)
	}
}

func TestNeedFencingAndLeaseLoss(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	if _, err := s.submit(submitOf(clk, "n1", time.Hour)); err != nil {
		t.Fatal(err)
	}
	old, err := s.claim("n1", "resolver-x", 10_000)
	if err != nil {
		t.Fatal(err)
	}
	clk.add(11 * time.Second) // x lost its lease without knowing
	_, err = s.resolve("n1", "resolver-x", old.Claim.Fence, json.RawMessage(`{"port":1}`))
	wantStatus(t, err, 409, "lease")
	cur, err := s.claim("n1", "resolver-y", 10_000)
	if err != nil {
		t.Fatal(err)
	}
	_, err = s.resolve("n1", "resolver-x", old.Claim.Fence, json.RawMessage(`{"port":1}`))
	wantStatus(t, err, 409, "fence")
	_, err = s.resolve("n1", "resolver-x", cur.Claim.Fence, json.RawMessage(`{"port":1}`))
	wantStatus(t, err, 409, "resolver")
	r, err := s.resolve("n1", "resolver-y", cur.Claim.Fence, json.RawMessage(`{"port":2}`))
	if err != nil || r.State != "resolved" || r.Rev != cur.Rev+1 {
		t.Fatalf("resolve with the current fence: %+v %v", r, err)
	}
}

func TestNeedResolutionCorrelatedRefusedAndLate(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	if _, err := s.submit(submitOf(clk, "n1", time.Hour)); err != nil {
		t.Fatal(err)
	}
	c, _ := s.claim("n1", "resolver-x", 60_000)
	if _, err := s.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`{"port":8080}`)); err != nil {
		t.Fatal(err)
	}
	mine, err := s.status("n1", "task-a", "inc-1", "")
	if err != nil || string(mine.Answer) != `{"port":8080}` || mine.Resolution == nil || mine.Resolution.Fence != c.Claim.Fence {
		t.Fatalf("requester's status: %+v %v", mine, err)
	}
	for _, who := range [][2]string{{"task-b", "inc-1"}, {"task-a", "inc-2"}} {
		r, err := s.status("n1", who[0], who[1], "")
		if err != nil || r.State != "resolved" || r.Answer != nil {
			t.Fatalf("status for %v: %+v %v, want the state without the answer", who, r, err)
		}
	}
	// A second answer to a resolved need is refused, never a second resolution.
	_, err = s.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`{"port":9}`))
	wantStatus(t, err, 409, "resolved")

	// Refusal is its own terminal state.
	s.submit(submitOf(clk, "n2", time.Hour))
	c2, _ := s.claim("n2", "resolver-x", 60_000)
	r, err := s.refuse("n2", "resolver-x", c2.Claim.Fence, "not mine to give")
	if err != nil || r.State != "refused" || r.Reason != "not mine to give" {
		t.Fatalf("refuse: %+v %v", r, err)
	}

	// A late answer is refused and the need is expired, not resolved.
	s.submit(submitOf(clk, "n3", time.Minute))
	c3, _ := s.claim("n3", "resolver-x", 600_000)
	clk.add(2 * time.Minute)
	_, err = s.resolve("n3", "resolver-x", c3.Claim.Fence, json.RawMessage(`{"port":1}`))
	wantStatus(t, err, 409, "expired")
	r, _ = s.status("n3", "task-a", "inc-1", "")
	if r.State != "expired" || r.Answer != nil {
		t.Fatalf("late answer: %+v, want expired without answer", r)
	}
}

func TestNeedWaitTimeoutIsNotExpiryNorCancel(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	s.submit(submitOf(clk, "n1", time.Hour))

	r, err := s.wait(context.Background(), "n1", "task-a", "inc-1", 30*time.Millisecond)
	if err != nil || r.Outcome != "wait-timeout" || r.State != "open" {
		t.Fatalf("wait past its own timeout: %+v %v, want outcome wait-timeout, need still open", r, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = s.wait(ctx, "n1", "task-a", "inc-1", time.Second)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled wait: %v, want context.Canceled", err)
	}
	if r, _ := s.status("n1", "task-a", "inc-1", ""); r.State != "open" {
		t.Fatalf("after a cancelled wait the need is %q, want open", r.State)
	}

	// A wait sees the resolution, with the answer for its requester.
	go func() {
		time.Sleep(20 * time.Millisecond)
		c, _ := s.claim("n1", "resolver-x", 60_000)
		s.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`{"ok":true}`))
	}()
	r, err = s.wait(context.Background(), "n1", "task-a", "inc-1", 5*time.Second)
	if err != nil || r.Outcome != "resolved" || string(r.Answer) != `{"ok":true}` {
		t.Fatalf("wait for the resolution: %+v %v", r, err)
	}

	// And an expiry, as an expiry.
	s.submit(submitOf(clk, "n2", time.Minute))
	go func() { time.Sleep(20 * time.Millisecond); clk.add(2 * time.Minute) }()
	r, err = s.wait(context.Background(), "n2", "task-a", "inc-1", 5*time.Second)
	if err != nil || r.Outcome != "expired" || r.State != "expired" {
		t.Fatalf("wait across the deadline: %+v %v", r, err)
	}
}

func TestNeedInvalidRefusedBeforeEffect(t *testing.T) {
	s, clk, path := newNeedTest(t)
	ok := submitOf(clk, "n1", time.Hour)
	bad := []struct {
		name string
		edit func(*needSubmit)
	}{
		{"id", func(x *needSubmit) { x.ID = "Not An ID" }},
		{"kind", func(x *needSubmit) { x.Kind = "anything" }},
		{"requester", func(x *needSubmit) { x.Requester = "" }},
		{"incarnation", func(x *needSubmit) { x.Incarnation = strings.Repeat("i", 200) }},
		{"body", func(x *needSubmit) { x.Body = json.RawMessage(`"` + strings.Repeat("b", maxNeedBody) + `"`) }},
		{"body", func(x *needSubmit) { x.Body = nil }},
		{"deadline", func(x *needSubmit) { x.DeadlineMS = clk.now().UnixMilli() }},
		{"deadline", func(x *needSubmit) { x.DeadlineMS = clk.now().Add(maxNeedHorizon + time.Second).UnixMilli() }},
	}
	for _, b := range bad {
		x := ok
		b.edit(&x)
		_, err := s.submit(x)
		wantStatus(t, err, 400, b.name)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("refused submits wrote the store: %v", err)
	}
	s.submit(ok)
	c, _ := s.claim("n1", "resolver-x", 60_000)
	_, err := s.claim("n2", "resolver-x", 60_000)
	wantStatus(t, err, 404, "n2")
	_, err = s.claim("n1", "resolver-y", 0)
	wantStatus(t, err, 400, "lease")
	_, err = s.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`[`+strings.Repeat("1,", maxNeedBody)+`1]`))
	wantStatus(t, err, 400, "answer")
	_, err = s.refuse("n1", "resolver-x", c.Claim.Fence, strings.Repeat("r", maxNeedReason+1))
	wantStatus(t, err, 400, "reason")
	if r, _ := s.status("n1", "task-a", "inc-1", ""); r.State != "claimed" || r.Rev != c.Rev {
		t.Fatalf("refused answers moved the need: %+v", r)
	}
}

func TestNeedAuthorizationIsNotSelfResolved(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	sub := submitOf(clk, "n1", time.Hour)
	sub.Kind = "authorization"
	s.submit(sub)
	_, err := s.claim("n1", "task-a", 60_000)
	wantStatus(t, err, 403, "requester")
}

// upPlane runs `ixctl up` in process on dir and returns its state file and a stop that waits for it.
func upPlane(t *testing.T, dir string) (stateFile, func()) {
	t.Helper()
	specPath := filepath.Join(dir, "spec.json")
	if err := os.WriteFile(specPath, []byte(`{"name":"need-up","services":[]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	pr, pw := io.Pipe()
	done := make(chan error, 1)
	go func() {
		err := cmdUp([]string{"--state", filepath.Join(dir, "state"), specPath}, pw)
		pw.Close()
		done <- err
	}()
	line, err := bufio.NewReader(pr).ReadString('\n')
	if err != nil {
		t.Fatalf("ready line: %v", err)
	}
	go io.Copy(io.Discard, pr)
	var ready readyLine
	if err := json.Unmarshal([]byte(line), &ready); err != nil {
		t.Fatalf("ready line %q: %v", line, err)
	}
	st, err := readStateFile(ready.StateFile)
	if err != nil {
		t.Fatal(err)
	}
	stopped := false
	stop := func() {
		if stopped {
			return
		}
		stopped = true
		req, _ := http.NewRequest("POST", "http://"+st.Addr+"/v1/shutdown", strings.NewReader(`{}`))
		req.Header.Set("Authorization", "Bearer "+st.Token)
		if resp, err := http.DefaultClient.Do(req); err == nil {
			resp.Body.Close()
		}
		select {
		case err := <-done:
			if err != nil {
				t.Errorf("cmdUp: %v", err)
			}
		case <-time.After(15 * time.Second):
			t.Error("cmdUp had not returned 15 s after /v1/shutdown")
		}
	}
	t.Cleanup(stop)
	return st, stop
}

// needCLI runs `ixctl need <op> --state <dir>` with body on stdin and decodes its JSON reply.
func needCLI(t *testing.T, dir, op, body string) (needReply, error) {
	t.Helper()
	var out bytes.Buffer
	err := cmdNeed([]string{op, "--state", filepath.Join(dir, "state"), "--name", "need-up"}, strings.NewReader(body), &out)
	if err != nil {
		return needReply{}, err
	}
	var r needReply
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("need %s printed %q: %v", op, out.String(), err)
	}
	return r, nil
}

func TestNeedCLIOverARealPlaneSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	_, stop := upPlane(t, dir)
	deadline := time.Now().Add(time.Hour).UnixMilli()
	sub := fmt.Sprintf(`{"id":"port-q","kind":"information","requester":"task-a","incarnation":"inc-1","body":{"q":"port"},"deadline_ms":%d}`, deadline)
	r, err := needCLI(t, dir, "submit", sub)
	if err != nil || r.ID != "port-q" || r.State != "open" || r.Rev != 1 {
		t.Fatalf("submit: %+v %v", r, err)
	}
	if r, err = needCLI(t, dir, "submit", sub); err != nil || !r.Duplicate {
		t.Fatalf("resubmit: %+v %v, want a duplicate", r, err)
	}
	w, err := needCLI(t, dir, "wait", `{"id":"port-q","requester":"task-a","incarnation":"inc-1","timeout_ms":50}`)
	if err != nil || w.Outcome != "wait-timeout" || w.State != "open" {
		t.Fatalf("short wait: %+v %v", w, err)
	}
	c, err := needCLI(t, dir, "claim", `{"id":"port-q","resolver":"res-x","lease_ms":60000}`)
	if err != nil || c.Claim == nil || string(c.Body) != `{"q":"port"}` {
		t.Fatalf("claim: %+v %v, want the question in the winner's reply", c, err)
	}
	stop() // the plane goes down with the need claimed

	upPlane(t, dir)
	s, err := needCLI(t, dir, "status", `{"id":"port-q","requester":"task-a","incarnation":"inc-1"}`)
	if err != nil || s.State != "claimed" || s.DeadlineMS != deadline || s.Claim.Fence != c.Claim.Fence || s.Rev != c.Rev {
		t.Fatalf("after restart: %+v %v, want the claim and deadline kept", s, err)
	}
	// The resolver, restarted too, reads the question from the plane and answers from it alone.
	h, err := needCLI(t, dir, "status", `{"id":"port-q","resolver":"res-x"}`)
	var q struct{ Q string }
	if err != nil || json.Unmarshal(h.Body, &q) != nil || q.Q == "" || h.Answer != nil {
		t.Fatalf("holder's status: %+v %v, want the question and no answer", h, err)
	}
	if o, _ := needCLI(t, dir, "status", `{"id":"port-q","resolver":"res-y"}`); o.Body != nil {
		t.Fatalf("a resolver without the claim read the question: %s", o.Body)
	}
	answer := fmt.Sprintf(`{%q:8080}`, q.Q)
	if _, err := needCLI(t, dir, "resolve", fmt.Sprintf(`{"id":"port-q","resolver":"res-x","fence":%d,"answer":%s}`, h.Claim.Fence, answer)); err != nil {
		t.Fatal(err)
	}
	w, err = needCLI(t, dir, "wait", `{"id":"port-q","requester":"task-a","incarnation":"inc-1","timeout_ms":5000}`)
	if err != nil || w.Outcome != "resolved" || string(w.Answer) != `{"port":8080}` {
		t.Fatalf("wait after restart: %+v %v", w, err)
	}
	_, err = needCLI(t, dir, "resolve", fmt.Sprintf(`{"id":"port-q","resolver":"res-x","fence":%d,"answer":{"port":9}}`, c.Claim.Fence))
	if err == nil || !strings.Contains(err.Error(), "resolved") {
		t.Fatalf("a second answer: %v, want refused as resolved", err)
	}
	if _, err := needCLI(t, dir, "submit", `{"id":"Bad"}`); err == nil {
		t.Fatal("an invalid submit passed the CLI")
	}
}

func TestNeedPersistenceInterruption(t *testing.T) {
	t.Run("temp write fails: nothing changes", func(t *testing.T) {
		s, clk, path := newNeedTest(t)
		s.submit(submitOf(clk, "n1", time.Hour))
		s.failTemp = errors.New("disk full")
		_, err := s.claim("n1", "resolver-x", 60_000)
		wantStatus(t, err, 500, "not persisted")
		r, _ := s.status("n1", "task-a", "inc-1", "")
		if r.State != "open" {
			t.Fatalf("after a failed write the need is %q, want open", r.State)
		}
		if r, _ := newNeedStore(path, clk.now).status("n1", "task-a", "inc-1", ""); r.State != "open" {
			t.Fatalf("reopened: %q, want open", r.State)
		}
	})
	t.Run("rename fails: unknown, then reconciled from disk, never replayed", func(t *testing.T) {
		s, clk, _ := newNeedTest(t)
		s.submit(submitOf(clk, "n1", time.Hour))
		s.failRename = errors.New("access denied")
		_, err := s.claim("n1", "resolver-x", 60_000)
		wantStatus(t, err, 503, "unknown")
		s.failRename = nil
		r, err := s.status("n1", "task-a", "inc-1", "")
		if err != nil || r.State != "open" || r.Rev != 1 {
			t.Fatalf("after reconcile: %+v %v, want the disk's open rev 1 (the claim not replayed)", r, err)
		}
	})
	t.Run("unreadable file: every op unknown, file kept", func(t *testing.T) {
		_, clk, path := newNeedTest(t)
		os.WriteFile(path, []byte(`{"schema":"ixctl-needs/0","needs":[{`), 0o600)
		s := newNeedStore(path, clk.now)
		_, err := s.submit(submitOf(clk, "n1", time.Hour))
		wantStatus(t, err, 503, "unknown")
		_, err = s.status("n1", "task-a", "inc-1", "")
		wantStatus(t, err, 503, "unknown")
		if b, _ := os.ReadFile(path); string(b) != `{"schema":"ixctl-needs/0","needs":[{` {
			t.Fatalf("the unreadable file was rewritten: %q", b)
		}
	})
}

// A file that decodes but breaks the ops' invariants is unknown: no op uses it, none panics,
// and the file stays as it was for the operator.
func TestNeedCorruptRecordsAreUnknownNotPanic(t *testing.T) {
	s, clk, path := newNeedTest(t)
	s.submit(submitOf(clk, "n1", time.Hour))
	c, _ := s.claim("n1", "resolver-x", 60_000)
	s.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`{"port":1}`))
	good, _ := os.ReadFile(path)
	edits := map[string]func(*needFile){
		"claimed without claim": func(f *needFile) {
			f.Needs[0].State, f.Needs[0].Answer, f.Needs[0].Resolution, f.Needs[0].Claim = "claimed", nil, nil, nil
		},
		"resolved without answer": func(f *needFile) { f.Needs[0].Answer = nil },
		"digest not its own":      func(f *needFile) { f.Needs[0].Digest = strings.Repeat("0", 64) },
		"fence above the file":    func(f *needFile) { f.Fence = 0 },
		"resolution unbound":      func(f *needFile) { f.Needs[0].Resolution.Fence = 99 },
		"deadline before created": func(f *needFile) { f.Needs[0].DeadlineMS = f.Needs[0].CreatedMS },
		"rev zero":                func(f *needFile) { f.Needs[0].Rev = 0 },
		"resolution at zero":      func(f *needFile) { f.Needs[0].Resolution.AtMS = 0 },
		"resolution past deadline": func(f *needFile) {
			f.Needs[0].Resolution.AtMS, f.Needs[0].Claim.LeaseUntilMS = f.Needs[0].DeadlineMS, f.Needs[0].DeadlineMS+1
		},
		"resolution past lease": func(f *needFile) { f.Needs[0].Resolution.AtMS = f.Needs[0].Claim.LeaseUntilMS },
	}
	for name, edit := range edits {
		t.Run(name, func(t *testing.T) {
			var f needFile
			if err := json.Unmarshal(good, &f); err != nil {
				t.Fatal(err)
			}
			edit(&f)
			b, _ := json.Marshal(f)
			os.WriteFile(path, b, 0o600)
			s := newNeedStore(path, clk.now)
			for _, op := range []func() error{
				func() error { _, err := s.status("n1", "task-a", "inc-1", ""); return err },
				func() error { _, err := s.claim("n1", "resolver-y", 60_000); return err },
				func() error { _, err := s.resolve("n1", "resolver-x", 1, json.RawMessage(`{}`)); return err },
			} {
				wantStatus(t, op(), 503, "unknown")
			}
			if after, _ := os.ReadFile(path); !bytes.Equal(after, b) {
				t.Fatal("the corrupt file was rewritten")
			}
		})
	}
	t.Run("over the byte bound", func(t *testing.T) {
		os.WriteFile(path, bytes.Repeat([]byte(" "), maxNeedsFile+1), 0o600)
		_, err := newNeedStore(path, clk.now).status("n1", "task-a", "inc-1", "")
		wantStatus(t, err, 503, "over")
	})
}

// os.Rename promises no atomicity off Unix: a temporary with no file beside it may be a write
// that lost the old file first, so the store is unknown, not empty.
func TestNeedLoneTemporaryIsUnknownNotEmpty(t *testing.T) {
	s, clk, path := newNeedTest(t)
	s.submit(submitOf(clk, "n1", time.Hour))
	b, _ := os.ReadFile(path)
	os.Remove(path)
	os.WriteFile(path+".tmp", b, 0o600)
	_, err := newNeedStore(path, clk.now).submit(submitOf(clk, "n1", time.Hour))
	wantStatus(t, err, 503, "missing")
}

// A resubmit after a crash, a restart or the deadline finds the same need: its receipt, never a
// new need, and a changed content stays a conflict.
func TestNeedResubmitAfterRestartAndExpiry(t *testing.T) {
	s, clk, path := newNeedTest(t)
	sub := submitOf(clk, "n1", time.Minute)
	first, _ := s.submit(sub)
	s = newNeedStore(path, clk.now) // the process died after the commit, before its reply
	r, err := s.submit(sub)
	if err != nil || !r.Duplicate || r.Rev != first.Rev || r.CreatedMS != first.CreatedMS {
		t.Fatalf("resubmit after restart: %+v %v, want the committed need as a duplicate", r, err)
	}
	clk.add(2 * time.Minute)
	r, err = s.submit(sub)
	if err != nil || !r.Duplicate || r.State != "expired" || r.DeadlineMS != sub.DeadlineMS {
		t.Fatalf("resubmit past the deadline: %+v %v, want the expired need as a duplicate", r, err)
	}
	other := sub
	other.Body = json.RawMessage(`{"other":1}`)
	_, err = s.submit(other)
	wantStatus(t, err, 409, "digest")
}

// After a failed rename nothing is adopted: before any reconcile, memory still holds the old
// list and no waiter was woken.
func TestNeedFailedRenameAdoptsNothing(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	s.submit(submitOf(clk, "n1", time.Hour))
	wake := s.wake
	s.failRename = errors.New("access denied")
	s.claim("n1", "resolver-x", 60_000)
	if len(s.needs) != 1 || s.needs[0].State != "open" || s.fence != 0 {
		t.Fatalf("memory after a failed rename: state %q fence %d, want open and 0", s.needs[0].State, s.fence)
	}
	select {
	case <-wake:
		t.Fatal("a failed rename woke the waiters")
	default:
	}
}

// At the real entry points: a second plane on the same state dir is refused (the instance lock
// keeps one writer of the needs file), a need op needs the bearer, and none is forwarded.
// A budget field the plane does not know (`token_budget` rather than `budget`) is refused before
// anything is written, never silently dropped, so no caller believes in a budget that is not kept.
func TestNeedTokenBudgetIsRefusedNotIgnored(t *testing.T) {
	dir := t.TempDir()
	st, _ := upPlane(t, dir)
	sub := fmt.Sprintf(`{"id":"budget-q","kind":"information","requester":"task-a","incarnation":"i1","body":{},"deadline_ms":%d,"token_budget":{"total":1000}}`,
		time.Now().Add(time.Hour).UnixMilli())
	if _, err := needCLI(t, dir, "submit", sub); err == nil || !strings.Contains(err.Error(), "token_budget") {
		t.Fatalf("CLI submit with a token budget: %v, want refused naming the field", err)
	}
	req, _ := http.NewRequest("POST", "http://"+st.Addr+"/v1/need/submit", strings.NewReader(sub))
	req.Header.Set("Authorization", "Bearer "+st.Token)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("HTTP submit with a token budget: %d, want 400", resp.StatusCode)
	}
	if _, err := needCLI(t, dir, "status", `{"id":"budget-q"}`); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("status after the refusals: %v, want not found", err)
	}
}

func TestNeedEntryPointsLockAuthAndNoForward(t *testing.T) {
	dir := t.TempDir()
	st, _ := upPlane(t, dir)
	needCLI(t, dir, "submit", fmt.Sprintf(`{"id":"q1","kind":"capacity","requester":"task-a","incarnation":"i1","body":{},"deadline_ms":%d}`,
		time.Now().Add(time.Hour).UnixMilli()))
	needsPath := filepath.Join(dir, "state", "need-up.needs.json")
	before, err := os.ReadFile(needsPath)
	if err != nil {
		t.Fatal(err)
	}
	specPath := filepath.Join(dir, "spec.json")
	if err := cmdUp([]string{"--state", filepath.Join(dir, "state"), specPath}, io.Discard); err == nil ||
		!strings.Contains(err.Error(), "already") {
		t.Fatalf("a second plane on the same state dir: %v, want refused", err)
	}
	if after, _ := os.ReadFile(needsPath); !bytes.Equal(before, after) {
		t.Fatal("the refused plane touched the needs file")
	}
	resp, err := http.Post("http://"+st.Addr+"/v1/need/status", "application/json", strings.NewReader(`{"id":"q1"}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Fatalf("need op without the bearer: %d, want 401", resp.StatusCode)
	}
	p, err := NewPlane(&Spec{Name: "x"}, "x", "", "")
	if err != nil {
		t.Fatal(err)
	}
	p.SetNeedStore(filepath.Join(t.TempDir(), "x.needs.json"))
	_, err = p.Request(context.Background(), Request{Op: "need.status", Peer: "other", Need: &needOp{ID: "q1"}})
	if err == nil || !strings.Contains(err.Error(), "peer") && !strings.Contains(err.Error(), "forward") {
		t.Fatalf("a need op naming a peer: %v, want refused", err)
	}
	// A declared identity that is not the asker's reads the state, never the answer.
	r, err := needCLI(t, dir, "status", `{"id":"q1","requester":"task-b","incarnation":"i1"}`)
	if err != nil || r.Answer != nil {
		t.Fatalf("status for another requester: %+v %v", r, err)
	}
}

// timeout_ms is bounded before it becomes a duration, at the plane's entry point and in the CLI
// (before any dial): huge values must not wrap into a short or negative wait.
func TestNeedWaitTimeoutBoundedBeforeConversion(t *testing.T) {
	p, err := NewPlane(&Spec{Name: "x"}, "x", "", "")
	if err != nil {
		t.Fatal(err)
	}
	p.SetNeedStore(filepath.Join(t.TempDir(), "x.needs.json"))
	ctx := context.Background()
	if _, err := p.Request(ctx, Request{Op: "need.submit", Need: &needOp{ID: "n1", Kind: "information", Requester: "a",
		Incarnation: "i", Body: json.RawMessage(`{}`), DeadlineMS: time.Now().Add(time.Hour).UnixMilli()}}); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	for _, ms := range []int64{18446744073710, 9223372036856, 0, -1, maxNeedLease.Milliseconds() + 1} {
		_, err := p.Request(ctx, Request{Op: "need.wait", Need: &needOp{ID: "n1", Requester: "a", Incarnation: "i", TimeoutMS: ms}})
		wantStatus(t, err, 400, "timeout_ms")
		err = cmdNeed([]string{"wait", "--state", filepath.Join(t.TempDir(), "none"), "--name", "x"},
			strings.NewReader(fmt.Sprintf(`{"id":"n1","timeout_ms":%d}`, ms)), io.Discard)
		if err == nil || !strings.Contains(err.Error(), "timeout_ms") {
			t.Fatalf("CLI wait with timeout_ms %d: %v, want refused before any dial", ms, err)
		}
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("refusals took %s: one of them waited", d)
	}
}

// The single writer is the instance lock: a plane in another process holds it, a second `up`
// on the same state dir is refused, and a crash (a kill, no shutdown) releases it to the OS;
// the next plane finds the needs, their claim and fence as committed.
func TestNeedSingleWriterAcrossProcessesAndCrash(t *testing.T) {
	dir := t.TempDir()
	specPath := filepath.Join(dir, "spec.json")
	if err := os.WriteFile(specPath, []byte(`{"name":"need-up","services":[]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testExe(t), "up", "--state", filepath.Join(dir, "state"), specPath)
	cmd.Env = append(os.Environ(), "IXCTL_TEST_ROLE=main")
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cmd.Process.Kill(); cmd.Wait() })
	if _, err := bufio.NewReader(out).ReadString('\n'); err != nil {
		t.Fatalf("child plane ready line: %v", err)
	}
	go io.Copy(io.Discard, out)

	sub := fmt.Sprintf(`{"id":"q1","kind":"information","requester":"task-a","incarnation":"i1","body":{"q":1},"deadline_ms":%d}`,
		time.Now().Add(time.Hour).UnixMilli())
	if _, err := needCLI(t, dir, "submit", sub); err != nil {
		t.Fatal(err)
	}
	c, err := needCLI(t, dir, "claim", `{"id":"q1","resolver":"res-x","lease_ms":60000}`)
	if err != nil {
		t.Fatal(err)
	}
	if err := cmdUp([]string{"--state", filepath.Join(dir, "state"), specPath}, io.Discard); err == nil ||
		!strings.Contains(err.Error(), "already") {
		t.Fatalf("an up beside another process's plane: %v, want refused", err)
	}
	if err := cmd.Process.Kill(); err != nil { // a crash: no shutdown, no cleanup
		t.Fatal(err)
	}
	cmd.Wait()

	upPlane(t, dir) // this process now: the OS released the lock
	s, err := needCLI(t, dir, "status", `{"id":"q1","requester":"task-a","incarnation":"i1"}`)
	if err != nil || s.State != "claimed" || s.Claim.Fence != c.Claim.Fence || s.Rev != c.Rev {
		t.Fatalf("after the crash: %+v %v, want claimed with fence %d rev %d", s, err, c.Claim.Fence, c.Rev)
	}
}

// Committed on disk, the result lost before memory and reply (a crash right after the rename,
// injected): a reopen sees the new revision, fence and answer, and a retried answer is refused.
// This is the model's proof, not a proof of the OS's rename or of a power loss.
func TestNeedResultLostAfterRenameIsNotAnsweredTwice(t *testing.T) {
	s, clk, path := newNeedTest(t)
	s.submit(submitOf(clk, "n1", time.Hour))
	c, _ := s.claim("n1", "resolver-x", 60_000)
	s.failAfterRename = errors.New("crash after commit")
	_, err := s.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`{"port":1}`))
	wantStatus(t, err, 503, "unknown")

	re := newNeedStore(path, clk.now)
	r, err := re.status("n1", "task-a", "inc-1", "")
	if err != nil || r.State != "resolved" || r.Rev != c.Rev+1 || r.Resolution.Fence != c.Claim.Fence || string(r.Answer) != `{"port":1}` {
		t.Fatalf("reopened: %+v %v, want the committed resolution", r, err)
	}
	_, err = re.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`{"port":2}`))
	wantStatus(t, err, 409, "resolved")
	s.failAfterRename = nil
	if r, err := s.status("n1", "task-a", "inc-1", ""); err != nil || r.State != "resolved" {
		t.Fatalf("the same store after reconcile: %+v %v", r, err)
	}
}

// The question goes to the asker and to the resolver holding the claim, nobody else; the answer
// only to the asker.
func TestNeedQuestionGoesToAskerAndHolderOnly(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	sub := submitOf(clk, "n1", time.Hour)
	a, _ := s.submit(sub)
	if string(a.Body) != string(sub.Body) {
		t.Fatalf("submit reply body %s", a.Body)
	}
	c, _ := s.claim("n1", "resolver-x", 60_000)
	if string(c.Body) != string(sub.Body) {
		t.Fatalf("claim winner got body %s, want the question", c.Body)
	}
	if r, _ := s.status("n1", "", "", "resolver-x"); string(r.Body) != string(sub.Body) || r.Answer != nil {
		t.Fatalf("holder's status: body %s answer %s", r.Body, r.Answer)
	}
	if r, _ := s.status("n1", "task-b", "inc-1", "resolver-y"); r.Body != nil {
		t.Fatalf("a stranger read the question: %s", r.Body)
	}
	r, _ := s.resolve("n1", "resolver-x", c.Claim.Fence, json.RawMessage(`{"ok":1}`))
	if r.Body != nil || r.Answer != nil {
		t.Fatalf("resolve reply carries body %s answer %s", r.Body, r.Answer)
	}
	if r, _ := s.status("n1", "", "", "resolver-x"); r.Body != nil {
		t.Fatalf("the holder still reads the question after the need ended: %s", r.Body)
	}
}

// A lapsed lease ends the right to read the question, before the deadline and with no reclaim:
// the claim is still on record, but it is no longer live. The asker still reads it.
func TestNeedQuestionHiddenFromTheHolderOnceItsLeaseLapses(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	sub := submitOf(clk, "n1", time.Hour)
	s.submit(sub)
	s.claim("n1", "resolver-x", 60_000)
	clk.add(59 * time.Second)
	if r, _ := s.status("n1", "", "", "resolver-x"); string(r.Body) != string(sub.Body) {
		t.Fatalf("holder inside its lease: body %s, want the question", r.Body)
	}
	clk.add(time.Second) // now == LeaseUntilMS: the lease has lapsed, the deadline is 59 min away
	r, err := s.status("n1", "", "", "resolver-x")
	if err != nil || r.State != "claimed" || r.Claim == nil || r.Claim.Resolver != "resolver-x" {
		t.Fatalf("status after the lease: %+v %v, want the lapsed claim still on record", r, err)
	}
	if r.Body != nil {
		t.Fatalf("the ex-holder still reads the question after its lease lapsed: %s", r.Body)
	}
	if r, _ := s.status("n1", "task-a", "inc-1", ""); string(r.Body) != string(sub.Body) {
		t.Fatalf("the asker after the lease: body %s, want the question", r.Body)
	}
}

// --- token budget: accounting and admission, no provider (fixtures prove the core, not a cap) ---

func budgetOf(clk *fakeClock, id string, total int64, parent string) needSubmit {
	sub := submitOf(clk, id, time.Hour)
	sub.Budget = &needBudget{TotalTokens: total, Parent: parent}
	return sub
}

func i64(v int64) *int64 { return &v }

func usageOf(in, out, cacheWrite, cacheRead int64) needUsage {
	return needUsage{Unit: "tokens", Source: "fixture:deterministic", InputTokens: i64(in), OutputTokens: i64(out),
		CacheWriteTokens: i64(cacheWrite), CacheReadTokens: i64(cacheRead)}
}

// budgetedClaim submits a budgeted need and claims it for resolver; it returns the claim's fence.
func budgetedClaim(t *testing.T, s *needStore, clk *fakeClock, id string, total int64, parent, resolver string) uint64 {
	t.Helper()
	if _, err := s.submit(budgetOf(clk, id, total, parent)); err != nil {
		t.Fatalf("submit %s: %v", id, err)
	}
	c, err := s.claim(id, resolver, 60_000)
	if err != nil {
		t.Fatalf("claim %s: %v", id, err)
	}
	return c.Claim.Fence
}

func budgetOfNeed(t *testing.T, s *needStore, id string) needBudgetView {
	t.Helper()
	r, err := s.status(id, "", "", "")
	if err != nil || r.Budget == nil {
		t.Fatalf("status %s: %+v %v, want a budget", id, r, err)
	}
	return *r.Budget
}

func TestNeedBudgetConcurrentReservationsNeverOverAllocate(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
	var wg sync.WaitGroup
	var mu sync.Mutex
	won, refused := 0, 0
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			_, err := s.reserve("n1", "res-x", fence, fmt.Sprintf("r%d", g), "deterministic", 30)
			mu.Lock()
			defer mu.Unlock()
			var oe *opError
			switch {
			case err == nil:
				won++
			case errors.As(err, &oe) && oe.status == 409 && strings.Contains(oe.msg, "over budget"):
				refused++
			}
		}(g)
	}
	wg.Wait()
	b := budgetOfNeed(t, s, "n1")
	if won != 3 || refused != 5 || b.CommittedTokens != 90 || b.ReservedTokens != 90 || b.RemainingTokens != 10 || len(b.Reservations) != 3 {
		t.Fatalf("8 x 30 against 100: %d won, %d refused, budget %+v; want 3 won, 5 refused, 90 committed", won, refused, b)
	}
	if b.Enforcement != "accounting" {
		t.Fatalf("enforcement %q, want accounting", b.Enforcement)
	}
}

func TestNeedBudgetReservationDuplicateAndConflict(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
	a, err := s.reserve("n1", "res-x", fence, "r1", "deterministic", 40)
	if err != nil || a.Duplicate {
		t.Fatalf("reserve: %+v %v", a, err)
	}
	b, err := s.reserve("n1", "res-x", fence, "r1", "deterministic", 40)
	if err != nil || !b.Duplicate || b.Rev != a.Rev {
		t.Fatalf("same reservation again: %+v %v, want a duplicate at rev %d", b, err, a.Rev)
	}
	_, err = s.reserve("n1", "res-x", fence, "r1", "deterministic", 41)
	wantStatus(t, err, 409, "another digest")
	if v := budgetOfNeed(t, s, "n1"); v.CommittedTokens != 40 || len(v.Reservations) != 1 {
		t.Fatalf("after a duplicate and a conflict: %+v, want 40 committed once", v)
	}
}

func TestNeedBudgetCrashAroundReserveAndSettle(t *testing.T) {
	s, clk, path := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")

	s.failRename = errors.New("crash at rename")
	_, err := s.reserve("n1", "res-x", fence, "r1", "deterministic", 60)
	wantStatus(t, err, 503, "unknown")
	s = newNeedStore(path, clk.now) // restart: the reservation was not written
	if v := budgetOfNeed(t, s, "n1"); v.CommittedTokens != 0 {
		t.Fatalf("after a reserve lost at rename: %+v, want nothing committed", v)
	}
	if _, err := s.reserve("n1", "res-x", fence, "r1", "deterministic", 60); err != nil {
		t.Fatalf("retry: %v", err)
	}

	s.failAfterRename = errors.New("crash after commit")
	_, err = s.reserve("n1", "res-x", fence, "r2", "deterministic", 30)
	wantStatus(t, err, 503, "unknown")
	s = newNeedStore(path, clk.now) // committed, reply lost
	r, err := s.reserve("n1", "res-x", fence, "r2", "deterministic", 30)
	if err != nil || !r.Duplicate {
		t.Fatalf("retried reserve after a lost reply: %+v %v, want the duplicate", r, err)
	}
	if v := budgetOfNeed(t, s, "n1"); v.CommittedTokens != 90 {
		t.Fatalf("after the retry: %+v, want 90 committed (no double reservation)", v)
	}

	s.failAfterRename = errors.New("crash after commit")
	_, err = s.settle("n1", "res-x", "r1", usageOf(10, 5, 0, 0))
	wantStatus(t, err, 503, "unknown")
	s = newNeedStore(path, clk.now)
	r, err = s.settle("n1", "res-x", "r1", usageOf(10, 5, 0, 0))
	if err != nil || !r.Duplicate {
		t.Fatalf("retried settle after a lost reply: %+v %v, want the duplicate", r, err)
	}
	if v := budgetOfNeed(t, s, "n1"); v.SettledTokens != 15 || v.ReservedTokens != 30 || v.CommittedTokens != 45 {
		t.Fatalf("after the settle: %+v, want 15 settled + 30 reserved, counted once", v)
	}
}

func TestNeedBudgetSurvivesRestartRenewAndRetry(t *testing.T) {
	s, clk, path := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
	big := budgetOf(clk, "n1", 1000, "") // the same submit, deadline included, with a larger total
	if _, err := s.reserve("n1", "res-x", fence, "r1", "deterministic", 60); err != nil {
		t.Fatal(err)
	}
	s = newNeedStore(path, clk.now)
	clk.add(61 * time.Second) // the lease lapses; another resolver claims: a new fence, the same budget
	c, err := s.claim("n1", "res-y", 60_000)
	if err != nil {
		t.Fatal(err)
	}
	_, err = s.reserve("n1", "res-y", c.Claim.Fence, "r2", "deterministic", 50)
	wantStatus(t, err, 409, "over budget")
	_, err = s.reserve("n1", "res-x", fence, "r3", "deterministic", 10)
	wantStatus(t, err, 409, "stale fence")
	if _, err := s.reserve("n1", "res-y", c.Claim.Fence, "r2", "deterministic", 40); err != nil {
		t.Fatalf("reserve inside what is left: %v", err)
	}
	// A larger total under the same id cannot raise the limit.
	_, err = s.submit(big)
	wantStatus(t, err, 409, "another digest")
	if v := budgetOfNeed(t, s, "n1"); v.TotalTokens != 100 || v.CommittedTokens != 100 || v.RemainingTokens != 0 {
		t.Fatalf("after restart, renew and retries: %+v", v)
	}
}

func TestNeedBudgetExhaustionRefusedBeforeEffect(t *testing.T) {
	s, clk, path := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
	if _, err := s.reserve("n1", "res-x", fence, "r1", "deterministic", 100); err != nil {
		t.Fatal(err)
	}
	before, _ := os.ReadFile(path)
	_, err := s.reserve("n1", "res-x", fence, "r2", "deterministic", 1)
	wantStatus(t, err, 409, "over budget")
	for _, bad := range []int64{0, -1, maxNeedTokens + 1} {
		_, err := s.reserve("n1", "res-x", fence, "r3", "deterministic", bad)
		wantStatus(t, err, 400, "tokens")
	}
	_, err = s.reserve("n1", "res-x", fence, "bad id!", "deterministic", 1)
	wantStatus(t, err, 400, "reservation_id")
	if after, _ := os.ReadFile(path); !bytes.Equal(before, after) {
		t.Fatal("a refused reservation touched the file")
	}
	// A need without a budget admits no reservation.
	s.submit(submitOf(clk, "plain", time.Hour))
	c, _ := s.claim("plain", "res-x", 60_000)
	_, err = s.reserve("plain", "res-x", c.Claim.Fence, "r1", "deterministic", 1)
	wantStatus(t, err, 409, "no budget")
	// Bad budgets are refused at submit.
	for _, total := range []int64{0, -5, maxNeedTokens + 1} {
		_, err := s.submit(budgetOf(clk, "bad-total", total, ""))
		wantStatus(t, err, 400, "total_tokens")
	}
}

func TestNeedBudgetChildDrawsFromItsParent(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	pf := budgetedClaim(t, s, clk, "parent", 100, "", "res-p")
	cf := budgetedClaim(t, s, clk, "child-a", 60, "parent", "res-a")
	_, err := s.submit(budgetOf(clk, "child-b", 50, "parent"))
	wantStatus(t, err, 409, "over budget")
	_, err = s.reserve("parent", "res-p", pf, "p1", "deterministic", 41)
	wantStatus(t, err, 409, "over budget")
	if _, err := s.reserve("parent", "res-p", pf, "p1", "deterministic", 40); err != nil {
		t.Fatal(err)
	}
	// The child spends inside its own total; the parent counts the child's total once, or what the
	// child really used when that is more.
	if _, err := s.reserve("child-a", "res-a", cf, "a1", "deterministic", 60); err != nil {
		t.Fatal(err)
	}
	_, err = s.reserve("child-a", "res-a", cf, "a2", "deterministic", 1)
	wantStatus(t, err, 409, "over budget")
	if v := budgetOfNeed(t, s, "parent"); v.CommittedTokens != 100 || v.ChildrenTokens != 60 {
		t.Fatalf("parent with a 60-token child: %+v, want 100 committed, 60 from children", v)
	}
	if _, err := s.settle("child-a", "res-a", "a1", usageOf(50, 20, 0, 0)); err != nil { // 70 used: an overrun
		t.Fatal(err)
	}
	if v := budgetOfNeed(t, s, "child-a"); !v.Overrun || v.CommittedTokens != 70 {
		t.Fatalf("child after an overrun: %+v", v)
	}
	if v := budgetOfNeed(t, s, "parent"); v.CommittedTokens != 110 || !v.Overrun || v.RemainingTokens != 0 {
		t.Fatalf("parent after its child's overrun: %+v, want 110 committed, overrun", v)
	}
	// A parent outside this store, or one without a budget, is refused.
	_, err = s.submit(budgetOf(clk, "orphan", 10, "elsewhere"))
	wantStatus(t, err, 404, "unsupported")
	_, err = s.submit(budgetOf(clk, "self", 10, "self"))
	wantStatus(t, err, 404, "unsupported")
	s.submit(submitOf(clk, "plain", time.Hour))
	_, err = s.submit(budgetOf(clk, "under-plain", 10, "plain"))
	wantStatus(t, err, 409, "no budget")
}

func TestNeedBudgetUnknownUsageKeepsTheReservation(t *testing.T) {
	s, clk, _ := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
	s.reserve("n1", "res-x", fence, "r1", "deterministic", 50)
	unknown := needUsage{Unit: "tokens", Source: "fixture:deterministic", Unknown: true}
	if _, err := s.settle("n1", "res-x", "r1", unknown); err != nil {
		t.Fatal(err)
	}
	if v := budgetOfNeed(t, s, "n1"); v.UnknownTokens != 50 || v.CommittedTokens != 50 || v.RemainingTokens != 50 {
		t.Fatalf("after an unknown usage: %+v, want the 50 still committed as unknown", v)
	}
	if r, err := s.settle("n1", "res-x", "r1", unknown); err != nil || !r.Duplicate {
		t.Fatalf("unknown again: %+v %v, want a duplicate", r, err)
	}
	_, err := s.settle("n1", "res-x", "r1", usageOf(10, 0, 0, 0))
	wantStatus(t, err, 409, "effects")
	_, err = s.settle("n1", "res-y", "r1", usageOf(10, 0, 0, 0))
	wantStatus(t, err, 409, "resolver")
	known := usageOf(10, 0, 0, 0)
	known.EffectsClarified = true
	if _, err := s.settle("n1", "res-x", "r1", known); err != nil {
		t.Fatal(err)
	}
	if v := budgetOfNeed(t, s, "n1"); v.UnknownTokens != 0 || v.SettledTokens != 10 || v.CommittedTokens != 10 {
		t.Fatalf("after the clarified usage: %+v", v)
	}
	_, err = s.settle("n1", "res-x", "r1", unknown)
	wantStatus(t, err, 409, "settled")
}

func TestNeedBudgetUsageCategoriesCountedOnce(t *testing.T) {
	s, clk, path := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
	s.reserve("n1", "res-x", fence, "r1", "deterministic", 50)
	before, _ := os.ReadFile(path)
	bad := map[string]needUsage{}
	u := usageOf(1, 1, 1, 1)
	u.InputTokens = i64(-1)
	bad["negative"] = u
	u = usageOf(1, 1, 1, 1)
	u.CacheReadTokens = nil
	bad["missing category"] = u
	u = usageOf(1, 1, 1, 1)
	u.OutputTokens = i64(maxNeedTokens + 1)
	bad["over bound"] = u
	u = usageOf(1, 1, 1, 1)
	u.Unit = "characters"
	bad["unit"] = u
	u = usageOf(1, 1, 1, 1)
	u.Source = ""
	bad["no source"] = u
	u = usageOf(1, 1, 1, 1)
	u.Unknown = true
	bad["unknown with counts"] = u
	for name, u := range bad {
		_, err := s.settle("n1", "res-x", "r1", u)
		wantStatus(t, err, 400, "")
		if after, _ := os.ReadFile(path); !bytes.Equal(before, after) {
			t.Fatalf("%s: a refused settle touched the file", name)
		}
	}
	if _, err := s.settle("n1", "res-x", "r1", usageOf(10, 5, 3, 2)); err != nil {
		t.Fatal(err)
	}
	if r, err := s.settle("n1", "res-x", "r1", usageOf(10, 5, 3, 2)); err != nil || !r.Duplicate {
		t.Fatalf("the same usage again: %+v %v", r, err)
	}
	_, err := s.settle("n1", "res-x", "r1", usageOf(20, 10, 6, 4)) // a cumulative total re-sent
	wantStatus(t, err, 409, "settled")
	if v := budgetOfNeed(t, s, "n1"); v.SettledTokens != 20 || v.CommittedTokens != 20 {
		t.Fatalf("10+5+3+2 settled: %+v, want 20 counted once", v)
	}
	_, err = s.settle("n1", "res-x", "nope", usageOf(1, 0, 0, 0))
	wantStatus(t, err, 404, "reservation")
}

func TestNeedBudgetModelRunnerIsUnsupported(t *testing.T) {
	s, clk, path := newNeedTest(t)
	fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
	before, _ := os.ReadFile(path)
	for _, runner := range []string{"model", "llm", ""} {
		_, err := s.reserve("n1", "res-x", fence, "r1", runner, 10)
		wantStatus(t, err, 501, "unsupported")
	}
	if after, _ := os.ReadFile(path); !bytes.Equal(before, after) {
		t.Fatal("an unsupported runner touched the file")
	}
}

func TestNeedBudgetCorruptRecordsAreUnknown(t *testing.T) {
	cases := map[string]func(nf *needFile){
		"reservation digest": func(nf *needFile) { nf.Needs[0].Reservations[0].Tokens = 99 },
		"reservation state":  func(nf *needFile) { nf.Needs[0].Reservations[0].State = "free" },
		"settled no usage":   func(nf *needFile) { nf.Needs[0].Reservations[0].State = "settled" },
		"budget digest":      func(nf *needFile) { nf.Needs[0].Budget.TotalTokens = 1000 },
		"missing parent":     func(nf *needFile) { nf.Needs = nf.Needs[1:] },
		"child over parent":  func(nf *needFile) { nf.Needs[1].Budget.TotalTokens = 101 },
	}
	for name, corrupt := range cases {
		t.Run(name, func(t *testing.T) {
			s, clk, path := newNeedTest(t)
			fence := budgetedClaim(t, s, clk, "n1", 100, "", "res-x")
			if _, err := s.reserve("n1", "res-x", fence, "r1", "deterministic", 10); err != nil {
				t.Fatal(err)
			}
			if _, err := s.submit(budgetOf(clk, "kid", 10, "n1")); err != nil {
				t.Fatal(err)
			}
			b, _ := os.ReadFile(path)
			var nf needFile
			if err := json.Unmarshal(b, &nf); err != nil {
				t.Fatal(err)
			}
			corrupt(&nf)
			b, _ = json.Marshal(nf)
			os.WriteFile(path, b, 0o600)
			_, err := newNeedStore(path, clk.now).status("kid", "", "", "")
			wantStatus(t, err, 503, "unknown")
		})
	}
}

func TestNeedBudgetOverTheCLI(t *testing.T) {
	dir := t.TempDir()
	upPlane(t, dir)
	sub := fmt.Sprintf(`{"id":"b1","kind":"capacity","requester":"task-a","incarnation":"i1","body":{"q":1},"deadline_ms":%d,"budget":{"total_tokens":100}}`,
		time.Now().Add(time.Hour).UnixMilli())
	if _, err := needCLI(t, dir, "submit", sub); err != nil {
		t.Fatal(err)
	}
	c, err := needCLI(t, dir, "claim", `{"id":"b1","resolver":"res-x","lease_ms":60000}`)
	if err != nil {
		t.Fatal(err)
	}
	reserve := `{"id":"b1","resolver":"res-x","fence":%d,"reservation_id":"r1","runner":"%s","tokens":10}`
	if _, err := needCLI(t, dir, "reserve", fmt.Sprintf(reserve, c.Claim.Fence, "model")); err == nil ||
		!strings.Contains(err.Error(), "unsupported") {
		t.Fatalf("a model runner over the CLI: %v, want unsupported", err)
	}
	if _, err := needCLI(t, dir, "reserve", fmt.Sprintf(reserve, c.Claim.Fence, "deterministic")); err != nil {
		t.Fatal(err)
	}
	settle := `{"id":"b1","resolver":"res-x","reservation_id":"r1","usage":{"unit":"tokens","source":"fixture","input_tokens":%s,"output_tokens":3,"cache_write_tokens":2,"cache_read_tokens":1}}`
	if _, err := needCLI(t, dir, "settle", fmt.Sprintf(settle, "1.5")); err == nil {
		t.Fatal("a fractional count was accepted")
	}
	r, err := needCLI(t, dir, "settle", fmt.Sprintf(settle, "4"))
	if err != nil || r.Budget == nil || r.Budget.SettledTokens != 10 || r.Budget.Enforcement != "accounting" {
		t.Fatalf("settle over the CLI: %+v %v", r, err)
	}
}
