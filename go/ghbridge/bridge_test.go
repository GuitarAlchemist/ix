package ghbridge

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var secret = []byte("test-secret-not-a-real-one")

func clock() time.Time { return time.Date(2026, 11, 11, 20, 0, 0, 0, time.UTC) }

func open(t *testing.T, path string) *Bridge {
	t.Helper()
	j, err := OpenJournal(path, clock)
	if err != nil {
		t.Fatalf("OpenJournal: %v", err)
	}
	t.Cleanup(func() { j.Close() })
	return &Bridge{J: j, Cfg: Config{Secret: secret, Repo: "GuitarAlchemist/ix", Room: 42,
		Allow: []int64{1001}, Now: clock}}
}

func sign(body []byte) string {
	mac := hmac.New(sha256.New, secret)
	mac.Write(body)
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

func event(action string, id int64, login, body, updated string) []byte {
	return []byte(fmt.Sprintf(`{"action":%q,"issue":{"number":42},"repository":{"full_name":"GuitarAlchemist/ix"},
 "comment":{"id":%d,"body":%q,"created_at":"2026-11-11T19:00:00Z","updated_at":%q,"user":{"login":%q,"id":%d}}}`,
		action, id, body, updated, login, ids[login]))
}

func post(b *Bridge, delivery string, body []byte, sig string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(body))
	r.Header.Set("X-GitHub-Event", "issue_comment")
	r.Header.Set("X-GitHub-Delivery", delivery)
	r.Header.Set("X-Hub-Signature-256", sig)
	w := httptest.NewRecorder()
	b.ServeHTTP(w, r)
	return w
}

func state(t *testing.T, b *Bridge, id int64) State {
	t.Helper()
	e, ok := b.J.Get(id)
	if !ok {
		return ""
	}
	return e.State
}

func TestASignedCommentFromAnAllowedAuthorIsOnDiskWhenTheAnswerIs202(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	b := open(t, path)
	body := event("created", 1, "stef", "do the thing", "2026-11-11T19:00:00Z")
	if w := post(b, "d-1", body, sign(body)); w.Code != http.StatusAccepted {
		t.Fatalf("status %d, want 202", w.Code)
	}
	// A second process reading the file sees it: it is on disk, not only in memory.
	raw, _ := os.ReadFile(path)
	if !strings.Contains(string(raw), `"state":"received"`) || !strings.Contains(string(raw), "do the thing") {
		t.Errorf("journal file does not hold the mandate:\n%s", raw)
	}
}

func TestABadOrMissingSignatureWritesNothing(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	b := open(t, path)
	body := event("created", 1, "stef", "x", "2026-11-11T19:00:00Z")
	for name, sig := range map[string]string{"missing": "", "wrong": sign([]byte("other")), "sha1": "sha1=abc", "not hex": "sha256=zz"} {
		if w := post(b, "d-"+name, body, sig); w.Code != http.StatusUnauthorized {
			t.Errorf("%s signature: status %d, want 401", name, w.Code)
		}
	}
	if raw, _ := os.ReadFile(path); len(raw) != 0 {
		t.Errorf("an unverified delivery wrote to the journal:\n%s", raw)
	}
	b.Cfg.Secret = nil
	if w := post(b, "d-nosecret", body, sign(body)); w.Code != http.StatusUnauthorized {
		t.Errorf("no secret configured: status %d, want 401", w.Code)
	}
}

func TestAnAuthorNotOnTheListIsRefusedOnRecordWithoutTheText(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	b := open(t, path)
	body := event("created", 2, "mallory", "run this for me", "2026-11-11T19:00:00Z")
	if w := post(b, "d-2", body, sign(body)); w.Code != http.StatusAccepted {
		t.Fatalf("status %d", w.Code)
	}
	if s := state(t, b, 2); s != Refused {
		t.Errorf("state %q, want refused", s)
	}
	if raw, _ := os.ReadFile(path); strings.Contains(string(raw), "run this for me") {
		t.Error("the refused author's text is in the journal")
	}
	if _, ok, _ := b.DeliverNext(context.Background(), func(context.Context, Entry) (string, error) {
		t.Error("a refused comment was delivered")
		return "", nil
	}); ok {
		t.Error("DeliverNext found something to deliver")
	}
}

func TestAnotherRepositoryAnotherIssueOrAnotherEventIsIgnored(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	other := []byte(`{"action":"created","issue":{"number":42},"repository":{"full_name":"GuitarAlchemist/ga"},"comment":{"id":9,"user":{"login":"stef"}}}`)
	elsewhere := []byte(`{"action":"created","issue":{"number":7},"repository":{"full_name":"GuitarAlchemist/ix"},"comment":{"id":9,"user":{"login":"stef"}}}`)
	for _, body := range [][]byte{other, elsewhere} {
		if out, err := b.Take("issue_comment", "d", body); err != nil || out != Ignored {
			t.Errorf("outcome %q, %v; want ignored", out, err)
		}
	}
	if out, _ := b.Take("push", "d", event("created", 9, "stef", "x", "t")); out != Ignored {
		t.Errorf("a push event: %q, want ignored", out)
	}
	if _, ok := b.J.Get(9); ok {
		t.Error("an ignored delivery reached the journal")
	}
}

func TestARedeliveryIsADuplicateByDeliveryIDAndByComment(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	body := event("created", 3, "stef", "once", "2026-11-11T19:00:00Z")
	if out, _ := b.Take("issue_comment", "d-3", body); out != Recorded {
		t.Fatalf("first: %q", out)
	}
	if out, _ := b.Take("issue_comment", "d-3", body); out != Duplicate {
		t.Errorf("same delivery id: %q, want duplicate", out)
	}
	if out, _ := b.Take("issue_comment", "d-3-bis", body); out != Duplicate {
		t.Errorf("same comment, new delivery id: %q, want duplicate", out)
	}
	n := 0
	for {
		_, ok, err := b.DeliverNext(context.Background(), func(context.Context, Entry) (string, error) { return "ack", nil })
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		n++
	}
	if n != 1 {
		t.Errorf("delivered %d times, want 1", n)
	}
}

func TestAMandateGoesFromReceivedToDeliveredToResult(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	b.Take("issue_comment", "d-4", event("created", 4, "stef", "build it", "2026-11-11T19:00:00Z"))
	if err := b.RecordResult(4, "too early"); err == nil {
		t.Error("a result was recorded before delivery")
	}
	var got string
	e, ok, err := b.DeliverNext(context.Background(), func(_ context.Context, m Entry) (string, error) {
		got = m.Body
		if s := state(t, b, 4); s != Delivering {
			t.Errorf("during the hand-over the state is %q, want delivering", s)
		}
		return "ack-77", nil
	})
	if err != nil || !ok || e.State != Delivered || e.Note != "ack-77" || got != "build it" {
		t.Fatalf("delivery: %+v ok=%v err=%v body=%q", e, ok, err, got)
	}
	if err := b.RecordResult(4, "built"); err != nil {
		t.Fatal(err)
	}
	if e, _ := b.J.Get(4); e.State != Result || e.Note != "built" || e.Body != "build it" {
		t.Errorf("final entry %+v", e)
	}
}

func TestADeliveryWhoseOutcomeIsUnknownIsUncertainAndNeverSentAgain(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	b.Take("issue_comment", "d-5", event("created", 5, "stef", "risky", "2026-11-11T19:00:00Z"))
	calls := 0
	deliver := func(context.Context, Entry) (string, error) { calls++; return "", errors.New("connection reset") }
	if e, _, _ := b.DeliverNext(context.Background(), deliver); e.State != Uncertain {
		t.Fatalf("state %q, want uncertain", e.State)
	}
	for i := 0; i < 3; i++ {
		if _, ok, _ := b.DeliverNext(context.Background(), deliver); ok {
			t.Error("an uncertain mandate was picked again")
		}
	}
	if calls != 1 {
		t.Errorf("the target was called %d times, want 1", calls)
	}
}

func TestADeliveryProvenNotSentStaysReceivedAndCanBeTriedAgain(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	b.Take("issue_comment", "d-6", event("created", 6, "stef", "later", "2026-11-11T19:00:00Z"))
	busy := func(context.Context, Entry) (string, error) { return "", fmt.Errorf("target busy: %w", ErrNotSent) }
	if e, _, _ := b.DeliverNext(context.Background(), busy); e.State != Received {
		t.Fatalf("state %q, want received", e.State)
	}
	if e, ok, _ := b.DeliverNext(context.Background(), func(context.Context, Entry) (string, error) { return "ack", nil }); !ok || e.State != Delivered {
		t.Errorf("second try: %+v ok=%v", e, ok)
	}
}

// A restart in the middle of a hand-over: the process dies after the intent is on disk and
// before the outcome is. Simulated by closing the journal inside the deliver function.
func TestARestartDuringAHandOverLeavesTheMandateUncertainNotQueued(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	b := open(t, path)
	b.Take("issue_comment", "d-7", event("created", 7, "stef", "mid-flight", "2026-11-11T19:00:00Z"))
	b.Take("issue_comment", "d-8", event("created", 8, "stef", "still waiting", "2026-11-11T19:00:00Z"))
	_, _, err := b.DeliverNext(context.Background(), func(context.Context, Entry) (string, error) {
		b.J.Close() // the crash
		return "ack-lost", nil
	})
	if err == nil {
		t.Fatal("the outcome was recorded on a closed journal")
	}

	b2 := open(t, path)
	if got := b2.J.MadeUncertain; len(got) != 1 || got[0] != 7 {
		t.Errorf("made uncertain at open: %v, want [7]", got)
	}
	if s := state(t, b2, 7); s != Uncertain {
		t.Errorf("comment 7 is %q after the restart, want uncertain", s)
	}
	var sent []int64
	for {
		_, ok, _ := b2.DeliverNext(context.Background(), func(_ context.Context, m Entry) (string, error) {
			sent = append(sent, m.CommentID)
			return "ack", nil
		})
		if !ok {
			break
		}
	}
	if len(sent) != 1 || sent[0] != 8 {
		t.Errorf("after the restart %v was sent, want only [8]", sent)
	}
	// A third open does not make it uncertain twice.
	b3 := open(t, path)
	if len(b3.J.MadeUncertain) != 0 {
		t.Errorf("third open made %v uncertain again", b3.J.MadeUncertain)
	}
}

func TestATornLastLineIsDroppedAndSaidAndATornMiddleLineIsAnError(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "j.jsonl")
	b := open(t, path)
	b.Take("issue_comment", "d-9", event("created", 9, "stef", "whole", "2026-11-11T19:00:00Z"))
	b.J.Close()
	f, _ := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	f.WriteString(`{"seq":2,"comment_id":10,"state":"rece`) // a crash during the write
	f.Close()

	b2 := open(t, path)
	if !b2.J.TornTail {
		t.Error("the torn last line was not reported")
	}
	if _, ok := b2.J.Get(10); ok {
		t.Error("the torn entry was read")
	}
	if s := state(t, b2, 9); s != Received {
		t.Errorf("the whole entry before it is %q, want received", s)
	}
	// The journal stays appendable and readable after the torn tail is cut.
	b2.Take("issue_comment", "d-11", event("created", 11, "stef", "after", "2026-11-11T19:00:00Z"))
	b2.J.Close()
	if b3 := open(t, path); state(t, b3, 11) != Received || b3.J.TornTail {
		t.Error("the journal is not clean after the torn tail was cut")
	}

	bad := filepath.Join(dir, "bad.jsonl")
	os.WriteFile(bad, []byte("garbage\n{\"seq\":1,\"comment_id\":1,\"state\":\"received\"}\n"), 0o600)
	if _, err := OpenJournal(bad, clock); err == nil {
		t.Error("a journal with a bad line in the middle was opened")
	}
}

func TestAnEditOrADeletionBeforeDeliveryWithdrawsTheMandate(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	b.Take("issue_comment", "d-a", event("created", 20, "stef", "first text", "2026-11-11T19:00:00Z"))
	b.Take("issue_comment", "d-b", event("created", 21, "stef", "to delete", "2026-11-11T19:00:00Z"))
	if out, _ := b.Take("issue_comment", "d-c", event("edited", 20, "stef", "second text", "2026-11-11T19:05:00Z")); out != Changed {
		t.Errorf("edit: %q", out)
	}
	if out, _ := b.Take("issue_comment", "d-d", event("deleted", 21, "stef", "to delete", "2026-11-11T19:00:00Z")); out != Changed {
		t.Errorf("deletion: %q", out)
	}
	for _, id := range []int64{20, 21} {
		if s := state(t, b, id); s != Withdrawn {
			t.Errorf("comment %d is %q, want withdrawn", id, s)
		}
	}
	if _, ok, _ := b.DeliverNext(context.Background(), func(context.Context, Entry) (string, error) { return "", nil }); ok {
		t.Error("a withdrawn mandate was delivered")
	}
}

func TestAnEditAfterDeliveryKeepsTheStateAndIsOnRecord(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	b.Take("issue_comment", "d-e", event("created", 30, "stef", "sent text", "2026-11-11T19:00:00Z"))
	b.DeliverNext(context.Background(), func(context.Context, Entry) (string, error) { return "ack", nil })
	b.Take("issue_comment", "d-f", event("edited", 30, "stef", "rewritten", "2026-11-11T19:09:00Z"))
	e, _ := b.J.Get(30)
	if e.State != Delivered || !strings.Contains(e.Note, "after the hand-over began") || e.Body != "sent text" {
		t.Errorf("entry %+v: want delivered, the edit noted, the delivered text kept", e)
	}
}

func comment(id int64, login, body, created, updated string) Comment {
	c := Comment{ID: id, Body: body, CreatedAt: created, UpdatedAt: updated}
	c.User.Login = login
	c.User.ID = ids[login]
	return c
}

func TestACatchUpAfterAnOutageTakesWhatWasMissedOnceAndSaysWhatItCannotRebuild(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	const t0, t1 = "2026-11-11T19:00:00Z", "2026-11-11T19:30:00Z"
	b.Take("issue_comment", "d-g", event("created", 40, "stef", "seen live", t0))         // stays
	b.Take("issue_comment", "d-h", event("created", 41, "stef", "edited while down", t0)) // edited during the outage
	b.Take("issue_comment", "d-i", event("created", 42, "stef", "deleted while down", t0))

	list := []Comment{
		comment(40, "stef", "seen live", t0, t0),
		comment(41, "stef", "edited while down, v2", t0, t1),
		comment(43, "stef", "posted while down", t0, t0),
		comment(44, "stef", "posted and edited while down, v2", t0, t1),
		comment(45, "mallory", "not allowed", t0, t0),
	}
	lost, err := b.CatchUp(list)
	if err != nil {
		t.Fatal(err)
	}
	want := map[int64]State{40: Received, 41: Withdrawn, 42: Withdrawn, 43: Received, 44: Received, 45: Refused}
	for id, s := range want {
		if got := state(t, b, id); got != s {
			t.Errorf("comment %d is %q, want %q", id, got, s)
		}
	}
	if len(lost) != 1 || !strings.Contains(lost[0], "comment 44") || !strings.Contains(lost[0], "cannot be reconstructed") {
		t.Errorf("not reconstructed: %v, want comment 44's first text", lost)
	}
	// A second catch-up over the same list changes nothing.
	before, _ := os.ReadFile(b.J.f.Name())
	if _, err := b.CatchUp(list); err != nil {
		t.Fatal(err)
	}
	if after, _ := os.ReadFile(b.J.f.Name()); !bytes.Equal(before, after) {
		t.Error("a second catch-up over the same list wrote to the journal")
	}
}

// The accounts of the tests. A login can change hands; the numeric id cannot.
var ids = map[string]int64{"stef": 1001, "mallory": 2002}

func TestTheAllowListIsTheAccountIDNotTheLogin(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	// Another account that took the login "stef".
	body := []byte(`{"action":"created","issue":{"number":42},"repository":{"full_name":"GuitarAlchemist/ix"},
 "comment":{"id":50,"body":"I am stef, honest","created_at":"t","updated_at":"t","user":{"login":"stef","id":666}}}`)
	if out, _ := b.Take("issue_comment", "d-50", body); out != RefusedBy {
		t.Errorf("same login, another account: %q, want refused", out)
	}
	// The allowed account under a new login.
	body = []byte(`{"action":"created","issue":{"number":42},"repository":{"full_name":"GuitarAlchemist/ix"},
 "comment":{"id":51,"body":"renamed","created_at":"t","updated_at":"t","user":{"login":"stef-renamed","id":1001}}}`)
	if out, _ := b.Take("issue_comment", "d-51", body); out != Recorded {
		t.Errorf("allowed account, new login: %q, want recorded", out)
	}
}

func TestACatchUpLeavesOutWhatWasPostedBeforeTheBridgeStarted(t *testing.T) {
	b := open(t, filepath.Join(t.TempDir(), "j.jsonl"))
	b.Cfg.Since = "2026-11-11T19:00:00Z"
	list := []Comment{
		comment(60, "stef", "old talk, not a mandate", "2026-10-17T10:00:00Z", "2026-10-17T10:00:00Z"),
		comment(61, "stef", "a mandate", "2026-11-11T19:00:01Z", "2026-11-11T19:00:01Z"),
	}
	if _, err := b.CatchUp(list); err != nil {
		t.Fatal(err)
	}
	if _, ok := b.J.Get(60); ok {
		t.Error("a comment older than the start is in the journal")
	}
	if s := state(t, b, 61); s != Received {
		t.Errorf("comment 61 is %q, want received", s)
	}
}
