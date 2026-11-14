package workshop

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestOwnerSessionIsRuntimeBoundSingleUseAndExpiring(t *testing.T) {
	key := []byte(strings.Repeat("x", 32))
	origin := "http://127.0.0.1:12345"
	a, err := newOwnerAuthority(key, origin)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 11, 5, 0, 0, 0, 0, time.UTC)
	a.now = func() time.Time { return now }
	nonce := strings.Repeat("b", 64)
	challenge, err := a.challenge(nonce)
	if err != nil {
		t.Fatal(err)
	}
	epoch := challenge["epoch"].(string)
	expires := challenge["expires"].(int64)
	sign := func(purpose string) string {
		mac := hmac.New(sha256.New, key)
		mac.Write([]byte(strings.Join([]string{"ixforge-owner/1", purpose, origin, nonce, epoch, strconv.FormatInt(expires, 10)}, "\n")))
		return hex.EncodeToString(mac.Sum(nil))
	}
	if challenge["proof"] != sign("challenge") {
		t.Fatal("challenge protocol mismatch")
	}
	if _, err = a.session(nonce, epoch, expires, sign("challenge")); err == nil {
		t.Fatal("challenge proof granted owner authority")
	}
	session, err := a.session(nonce, epoch, expires, sign("session"))
	if err != nil {
		t.Fatal(err)
	}
	token := session["session"].(string)
	if !a.valid(token) {
		t.Fatal("session not active")
	}
	if a.valid(hex.EncodeToString(key)) {
		t.Fatal("persistent owner key accepted as bearer")
	}
	if _, err = a.session(nonce, epoch, expires, sign("session")); err == nil {
		t.Fatal("session bootstrap replayed")
	}
	if _, err = a.challenge(nonce); err == nil {
		t.Fatal("used nonce resurrected")
	}
	restarted, _ := newOwnerAuthority(key, origin)
	if restarted.valid(token) {
		t.Fatal("session survived process epoch")
	}
	if _, err = restarted.session(nonce, epoch, expires, sign("session")); err == nil {
		t.Fatal("captured bootstrap survived epoch")
	}
	now = now.Add(16 * time.Minute)
	if a.valid(token) {
		t.Fatal("expired owner session accepted")
	}
	if _, err = a.session(nonce, epoch, expires, sign("session")); err == nil {
		t.Fatal("old expired proof accepted")
	}
	if _, err = a.challenge(nonce); err != nil {
		t.Fatal(err)
	}
	if _, err = a.session(nonce, epoch, expires, sign("session")); err == nil {
		t.Fatal("reissued nonce accepted old deadline proof")
	}
}
