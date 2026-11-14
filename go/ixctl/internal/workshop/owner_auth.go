package workshop

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"
	"sync"
	"time"
)

type ownerNonce struct {
	expires int64
	used    bool
}
type ownerAuthority struct {
	mu            sync.Mutex
	key           []byte
	origin, epoch string
	nonces        map[string]ownerNonce
	sessions      map[string]time.Time
	now           func() time.Time
}

func newOwnerAuthority(key []byte, origin string) (*ownerAuthority, error) {
	epoch, err := randomID()
	if err != nil {
		return nil, err
	}
	return &ownerAuthority{key: key, origin: origin, epoch: epoch, nonces: map[string]ownerNonce{}, sessions: map[string]time.Time{}, now: time.Now}, nil
}
func (a *ownerAuthority) sign(purpose, nonce string, expires int64, extra ...string) string {
	fields := []string{"ixforge-owner/1", purpose, a.origin, nonce, a.epoch, strconv.FormatInt(expires, 10)}
	fields = append(fields, extra...)
	mac := hmac.New(sha256.New, a.key)
	mac.Write([]byte(strings.Join(fields, "\n")))
	return hex.EncodeToString(mac.Sum(nil))
}
func (a *ownerAuthority) cleanup() {
	now := a.now()
	for nonce, item := range a.nonces {
		if now.Unix() >= item.expires {
			delete(a.nonces, nonce)
		}
	}
	for token, until := range a.sessions {
		if !now.Before(until) {
			delete(a.sessions, token)
		}
	}
}
func (a *ownerAuthority) challenge(nonce string) (map[string]any, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cleanup()
	raw, err := hex.DecodeString(nonce)
	if err != nil || len(raw) != 32 {
		return nil, ErrInvalid
	}
	item, exists := a.nonces[nonce]
	if exists && item.used {
		return nil, ErrInvalid
	}
	if !exists {
		if len(a.nonces) >= 128 {
			return nil, ErrInvalid
		}
		item = ownerNonce{expires: a.now().Add(time.Minute).Unix()}
		a.nonces[nonce] = item
	}
	return map[string]any{"epoch": a.epoch, "expires": item.expires, "proof": a.sign("challenge", nonce, item.expires)}, nil
}
func (a *ownerAuthority) session(nonce, epoch string, expires int64, proof string) (map[string]any, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cleanup()
	item, ok := a.nonces[nonce]
	if !ok || item.used || item.expires != expires || epoch != a.epoch {
		return nil, ErrUnauthorized
	}
	expected, _ := hex.DecodeString(a.sign("session", nonce, expires))
	supplied, err := hex.DecodeString(proof)
	if err != nil || !hmac.Equal(supplied, expected) {
		return nil, ErrUnauthorized
	}
	if len(a.sessions) >= 128 {
		return nil, ErrInvalid
	}
	token, err := randomID()
	if err != nil {
		return nil, err
	}
	item.used = true
	a.nonces[nonce] = item
	until := a.now().Add(15 * time.Minute)
	a.sessions[token] = until
	return map[string]any{"session": token, "expires": until.Unix(), "proof": a.sign("receipt", nonce, expires, token, strconv.FormatInt(until.Unix(), 10))}, nil
}
func (a *ownerAuthority) valid(token string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cleanup()
	until, ok := a.sessions[token]
	return ok && a.now().Before(until)
}
