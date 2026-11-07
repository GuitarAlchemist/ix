package server

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
)

// The hello challenge: a client that holds the token proves the server holds it too before it
// sends it. POST /v1/hello {nonce} needs no token; it answers {mac}, the HMAC-SHA-256 of the
// nonce's 32 bytes under a key derived from the token in its own domain. That key signs nothing
// else, and the journal cursors are signed by another key (Server.key, random per process): a mac
// of /v1/hello is never a signature of anything else.
const (
	HelloDomain     = "ixmux/hello/v1"
	HelloNonceBytes = 32
	helloMaxBody    = 1 << 10
)

// HelloKey derives the key of the hello challenge from the token.
func HelloKey(token []byte) []byte {
	m := hmac.New(sha256.New, token)
	m.Write([]byte(HelloDomain))
	return m.Sum(nil)
}

// HelloMAC is the answer to nonce (its bytes) of a server that holds token.
func HelloMAC(token, nonce []byte) []byte {
	m := hmac.New(sha256.New, HelloKey(token))
	m.Write(nonce)
	return m.Sum(nil)
}

// hello answers the challenge. It keeps the checks of every route but the token: a loopback Host,
// no Origin, a bounded body; the listener's bound on connections holds as for any request.
// @ai:invariant /v1/hello answers only the mac of a 32-byte hex nonce, under the derived hello key [T:test conf:0.85 src:go/ixmux/cmd/ixmux/s6e_test.go:TestHelloEndpoint]
func (s *Server) hello(w http.ResponseWriter, r *http.Request) {
	switch {
	case !loopbackHost(r.Host):
		refuse(w, http.StatusForbidden, "host-not-loopback", "")
		return
	case r.Header.Get("Origin") != "":
		refuse(w, http.StatusForbidden, "origin-refused", "")
		return
	}
	var req struct {
		Nonce string `json:"nonce"`
	}
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, helloMaxBody))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		refuse(w, http.StatusBadRequest, "bad-nonce", "the body is {\"nonce\": <64 hex digits>}")
		return
	}
	nonce, err := hex.DecodeString(req.Nonce)
	if err != nil || len(nonce) != HelloNonceBytes || hex.EncodeToString(nonce) != req.Nonce {
		refuse(w, http.StatusBadRequest, "bad-nonce", "the nonce is 32 bytes in lower-case hex")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"mac": hex.EncodeToString(HelloMAC(s.token, nonce))})
}
