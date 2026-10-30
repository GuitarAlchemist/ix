package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const secret = "5ec2e75ec2e75ec2e75ec2e75ec2e75ec2e75ec2e75ec2e75ec2e75ec2e75ec2"

// (a) The token never leaves for a name or a non-loopback address: DialWS
// refuses before any connection is attempted, and says nothing of the token.
func TestDialWSRefusesNonLoopback(t *testing.T) {
	tr := loopbackHTTP.Transport.(*http.Transport)
	var dials atomic.Int32
	saved := tr.DialContext
	tr.DialContext = func(context.Context, string, string) (net.Conn, error) {
		dials.Add(1)
		return nil, errors.New("dial attempted")
	}
	defer func() { tr.DialContext = saved }()

	for _, addr := range []string{"example.com:80", "10.0.0.1:80", "localhost:80", "0.0.0.0:80", "127.0.0.1", "[::1]", ""} {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err := DialWS(ctx, addr, secret)
		cancel()
		if err == nil || !strings.Contains(err.Error(), "loopback") && !strings.Contains(err.Error(), "address") {
			t.Errorf("%q: %v", addr, err)
		}
		if err != nil && strings.Contains(err.Error(), secret) {
			t.Errorf("%q: the error carries the token", addr)
		}
	}
	if n := dials.Load(); n != 0 {
		t.Fatalf("%d connection attempts", n)
	}
}

// (b) A redirect is never followed: a 307 from /v1/ws to another loopback
// listener would otherwise carry the Authorization header there.
func TestDialWSDoesNotFollowRedirects(t *testing.T) {
	var second atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		second.Add(1)
		w.WriteHeader(http.StatusTeapot)
	}))
	defer other.Close()
	var first atomic.Int32
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer "+secret {
			first.Add(1)
		}
		http.Redirect(w, r, other.URL+"/v1/ws", http.StatusTemporaryRedirect)
	}))
	defer redirector.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := DialWS(ctx, strings.TrimPrefix(redirector.URL, "http://"), secret)
	if err == nil {
		t.Fatal("DialWS followed the redirect and succeeded")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatal("the error carries the token")
	}
	if first.Load() != 1 {
		t.Fatalf("the first listener saw %d authorized requests", first.Load())
	}
	if n := second.Load(); n != 0 {
		t.Fatalf("the redirect target saw %d requests", n)
	}

	// The plain HTTP client refuses to follow too.
	req, _ := http.NewRequest("POST", redirector.URL+"/v1/shutdown", nil)
	req.Header.Set("Authorization", "Bearer "+secret)
	resp, err := loopbackHTTP.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusTemporaryRedirect || second.Load() != 0 {
		t.Fatalf("HTTP client: status %d, target saw %d", resp.StatusCode, second.Load())
	}
}
