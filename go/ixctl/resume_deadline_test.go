package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// The health probe's deadline (GAIA-IX-WINSVC-DEADLINE-R6). Fake transports only: no socket.
// Each stalls the way a real server can (no headers, or headers then no body) and wakes only when
// the request's context ends, as the real transport does.

// stalledBody yields its prefix, then blocks until ctx ends.
type stalledBody struct {
	ctx    context.Context
	prefix *strings.Reader
}

func (b *stalledBody) Read(p []byte) (int, error) {
	if b.prefix.Len() > 0 {
		return b.prefix.Read(p)
	}
	<-b.ctx.Done()
	return 0, b.ctx.Err()
}

func (b *stalledBody) Close() error { return nil }

var (
	stallHeaders = roundTrip(func(r *http.Request) (*http.Response, error) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	stallBody = roundTrip(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Request: r, Header: http.Header{},
			Body: &stalledBody{ctx: r.Context(), prefix: strings.NewReader(`{"name":"IxForge"`)}}, nil
	})
)

// probeWithin runs one probe and fails the test if it outlives guard: the deadline is the
// contract, so a probe that does not return is the defect, not a slow test.
func probeWithin(t *testing.T, probe func(u, host string) ([]byte, error), guard time.Duration) ([]byte, error, time.Duration) {
	t.Helper()
	type result struct {
		body []byte
		err  error
	}
	done := make(chan result, 1)
	start := time.Now()
	go func() {
		b, err := probe("http://127.0.0.1:8761/workshop/branding.json", "")
		done <- result{b, err}
	}()
	select {
	case r := <-done:
		return r.body, r.err, time.Since(start)
	case <-time.After(guard):
		t.Fatalf("the health probe did not return within %v: no deadline", guard)
		return nil, nil, 0
	}
}

// The documented contract: the probes the check really uses give up after 2 s.
func TestResumeHealthDeadlineIsTwoSeconds(t *testing.T) {
	if resumeHealthTimeout != 2*time.Second {
		t.Fatalf("resumeHealthTimeout = %v, the documented contract is 2 s", resumeHealthTimeout)
	}
	_, err, took := probeWithin(t, realResumeProbesOver(stallHeaders).health, 4*time.Second)
	if err == nil || took < resumeHealthTimeout {
		t.Fatalf("stalled headers: %v after %v, want an error after 2 s", err, took)
	}
}

// A stall before the headers, or after them inside the body, ends in an error within the
// deadline, and the partial body is never handed back as health.
func TestResumeHealthDeadlineCoversHeadersAndBody(t *testing.T) {
	for _, c := range []struct {
		name string
		rt   http.RoundTripper
	}{{"headers", stallHeaders}, {"body", stallBody}} {
		_, err, took := probeWithin(t, healthProbe(c.rt, 50*time.Millisecond), time.Second)
		if err == nil || !errors.Is(err, context.DeadlineExceeded) && !strings.Contains(err.Error(), "Client.Timeout") {
			t.Errorf("%s stall: %v after %v, want a deadline error", c.name, err, took)
		}
	}
}

// A cancelled request or a transport error is down, never ok: even when the partial body already
// holds health_expect's text.
func TestResumeHealthStallOrErrorIsNeverUp(t *testing.T) {
	inv := mustInventory(t, `{"components":[{"name":"workshop","health":"http://127.0.0.1:8761/workshop/branding.json","health_expect":"\"name\":\"IxForge\""}]}`)
	for _, c := range []struct {
		name string
		rt   http.RoundTripper
	}{
		{"body stall after the expected text", stallBody},
		{"cancelled", roundTrip(func(*http.Request) (*http.Response, error) { return nil, context.Canceled })},
		{"transport error", roundTrip(func(*http.Request) (*http.Response, error) { return nil, io.ErrUnexpectedEOF })},
	} {
		r := checkResume(inv, resumeProbes{health: healthProbe(c.rt, 50*time.Millisecond)})[0]
		if r.Verdict != verdictDown || !strings.Contains(resumeChecks(r), "health http://127.0.0.1:8761/workshop/branding.json: ") {
			t.Errorf("%s: %s %q, want down with the probe error", c.name, r.Verdict, r.Checks)
		}
	}
}
